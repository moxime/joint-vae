import os
from contextlib import contextmanager
import logging
import time

import torch
from cvae import ClassificationVariationalNetwork as M
from module.priors import build_prior

from utils.save_load import MissingKeys, save_json, load_json
import utils.torch_load as torchdl
from utils.print_log import EpochOutput


class WIMVariationalNetwork(M):

    def __init__(self, *a, alternate_prior=None, **kw):

        super().__init__(*a, **kw)
        self._original_prior = self.encoder.prior

        for p in self._original_prior.parameters():
            p.requires_grad_(False)

        self._alternate_prior = None
        if alternate_prior is not None:
            self.set_alternate_prior(alternate_prior)

        self._is_alternate_prior = False

    @classmethod
    def is_wim(cls, d):
        return os.path.exists(os.path.join(d, 'wim.json'))

    @property
    def is_alternate_prior(self):
        return self._is_alternate_prior

    @property
    def is_original_prior(self):
        return not self.is_alternate_prior

    def _switch_to_alternate_prior(self, b):
        if b:
            if self._alternate_prior is None:
                raise AttributeError('Model still not has alternat prior')
            self.encoder.prior = self._alternate_prior
            logging.debug('Switching to alternate prior: {}'.format(self.encoder.prior))
            self._is_alternate_prior = True
        else:
            self.encoder.prior = self._original_prior
            logging.debug('Switching to original prior: {}'.format(self.encoder.prior))
            self._is_alternate_prior = False
        return self.encoder.prior

    @property
    @contextmanager
    def original_prior(self):
        state = self.is_original_prior
        try:
            yield self._switch_to_alternate_prior(False)
        finally:
            self.original_prior = state

    @property
    @contextmanager
    def alternate_prior(self):
        state = self.is_alternate_prior
        try:
            yield self._switch_to_alternate_prior(True)
        finally:
            self.alternate_prior = state

    @alternate_prior.setter
    def alternate_prior(self, b):
        self._switch_to_alternate_prior(b)

    @original_prior.setter
    def original_prior(self, b):
        self.alternate_prior = not b

    def set_alternate_prior(self, p):

        logging.debug('Setting alternate prior')
        assert self._alternate_prior is None
        self._alternate_prior = build_prior(**p)
        self.wim_params = p.copy()

        for p in self._alternate_prior.parameters():
            p.requires_grad_(False)

    @classmethod
    def load(cls, dir_name, load_net=True, **kw):

        try:
            model = super().load(dir_name, strict=False, load_net=load_net, **kw)
        except MissingKeys as e:
            logging.debug('Model loaded has been detected as not wim')
            logging.debug('Missing keys: {}'.format(', '.join(e.args[-1])))
            model = e.args[0]
            s = e.args[1]  # state_dict
            logging.debug('Creating fake params prior means')
            s['_original_prior.mean'] = torch.clone(s['encoder.prior.mean'])
            s['_original_prior._var_parameter'] = torch.clone(s['encoder.prior._var_parameter'])

            model.load_state_dict(s)

            logging.debug('Reset results')
            model.ood_results = {}

        try:
            wim_params = load_json(dir_name, 'wim.json')
            logging.debug('Model was already a wim')
            alternate_prior_params = wim_params.copy()
            for k in ('sets', 'alpha', 'epochs', 'from'):
                k, alternate_prior_params.pop(k, None)
            if load_net:
                model.set_alternate_prior(alternate_prior_params)
            model.wim_params = wim_params

        except FileNotFoundError:
            logging.debug('Model loaded has been detected as not wim')
            logging.debug('Reset results')
            model.ood_results = {}

        return model

    def save(self, *a, **kw):

        with self.original_prior:
            dir_name = super().save(*a, **kw)
        save_json(self.wim_params, dir_name, 'wim.json')
        return dir_name

    def finetune(self, *sets,
                 epochs=5, alpha=0.1,
                 test_batch_size=8192,
                 optimizer=None,
                 outputs=EpochOutput(),
                 ):

        if optimizer is None:
            optimizer = self.optimizer

        logging.debug('Learning rate: {}'.format(optimizer.lr))

        self.wim_params['sets'] = sets
        self.wim_params['alpha'] = alpha
        self.wim_params['epochs'] = epochs

        for p in self._alternate_prior.parameters():
            assert not p.requires_grad, 'prior parameter queires grad'

        for p in self._original_prior.parameters():
            assert not p.requires_grad, 'prior parameter queires grad'

        max_batch_sizes = self.max_batch_sizes

        test_batch_size = min(max_batch_sizes['test'], test_batch_size)

        _s = 'Test batch size wanted {} / max {}'
        logging.info(_s.format(test_batch_size, max_batch_sizes['test']))

        set_name = self.training_parameters['set']
        transformer = self.training_parameters['transformer']
        data_augmentation = self.training_parameters['data_augmentation']
        batch_size = self.training_parameters['batch_size']

        logging.info('Finetune batch size = {}'.format(batch_size))

        trainset, testset = torchdl.get_dataset(set_name,
                                                transformer=transformer,
                                                data_augmentation=data_augmentation)

        moving_sets = {_: torchdl.get_dataset(_, transformer=transformer, splits=['test'])[1] for _ in sets}
        moving_sets['test'] = testset

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  # pin_memory=True,
                                                  shuffle=True,
                                                  num_workers=0)

        moving_loaders = {_: torch.utils.data.DataLoader(moving_sets[_],
                                                         batch_size=batch_size,
                                                         # pin_memory=True,
                                                         shuffle=True,
                                                         num_workers=0)
                          for _ in moving_sets}

        current_measures = {}
        moving_current_measures_on_alternate = {}
        moving_current_measures_on_original = {}

        device = next(self.parameters()).device

        self.eval()
        sample_dirs = [os.path.join(self.saved_dir, 'samples', '{:04d}'.format(self.trained), 'init')]
        for d in sample_dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                assert os.path.isdir(d), '{} exists and is not a dir'.format(d)

        self.original_prior = True
        with torch.no_grad():
            self.ood_detection_rates(batch_size=test_batch_size,
                                     oodsets=[moving_sets[_] for _ in moving_sets if _ != 'test'],
                                     num_batch=1024 // test_batch_size,
                                     outputs=outputs,
                                     sample_dirs=sample_dirs,
                                     recorders={},
                                     print_result='*')
            self.ood_results = {}

        acc_methods = self.predict_methods

        printed_losses = ['train_zdist']
        for s in moving_batches:
            printed_losses.append('{}_zdist'.format(s))
            printed_losses.append('{}_zdist*'.format(s))

        for epoch in range(epochs):

            self.eval()

            moving_iters = {_: iter(moving_loaders[_]) for _ in moving_loaders}
            t0 = time.time()
            time_per_i = 1e-9

            # EVAL
            self.original_prior = True
            val_batches = 1024 // batch_size
            t0 = time.time()
            time_per_i = 1e-9

            for i in range(val_batches):
                if i:
                    time_per_i = (time.time() - t0) / i
                moving_batches = {}
                try:
                    for _ in moving_iters:
                        moving_batches[_] = next(moving_iters[_])
                except StopIteration:
                    break

                for s in moving_iters:
                    with torch.no_grad:
                        (_, y_est, batch_losses, measures) = self.evaluate(x.to(device), y.to(device),
                                                                           current_measures=current_measures,
                                                                           batch=i,
                                                                           with_beta=True)

                train_running_loss = {'{}_{}': batch_losses[k].mean().item() for k in batch_losses}
                if not i:
                    train_mean_loss = train_running_loss
                else:
                    train_mean_loss = {k: (train_mean_loss[k] * i + train_running_loss[k]) / (i + 1)
                                       for k in train_mean_loss}

            t0 = time.time()
            time_per_i = 1e-9

            self.train()

            # logging.info('Starting epoch {}'.format(epoch + 1))

            moving_iters = {_: iter(moving_loaders[_]) for _ in moving_loaders}

            per_epoch = len(trainloader)
            train_running_loss = {}
            for i, (x, y) in enumerate(trainloader):

                if i:
                    time_per_i = (time.time() - t0) / i

                logging.debug('Epoch {} Batch {}'.format(epoch + 1, i + 1))

                optimizer.zero_grad()

                self.original_prior = True
                with_torch.no_grad():
                    (_, y_est, batch_losses, measures) = self.evaluate(x.to(device), y.to(device),
                                                                       current_measures=current_measures,
                                                                       batch=i,
                                                                       with_beta=True)

                train_running_loss = {'train_' + k: batch_losses[k].mean().item() for k in batch_losses}

                L = batch_losses['total'].mean()

                moving_batches = {}
                for _ in moving_iters:
                    try:
                        moving_batches[_] = next(moving_iters[_])
                    except StopIteration:
                        moving_iters[_] = iter(moving_loaders[_])
                        moving_batches[_] = next(moving_iters[_])

                self.alternate_prior = True

                for s in moving_batches:

                    if not alpha:
                        break

                    self.train()

                    logging.debug('Epoch {} Batch {} -- set {}'.format(epoch + 1, i + 1, s))

                    x, y = moving_batches[s]
                    x = x.to(device)
                    y_zero = torch.zeros_like(y, dtype=int)
                    o = self.evaluate(x, y_zero.to(device),
                                      current_measures=moving_current_measures_on_alternate,
                                      batch=i,
                                      with_beta=True)

                    _, y_est, batch_losses, measures = o

                    train_running_loss.update({'{}_{}*'.format(s, k):
                                               batch_losses[k].mean().item() for k in batch_losses})

                    L += alpha * batch_losses['total'].mean()

                #     self.eval()
                #     self.original_prior = True
                #     if not i % 10 and False:
                #         with torch.no_grad():
                #             o = self.evaluate(x, y.to(device),
                #                               current_measures=moving_current_measures_on_original,
                #                               batch=i,
                #                               with_beta=True)

                #         _, y_est, batch_losses_eval, measures = o
                #     if False:
                #         train_running_loss.update({'{}_{}'.format(s, k):
                #                                    batch_losses_eval[k].mean().item() for k in batch_losses})
                # # print('\n*** running', i, ':', *train_running_loss)  #
                if not i:
                    train_mean_loss = train_mean_loss.update(train_running_loss)
                else:
                    train_mean_loss.update({k: (train_mean_loss[k] * i + train_running_loss[k]) / (i + 1)
                                            for k in train_running_loss})

                outputs.results(i, per_epoch, epoch + 1, epochs,
                                preambule='finetune',
                                losses={k: train_mean_loss[k] for k in train_mean_loss if k in printed_losses},
                                batch_size=batch_size * (1 + len(moving_iters)),
                                time_per_i=time_per_i,
                                end_of_epoch='\n')

                logging.debug('Epoch {} Batch {} -- backprop'.format(epoch + 1, i + 1))

                L.backward()
                optimizer.clip(self.parameters())

                logging.debug('Epoch {} Batch {} -- step'.format(epoch + 1, i + 1))

                optimizer.step()

        sample_dirs = [os.path.join(self.saved_dir, 'samples', '{:04d}'.format(self.trained))]
        for d in sample_dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                assert os.path.isdir(d), '{} exists and is not a dir'.format(d)

        logging.info('Computing ood fprs')

        self.original_prior = True

        self.eval()
        with torch.no_grad():
            self.ood_detection_rates(batch_size=test_batch_size,
                                     oodsets=[moving_sets[_] for _ in moving_sets if _ != 'test'],
                                     num_batch='all',
                                     outputs=outputs,
                                     sample_dirs=sample_dirs,
                                     recorders={},
                                     print_result='*')


if __name__ == '__main__':

    import sys
    import argparse
    import configparser
    from utils.save_load import find_by_job_number
    from utils.parameters import next_jobnumber, set_log
    from module.optimizers import Optimizer

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--debug', action='store_true')
    conf_parser.add_argument('--verbose', '-v', action='count', default=0)
    conf_parser.add_argument('--config-file', default='config.ini')

    conf_args, remaining_args = conf_parser.parse_known_args()

    config = configparser.ConfigParser()
    config.read(conf_args.config_file)

    config_params = config['wim-default']

    defaults = {}

    defaults.update(config_params)

    parser = argparse.ArgumentParser(parents=[conf_parser],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device')
    parser.add_argument('job', type=int)
    parser.add_argument('-J', '--source-job-dir')
    parser.add_argument('--target-job-dir')
    parser.add_argument('--job-number', '-j', type=int)

    parser.add_argument('--wim-sets', nargs='*', default=[])
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--epochs', type=int)

    parser.add_argument('--test-batch-size', type=int)

    parser.add_argument('--prior', choices=['gaussian', 'tilted', 'uniform'])
    parser.add_argument('--prior-means', type=float)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)

    parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_args)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    job_number = args.job_number
    if not job_number:
        job_number = next_jobnumber()

    log_dir = os.path.join(args.output_dir, 'log')
    log = set_log(conf_args.verbose, conf_args.debug, log_dir, job_number=job_number)

    log.debug('$ ' + ' '.join(sys.argv))

    args.target_job_dir = args.target_job_dir or args.source_job_dir

    model_dict = find_by_job_number(args.job, job_dir=args.source_job_dir)

    if model_dict is None:
        log.debug('Model not found, reollecting models')
        model_dict = find_by_job_number(args.job, job_dir=args.source_job_dir, flash=False)

    if model_dict is None:
        log.error('Model not found')
        sys.exit(1)

    log.info('Model found')

    dataset = model_dict['set']

    model = WIMVariationalNetwork.load(model_dict['dir'], load_net=True, load_state=True)

    log.info('Job #{}'.format(job_number))

    log.debug('$ ' + ' '.join(sys.argv))

    save_dir_root = os.path.join(args.target_job_dir, dataset,
                                 model.print_architecture(sampling=False),
                                 'wim')

    save_dir = os.path.join(save_dir_root, f'{job_number:06d}')

    output_file = os.path.join(args.output_dir, f'train-{job_number:06d}.out')

    log.debug(f'Outputs registered in {output_file}')
    outputs = EpochOutput()
    outputs.add_file(output_file)

    model.job_number = job_number
    model.saved_dir = save_dir

    model.encoder.prior.mean.requires_grad_(False)
    alternate_prior_params = model.encoder.prior.params.copy()
    alternate_prior_params['learned_means'] = False

    alternate_prior_params['init_mean'] = args.prior_means
    if args.prior:
        alternate_prior_params['distribution'] = args.prior
    alternate_prior_params['tau'] = args.tau

    model.set_alternate_prior(alternate_prior_params)
    model.wim_params['from'] = args.job

    with model.original_prior as p1:
        with model.alternate_prior as p2:
            log.info('WIM from {} to {}'.format(p1, p2))

            if p1.num_priors > 1:
                log.info('Means from {:.3} to {:.3}'.format(p1.mean.std(0).mean(),
                                                            p2.mean.std(0).mean()))

    try:
        model.to(device)
    except Exception:
        log.warning('Something went wrong when trying to send to {}'.format(device))

    optimizer = None

    if args.lr:
        logging.info('New optimizer')
        optimizer = Optimizer(model.parameters(), optim_type='adam', lr=args.lr, weight_decay=args.weight_decay)

    model.finetune(*args.wim_sets,
                   epochs=args.epochs,
                   test_batch_size=args.test_batch_size,
                   alpha=args.alpha,
                   optimizer=optimizer,
                   outputs=outputs)

    model.save(model.saved_dir)
