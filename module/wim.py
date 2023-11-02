import os
from contextlib import contextmanager
import logging
import time

from itertools import cycle, product

import numpy as np

import torch
from cvae import ClassificationVariationalNetwork as M
from module.priors import build_prior

from utils.save_load import MissingKeys, save_json, load_json, LossRecorder
import utils.torch_load as torchdl
from utils.torch_load import MixtureDataset, EstimatedLabelsDataset, collate
from utils.print_log import EpochOutput


class WIMVariationalNetwork(M):

    ood_methods_per_type = {'vae': ['zdist'], 'cvae': ['zdist', 'zdist~']}
    predict_methods_per_type = {'vae': [], 'cvae': ['already']}
    misclass_methods_per_type = {'cvae': ['softzdist~', 'zdist~'],
                                 'vae': [], }

    def __init__(self, *a, alternate_prior=None, **kw):

        super().__init__(*a, **kw)
        self._original_prior = self.encoder.prior

        for p in self._original_prior.parameters():
            p.requires_grad_(False)

        self._alternate_prior = None
        if alternate_prior is not None:
            self.set_alternate_prior(**alternate_prior)

        self._is_alternate_prior = False

        self._with_estimated_labels = False

        self.ood_methods = tuple(_ for _ in self.ood_methods_per_type[self.type] if _[-1] != '~')

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

    def set_alternate_prior(self, **p):

        logging.debug('Setting alternate prior')
        assert self._alternate_prior is None
        self._alternate_prior = build_prior(**p)
        self.wim_params = p.copy()

        for p in self._alternate_prior.parameters():
            p.requires_grad_(False)

    @contextmanager
    def estimated_labels(self, v=True):
        if v:
            try:
                self._with_estimated_labels = True
                self.ood_methods = self.ood_methods_per_type[self.type]
                logging.debug('With estimated labels ood methods: {}'.format(','.join(self.ood_methods)))
                yield
            finally:
                self._with_estimated_labels = False
                self.ood_methods = tuple(_ for _ in self.ood_methods_per_type[self.type] if _[-1] != '~')
                logging.debug('Back to non estimated labels ood methods: {}'.format(','.join(self.ood_methods)))
        else:
            try:
                logging.debug('Will not switch to estimated labels')
                yield
            finally:
                pass

    def evaluate(self, x, *a, **kw):

        if self._with_estimated_labels:
            x, y_ = x
            o = super().evaluate(x, *a, **kw)
            # losses is o[2]
            k_ = ('kl', 'zdist')
            if self.is_cvae:
                o[2].update({k + '~': o[2][k].gather(0, y_.squeeze().unsqueeze(0)).squeeze() for k in k_})
                o[2].update({'soft{}~'.format(k):
                             (-o[2][k]).softmax(0).gather(0, y_.squeeze().unsqueeze(0)).squeeze() for k in k_})
                o[2].update({'y_est_already': y_})
            return o

        return super().evaluate(x, *a, **kw)

    @ classmethod
    def _recurse_train(cls, module):

        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            return 1

        return sum([cls._recurse_train(child) for child in module.children()])

    def train(self, *a, **kw):

        super().train(*a, **kw)
        if self.training:
            n = self._recurse_train(self)
            logging.debug('Kept {} bn layers in eval mode'.format(n))

    @ classmethod
    def load(cls, dir_name, build_module=True, **kw):

        try:
            model = super().load(dir_name, strict=False, build_module=build_module, **kw)
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
            for k in ('sets', 'alpha', 'train_size', 'moving_size', 'from', 'mix'):
                k, alternate_prior_params.pop(k, None)
            if build_module:
                model.set_alternate_prior(**alternate_prior_params)
            model.wim_params = wim_params

        except FileNotFoundError:
            logging.debug('Model loaded has been detected as not wim')
            logging.debug('Reset results')
            model.ood_results = {}

        return model

    def save(self, *a, **kw):

        kw['except_optimizer'] = True
        with self.original_prior:
            dir_name = super().save(*a, **kw)
        save_json(self.wim_params, dir_name, 'wim.json')
        return dir_name

    def finetune(self, *sets,
                 train_size=100000,
                 epochs=None,
                 moving_size=10000,
                 alpha=0.1,
                 ood_mix=0.5,
                 test_batch_size=8192,
                 optimizer=None,
                 outputs=EpochOutput(),
                 ):

        # logging.warning('DEBUG MODE MODEL IN MODE EVAL')

        def zdbg(*a):           #
            debug_str = '### {:10} epoch {:2} batch {:2} set {:8} {:10} prior <zdist> = {:9.4g}'
            logging.debug(debug_str.format(*a))

        if optimizer is None:
            optimizer = self.optimizer

        logging.debug('Learning rate: {}'.format(optimizer.lr))

        self.wim_params['sets'] = sets
        self.wim_params['alpha'] = alpha
        self.wim_params['train_size'] = train_size
        self.wim_params['moving_size'] = moving_size
        self.wim_params['mix'] = ood_mix

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

        ood_sets = {_: torchdl.get_dataset(_, transformer=transformer, splits=['test'])[1] for _ in sets}

        try:
            subset_idx_shift_key = model.job_number % 100 + 7
        except AttributeError:
            logging.warning('Will not attribute a pseudo randomization on subsets indices')
            subset_idx_shift_key = 0

        logging.debug('Pseudo randomization of subdatasets idx with key {}'.format(subset_idx_shift_key))

        ood_set = MixtureDataset(**ood_sets, mix=1, shift_key=subset_idx_shift_key,)

        logging.debug('ood set with sets {}'.format(','.join(ood_set.classes)))

        moving_set = MixtureDataset(ood=ood_set, ind=testset,
                                    mix={'ood': ood_mix, 'ind': 1 - ood_mix},
                                    shift_key=subset_idx_shift_key,
                                    length=moving_size)

        _s = 'Moving set of length {}, with mixture {}'
        _s = _s.format(len(moving_set), ', '.join('{}:{:.1%}'.format(n, m)
                                                  for n, m in zip(moving_set.classes, moving_set.mix)))
        logging.info(_s)

        if len(moving_set) < moving_size:
            self.wim_params['moving_size'] = moving_size

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  # pin_memory=True,
                                                  shuffle=True,
                                                  num_workers=0)

        moving_loader = torch.utils.data.DataLoader(moving_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

        current_measures = {}

        device = next(self.parameters()).device

        self.eval()
        sample_dirs = [os.path.join(self.saved_dir, 'samples', '{:04d}'.format(self.trained), 'init')]
        for d in sample_dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                assert os.path.isdir(d), '{} exists and is not a dir'.format(d)

        """
        Compute ood fprs before wim finetuning

        """
        self.original_prior = True
        recorders = {_: LossRecorder(test_batch_size) for _ in list(sets) + [set_name]}

        with torch.no_grad():
            ood_ = moving_set.extract_subdataset('ood')
            logging.debug('OOD set of size {}'.format(len(ood_)))
            self.ood_detection_rates(batch_size=test_batch_size,
                                     testset=moving_set.extract_subdataset('ind', new_name=testset.name),
                                     oodsets=[ood_.extract_subdataset(_) for _ in ood_sets],
                                     # num_batch=1024 // test_batch_size,  #
                                     outputs=outputs,
                                     sample_dirs=sample_dirs,
                                     recorders=recorders,
                                     print_result='*')
            self.ood_results = {}

        printed_losses = ['zdist']
        # for s in ('ind', 'ood' 'train'):
        #     printed_losses.append('{}_zdist'.format(s))
        #     printed_losses.append('{}_zdist*'.format(s))

        if epochs:
            train_size = epochs * len(trainset)
            logging.debug('Train size override by epochs: {}'.format(train_size))
            self.wim_params['train_size'] = train_size
        epochs = int(np.ceil(train_size / len(trainset)))
        logging.debug('Epochs: {} / {} = {}'.format(train_size, len(trainset), epochs))
        for epoch in range(epochs):

            per_epoch = min(train_size, len(trainset)) // batch_size
            train_size -= per_epoch * batch_size
            running_loss = {}
            self.eval()

            t0 = time.time()
            time_per_i = 1e-9

            n_ = {'ind': 0, 'ood': 0, 'train': 0}

            train_iter = iter(trainloader)
            moving_iter = iter(moving_loader)

            for batch in range(per_epoch):

                x_a, y_a = next(train_iter)
                try:
                    x_u, y_u = next(moving_iter)
                except StopIteration:
                    moving_iter = iter(moving_loader)
                    x_u, y_u = next(moving_iter)

                if not batch and not epoch:
                    logging.debug('First epoch / first batch')
                    logging.debug('First labels for train: {}'.format(' '.join(str(_.item()) for _ in y_a[:10])))
                    logging.debug('First labels for unknown: {}'.format(' '.join(str(_.item()) for _ in y_u[:10])))

                val_batch = not (batch % max(per_epoch * batch_size // 3000, 1))
                if batch:
                    time_per_i = (time.time() - t0) / batch

                i_ = {}
                i_['ind'] = list(moving_set.which_subsets(*y_u, which='ind'))
                i_['ood'] = [~_ for _ in i_['ind']]

                n_per_i_ = {_: sum(i_[_]) for _ in i_}
                n_per_i_['train'] = len(x_a)

                optimizer.zero_grad()

                """

                On original prior

                """

                self.original_prior = True
                self.train()
                _s = 'Epoch {:2} Batch {:2} -- set {} --- prior {}'
                logging.debug(_s.format(epoch + 1, batch + 1, 'train', 'original'))

                (_, y_est, batch_losses, _) = self.evaluate(x_a.to(device), y_a.to(device),
                                                            batch=batch,
                                                            with_beta=True)

                zdbg('finetune', epoch + 1, batch + 1, 'train', 'original', batch_losses['zdist'].mean())

                running_loss = {'train_' + k: batch_losses[k].mean().item() for k in printed_losses}

                L = batch_losses['total'].mean()

                y_u_est = torch.zeros(batch_size, device=device, dtype=int)

                if val_batch:  # or self.is_cvae:
                    self.eval()
                    with torch.no_grad():

                        # Eval on unknown batch

                        (_, _, batch_losses, _) = self.evaluate(x_u.to(device),
                                                                batch=batch,
                                                                with_beta=True)

                        if self.is_cvae:
                            y_u_est = torch.zeros(batch_size, device=device, dtype=int)
                            # y_u_est = batch_losses['zdist'].min(0)[1]
                            logging.debug('zdist shape: {}'.format(batch_losses['zdist'].shape))
                            batch_losses = {k: batch_losses[k].min(0)[0] for k in printed_losses}

                        else:
                            y_u_est = torch.zeros(batch_size, device=device, dtype=int)

                        running_loss.update({_ + '_' + k: batch_losses[k][i_[_]].mean().item()
                                             for _, k in product(i_, printed_losses)})

                        for _ in i_:
                            zdbg('eval', epoch + 1, batch + 1, _, 'original', batch_losses['zdist'][i_[_]].mean())

                        # Eval on train batch

                        (_, _, batch_losses, _) = self.evaluate(x_a.to(device),
                                                                batch=batch,
                                                                with_beta=True)

                        if self.is_cvae:
                            # y_u_est = torch.zeros(batch_size, device=device, dtype=int)
                            y_a_est = batch_losses['zdist'].min(0)[1]
                            acc = (y_a.to(device) == y_a_est).float().mean()
                            logging.debug('Batch train acc: {:.1%}'.format(acc))
                            batch_losses = {k: batch_losses[k].min(0)[0] for k in printed_losses}

                        else:
                            y_a_est = torch.zeros(batch_size, device=device, dtype=int)

                        # running_loss.update({_ + '_' + k: batch_losses[k][i_[_]].mean().item()
                        #                      for _, k in product(i_, printed_losses)})

                        zdbg('eval', epoch + 1, batch + 1, 'train', 'original', batch_losses['zdist'].mean())

                """

                On alternate prior

                """

                self.alternate_prior = True

                _s = 'Epoch {:2} Batch {:2} -- set {} --- prior {}'
                logging.debug(_s.format(epoch + 1, batch + 1, 'moving', 'alternate'))

                logging.debug('x_u shape: {} y_u_est shape {}'.format(x_u.shape, y_u_est.shape))

                self.train()
                o = self.evaluate(x_u.to(device), y_u_est,
                                  batch=batch,
                                  with_beta=True)

                _, _, batch_losses, _ = o
                L += alpha * batch_losses['total'].mean()

                L.backward()
                optimizer.step()
                optimizer.clip(self.parameters())

                for _ in i_:
                    zdbg('finetune', epoch + 1, batch + 1, _, 'alternate', batch_losses['zdist'][i_[_]].mean())

                running_loss.update({_ + '_' + k + '*': batch_losses[k][i_[_]].mean().item()
                                     for _, k in product(i_, printed_losses)})

                self.eval()
                # _s = 'Val   {:2} Batch {:2} -- set {} --- prior {}'
                # logging.debug(_s.format(epoch + 1, batch + 1, 'train', 'alternate'))

                # with torch.no_grad():
                #     (_, _, batch_losses, _) = self.evaluate(x_a.to(device),
                #                                             y_a.to(device),
                #                                             batch=batch,
                #                                             with_beta=True)

                # zdbg('eval', epoch + 1, batch + 1, 'train', 'alternate', batch_losses['zdist'].mean())

                # running_loss.update({'train_' + k + '*': batch_losses[k].mean().item()
                #                      for k in printed_losses})

                if not batch:
                    mean_loss = running_loss
                else:
                    for _, k, suf in product(n_per_i_, printed_losses, ('*', '')):
                        k_ = _ + '_' + k + suf
                        if k in running_loss:
                            mean_loss[k_] = (mean_loss[k_] * n_[_] + running_loss[k_] * n_per_i_[_]) / n_[_]

                for _ in n_:
                    n_[_] += n_per_i_[_]

                outputs.results(batch, per_epoch, epoch + 1, epochs,
                                preambule='finetune',
                                losses=mean_loss,
                                batch_size=2 * batch_size,
                                time_per_i=time_per_i,
                                end_of_epoch='\n')

            logging.debug('During this epoch we went through {} inds and {} oods'.format(n_['ind'], n_['ood']))

        sample_dirs = [os.path.join(self.saved_dir, 'samples', '{:04d}'.format(self.trained))]
        for d in sample_dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                assert os.path.isdir(d), '{} exists and is not a dir'.format(d)

        logging.info('Computing ood fprs')

        self.eval()
        testset = EstimatedLabelsDataset(moving_set.extract_subdataset('ind', new_name=testset.name))
        oodsets = [EstimatedLabelsDataset(ood_.extract_subdataset(_)) for _ in ood_sets]

        _s = 'Collecting loss for {} with {} of size {}'
        logging.debug(_s.format(testset.name, recorders[testset.name], len(recorders[testset.name])))
        if self.is_cvae:
            y_est = recorders[testset.name]['kl'].argmin(0)
            testset.append_estimated(y_est)
            testset.return_estimated = True
            for s in oodsets:
                if not s:
                    continue
                y_est = recorders[s.name]['kl'].argmin(0)
                s.append_estimated(y_est)
                s.return_estimated = True

                #        if True:  # debug

            loader = torch.utils.data.DataLoader(testset, collate_fn=collate, batch_size=100)
            for i, batch in enumerate(loader):
                (x, y_), y = batch
                logging.debug('y = y_ with {:.1%}'.format((y == y_).float().mean()))
                if not i:
                    logging.debug('First labels: {}'.format(' '.join(str(_.item()) for _ in y[:10])))

        with self.estimated_labels(self.is_cvae):
            with torch.no_grad():
                self.original_prior = True
                outputs.write('With original prior\n')
                self.ood_detection_rates(batch_size=test_batch_size,
                                         testset=testset,
                                         oodsets=oodsets,
                                         num_batch='all',
                                         outputs=outputs,
                                         sample_dirs=sample_dirs,
                                         recorders={},
                                         print_result='*')
                logging.info('Computing misclass detection rates')
                self.misclassification_detection_rates(print_result='~')
                logging.info('Computing misclass detection rates: done')

            # self.alternate_prior = True
            # outputs.write('With alternate prior\n')
            # self.ood_detection_rates(batch_size=test_batch_size,
            #                          oodsets=[moving_sets[_] for _ in moving_sets if _ != 'test'],
            #                          num_batch='all',
            #                          outputs=outputs,
            #                          sample_dirs=sample_dirs,
            #                          recorders={},
            #                          update_self_ood=False,
            #                          print_result='*')


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

    parser.add_argument('--mix', type=float)

    parser.add_argument('-N', '--train-size', type=int)
    parser.add_argument('-n', '--moving-size', type=int)
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

    log.info('Model found of type {}'.format(model_dict['type']))

    dataset = model_dict['set']

    model = WIMVariationalNetwork.load(model_dict['dir'], build_module=True, load_state=True)

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

    alternate_prior_params['mean_shift'] = args.prior_means
    if args.prior:
        alternate_prior_params['distribution'] = args.prior
    alternate_prior_params['tau'] = args.tau

    model.set_alternate_prior(**alternate_prior_params)
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

    wim_sets = sum((_.split('-') for _ in args.wim_sets), [])
    model.finetune(*wim_sets,
                   train_size=args.train_size,
                   epochs=args.epochs,
                   moving_size=args.moving_size,
                   test_batch_size=args.test_batch_size,
                   alpha=args.alpha,
                   ood_mix=args.mix,
                   optimizer=optimizer,
                   outputs=outputs)

    model.save(model.saved_dir)
