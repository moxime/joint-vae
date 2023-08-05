import os
import torch
import logging
from cvae import ClassificationVariationalNetwork as M
from utils.save_load import MissingKeys
import utils.torch_load as torchdl
from module.priors import build_prior
from utils.print_log import EpochOutput


class WIMVariationalNetwork(M):

    def __init__(self, *a, alternate_prior=None, **kw):

        super().__init__(*a, **kw)
        self._original_prior = self.encoder.prior

        self._alternate_prior = None
        if alternate_prior is not None:
            self.alternate_prior = alternate_prior

    @property
    def alternate_prior(self):
        return self._alternate_prior

    @alternate_prior.setter
    def alternate_prior(self, p):

        assert self.alternate_prior is None
        self._alternate_prior = build_prior(**p)

    @classmethod
    def load(cls, *a, **kw):

        try:
            super().load(*a, strict=False, **kw)
        except MissingKeys as e:
            model = e.args[0]
            s = e.args[1]  # state_dict
            logging.debug('Creating fake params prior means')
            s['_original_prior.mean'] = torch.zeros_like(s['encoder.prior.mean'])
            s['_original_prior._var_parameter'] = torch.ones_like(s['encoder.prior._var_parameter'])

            model.load_state_dict(s)

        return model

    def finetune(self, *sets,
                 epochs=5, alpha=0.1,
                 test_batch_size=8192,
                 optimizer=None,
                 outputs=EpochOutput(),
                 ):

        if optimizer is None:
            optimizer = self.optimizer

        max_batch_sizes = self.max_batch_sizes

        test_batch_size = min(max_batch_sizes['test'], test_batch_size)

        logging.info(
            'Test batch size wanted {} / max {}'.format(test_batch_size, max_batch_sizes['test']))

        self.train()

        set_name = self.training_parameters['set']
        transformer = self.training_parameters['transformer']
        data_augmentation = self.training_parameters['data_augmentation']
        batch_size = self.training_parameters['batch_size']

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
        moving_current_measures = {}

        device = next(self.parameters()).device

        acc_methods = self.predict_methods

        outputs.results(0, 0, -2, epochs,
                        metrics=self.metrics,
                        loss_components=self.loss_components,
                        acc_methods=acc_methods)
        outputs.results(0, 0, -1, epochs,
                        metrics=self.metrics,
                        loss_components=self.loss_components,
                        acc_methods=acc_methods)

        for epoch in range(epochs):

            logging.info('Starting epoch {}'.format(epoch + 1))

            moving_iters = {_: iter(moving_loaders[_]) for _ in moving_loaders}
            moving_batches = {_: next(moving_iters[_]) for _ in moving_iters}

            per_epoch = len(trainloader)
            for i, (x, y) in enumerate(trainloader):

                logging.debug('Epoch {} Batch {}'.format(epoch + 1, i + 1))

                optimizer.zero_grad()

                self.encoder.prior = self._original_prior
                (_, y_est, batch_losses, measures) = self.evaluate(x.to(device), y.to(device),
                                                                   current_measures=current_measures,
                                                                   batch=i,
                                                                   with_beta=True)
                outputs.results(i, per_epoch, epoch + 1, epochs,
                                preambule='train',
                                acc_methods=acc_methods,
                                loss_components=self.loss_components,
                                # losses=train_mean_loss,
                                metrics=self.metrics,
                                measures=measures,
                                # time_per_i=t_per_i,
                                batch_size=batch_size,
                                end_of_epoch='\n')

                self.encoder.prior = self._alternate_prior

                L = batch_losses['total'].mean()

                for _ in moving_batches:

                    logging.debug('Epoch {} Batch {} -- set {}'.format(epoch + 1, i + 1, _))

                    x, y = moving_batches[_]
                    y = torch.zeros_like(y, dtype=int)
                    o = self.evaluate(x.to(device), y.to(device),
                                      current_measures=moving_current_measures,
                                      batch=i,
                                      with_beta=True)

                    _, y_est, batch_losses, measures = o

                    L += alpha * batch_losses['total'].mean()

                logging.debug('Epoch {} Batch {} -- backprop'.format(epoch + 1, i + 1))

                L.backward()
                optimizer.clip(self.parameters())

                logging.debug('Epoch {} Batch {} -- step'.format(epoch + 1, i + 1))

                optimizer.step()

        sample_dirs = [os.path.join(self.saved_dir, 'samples', '{:04d}'.format(self.trained))]

        logging.info('Computinog ood fprs')

        self.ood_detection_rates(batch_size=test_batch_size,
                                 num_batch='all',
                                 outputs=outputs,
                                 sample_dirs=sample_dirs,
                                 print_result='*')


if __name__ == '__main__':

    import sys
    import argparse
    import configparser
    from utils.save_load import find_by_job_number
    from utils.parameters import get_last_jobnumber, register_last_jobnumber, set_log

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--debug', action='store_true')
    conf_parser.add_argument('--verbose', '-v', action='count', default=0)
    conf_parser.add_argument('--config-file', default='config.ini')

    conf_args, remaining_args = conf_parser.parse_known_args()

    config = configparser.ConfigParser()
    config.read(conf_args.config_file)

    config_params = config['DEFAULT']

    defaults = {}

    defaults.update(config_params)

    parser = argparse.ArgumentParser(parents=[conf_parser],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('job', type=int)
    parser.add_argument('--job-dir', default='./jobs')
    parser.add_argument('--job-number', '-j', type=int)

    parser.add_argument('--wim-sets', nargs='*')

    parser.add_argument('--test-batch-size', type=int)

    parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_args)

    model_dict = find_by_job_number(args.job, job_dir=args.job_dir)

    if model_dict is None:
        logging.debug('Model not found, reollecting models')
        model_dict = find_by_job_number(args.job, job_dir=args.job_dir, flash=False)

    if model_dict is None:
        logging.error('Model not found')
        sys.exit(1)

    logging.info('Model found')

    dataset = model_dict['set']

    model = WIMVariationalNetwork.load(model_dict['dir'], load_net=True, load_state=True)

    job_number = args.job_number
    if not job_number:
        job_number = get_last_jobnumber() + 1

    log_dir = os.path.join(defaults['output_dir'], 'log')

    log = set_log(conf_args.verbose, conf_args.debug, log_dir, job_number=job_number)

    register_last_jobnumber(job_number)

    save_dir_root = os.path.join(args.job_dir, dataset,
                                 model.print_architecture(sampling=False),
                                 'wim')

    save_dir = os.path.join(save_dir_root, f'{job_number:06d}')

    output_file = os.path.join(args.output_dir, f'train-{job_number:06d}.out')

    logging.debug(f'Outputs registered in {output_file}')
    outputs = EpochOutput()
    outputs.add_file(output_file)

    model.job_number = job_number
    model.saved_dir = save_dir

    model.encoder.prior.mean.requires_grad_(False)
    alternate_prior_params = model.encoder.prior.params
    alternate_prior_params['learned_means'] = False
    alternate_prior_params['init_mean'] = 0.

    model.alternate_prior = alternate_prior_params

    model.finetune(*args.wim_sets,
                   test_batch_size=args.test_batch_size,
                   outputs=outputs)