import os
import torch
import logging
from cvae import ClassificationVariationalNetwork as M
import utils.torch_load as torchdl
from module.priors import build_prior


class WIMVariationalNetwork(M):

    def __init__(self, *a, alternate_prior=None, **kw):

        super().__init__(*a, **kw)
        self._original_prior = self.encoder.prior,

        self._alternate_prior = None
        if alternate_prior is not None:
            self.alternate_prior = build_prior(**alternate_prior)

    @property
    def alternate_prior(self):
        return self._alternate_prior

    @alternate_prior.setter
    def alternate_prior(self, p):

        assert self.alternate_prior is None
        self._alternate_prior = p

    def finetune(self, *sets, epochs=5, alpha=0.1, optimizer=None):

        if optimizer is None:
            optimizer = self.optimizer

        self.train()

        set_name = self.training_parameters['set']
        transformer = self.training_parameters['transoformer']
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

        for epoch in range(epochs):

            moving_iters = {_: iter(moving_loaders[_]) for _ in moving_loaders}
            moving_batches = {_: next(moving_iters[_]) for _ in moving_iters}

            for i, (x, y) in trainloader:

                optimizer.zero_grad()

                self.encoder.prior = self._original_prior
                (_, y_est, batch_losses, measures) = self.evaluate(x.to(device), y.to(device),
                                                                   current_measures=current_measures,
                                                                   batch=i,
                                                                   with_beta=True)

                self.encoder.prior = self._alternate_prior

                L = batch_losses['total'].mean()

                for x, y in moving_batches:

                    y = torch.zeros_like(y, dtype=int)
                    o = self.evaluate(x.to(device), y.to(device),
                                      current_measures=moving_current_measures,
                                      batch=i,
                                      with_beta=True)

                    _, y_est, batch_losses, measures = o

                    L += alpha * batch_losses['total'].mean()

                L.backward()
                optimizer.clip(self.parameters())
                optimizer.step()

        sample_dirs = [os.path.join(self.saved_dir, 'samples', d)
                       for d in ('last', f'{epoch:04d}')]

        self.ood_detection_rate()


if __name__ == '__main__':

    import sys
    import argparse
    import configparser
    from utils.save_load import find_by_job_number
    from utils.parameters import get_last_jobnumber, register_last_jobnumber
    from utils.print_log import EpochOutput

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
    parser.add_argument('--job-number', type=int)

    parser.add_argument('--wim-sets', nargs='*')

    parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_args)

    model_dict = find_by_job_number(args.job, job_dir=args.job_dir)

    dataset = model_dict['set']

    model = WIMVariationalNetwork.load(model['dir'], load_net=True, load_state=True)

    job_number = args.job_number
    if not job_number:
        job_number = get_last_jobnumber() + 1
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

    model.finetune(*args.wim_sets)
