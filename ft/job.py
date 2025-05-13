from abc import ABC, abstractmethod, abstractclassmethod
import os
from contextlib import contextmanager
import logging
import time

from itertools import product

import numpy as np

import torch
from cvae import ClassificationVariationalNetwork as M

from utils.save_load import MissingKeys, save_json, load_json, fetch_models, make_dict_from_model
from utils.save_load import LossRecorder
from utils.filters import DictOfListsOfParamFilters, ParamFilter, get_filter_keys
import utils.torch_load as torchdl
from utils.torch_load import collate
from .datasets import MixtureDataset, EstimatedLabelsDataset, create_moving_set
from utils.print_log import EpochOutput


class DontDoFineTuning(Exception):

    def __init__(self, continue_as_array):

        self.continue_as_array = continue_as_array


class FTJob(M, ABC):

    predict_methods_per_type = {'vae': [], 'cvae': ['already'], 'vib': ['already']}
    added_loss_components_per_type = {'cvae': ('y_est_already',), 'vae': (), 'vib': ('y_est_already',)}

    """  to be overridden by child class
    """
    ood_methods_per_type = {'vae': ['zdist', 'elbo', 'kl'],
                            'cvae': ['zdist', 'zdist~', 'zdist@', 'zdist~@',
                                     'elbo', 'elbo~', 'elbo@', 'elbo~@']}

    printed_loss = ('zdist',)

    @classmethod
    def is_one(cls, d):
        try:
            return os.path.exists(os.path.join(d, cls.ft_param_file))
        except AttributeError:
            # FTJob has no ft_param_file
            return True

    @abstractmethod
    def update_loss_components(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def misclass_methods_per_type(self):
        raise NotImplementedError

    _generalize = False

    def __init__(self, *a, **kw):

        super().__init__(*a, **kw)

        self.update_loss_components()
        self._original_num_labels = self.num_labels

        self._with_estimated_labels = self.is_cvae or self.is_vib

        self.ood_methods = self.ood_methods_per_type[self.type].copy()

    @contextmanager
    def no_estimated_labels(self):
        prior_state = self._with_estimated_labels
        try:
            self.ood_methods = [_ for _ in self.ood_methods_per_type[self.type] if _[-1] not in '@~']
            self._with_estimated_labels = False
            logging.debug('Without estimated labels ood methods: {}'.format(','.join(self.ood_methods)))
            yield
        finally:
            self.ood_methods = self.ood_methods_per_type[self.type].copy()
            self._with_estimated_labels = prior_state
            if prior_state:
                logging.debug('Back to estimated labels ood methods: {}'.format(','.join(self.ood_methods)))

    def evaluate(self, x, *a, **kw):
        if self._with_estimated_labels:
            x, y_ = x
            o = super().evaluate(x, *a, **kw)
            # losses is o[2]
            o[2].update({'y_est_already': y_})
            return o

        return super().evaluate(x, *a, **kw)

    @abstractmethod
    def batch_dist_measures(self, logits, losses, methods, to_cpu=False):
        return super().batch_dist_measures(logits, losses, methods, to_cpu=to_cpu)

    @classmethod
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

    @abstractmethod
    def transfer_from_model(self, state):
        raise NotImplementedError

    @abstractmethod
    def load_post_hook(self, **ft_params):
        raise NotImplementedError

    @classmethod
    def load(cls, dir_name, build_module=True, **kw):

        if cls is FTJob:

            for c in cls.__subclasses__():

                if c.is_one(dir_name):
                    return c.load(dir_name, build_module=build_module, **kw)

            raise ValueError('{} is neither {}'.format(dir_name,
                                                       ', '.join([c.__name__ for c in cls.__subclasses__()])))

        try:
            model = super().load(dir_name, strict=False, build_module=build_module, **kw)
        except MissingKeys as e:
            logging.debug('Model loaded has been detected as not wim')
            logging.debug('Missing keys: {}'.format(', '.join(e.args[-1])))
            model = e.args[0]
            s = e.args[1]  # state_dict
            logging.debug('Creating fake params from original state')
            model.transfer_from_model(s)

            model.load_state_dict(s)

            logging.debug('Reset results')

            model.ood_results = {}

        try:
            ft_params = load_json(dir_name, cls.ft_param_file)
            logging.debug('Model was already a ft')
            model.ft_params = ft_params
            if build_module:
                model.load_post_hook(**ft_params)

        except FileNotFoundError:
            logging.debug('Model loaded has been detected as not ft job')
            logging.debug('Reset results')
            model.ood_results = {}

        return model

    def save(self, *a, except_state=True, **kw):
        logging.debug('Saving ft model')
        kw['except_optimizer'] = True
        dir_name = super().save(*a, except_state=except_state, **kw)
        save_json(self.ft_params, dir_name, self.ft_param_file)
        logging.debug('Model saved in {}'.format(dir_name))
        return dir_name

    @abstractmethod
    def finetune_batch(self, epoch, batch, x_in, y_in, x_mix, **kw):
        """
        Has to return a tuple (L, in_loss, mix_loss) where L is the loss to retroproragate on, loss is a dict of loss
        """
        raise NotImplementedError

    def finetune(self, *sets,
                 train_size=100000,
                 epochs=None,
                 moving_size=10000,
                 padding=0.,
                 padding_sets=[],
                 mix_padding=0.,
                 ood_mix=0.5,
                 test_batch_size=8192,
                 optimizer=None,
                 outputs=EpochOutput(),
                 seed=0,
                 task=0,
                 sample_recorders={},
                 generalize=_generalize,
                 **kw
                 ):

        # logging.warning('DEBUG MODE MODEL IN MODE EVAL')

        if optimizer is None:
            optimizer = self.optimizer

        logging.debug('Learning rate: {}'.format(optimizer.lr))

        self.ft_params['sets'] = sets
        self.ft_params['train_size'] = train_size
        self.ft_params['moving_size'] = moving_size
        self.ft_params['mix'] = ood_mix
        self.ft_params['padding'] = padding
        self.ft_params['padding_sets'] = padding_sets
        self.ft_params['mix_padding'] = mix_padding
        self.ft_params.update(**kw)

        transformer = self.training_parameters['transformer']
        data_augmentation = self.training_parameters['data_augmentation']
        batch_size = self.training_parameters['batch_size']

        logging.info('Finetune batch size = {}'.format(batch_size))

        subset_idx_seed = seed
        subset_idx_task = 0 if task == 'array' else task
        if not subset_idx_seed:
            logging.warning('Will not attribute a pseudo randomization on subsets indices')

        logging.info('Pseudo randomization of subdatasets idx with seed/task {} {}'.format(subset_idx_seed,
                                                                                           subset_idx_task))

        ood_sets = {_: torchdl.get_dataset(_, transformer=transformer, splits=['test'])[1] for _ in sets}
        ood_set = MixtureDataset(**ood_sets, mix=1, seed=subset_idx_seed, task=subset_idx_task)

        number_of_tasks = int(len(ood_set) // (ood_mix * moving_size))

        set_name = self.training_parameters['set']

        default_padding_sets = {d: [_ for _ in torchdl.get_same_size_by_name(set_name)
                                    if _.startswith(d)][0] for d in ('const', 'uniform')}

        if not padding_sets:
            padding_sets = ['uniform', 'const']

        padding_sets = [default_padding_sets.get(_, _) for _ in padding_sets]

        if not padding:
            self.ft_params['padding_sets'] = []
            logging.debug('Will not pad moving batch')

        else:
            self.ft_params['padding_sets'] = padding_sets
            tmpstr = 'Will pad moving batch with {:.0%} more of {}'
            logging.info(tmpstr.format(padding, '-'.join(padding_sets)))

        moving_set = create_moving_set(set_name, transformer, data_augmentation, moving_size, ood_mix, sets,
                                       padding_sets, padding=padding, mix_padding=mix_padding,
                                       seed=subset_idx_seed, task=subset_idx_task)

        if epochs:
            train_size = epochs * len(moving_set)
            logging.debug('Train size override by epochs: {}'.format(train_size))
            self.ft_params['train_size'] = train_size

        max_batch_sizes = self.max_batch_sizes

        test_batch_size = min(max_batch_sizes['test'], test_batch_size)

        _s = 'Test batch size wanted {} / max {}'
        logging.info(_s.format(test_batch_size, max_batch_sizes['test']))

        trainset, testset = torchdl.get_dataset(set_name,
                                                transformer=transformer,
                                                data_augmentation=data_augmentation)

        logging.info('Will do finetune (task {}/{})'.format(task, number_of_tasks))

        _s = 'Moving set of length {}, with mixture {}'
        _s = _s.format(len(moving_set), ', '.join('{}:{:.1%}'.format(n, m)
                                                  for n, m in zip(moving_set.classes, moving_set.mix)))
        logging.info(_s)

        actual_moving_size = int(len(moving_set) // (1 + padding + mix_padding))
        if actual_moving_size < moving_size:
            self.ft_params['moving_size'] = actual_moving_size
            logging.warning('Moving size reduced to {} (instead of {})'.format(actual_moving_size, moving_size))

        if task is not None:
            if task == 'array' or task == number_of_tasks:
                logging.info('Is an array, will not do finetuning')
                raise DontDoFineTuning(True)
            if task > number_of_tasks:
                logging.info('All is done, will stop here')
                raise DontDoFineTuning(False)

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
        Compute ood fprs before wim tuning

        """
        recorders = {_: LossRecorder(test_batch_size) for _ in list(sets) + [set_name]}

        logging.debug(str(moving_set))
        ood_ = moving_set.extract_subdataset('ood')

        if self._generalize:
            moving_set.bar(True)

        with self.no_estimated_labels():
            with torch.no_grad():
                self.ood_detection_rates(batch_size=test_batch_size,
                                         testset=moving_set.extract_subdataset('ind', new_name=testset.name),
                                         oodsets=[ood_.extract_subdataset(_) for _ in ood_sets],
                                         # num_batch=1024 // test_batch_size,  #
                                         outputs=outputs,
                                         sample_dirs=sample_dirs,
                                         recorders=recorders,
                                         sample_recorders=sample_recorders,
                                         print_result='*')
                self.ood_results = {}

        moving_set.bar(False)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   # pin_memory=True,
                                                   shuffle=True,
                                                   num_workers=0)

        moving_loader = torch.utils.data.DataLoader(moving_set,
                                                    drop_last=True,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

        epochs = int(np.ceil(train_size / len(moving_set)))
        logging.info('Epochs: {} / {} = {}'.format(train_size, len(moving_set), epochs))
        for epoch in range(epochs):

            per_epoch = min(train_size, len(moving_set)) // batch_size

            if not epoch:
                tmp_s = '{} epochs of {} batches of size {} ({} samples)'
                logging.info(tmp_s.format(epochs,
                                          per_epoch,
                                          batch_size,
                                          epochs * per_epoch * batch_size))

            train_size -= per_epoch * batch_size
            self.eval()

            t0 = time.time()
            time_per_i = 1e-9

            n_ = {'ind': 0, 'ood': 0, 'train': 0}

            train_iter = iter(train_loader)
            moving_iter = iter(moving_loader)

            for batch in range(per_epoch):

                x_u, y_u = next(moving_iter)
                try:
                    x_a, y_a = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x_a, y_a = next(train_iter)

                if not batch and not epoch:
                    logging.debug('First epoch / first batch')
                    logging.debug('First labels for train: {}'.format(' '.join(str(_.item()) for _ in y_a[:10])))
                    logging.debug('First labels for unknown: {}'.format(' '.join(str(_.item()) for _ in y_u[:10])))

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

                L, in_batch_loss, mix_batch_loss = self.finetune_batch(epoch, batch,
                                                                       x_a.to(device), y_a.to(device),
                                                                       x_u.to(device), **kw)

                L.backward()
                # if not batch:
                #     for p in self.parameters():
                #         if p.grad is not None:
                #             print(p.shape, p.grad.shape, p.grad.norm())
                optimizer.step()
                optimizer.clip(self.parameters())

                running_loss = {'{}_{}'.format(_, k): mix_batch_loss[k][i_[_]].mean().item()
                                for _, k in product(i_, mix_batch_loss) if k in self.printed_loss}

                running_loss.update({'in_{}'.format(k): in_batch_loss[k].mean().item()
                                     for k in in_batch_loss if k in self.printed_loss})

                running_loss.update({'mix_{}'.format(k): in_batch_loss[k].mean().item()
                                     for k in in_batch_loss if k in self.printed_loss})

                if not batch:
                    mean_loss = {k: 0. for k in running_loss}
                mean_loss = {k: (mean_loss[k] * batch + running_loss[k])/(batch + 1)
                             for k in running_loss}

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
        if self._generalize:
            moving_set.bar(True)

        testset = EstimatedLabelsDataset(moving_set.extract_subdataset('ind', new_name=testset.name))
        oodsets = [EstimatedLabelsDataset(ood_.extract_subdataset(_)) for _ in ood_sets]

        _s = 'Collecting loss for {} with {} of size {}'
        if self._with_estimated_labels:
            for s in [testset, *oodsets]:
                if not s:
                    continue
                logging.info(_s.format(s.name, recorders[s.name], len(recorders[s.name])))
                if self.is_cvae:
                    y_est = recorders[s.name]['kl'].argmin(0)
                elif self.is_vib:
                    y_est = recorders[s.name]['cross_y'].argmin(0)
                s.append_estimated(y_est)
                s.return_estimated = True

            loader = torch.utils.data.DataLoader(testset, collate_fn=collate, batch_size=100)
            for i, batch in enumerate(loader):
                (x, y_), y = batch
                logging.debug('y = y_ with {:.1%}'.format((y == y_).float().mean()))
                if not i:
                    logging.debug('First labels: {}'.format(' '.join(str(_.item()) for _ in y[:10])))

        with torch.no_grad():
            for s in sample_recorders:
                sample_recorders[s].reset()
            self.ood_detection_rates(batch_size=test_batch_size,
                                     testset=testset,
                                     oodsets=oodsets,
                                     num_batch='all',
                                     outputs=outputs,
                                     sample_dirs=sample_dirs,
                                     recorders={},
                                     sample_recorders=sample_recorders,
                                     print_result='*')
            logging.info('Computing misclass detection rates')
            self.misclassification_detection_rates(print_result='~')
            logging.info('Computing misclass detection rates: done')

    def fetch_jobs_alike(self, job_dir=None, models=None, flash=False):

        assert (job_dir is None) ^ (models is None), 'Either job_dir or models is None'

        wim_filter_keys = get_filter_keys()
        wim_filter_keys = {_: wim_filter_keys[_] for _ in wim_filter_keys if _.startswith('wim')}
        wim_filter_keys.pop('wim_array_size', None)

        filter = DictOfListsOfParamFilters()

        self_dict = make_dict_from_model(self, '')

        for k, f in wim_filter_keys.items():
            filter.add(k, ParamFilter(type=f['type'], values=[self_dict[k]]))

        if job_dir:
            fetched_jobs = fetch_models(job_dir, flash=flash,
                                        build_module=False, filter=filter,
                                        load_state=False, show_debug=False)
        else:
            logging.debug('Looking jobs alike in a list of models of size {}'.format(len(models)))
            fetched_jobs = [m for m in models if filter.filter(m)]

        logging.debug('Fetched {} models with filters'.format(len(fetched_jobs)))

        # fetched_jobs = [_ for _ in fetched_jobs if self == _['net']]
        # logging.debug('Kept {} models with eq'.format(len(fetched_jobs)))

        return fetched_jobs
