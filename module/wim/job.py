import os
from contextlib import contextmanager
import logging
import time

from itertools import cycle, product

import numpy as np

import torch
from cvae import ClassificationVariationalNetwork as M
from module.priors import build_prior

from utils.save_load import MissingKeys, save_json, load_json, LossRecorder, fetch_models, make_dict_from_model
from utils.filters import DictOfListsOfParamFilters, ParamFilter, get_filter_keys
import utils.torch_load as torchdl
from utils.torch_load import MixtureDataset, EstimatedLabelsDataset, collate
from utils.print_log import EpochOutput


class WIMJob(M):

    ood_methods_per_type = {'vae': ['zdist'], 'cvae': ['zdist', 'zdist~', 'zdist*', 'zdist*~']}
    predict_methods_per_type = {'vae': [], 'cvae': ['already']}
    misclass_methods_per_type = {'cvae': ['softzdist~', 'zdist~'],
                                 'vae': [], }

    added_loss_components_per_type = {'cvae': ('kl~', 'softkl~', 'zdist~', 'softzdist~', 'y_est_already')}

    def __init__(self, *a, alternate_prior=None, **kw):

        super().__init__(*a, **kw)
        self.loss_components += self.added_loss_components_per_type.get(self.type, [])
        self._original_prior = self.encoder.prior

        for p in self._original_prior.parameters():
            p.requires_grad_(False)

        self._alternate_prior = None
        if alternate_prior is not None:
            self.set_alternate_prior(**alternate_prior)

        self._is_alternate_prior = False

        self._with_estimated_labels = True

        self._evaluate_on_both_priors = False

        self.ood_methods = self.ood_methods_per_type[self.type].copy()

    @ classmethod
    def is_wim(cls, d):
        return os.path.exists(os.path.join(d, 'wim.json'))

    @ property
    def is_alternate_prior(self):
        return self._is_alternate_prior

    @ property
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

    @ property
    @ contextmanager
    def original_prior(self):
        state = self.is_original_prior
        try:
            yield self._switch_to_alternate_prior(False)
        finally:
            self.original_prior = state

    @ property
    @ contextmanager
    def alternate_prior(self):
        state = self.is_alternate_prior
        try:
            yield self._switch_to_alternate_prior(True)
        finally:
            self.alternate_prior = state

    @ alternate_prior.setter
    def alternate_prior(self, b):
        self._switch_to_alternate_prior(b)

    @ original_prior.setter
    def original_prior(self, b):
        self.alternate_prior = not b

    def set_alternate_prior(self, **p):

        logging.debug('Setting alternate prior')
        assert self._alternate_prior is None
        self._alternate_prior = build_prior(**p)

        if not hasattr(self, 'wim_params'):
            self.wim_params = p.copy()

        for p in self._alternate_prior.parameters():
            p.requires_grad_(False)

    @ contextmanager
    def evaluate_on_both_priors(self):
        state = self._evaluate_on_both_priors
        self._evaluate_on_both_priors = True
        yield
        self._evaluate_on_both_priors = state

    @ contextmanager
    def no_estimated_labels(self):
        try:
            self.ood_methods = [_ for _ in self.ood_methods_per_type[self.type] if _[-1] not in '*~']
            self._with_estimated_labels = False
            logging.debug('Without estimated labels ood methods: {}'.format(','.join(self.ood_methods)))
            yield
        finally:
            self.ood_methods = self.ood_methods_per_type[self.type].copy()
            self._with_estimated_labels = True
            logging.debug('Back to estimated labels ood methods: {}'.format(','.join(self.ood_methods)))

    def evaluate(self, x, *a, **kw):

        if not self._evaluate_on_both_priors:
            if self._with_estimated_labels:
                x, y_ = x
                o = super().evaluate(x, *a, **kw)
                # losses is o[2]
                if self.is_cvae:
                    o[2].update({'y_est_already': y_})
                    return o

            return super().evaluate(x, *a, **kw)

        else:
            logging.debug('Will evaluate on both prior')
            self._evaluate_on_both_priors = False
            with self.alternate_prior:
                o = self.evaluate(x, *a, **kw)
                alternate_loss = {k + '*': o[2][k] for k in o[2] if not k.endswith('~')}
            with self.original_prior:
                o = self.evaluate(x, *a, **kw)
                o[2].update(alternate_loss)
            self._evaluate_on_both_priors = True
            return o

    def batch_dist_measures(self, logits, losses, methods, to_cpu=False):

        wim_methods = [_ for _ in methods if _[-1] in '~*']
        dist_methods = [_ for _ in methods if _ not in wim_methods]

        dist_measures = super().batch_dist_measures(logits, losses, dist_methods, to_cpu=to_cpu)

        if not wim_methods:
            return dist_measures

        losses['elbo'] = -losses['total']
        losses['elbo*'] = -losses['total*']
        k_ = {'kl': -1, 'zdist': -0.5, 'iws': 1, 'elbo': 1}

        k_.update({k + '*': k_[k] for k in k_})

        loss_ = {}
        if self.is_cvae:
            y_ = losses['y_est_already']

            loss_['y'] = {k: k_[k] * losses[k].gather(0, y_.squeeze().unsqueeze(0)).squeeze() for k in k_}
            loss_['soft'] = {k: (losses[k] * k_[k]).softmax(0) for k in k_}
            loss_['soft_y'] = {k: loss_['soft'][k].gather(0, y_.squeeze().unsqueeze(0)).squeeze() for k in k_}
            loss_['soft'] = {k: loss_['soft'][k].max(0)[0] for k in k_}
            loss_['logsumexp'] = {k: (losses[k] * k_[k]).logsumexp(0) for k in k_}

        wim_measures = {}
        for m in wim_methods:

            if m[-1] == '~':
                m_ = m[-1]
                prefix = 'soft_' if m.startswith('soft') else ''
                measures = loss_[prefix + 'y'][m_] * k_[m_] / abs(k_[m_])

            elif m[-1] == '*':
                m_ = m[-1]
                if m_[-1] == '~':
                    m_ = m_[:-1]
                    w = 'y'
                else:
                    w = 'logsumexp'

                measures = loss_[w][m_] - loss_[w][m_ + '*']

            wim_measures[m] = measures.cpu() if to_cpu else measures

        dist_measures.update(wim_measures)

        losses.pop('elbo', None)
        losses.pop('elbo*', None)

        return dist_measures

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
            model.wim_params = wim_params
            for k in ('sets', 'alpha', 'train_size', 'moving_size',
                      'augmentation', 'augmentation_dataset',
                      'from', 'mix', 'hash', 'array_size'):
                k, alternate_prior_params.pop(k, None)
            if build_module:
                model.set_alternate_prior(**alternate_prior_params)

        except FileNotFoundError:
            logging.debug('Model loaded has been detected as not wim')
            logging.debug('Reset results')
            model.ood_results = {}

        return model

    def save(self, *a, **kw):
        logging.debug('Saving wim model')
        kw['except_optimizer'] = True
        with self.original_prior:
            dir_name = super().save(*a, **kw)
        save_json(self.wim_params, dir_name, 'wim.json')
        logging.debug('Model saved in {}'.format(dir_name))
        return dir_name

    def finetune(self, *sets,
                 train_size=100000,
                 epochs=None,
                 moving_size=10000,
                 augmentation=0.,
                 augmentation_dataset=None,
                 alpha=0.1,
                 ood_mix=0.5,
                 test_batch_size=8192,
                 optimizer=None,
                 outputs=EpochOutput(),
                 do_it=True,
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
        self.wim_params['augmentation'] = augmentation

        set_name = self.training_parameters['set']

        if not augmentation_dataset:
            augmentation_dataset = [_ for _ in torchdl.get_same_size_by_name(set_name)
                                    if _.startswith('uniform')][0]

        if not augmentation:
            self.wim_params['augmentation_dataset'] = None
            logging.debug('Will not augment moving batch')

        else:
            self.wim_params['augmentation_dataset'] = augmentation_dataset
            tmpstr = 'Will augment moving batch with {:.0%} more of {}'
            logging.info(tmpstr.format(augmentation, augmentation_dataset))

        if not do_it:
            return

        for p in self._alternate_prior.parameters():
            assert not p.requires_grad, 'prior parameter queires grad'

        for p in self._original_prior.parameters():
            assert not p.requires_grad, 'prior parameter queires grad'

        max_batch_sizes = self.max_batch_sizes

        test_batch_size = min(max_batch_sizes['test'], test_batch_size)

        _s = 'Test batch size wanted {} / max {}'
        logging.info(_s.format(test_batch_size, max_batch_sizes['test']))

        transformer = self.training_parameters['transformer']
        data_augmentation = self.training_parameters['data_augmentation']
        batch_size = self.training_parameters['batch_size']

        logging.info('Finetune batch size = {}'.format(batch_size))

        trainset, testset = torchdl.get_dataset(set_name,
                                                transformer=transformer,
                                                data_augmentation=data_augmentation)

        ood_sets = {_: torchdl.get_dataset(_, transformer=transformer, splits=['test'])[1] for _ in sets}

        augmentation_set = torchdl.get_dataset(augmentation_dataset,
                                               transformer=transformer, splits=['test'])[1]

        try:
            subset_idx_shift_key = self.job_number % 100 + 7
        except AttributeError:
            logging.warning('Will not attribute a pseudo randomization on subsets indices')
            subset_idx_shift_key = 0

        logging.debug('Pseudo randomization of subdatasets idx with key {}'.format(subset_idx_shift_key))

        ood_set = MixtureDataset(**ood_sets, mix=1, shift_key=subset_idx_shift_key,)

        logging.debug('ood set with sets {}'.format(','.join(ood_set.classes)))

        moving_set = MixtureDataset(ood=ood_set, ind=testset, augment=augmentation_set,
                                    mix={'ood': ood_mix, 'ind': 1 - ood_mix, 'augment': augmentation},
                                    shift_key=subset_idx_shift_key,
                                    length=moving_size + int(augmentation * moving_size))

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

        with self.no_estimated_labels():
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

                with self.no_estimated_labels():
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
                        with self.no_estimated_labels():

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
                        with self.no_estimated_labels():
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
                with self.no_estimated_labels():

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

            loader = torch.utils.data.DataLoader(testset, collate_fn=collate, batch_size=100)
            for i, batch in enumerate(loader):
                (x, y_), y = batch
                logging.debug('y = y_ with {:.1%}'.format((y == y_).float().mean()))
                if not i:
                    logging.debug('First labels: {}'.format(' '.join(str(_.item()) for _ in y[:10])))

        with torch.no_grad():
            self.original_prior = True
            outputs.write('With original prior\n')
            with self.evaluate_on_both_priors():
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
                                        build_module=False, filter=filter, load_state=False, show_debug=False)
        else:
            logging.debug('Looking jobs alike in a list of models of size {}'.format(len(models)))
            fetched_jobs = [m for m in models if filter.filter(m)]

        logging.debug('Fetched {} models with filters'.format(len(fetched_jobs)))

        # fetched_jobs = [_ for _ in fetched_jobs if self == _['net']]
        # logging.debug('Kept {} models with eq'.format(len(fetched_jobs)))

        return fetched_jobs
