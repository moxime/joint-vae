import os
from contextlib import contextmanager
import logging
import time

from itertools import cycle, product

import numpy as np

import torch
from module.priors import build_prior
from ft.job import FTJob

from utils.save_load import MissingKeys, save_json, load_json, fetch_models, make_dict_from_model
from utils.save_load import LossRecorder, SampleRecorder
from utils.filters import DictOfListsOfParamFilters, ParamFilter, get_filter_keys
import utils.torch_load as torchdl
from utils.torch_load import MixtureDataset, EstimatedLabelsDataset, collate
from utils.print_log import EpochOutput


class WIMJob(FTJob):

    ood_methods_per_type = {'vae': ['zdist', 'elbo', 'kl'],
                            'cvae': ['zdist', 'zdist~', 'zdist@', 'zdist~@',
                                     'elbo', 'elbo~', 'elbo@', 'elbo~@']}
    misclass_methods_per_type = {'cvae': ['softzdist~', 'zdist~'],
                                 'vae': [], }

    ft_param_file = 'wim.json'

    def update_loss_components(self):
        self.loss_components += tuple(k + '@' for k in self.loss_components)
        self.loss_components += self.added_loss_components_per_type.get(self.type, ())

    def __init__(self, *a, alternate_prior=None, **kw):

        super().__init__(*a, **kw)

        self._original_prior = self.encoder.prior
        self._original_num_labels = self.num_labels

        for p in self._original_prior.parameters():
            p.requires_grad_(False)

        self._alternate_prior = None
        if alternate_prior is not None:
            self.set_alternate_prior(**alternate_prior)

        self._is_alternate_prior = False

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
            self.num_labels = 1
        else:
            self.encoder.prior = self._original_prior
            logging.debug('Switching to original prior: {}'.format(self.encoder.prior))
            self._is_alternate_prior = False
            self.num_labels = self._original_num_labels
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
                alternate_loss = {k + '@': o[2][k] for k in o[2] if not k.endswith('~')}
            with self.original_prior:
                o = self.evaluate(x, *a, **kw)
                o[2].update(alternate_loss)
            self._evaluate_on_both_priors = True
            return o

    def batch_dist_measures(self, logits, losses, methods, to_cpu=False):

        wim_methods = [_ for _ in methods if _[-1] in '~@']
        dist_methods = [_ for _ in methods if _ not in wim_methods]

        dist_measures = super().batch_dist_measures(logits, losses, dist_methods, to_cpu=to_cpu)

        if not wim_methods:
            return dist_measures

        logging.debug('Compute measures for {}'.format(','.join(wim_methods)))
        losses['elbo'] = -losses['total']

        k_ = {'kl': -1, 'zdist': -0.5, 'iws': 1, 'elbo': 1}

        loss_ = {}
        if self.is_cvae:
            y_ = losses['y_est_already']
            logging.debug('y, [{}]'.format(', '.join(map(str, y_.shape))))

            # for k in losses:
            #     logging.debug('*** {}: [{}]'.format(k, ', '.join(map(str, losses[k].shape))))

            loss_['y'] = {k: k_[k] * losses[k].gather(0, y_.unsqueeze(0)).squeeze(0) for k in k_}
            # for k in loss_['y']:
            #     logging.debug('*** {}: [{}]'.format(k, ', '.join(map(str, losses[k].shape))))
            #     logging.debug('*** {} y [{}]'.format(k, ', '.join(map(str, loss_['y'][k].shape))))
            loss_['soft'] = {'soft' + k: (losses[k] * k_[k]).softmax(0) for k in k_}
            loss_['soft_y'] = {k: loss_['soft'][k].gather(0, y_.unsqueeze(0)).squeeze(0)
                               for k in loss_['soft']}
            loss_['soft'] = {k: loss_['soft'][k].max(0)[0] for k in loss_['soft']}
            loss_['logsumexp'] = {k: (losses[k] * k_[k]).logsumexp(0) for k in k_}

        if any('@' in m for m in methods):
            losses['elbo@'] = -losses['total@']
            k_.update({k + '@': k_[k] for k in k_})

        wim_measures = {}
        for m in wim_methods:
            """
            -- (soft)k~: (soft)loss[k][y] on original prior with y predicted centroid

            -- k@: logsumexp(loss[k]) - logsumexp(loss@[k])

            -- k~@: loss[k][y] - loss@[k][y]

            """
            if m[-1] == '~':
                m_ = m[:-1]
                prefix = 'soft_' if m.startswith('soft') else ''
                measures = loss_[prefix + 'y'][m_]

            elif m[-1] == '@':
                m_ = m[:-1]
                if m_[-1] == '~':
                    m_ = m_[:-1]
                    w = 'y'
                else:
                    w = 'logsumexp'

                measures = loss_[w][m_] - k_[m_] * losses[m_ + '@']

            wim_measures[m] = measures.cpu() if to_cpu else measures
            logging.debug('{}: {}'.format(m, ', '.join(map(str, measures.shape))))
        dist_measures.update(wim_measures)

        losses.pop('elbo', None)
        losses.pop('elbo@', None)

        return dist_measures

    @ classmethod
    def transfer_from_model(cls, state):
        state['_original_prior.mean'] = torch.clone(state['encoder.prior.mean'])
        state['_original_prior._var_parameter'] = torch.clone(state['encoder.prior._var_parameter'])

    def load_post_hook(self, **wim_params):
        for k in ('sets', 'alpha', 'train_size', 'moving_size',
                  'augmentation', 'augmentation_sets',
                  'from', 'mix', 'hash', 'array_size'):
            wim_params.pop(k, None)
        self.set_alternate_prior(**wim_params)

    def finetune_batch(self, batch, epoch, x_in, y_in, x_mix, alpha=0.1):

        self.original_prior = True
        """

        On original prior

        """
        self.train()
        _s = 'Epoch {:2} Batch {:2} -- set {} --- prior {}'
        logging.debug(_s.format(epoch + 1, batch + 1, 'train', 'original'))

        with self.no_estimated_labels():
            (_, y_est, batch_loss, _) = self.evaluate(x_in, y_in,
                                                      batch=batch,
                                                      with_beta=True)

        L = batch_loss['total'].mean()
        batch_loss = {'train_{}'.format(k): batch_loss[k].mean().item() for k in batch_loss}

        self.alternate_prior = True
        """

        On alternate prior

        """

        batch_size = len(x_mix)
        device = x_mix.device
        y_u_est = torch.zeros(batch_size, device=device, dtype=int)

        self.train()
        with self.no_estimated_labels():
            assert not y_u_est.any()
            o = self.evaluate(x_mix, y_u_est,
                              batch=batch,
                              with_beta=True)

        _, _, mix_batch_loss, _ = o
        L += alpha * mix_batch_loss['total'].mean()

        batch_loss.update({'mix_{}'.format(k): batch_loss[k] for k in mix_batch_loss})

        return L, batch_loss
