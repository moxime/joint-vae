import os
from contextlib import contextmanager
import logging


import torch
from torch.nn import Linear, Dropout, Sequential, Parameter
from ft.job import FTJob


class PoscodJob(FTJob):

    added_loss_components_per_type = {'cvae': ('y_est_already',), 'vae': (), 'vib': ('y_est_already', 'llr')}

    ood_methods_per_type = {'vae': ['zdist', 'elbo', 'kl'],
                            'cvae': ['zdist', 'zdist~', 'zdist@', 'zdist~@',
                                     'elbo', 'elbo~', 'elbo@', 'elbo~@'],
                            'vib': ['llr'],
                            }
    misclass_methods_per_type = {'cvae': ['softzdist~', 'zdist~'],
                                 'vae': [], 'vib': []}

    ft_param_file = 'poscod.json'

    _generalize = True

    def update_loss_components(self):
        self.loss_components += tuple(k + '@' for k in self.loss_components)
        self.loss_components += self.added_loss_components_per_type.get(self.type, ())

    def __init__(self, *a, alternate_prior=None, **kw):

        super().__init__(*a, **kw)

        for p in self.parameters():
            p.requires_grad_(False)

        self.ood_head = Sequential(Dropout(p=0.2),
                                   Linear(in_features=self.latent_dim,
                                          out_features=1, bias=True))

        self.param_a = Parameter(torch.tensor(
            torch.rand(1), requires_grad=True))

        self._train_ood_head = False

    @ classmethod
    def is_poscod(cls, d):
        return os.path.exists(os.path.join(d, 'poscod.json'))

    @property
    @contextmanager
    def train_ood_head(self):
        self._train_ood_head = True
        try:
            yield
        finally:
            self._train_ood_head = False

    def evaluate(self, x, *a, y=None, z_output=False, **kw):

        if not self._train_ood_head:
            o = super().evaluate(x, *a, y=y, z_output=True, **kw)
        else:
            # y is 1 if ood, 0 if not ood
            o = super().evaluate(x, *a, y=None, z_output=True, **kw)

        o_z = o[-3:]
        z = o[-1]
        ood_logit = self.ood_head(z[1:]).mean(0)

        # loss is o[2]
        o[2]['ood_logit'] = ood_logit

        ood_sigmoid = torch.sigmoid(ood_logit)

        if z_output:
            return o

        return o[:-3]

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

            for k in losses:
                logging.debug('*** {}: [{}]'.format(k, ', '.join(map(str, losses[k].shape))))

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
        pass

    def load_post_hook(self, **ft_params):
        pass

    def finetune_batch(self, batch, epoch, x_in, y_in, x_mix, alpha=0.1):

        self.original_prior = True
        """

        On original prior

        """
        self.train()
        _s = 'Epoch {:2} Batch {:2} -- set {} --- prior {}'
        logging.debug(_s.format(epoch + 1, batch + 1, 'train', 'original'))

        with self.no_estimated_labels():
            (_, y_est, in_batch_loss, _) = self.evaluate(x_in, y_in,
                                                         batch=batch,
                                                         with_beta=True)

        L = in_batch_loss['total'].mean()

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

        self._evaluate_on_both_priors = True

        return L, in_batch_loss, mix_batch_loss
