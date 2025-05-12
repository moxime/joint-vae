import os
from contextlib import contextmanager
import logging


import torch
from torch.nn import Linear, Dropout, Sequential, Parameter, functional as F
from ft.job import FTJob


class PoscodJob(FTJob):

    added_loss_components_per_type = {'cvae': ('y_est_already',),
                                      'vae': (),
                                      'vib': ('y_est_already', 'g', 'mix_in_cbce')}

    ood_methods_per_type = {'vae': ['zdist', 'elbo', 'kl'],
                            'cvae': ['zdist', 'zdist~', 'zdist@', 'zdist~@',
                                     'elbo', 'elbo~', 'elbo@', 'elbo~@'],
                            'vib': ['g'],
                            }
    misclass_methods_per_type = {'cvae': ['softzdist~', 'zdist~'],
                                 'vae': [], 'vib': []}

    ft_param_file = 'poscod.json'

    _generalize = True

    def update_loss_components(self):
        self.loss_components += self.added_loss_components_per_type.get(self.type, ())

    def __init__(self, *a, **kw):

        super().__init__(*a, **kw)

        for p in self.parameters():
            p.requires_grad_(False)

        self.ood_head = Sequential(Dropout(p=0.2),
                                   Linear(in_features=self.latent_dim,
                                          out_features=1, bias=True))

        self.param_a = Parameter(torch.rand(1), requires_grad=True)

        self._train_ood_head = False

        self.ft_params = {}

    @classmethod
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
            # y is 0 if in, 1 if mix
            o = super().evaluate(x, *a, y=None, z_output=True, **kw)

        o_z = o[-3:]
        z = o[-1]

        mix_in_logit = self.ood_head(z[1:]).squeeze(-1).mean(0)

        # (12) & (13) in Franc (2024)
        p_z_mix_over_p_z_in = mix_in_logit.exp() + self.param_a.abs()

        p_u = 0.5

        p_o_train_in_mix = 1 + self.param_a.abs() - self.param_a.abs() / p_u

        # loss is o[2]
        o[2]['g'] = p_z_mix_over_p_z_in * (1 - p_u) / (p_u * p_o_train_in_mix)

        if self._train_ood_head:
            mix_in_corrected_logit = p_z_mix_over_p_z_in.log()
            mix_in_coorected_bce = F.binary_cross_entropy_with_logits(mix_in_corrected_logit, y)
            o[2]['mix_in_cbce'] = mix_in_coorected_bce

        if z_output:
            return o
        return o[:-3]

    def batch_dist_measures(self, logits, losses, methods, to_cpu=False):

        poscod_methods = ['g']
        dist_methods = [_ for _ in methods if _ not in poscod_methods]

        dist_measures = super().batch_dist_measures(logits, losses, dist_methods, to_cpu=to_cpu)

        if not poscod_methods:
            return dist_measures

        logging.debug('Compute measures for {}'.format(','.join(poscod_methods)))

        poscod_measures = {}
        for m in poscod_methods:
            """
            -- (soft)k~: (soft)loss[k][y] on original prior with y predicted centroid

            -- k@: logsumexp(loss[k]) - logsumexp(loss@[k])

            -- k~@: loss[k][y] - loss@[k][y]

            """
            if m == 'g':
                measures = -losses['g']

            poscod_measures[m] = measures.cpu() if to_cpu else measures
            logging.debug('{}: {}'.format(m, ', '.join(map(str, measures.shape))))
        dist_measures.update(poscod_measures)

        return dist_measures

    def transfer_from_model(self, state):

        state['param_a'] = self.param_a
        state['ood_head.1.weight'] = self.ood_head[1].weight
        state['ood_head.1.bias'] = self.ood_head[1].bias

    def load_post_hook(self, **ft_params):
        self.ft_params = {}

    def finetune_batch(self, batch, epoch, x_in, y_in, x_mix, alpha=0.1):

        self.train()

        y_mix = torch.ones(len(x_in), device=x_in.device)

        x_mix_in = torch.concat([x_in, x_mix])

        y_mix[:len(x_in)] = 0

        x = {0: x_mix_in[::2], 1: x_mix_in[1::2]}
        y = {0: y_mix[::2], 1: y_mix[1::2]}
        batch_losses = {}
        L = 0
        with self.no_estimated_labels():
            with self.train_ood_head():
                for w in x:
                    (_, y_est, batch_loss, _) = self.evaluate(x[w], y=y[w],
                                                              batch=batch,
                                                              with_beta=True)
                    batch_losses[_] = batch_loss

                    L += batch_loss['cbce'].mean()

        in_batch_loss = {_: torch.concat([batch_losses[w][_][y_mix == 0 for w in x]) for _ in batch_loss}
        mix_batch_loss = {_: torch.concat([batch_losses[w][_][y_mix == 1 for w in x]) for _ in batch_loss}
        return L, in_batch_loss, mix_batch_loss
