import os
from contextlib import contextmanager
import logging


import torch
from torch.nn import Linear, Dropout, Sequential, Parameter, functional as F
from ft.job import FTJob


class PoscodJob(FTJob):

    added_loss_components_per_type = {'cvae': ('y_est_already',),
                                      'vae': (),
                                      'vib': ('y_est_already', 'g', 'cbce')}

    ood_methods_per_type = {'vae': ['zdist', 'elbo', 'kl'],
                            'cvae': ['zdist', 'zdist~', 'zdist@', 'zdist~@',
                                     'elbo', 'elbo~', 'elbo@', 'elbo~@'],
                            'vib': ['g'],
                            }
    misclass_methods_per_type = {'cvae': ['softzdist~', 'zdist~'],
                                 'vae': [], 'vib': []}

    printed_loss = ('g', 'cbce')

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

        self.ft_params = {}

    @classmethod
    def is_poscod(cls, d):
        return os.path.exists(os.path.join(d, 'poscod.json'))

    def evaluate(self, x, *a, y=None, z_output=False, **kw):

        if self.training:
            # y is 0 if in, 1 if mix
            o = super().evaluate(x, *a, y=None, z_output=True, **kw)
        else:
            o = super().evaluate(x, *a, y=y, z_output=True, **kw)

        o_z = o[-3:]
        z = o[-1]

        mix_in_logit = self.ood_head(z[0]).squeeze(-1)

        # (12) & (13) in Franc (2024)
        p_z_mix_over_p_z_in = mix_in_logit.exp() + self.param_a.abs()

        p_u = 0.5

        p_o_train_in_mix = 1 + self.param_a.abs() - self.param_a.abs() / p_u

        # loss is o[2]
        o[2]['g'] = p_z_mix_over_p_z_in * (1 - p_u) / (p_u * p_o_train_in_mix)

        if self.training:
            mix_in_corrected_logit = p_z_mix_over_p_z_in.log()
            mix_in_coorected_bce = F.binary_cross_entropy_with_logits(mix_in_corrected_logit, y, reduction='none')
            o[2]['cbce'] = mix_in_coorected_bce

        if z_output:
            return o
        return o[:-3]

    def batch_dist_measures(self, logits, losses, methods, to_cpu=False):

        poscod_methods = ['g']
        dist_methods = [_ for _ in methods if _ not in poscod_methods]

        dist_measures = super().batch_dist_measures(logits, losses, dist_methods, to_cpu=to_cpu)

        if not poscod_methods:
            return dist_measures

        # logging.debug('Compute measures for {}'.format(','.join(poscod_methods)))

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
            #  logging.debug('{}: {}'.format(m, ', '.join(map(str, measures.shape))))
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

        y_mix_in = torch.ones(len(x_in) + len(x_mix), device=x_in.device)
        y_mix_in[:len(x_in)] = 0

        L = 0
        with self.no_estimated_labels():
            (_, y_est, in_batch_loss, _) = self.evaluate(x_in, y=torch.zeros(len(x_in), device=x_in.device),
                                                         batch=batch,
                                                         with_beta=True)
            (_, y_est, mix_batch_loss, _) = self.evaluate(x_mix, y=torch.ones(len(x_mix), device=x_mix.device),
                                                          batch=batch,
                                                          with_beta=True)

            L = len(x_in) * in_batch_loss['cbce'].mean() + len(x_mix) * mix_batch_loss['cbce'].mean()

            L /= (len(x_in) + len(x_mix))

        self.eval()
        return L, in_batch_loss, mix_batch_loss
