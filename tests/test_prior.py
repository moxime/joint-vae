import torch
from module.vae_layers import Prior
from matplotlib import pyplot as plt
import time
import numpy as np

var_type = 'diag'
var_type = 'full'
var_type = 'scalar'

K = 5
N = 50
labels = 10

var_per_dim = torch.stack([torch.ones(K) for i in range(labels)])
mu_per_dim = torch.randn(labels, K)

priors = {}

var_types = ('full', 'diag', 'scalar')
var_types = ('full',)
var_types = ('scalar',)
Cs = (10, 1)
Cs = (1,)
Cs = (10,)

print('\n'*20)
for var_type in var_types:
    priors[var_type] = {}
    for C in Cs:
        print('************************')
        print('***', var_type, C, '***')
        print()
        mean = torch.randn(C, K)
        prior = Prior(K, mean=mean, var_type=var_type, num_priors=C)
        priors[var_type][C] = prior

        if C > 1:
            params_to_be_modified = [_ for _ in prior._var_parameter]
        else:
            params_to_be_modified = [prior._var_parameter]
        if var_type == 'full':
            for p in params_to_be_modified:
                p.data[0][0] = 0.5
                p.data[1][0] = 1e-1

        if var_type == 'diag':
            for p in params_to_be_modified:
                p.data[0] = 0.5

        if var_type == 'scalar':
            for p in params_to_be_modified:
                pass  # p.data = torch.tensor(0.5)

        z = torch.randn(N, K)
        z0 = z[0]
        v = z.exp()
        v0 = v[0]
        y = torch.randint(C, (N,)) if C > 1 else None

        u = prior.whiten(z, y)

        inv_trans = prior.inv_trans
        if prior.conditional:
            inv_trans = inv_trans[y[0]]
        if inv_trans.ndim < 2:
            u0 = (inv_trans * z0.unsqueeze(0))
            m0 = (inv_trans * prior.mean[y[0]].unsqueeze(0))
        else:
            u0 = torch.matmul(inv_trans, z0.unsqueeze(-1)).squeeze(-1)
            m0 = torch.matmul(inv_trans, prior.mean[y[0]].unsqueeze(-1)).squeeze(-1)
        print('A^-1.z of shape', *u.shape)
        print('u err={:.2%}'.format((u[0] - u0).norm() / u0.norm()))

        print()
        d = prior.mahala(z, y)
        print('||z|| of shape', *d.shape)
        d0 = (u0 - m0).pow(2).sum()
        print('||.|| err={:.2e}'.format(d[0] - d0))

        print()
        tr = prior.trace_prod_by_var(v, y)

        inv_var = prior.inv_var
        if prior.conditional:
            inv_var = inv_var[y[0]]
        if inv_var.ndim > 1:
            inv_var = inv_var.diag()
        tr0 = (v0 * inv_var).sum()
        print('trace of shape', *tr.shape)
        print('trace err={:.2e}'.format(tr[0] - tr0))

        print()
        ld = prior.log_det_per_class()
        print('logDet of shape', *ld.shape)

        print()

        log_p = prior.log_density(z, y)
        p = log_p.exp()
        print('log dens shape', *log_p.shape)
        if p.isnan().any():
            print('p is nan')
        if p.isinf().any():
            print('p is inf')

        if C > 1:
            v0 = u0 - m0
            log_p0 = - np.log(2 * np.pi) * K / 2 - ld[0] / 2 - v0.norm() ** 2 / 2
            print('logp err: {:.2f}'.format((log_p0 - log_p[0]) / log_p0))
