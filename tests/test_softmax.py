import torch
from torch import nn
from torch.nn import functional as F
from module.vae_layers import Encoder
import time

prior_dist = 'uniform'
prior_dist = 'gaussian'

classif = 'softmax'
classif = 'linear'

torch.manual_seed(42)

P = 16

L = 1
L = 16
C = 10
N = 100
K = 16

coder = Encoder(P, C, latent_dim=K, sampling_size=L,
                intermediate_dims=[32, 16],
                prior={'num_priors': C, 'distribution': prior_dist, 'init_mean': 1})


classifier = nn.Linear(K, C)

my = coder.prior.mean

print(coder.prior)
print('my:', *my.shape)


mu_ = torch.randn(C, P)


f_ = 0.01 * torch.randn(N, P)
y = torch.randint(0, C, (N,))

f = f_ + torch.index_select(mu_, 0, y)

z_mean, z_log_var, z, e, _ = coder(f)

print(' z:', *z.shape)


def logp_z_y(z):
    ones = torch.ones(z.shape[:2], dtype=int)
    return torch.cat(tuple((coder.prior.log_density(z, _ * ones)).unsqueeze(-1) for _ in range(C)), dim=-1)


if prior_dist == 'uniform':
    logp = logp_z_y(z)

    print(logp.shape)

    p = F.log_softmax(logp, dim=-1)

optimizer = torch.optim.Adam([*coder.parameters(), *classifier.parameters()])
criterion = torch.nn.CrossEntropyLoss()

kl_w = 1


n_epochs = 200

t0 = time.time()

for epoch in range(n_epochs):

    optimizer.zero_grad()

    f_ = 0.1 * torch.randn(N, P)
    y = torch.randint(0, C, (N,))

    f = f_ + torch.index_select(mu_, 0, y)

    z_mean, z_log_var, z, e, _ = coder(f)

    if classif == 'linear':
        logp = classifier(z[1:])
    else:
        logp = logp_z_y(z[1:])

    y_ = y.view(1, -1).repeat(L, 1).view(-1)
    cross_y = criterion(logp.view(-1, C), y_)
    kl = coder.prior.kl(z_mean, z_log_var, y, output_dict=False).mean()
    loss = cross_y + kl_w * kl

    p = F.log_softmax(logp, dim=-1).mean(0)

    acc = (y == p.argmax(-1).view(-1)).float().mean()

    loss.backward()

    optimizer.step()

    if not epoch % 100:
        print('{e:5}: {l:7.3} = {c:7.3} + {w}x{k:7.3} -- acc={a:.1%}'.format(e=epoch,
                                                                             l=loss,
                                                                             c=cross_y,
                                                                             k=kl,
                                                                             a=acc,
                                                                             w=kl_w))

    p = F.log_softmax(logp, dim=-1)

t = (time.time() - t0) / (N * n_epochs)

print('Acc = {:.1%}'.format(acc)

print('time of {} {}: {:.0}ms/i'.format(classif, prior_dist, t * 1000))
