import torch
from module.vae_layers import Prior
from matplotlib import pyplot as plt
import time
var_type = 'scalar'
var_type = 'full'

learned_mean = True
learned_var = False
learned_var = True

K = 16
N = 50
C = 10

device = 'cuda'

prior = Prior(K, var_type=var_type, num_priors=C, learned_mean=learned_mean, learned_var=learned_var)

prior.to(device)

# var_per_dim = torch.randn(C, K) ** 2
var_per_dim = torch.stack([(i + 1) * torch.ones(K) for i in range(C)])

mu_per_dim = torch.randn(C, K)

mu = torch.zeros(N, K)

optimizer = torch.optim.SGD(prior.parameters(), lr=0.01)

losses = []

dev = 0
show_every = 100

t0 = time.time()

for epoch in range(int(1e5)):

    if C > 1:
        y = torch.randint(C, (N,))
    else:
        y = None

    var = var_per_dim.index_select(0, y.view(-1)) * (1 + dev * torch.randn(N, K))
    mu = mu_per_dim.index_select(0, y.view(-1)) + dev * torch.randn(N, K)

    log_var = var.log()

    optimizer.zero_grad()
    
    distance, trace, log_det, log_det_prior, kl = prior.kl(mu.to(device), log_var.to(device), y.to(device))

    loss = kl.mean()

    loss.backward()

    loss = loss.cpu()
    if not epoch % show_every:
        t = (time.time() - t0) / N / show_every
        
        print('{:6d}: {:.3e} ({:f}s/i)'.format(epoch, loss.item(), t))
        t0 = time.time()
        
    optimizer.step()

    losses.append(loss.item())
    if loss < 1e-1:
        break

plt.plot(losses[100:])
plt.show()

for y in range(C):

    print('****', y, '****')
    s = prior.inv_var[y]
    for k in range(K):
        print(' '.join(['{:-8.2g}'.format(_.item()) for _ in s[k]]))
