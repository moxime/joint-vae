import torch
from module.vae_layers import Prior
from matplotlib import pyplot as plt
import time
import numpy as np

var_type = 'scalar'
var_type = 'full'

learned_mean = True
learned_var = False
learned_var = True

K = 512
N = 500
C = 10

lr = 1e-2

device = 'cpu'
device = 'cuda'

prior = Prior(K, var_type=var_type, num_priors=C, learned_mean=learned_mean, learned_var=learned_var)

prior.to(device)

# var_per_dim = torch.randn(C, K) ** 2
var_per_dim = torch.stack([(i + 1) * torch.ones(K) for i in range(C)])

mu_per_dim = torch.randn(C, K)

mu = torch.zeros(N, K)

optimizer = torch.optim.SGD(prior.parameters(), lr=lr)

losses = []

dev = 0
show_every = 10

t0 = time.time()

former_params = None

torch.autograd.set_detect_anomaly(True)

for epoch in range(int(1e5)):

    if C > 1:
        y = torch.randint(C, (N,))
    else:
        y = None

    var = var_per_dim.index_select(0, y.view(-1)) * (1 + dev * torch.randn(N, K))
    mu = mu_per_dim.index_select(0, y.view(-1)) + dev * torch.randn(N, K)

    log_var = var.log()
    
    losses_components = prior.kl(mu.to(device), log_var.to(device), y.to(device))

    if losses_components is None:
        print('*** max_grad = {:.1e}'.format(max_grad))
        break
        
    loss = losses_components['kl'].mean()

    max_eigen = []
    min_eigen = []
    
    inv_var = prior.inv_var

    if inv_var.isnan().any():
        inv_var = former_params
        break
    else:
        former_params = inv_var

    loss.backward()
    
    is_nan_before = any([_.isnan().any() for _ in prior.parameters()])
    for p in prior.parameters():
        if p.requires_grad:
            p.data = p.data - lr * p.grad
    is_nan_after = any([_.isnan().any() for _ in prior.parameters()])

    max_grad = max(_.grad.norm() for _ in prior.parameters())
    if is_nan_after:
        print('*** IS NAN NOW', is_nan_before)
        print(max_grad)
        break

    for p in prior.parameters():
        p.grad.data.zero_()
    
    loss = loss.cpu()
    if not epoch % show_every:
        t = (time.time() - t0) / N / show_every
        
        print('{:6d}: {:.3e} ({:.0f}us/i)'.format(epoch, loss.item(), t * 1e6))
        t0 = time.time()
            
    losses.append(loss.item())
    if loss < 1e-1:
        break

# plt.plot(losses[100:])
# plt.show(block=False)


for y in range(C):

    print('****', y, '****')
    s = inv_var[y].inverse()
    print('{}'.format(s.isnan().any()))
    # eig = s.eig()[0].norm(dim=1)
    # print('{:.1e} <= eig <= {:.1e}'.format(min(eig), max(eig)))

