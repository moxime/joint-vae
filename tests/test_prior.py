import torch
from module.vae_layers import Prior
from matplotlib import pyplot as plt
import time
import numpy as np

var_type = 'scalar'
var_type = 'full'

learned_mean = False
learned_mean = True
learned_var = False
learned_var = True

K = 256
N = 100
C = 10

lr = 1e-3

device = 'cpu'
device = 'cuda'

prior = Prior(K, var_type=var_type, num_priors=C, learned_mean=learned_mean, learned_var=learned_var)


prior.to(device)

# var_per_dim = torch.randn(C, K) ** 2
var_per_dim = torch.stack([(i * 0.5 + 1) * torch.ones(K) for i in range(C)])

mu_per_dim = torch.randn(C, K) 

mu = torch.zeros(N, K)

optimizer = torch.optim.SGD(prior.parameters(), lr=lr)

losses = []

dev = 0.1
show_every = 100

t0 = time.time()

former_params = None

# torch.autograd.set_detect_anomaly(True)

losses_components = None

for epoch in range(int(1e5)):

    optimizer.zero_grad()
    
    if C > 1:
        y = torch.randint(C, (N,))
    else:
        y = None

    var = var_per_dim.index_select(0, y.view(-1)) * (1 + dev * torch.randn(N, K))
    mu = mu_per_dim.index_select(0, y.view(-1)) + dev * torch.randn(N, K)

    log_var = var.log()

    previous_loss = losses_components
    losses_components = prior.kl(mu.to(device), log_var.to(device), y.to(device))

    # break
    if losses_components is None:
        break
        
    loss = losses_components['kl'].mean()

    loss.backward()

    optimizer.step()
    
    loss = loss.cpu()
    trace = losses_components['trace'].mean().cpu().item() - K
    distance = losses_components['distance'].mean().cpu().item()
    
    if not epoch % show_every:
        t = (time.time() - t0) / N / show_every
        
        print('{:6d}: L={:.3e} tr={:+.3e} d={:.3e} ({:.0f}us/i)'.format(epoch, loss.item(), trace, distance, t * 1e6))
        t0 = time.time()
            
    losses.append(loss.item())
    if loss < 1e-2:
        break

