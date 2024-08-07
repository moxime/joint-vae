import torch
from module.priors import build_prior

K = 16
N = 13
L = 11

C = 10

y = None

if C > 1:
    y = torch.randint(C, (N, L,))

p = build_prior(K, num_priors=C, init_mean=3)


z = torch.randn(N, L, K)

d = p.mahala(z, y)

print(*d.shape)
