import torch
from module.rst.module import InterpolExtract
import matplotlib.pyplot as plt

P = 32
T = 256

K = 2048
L = K

# image = torch.linspace(1, K, K)[:, None] + 100 * torch.linspace(1, L, L)[None, :]

C_k = (torch.linspace(0, K - 1, K)[:, None] + K // 2) % K - K // 2
C_l = (torch.linspace(0, L - 1, L)[None, :] + L // 2) % L - L // 2

C_r = ((C_k / K) ** 2 + (C_l / L) ** 2).sqrt()
C_theta = (torch.atan(C_l * K / C_k / L) * 180 / torch.pi - 0) % 180 - 0
C_theta[C_theta.isnan()] = 0.

q = (1/8) ** (1/P)
rho = torch.tensor([q ** _ for _ in range(1, P+1)]) * 0.5
# rho = torch.linspace(1/P, 1, P) * 0.5
theta = torch.linspace(0, 1-1/T, T) * torch.pi
theta_deg = theta * 180 / torch.pi
k_ = K * rho[:, None] * torch.cos(theta[None, :])
l_ = L * rho[:, None] * torch.sin(theta[None, :])

# x = torch.linspace(-0.5, 0.5, P)[:, None] + 0.5
# y = torch.linspace(-0.5, 0.5, T)[None, :] + 0.5

k_ = torch.clamp(k_ % K, 0, K - 1)
l_ = torch.clamp(l_ % L, 0, L - 1)

coordinates_k = torch.LongTensor(len(rho), len(theta), 2, 2)
coordinates_l = torch.LongTensor(len(rho), len(theta), 2, 2)

weights = torch.Tensor(len(rho), len(theta), 2, 2)

k_0 = torch.floor(k_).type(torch.LongTensor)
k_1 = torch.clamp(k_0 + 1, 0, K - 1)
l_0 = torch.floor(l_).type(torch.LongTensor)
l_1 = torch.clamp(l_0 + 1, 0, L - 1)


""" Plane coordinate description
________________________________
|                               |
|           l0          l1      |
|         ==========l========   |
| k_0 ||   [0, 0]      [0, 1]   |
|     ||                        |
|     k                         |
|     ||                        |
| k_1 ||  [1, 0]      [1, 1]    |
|_______________________________|

"""


coordinates_k[..., 0, 0] = k_0
coordinates_l[..., 0, 0] = l_0
coordinates_k[..., 1, 0] = k_1
coordinates_l[..., 1, 0] = l_0
coordinates_k[..., 0, 1] = k_0
coordinates_l[..., 0, 1] = l_1
coordinates_k[..., 1, 1] = k_1
coordinates_l[..., 1, 1] = l_1


weights_k_0 = (k_1 - k_)
weights_k_0[k_1 == k_0] = 1.
weights_k_1 = (k_ - k_0)

weights_l_0 = (l_1 - l_)
weights_l_0[l_1 == l_0] = 1.
weights_l_1 = (l_ - l_0)


weights[..., 0, 0] = weights_k_0 * weights_l_0
weights[..., 1, 0] = weights_k_1 * weights_l_0
weights[..., 0, 1] = weights_k_0 * weights_l_1
weights[..., 1, 1] = weights_k_1 * weights_l_1


""" [k, l] <=> L * k + l """
coordinates_ = coordinates_k * L + coordinates_l

interpolated = {}
image = {'theta': C_theta, 'r': C_r}
for _ in image:
    values = image[_].view(-1).index_select(0, coordinates_.view(-1)).view(*coordinates_.shape)
    interpolated[_] = (weights * values).sum((-1, -2))
