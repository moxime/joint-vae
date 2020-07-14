from cvae import ClassificationVariationalNetwork as Net
import utils.losses as lf
import torch
import torch.nn.functional as F
from itertools import product

C = 2
D = (1, 28, 28)
K = 7
L = 11

N = (5,)


x = torch.randn(*N, *D)
y = torch.randint(0, C, N)


beta = 1e-2
net = Net(D, C, latent_dim=K, latent_sampling=L, beta=beta)

_x_, logit_, mu_z, lv_z, z_ = net.forward(x, y)
print(logit_.shape)

y_ = F.softmax(logit_, -1)

loss_ = net.loss(x, y, _x_, y_, mu_z, lv_z, return_all_losses=True)

_x, logit, loss = net.evaluate(x, return_all_losses=True)
print(logit.shape)
