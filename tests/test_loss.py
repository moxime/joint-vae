from cvae import ClassificationVariationalNetwork as Net
import utils.losses as lf
import torch
import torch.nn.functional as F
from itertools import product

C = 2
D = (1, 28, 28)
K = 9
L = int(1e4)

N = (4,)


x = torch.randn(*N, *D)
y = torch.randint(0, C, N)


type_ = 'vae'
beta = 1e-2
net = Net(D, C, latent_dim=K, latent_sampling=L, beta=beta, type_of_net=type_)

_x_, logit_, mu_z, lv_z, z_ = net.forward(x, y)
print(logit_.shape)

y_ = F.softmax(logit_, -1)

loss_ = net.loss(x, y, _x_, y_, mu_z, lv_z, return_all_losses=True)

_x, logit, loss = net.evaluate(x, return_all_losses=True)
print(logit.shape)

if type_ != 'vae':
    y_pred = net.predict_after_evaluate(logit, loss)
