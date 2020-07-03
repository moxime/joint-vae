from cvae import ClassificationVariationalNetwork as Net
import utils.losses as losses
import torch
import torch.nn.functional as F

C = 2
D = (11, 5)
K = 7
L = 10000

N = 50


x = torch.randn(N, *D)

y = torch.randint(0, C, (N,))


beta = 1e-2
net = Net(D, C, latent_dim=K, latent_sampling=L, beta=beta)

# y_ = torch.cat([torch.zeros(1, N), torch.ones(1, N)], axis=0)

_x_, logit_, mu_z, lv_z, z_ = net(x, y)
y_ = F.softmax(logit_, -1)



mse_loss = ((_x_ - x) ** 2).mean()
x_loss = 0

for n in range(N):
    y_n = y[n]
    x_loss += - y_[:, n, y_n].log().mean(axis=0) / N

kl_loss = 0.5 * (lv_z.exp() + mu_z ** 2 - 1 - lv_z).sum(axis=-1).mean()

total_loss = mse_loss + 2 * beta * x_loss + 2 * beta * kl_loss

losses = net.loss(x, y, _x_, y_, mu_z, lv_z, return_all_losses=True)

_x, logit, loss = net.evaluate(x)



