import torch
from torch.nn import functional as F
from module.vae_layers import build_de_conv_layers
from module.losses import categorical_loss

K = 64

L = 8

deconv = build_de_conv_layers(K, 'deconv32', where='output', output_distribution='categorical')

conv = build_de_conv_layers((3, 32, 32), 'conv32')
lin = torch.nn.Linear(800, K)

image_shape = (3, 32, 32)
batch_shape = (10, 7)

device = 'cuda'
device = 'cpu'

conv.to(device)
deconv.to(device)
lin.to(device)


x = torch.rand(*batch_shape, *image_shape, device=device)


f = conv(x.view(-1, *image_shape))
z = lin(f.view(*batch_shape, -1))

z = z.expand(L, *z.shape).contiguous()

batch_shape = z.shape[:-1]
x_ = deconv(z.view(-1, K, 1, 1)).view(*batch_shape, -1, *image_shape).squeeze()

print(*x_.shape)
ce = categorical_loss(x_, x, batch_mean=False)

print('ce', *ce.shape)
