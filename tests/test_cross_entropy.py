import torch
from torch.nn import functional as F
from module.vae_layers import build_de_conv_layers

K = 512

deconv = build_de_conv_layers(K, 'deconv32', where='output', output_distribution='categorical')
conv = build_de_conv_layers((3, 32, 32), 'conv32')
lin = torch.nn.Linear(800, K)

image_shape = (3, 32, 32)
batch_shape = (10, 7)

device = 'cpu'
device = 'cuda'

conv.to(device)
deconv.to(device)
lin.to(device)

x = torch.rand(*batch_shape, *image_shape, device=device)

z = lin(conv(x.view(-1, *image_shape)).view(*batch_shape, -1))

x_ = deconv(z.view(-1, K, 1, 1)).view(-1, 256, *image_shape)


x_target = (x * 255).long().unsqueeze(-1)

reduction = 'mean'
reduction = 'none'

ce = F.cross_entropy(x_,
                     x_target.view(-1, *image_shape), reduction=reduction).sum((-3, -2, -1)).view(*batch_shape)

print('ce', *ce.shape)
