import torch
from torch import nn
from module.vae_layers.conv import _parse_conv_layer_name
from module.vae_layers.conv import _conv_layer_name

from module.vae_layers import ConvFeatures, ConvDecoder, build_de_conv_layers


shape = (1, 28, 28)
shape = (3, 32, 32)

conv_name = '4x3-ax2'
conv_name = '4x3:1+4-ax2'
conv_name = '64x3-Mx2-128x3-Mx2-256x3-256x3-Mx2-512x3-512x3-Mx2-512x3-512x3-Mx2-Ax1'

conv_name = '[x3-Mx2]64-M-128-M-256-256-M-512-512-M-512-512-M-Ax1'

conv = build_de_conv_layers(shape, conv_name)

f = conv(torch.randn(1, *shape))

print('output shape for', conv.name)
print(*f.shape[1:], '--', *conv.output_shape)
conv_ = str(conv).split('\n')


deconv_name = ('64x8-'
               '64x5+2-'
               '64x5:2+2++1-'
               '32x5+2-'
               '32x5:2+2++1-'
               '32x5+2-'
               '!3x5+2')

deconv = build_de_conv_layers((512, 1, 1), 'deconv32', where='output')

deconv.to('cuda')

for n in range(5, 20):
    print('***', 2**n)
    x = deconv(torch.randn(2**n, 512, 1, 1, device='cuda'))
print('output shape for', deconv.name)
print(*x.shape[1:], '--',  *deconv.output_shape)

# conv_name = '[x5+2]32-32:2-64-64:2-256x7+0'
# conv = make_de_conv_features((3, 32, 32), conv_name)
# f = conv(torch.randn(1, *shape))

# print('output shape for', conv.name)
# print(*f.shape[1:], '--', *conv.output_shape)
