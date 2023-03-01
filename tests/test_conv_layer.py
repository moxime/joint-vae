import torch
from torch import nn
from module.vae_layers.conv import _parse_conv_layer_name
from module.vae_layers.conv import _conv_layer_name

from module.vae_layers import VGGFeatures, ConvFeatures, ConvDecoder, make_de_conv_features


shape = (1, 28, 28)
shape = (3, 32, 32)

conv_name = '4x3-ax2'
conv_name = '4x3:1+4-ax2'
conv_name = '64x3-Mx2-128x3-Mx2-256x3-256x3-Mx2-512x3-512x3-Mx2-512x3-512x3-Mx2-Ax1'

conv_name = '[x3-Mx2]64-M-128-M-256-256-M-512-512-M-512-512-M-Ax1'

conv = make_de_conv_features(shape, conv_name)

f = conv(torch.randn(1, *shape))

print('output shape for', conv.name)
print(*f.shape[1:], '--', *conv.output_shape)

vgg = VGGFeatures('vgg11', (3, 32, 32))

vgg_ = str(vgg).split('\n')
conv_ = str(conv).split('\n')

for l1, l2 in zip(vgg_[1:], conv_[1:]):
    if l1 != l2:
        print('V', l1)
        print('C', l2)
        print()

deconv_name = ('64x8-'
               '64x5+2-'
               '64x5:2+2++1-'
               '32x5+2-'
               '32x5:2+2++1-'
               '32x5+2-'
               '3x5+2')

deconv = make_de_conv_features((512, 1, 1), deconv_name, where='output')

x = deconv(torch.randn(2, 512, 1, 1))
print('output shape for', deconv.name)
print(*x.shape[1:], '--',  *deconv.output_shape)

conv_name = '[x5+2]32-32:2-64-64:2-256x7+0'
conv = make_de_conv_features((3, 32, 32), conv_name)
f = conv(torch.randn(1, *shape))

print('output shape for', conv.name)
print(*f.shape[1:], '--', *conv.output_shape)
