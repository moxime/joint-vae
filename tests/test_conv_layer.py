import torch
from torch import nn
from module.vae_layers import build_de_conv_layers

device = 'cpu'

shape = (1, 28, 28)
shape = (3, 32, 32)

conv_name = '4x3-ax2'
conv_name = '4x3:1+4-ax2'
conv_name = '64x3-Mx2-128x3-Mx2-256x3-256x3-Mx2-512x3-512x3-Mx2-512x3-512x3-Mx2-Ax1'
conv_name = '[x3-Mx2]64-M-128-M-256-256-M-512-512-M-512-512-M-Ax1'
conv_name = '[x5+2]64-64:2-128-128:2-256-256:2-200x3+0'
conv_name = 'conv32+'

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

deconv_name = '[x5+2]64x4+0-64-64:2++1-32-32:2++1-32-32:2++1-!3x5+2'
deconv_name = '[x5+2]64x8+0-64-64:2++1-32-32:2++1-32-!3x5+2'
deconv_name = '[x5+2]256x4+0-256-256:2++1-128-128:2++1-64-64:2++1-32-!3x5+2'
deconv_name = 'deconv32+'

deconv = build_de_conv_layers((512, 1, 1), deconv_name, where='output')
deconv.to(device)

x = deconv(torch.randn(1, 512, 1, 1, device=device))
print('output shape for', deconv.name)
print(*x.shape[1:], '--', *deconv.output_shape)


conv_layers = [str(c) for c in conv.children() if 'conv' in str(c).lower()]

print('====X====')
print(*conv.shapes[0])
for c, s in zip(conv_layers, conv.shapes[1:]):

    print('        ', c)
    print(*s)
print('====Z====')

deconv_layers = [str(c) for c in deconv.children() if 'conv' in str(c).lower()]

print('====Z====')
print(*deconv.shapes[0])
for c, s in zip(deconv_layers, deconv.shapes[1:]):

    print('        ', c)
    print(*s)
print('====X====')
# conv_name = '[x5+2]32-32:2-64-64:2-256x7+0'
# conv = make_de_conv_features((3, 32, 32), conv_name)
# f = conv(torch.randn(1, *shape))

# print('output shape for', conv.name)
# print(*f.shape[1:], '--', *conv.output_shape)
