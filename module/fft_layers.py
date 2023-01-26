import torch
from torch import nn
import numpy as np
from torch.nn import functional as F, Parameter
from utils.print_log import texify_str
import logging
from torchvision import models


def phase(X):

    return X.angle()


def module(X):
    return X.abs()


def real(X):
    return X.real


def imag(X):
    return X.imag


def imodule(X):
    return torch.fft.ifft(X.abs()).real


def iphase(X):
    return torch.fft.ifft(X / X.abs()).real


transforms = {'module': module, 'phase': phase,
              'real': real, 'imag': imag,
              'imodule': imodule, 'iphase': iphase}

order = list(transforms)


class FFTFeatures(nn.Module):

    def __init__(self, input_shape, P=1, which=['module'], **kw):

        super().__init__(**kw)

        which = sorted(which, key=order.index)

        self.input_shape = input_shape
        self.P = P
        self.which = which

        self.output_shape = (P * input_shape[-2] * len(which), P * input_shape[-1])
        self.fft_size = (P * input_shape[-2], P * input_shape[-1])

        if input_shape[0] > 1:
            conv = nn.Conv2d(input_shape[0], 1, 1, stride=1, bias=False)
            conv.weight = nn.Parameter(torch.ones_like(conv.weight), requires_grad=False)
            self.gs = conv
        else:
            self.gs = None

    def forward(self, x):

        if self.gs:
            x = self.gs(x)

        # x.squeeze_(-3)

        X = torch.fft.fft2(x, s=self.fft_size)

        f_ = {_: transforms[_](X).squeeze(-3) for _ in self.which}

        # for _ in self.which:
        #     print(_, *f_[_].shape)

        return torch.cat(tuple(f_[_] for _ in self.which), dim=1)

    def __repr__(self):
        i = ','.join(str(_) for _ in self.input_shape)
        o = 'x'.join(str(_) for _ in self.output_shape)
        o_ = 1
        for _ in self.output_shape:
            o_ *= _
        w = '|'.join(self.which)
        return 'FFTFeatures(input_shape=({}), output_shape=({}={}), which={})'.format(i, o, o_, w)
