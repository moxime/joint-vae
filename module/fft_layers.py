import torch
from torch import nn
import numpy as np
from torch.nn import functional as F, Parameter
from utils.print_log import texify_str
import logging
from torchvision import models


def phase(X, half=False):

    L = X.shape[-1]
    if half:
        L //= 2
    return X.angle()[..., :L]


def module(X, half=False):
    L = X.shape[-1]
    if half:
        L //= 2
    return X.abs()[..., :L]


def real(X):
    L = X.shape[-1]
    if half:
        L //= 2
    return X.real[..., :L]


def imag(X):
    L = X.shape[-1]
    if half:
        L //= 2
    return X.imag[..., :L]


def imodule(X, half=True):
    L = X.shape[-1]
    if half:
        L //= 2
    return torch.fft.ifft(X.abs()).real[..., :L]


def iphase(X):
    return torch.fft.ifft(X / X.abs()).real


transforms = {'module': module, 'phase': phase,
              'real': real, 'imag': imag,
              'imodule': imodule, 'iphase': iphase}

order = list(transforms)


class FFTFeatures(nn.Module):

    def __init__(self, input_shape, P=1, which=['module'], **kw):

        super().__init__(**kw)

        which_ = []
        half_ = {}

        for w in which:
            if w.endswith('*'):
                which_.append(w[:-1])
                half_[w[:-1]] = True
            else:
                which_.append(w)
                half_[w] = False

        which_ = sorted(which_, key=order.index)

        self._which = which_
        self._half = half_
        
        self.input_shape = input_shape
        self.P = P

        self.which = [w + ('*' if half_[w] else '') for w in which_]

        shape_mult_last_dim = sum([P / (2 if half_[_] else 1) for _ in which_])

        self.output_shape = (P * input_shape[-2], int(shape_mult_last_dim * input_shape[-1]))
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

        f_ = {_: transforms[_](X, half=self._half[_]).squeeze(-3) for _ in self._which}

        # for _ in self.which:
        #     print(_, *f_[_].shape)

        return torch.cat(tuple(f_[_] for _ in self._which), dim=-1)

    def __repr__(self):
        i = ','.join(str(_) for _ in self.input_shape)
        o = 'x'.join(str(_) for _ in self.output_shape)
        o_ = 1
        for _ in self.output_shape:
            o_ *= _
        w = '|'.join(self.which)
        return 'FFTFeatures(input_shape=({}), output_shape=({}={}), which={})'.format(i, o, o_, w)
