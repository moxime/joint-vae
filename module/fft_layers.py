import torch
from torch import nn
import numpy as np
from torch.nn import functional as F, Parameter
from utils.print_log import texify_str
import logging
from torchvision import models


def phase(X):

    return X.angle()


def sup_module(X):

    return X / X.abs()


def module(X):
    return X.abs()


def imodule(X):
    return torch.fft.ifft(X.abs()).real


def iphase(X):
    return torch.fft.ifft(X / X.abs()).real


transforms = {'module': module, 'phase': phase, 'phase~': sup_module, 'imodule': imodule, 'iphase': iphase}


class FFTFeatures(nn.Module):

    def __init__(self, input_shape, P=1, which='module', **kw):

        super().__init__(**kw)

        self.input_shape = input_shape
        self.P = P
        self.which = which

        self.output_shape = (P * input_shape[-2], P * input_shape[-1])

        if input_shape[0] > 1:
            conv = nn.Conv2d(input_shape[0], 1, 1, stride=1, bias=False)
            conv.weight = nn.Parameter(torch.ones_like(conv.weight), requires_grad=False)
            self.gs = conv
        else:
            self.gs = None

    def forward(self, x):

        if self.gs:
            x = self.gs(x)

        x.squeeze_(-3)

        X = torch.fft.fft2(x, s=self.output_shape)

        return transforms[self.which](X)
