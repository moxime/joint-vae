import torch
from torch import nn


def onehot_encoding(y, C):

    s_y = y.shape
    s_ = s_y + (1,)
    s = s_y + (C,)

    # print('vae_l l. 16 y:', y.shape, 's_', s_)
    y_onehot = torch.zeros(s, device=y.device)
    # y_onehot = torch.LongTensor(s)
    # y_onehot.zero_()
    y_onehot.scatter_(-1, y.reshape(s_), 1)

    return y_onehot


def _no_activation(a):
    return a


activation_layers = {'linear': nn.Identity,
                     'sigmoid': nn.Sigmoid,
                     'relu': nn.ReLU,
                     'leaky': nn.LeakyReLU}
