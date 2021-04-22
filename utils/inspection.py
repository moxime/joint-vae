from cvae import ClassificationVariationalNetwork
import data.torch_load as dl
from utils.save_load import find_by_job_number
import torch
from sklearn.metrics import auc, roc_curve
import numpy as np
from matplotlib import pyplot as plt
import logging
import os
import sys

from torchvision.transforms import ToPILImage


def _create_output(o, pltf='plot'):

    def _close(*a):
        return
    
    def _write(a):
        return

    def _plot(*a, **kw):
        return
    
    if isinstance(o, type(sys.stdout)):

        def _write(*a):
            sys.stdout.write(*a)
            return
        
    elif isinstance(o, str):
        f = open(o, 'w')

        def _close():
            f.close()
            return
        
        def _write(a):
            f.write(a)
            return
        
    elif isinstance(o, plt.Axes):

        def _plot(*a, **kw):
            getattr(o, pltf)(*a, **kw)
            return
        
    return _plot, _write, _close
    

def latent_distribution(mu_z, var_z, result_type='hist_of_var',
                        output=sys.stdout, **options):
    r"""result_type can be:

        -- hist_of_var: hisotgram of variance, options can be options
           of numpy.histogram and log_scale=bool (default=False)

    """

    N = mu_z.shape[0]
    K = mu_z.shape[1]
    
    if result_type == 'hist_of_var':
        log_scale = options.pop('log_scale', False)
        data = var_z.log().cpu() if log_scale else var_z.cpu()
        hist, bins = np.histogram(data, **options)
        if log_scale:
            bins = np.exp(bins)

        plot, write, close = _create_output(output, pltf='bar')
        plot(bins[:-1], hist, align='edge')
        write('edge          num\n')
        for b, v in zip(bins[:-1], hist):
            write(f'{b:-13.6e} {v:-12g}\n')
        close()

    if result_type == 'scatter':
        per_dim = options.pop('per_dim', False)
        if per_dim:
            x_title = 'var_mu_z'
            y_title = 'mu_var_z'
            x = mu_z.var(0).cpu()
            y = var_z.mean(0).cpu()
        else:
            x_title = 'mu_z'
            y_title = 'var_z'
            x = mu_z.view(-1).cpu()
            y = var_z.view(-1).cpu()

        plot, write, close = _create_output(output, pltf='scatter')
        plot(x, y)
        write(f'{x_title}   {y_title}\n')
        for a, b in zip(x, y):
            write(f'{a:-13.6e} {b:-12g}\n')
        close()


if __name__ == '__main__':

    plt.clf()
    plt.close()
    N = 500
    K = 16
    mu_z = torch.randn(N, K)
    var_z = (1 - mu_z)**2 + torch.randn(N, K)**2

    output=None
    plot = True
    if plot:
        f, a = plt.subplots()
        output = a
    
    latent_distribution(mu_z, var_z,
                        # result_type='scatter',
                        # per_dim=True,
                        output=output)

    plt.show()
