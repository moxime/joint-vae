from cvae import ClassificationVariationalNetwork
import data.torch_load as dl
from utils.save_load import find_by_job_number, LossRecorder
import torch
from sklearn.metrics import auc, roc_curve
import numpy as np
from matplotlib import pyplot as plt
import logging
import os
import sys

# from torchvision.transforms import ToPILImag


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
    
    per_dim = options.pop('per_dim', False)
    if result_type == 'hist_of_var':
        log_scale = options.pop('log_scale', False)
        data = var_z.log().cpu() if log_scale else var_z.cpu()
        if per_dim:
            data = data.mean(0)
        hist, bins = np.histogram(data, **options)
        if log_scale:
            bins = np.exp(bins)

        plot, write, close = _create_output(output, pltf='bar')
        plot(bins[:-1], hist, align='edge')  # 
        
        write('edge          num\n')
        for b, v in zip(bins[:-1], hist):
            write(f'{b:-13.6e} {v:-12g}\n')
        write(f'{bins[-1]:-13.6e} {0:-12g}\n')

        close()

    if result_type == 'scatter':
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


def losses_history(dict_of_losses,
                   graph='histogram',  # or boxplot
                   output=sys.stdout,
                   **opt,):
    
    bins = opt.pop('bins', 10)
    quantiles = opt.pop('quantiles', [0.05, 0.25, 0.5, 0.75, 0.95])

    if graph == 'histogram':
        plot, write, close = _create_output(output,
                                            pltf='bar',
                                            # pltf='plot',
        )

        hist = {k: np.histogram(dict_of_losses[k], bins=bins) for k in dict_of_losses}        

        write(' '.join(['edge-{k:<8} num-{k:<7}'.format(k=k) for k in dict_of_losses]))
        write('\n')

        for b in range(bins - 1):
            write(' '.join(['{e:-13.6e} {v:-12g}'.format(e = hist[k][1][b],
                                                           v = hist[k][0][b])
                            for k in dict_of_losses]))
            write('\n')
            
        write(' '.join(['{e:-13.6e} {v:-12g}'.format(e = hist[k][1][-1],
                                                       v = 0)
                        for k in dict_of_losses]))
        write('\n')

        for k in dict_of_losses:
            plot(hist[k][1][:-1], hist[k][0], label=k) # , align='edge')
        
    if graph == 'boxplot':
        plot, write, close = _create_output(output, pltf='boxplot')

        
        
if __name__ == '__main__':

    plt.clf()
    plt.close()
    N = 500
    K = 16
    mu_z = torch.randn(N, K)
    var_z = (1 - mu_z)**2 + torch.randn(N, K)**2

    dict_of_losses = {'svhn': svhn_r._tensors['total'].cpu(),
                      'lsun': lsun_r._tensors['total'].cpu()}

    # dict_of_losses.pop('svhn')

    output=None
    plot = True
    if plot:
        f, a = plt.subplots()
        output = a
    losses_history(dict_of_losses, bins=20)
    losses_history(dict_of_losses, bins=200, output=output)

    a.legend()

    if plot:
        f, a = plt.subplots()
        output = a

    latent_distribution(mu_z, var_z,
                        # result_type='scatter',
                        result_type='hist_of_var',
                        per_dim=True,
                        output=output)

    plt.show()

    
