from cvae import ClassificationVariationalNetwork
import data.torch_load as dl
from utils.save_load import find_by_job_number, LossRecorder, job_to_str
import torch
from sklearn.metrics import auc, roc_curve
import numpy as np
from matplotlib import pyplot as plt
import logging
import os
import sys
import argparse

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

    
def loss_comparisons(net, root='results/%j/losses'):

    sample_directory = os.path.join(net.saved_dir, 'samples', 'last')
    root = job_to_str(net.job_number, root)
    
    if not os.path.exists(sample_directory):
        logging.warning(f'Net #{net.job_number} has no recorded loss')
        return
    if not os.path.exists(root):
        os.makedirs(root)

    testset = net.training['set']
    datasets = [testset] + list(net.ood_results.keys())

    losses = {}
    logits = {}
    y_pred = {}
    
    for s in datasets:
        try:
            r = LossRecorder.load(os.path.join(sample_directory, f'record-{s}.pth'))
            r.to('cpu')
            losses[s] = r._tensors
            logits[s] = losses[s].pop('logits').T
        except FileNotFoundError:
            logging.warning(f'Recorder for set {s} does not exist')
            return
        y_pred[s] = net.predict_after_evaluate(logits[s], losses[s])
        
    y_true = losses[testset].pop('y_true')
    i_miss = np.where(y_true != y_pred[testset])[0]
    i_true = np.where(y_true == y_pred[testset])[0]
    
    for s in losses:
        for k in losses[s]:
            if len(losses[s][k].shape) == 2:
                losses[s][k] = losses[s][k].gather(0, y_pred[s].unsqueeze(0)).squeeze()
                
    for w, i in zip(('correct', 'missed'), (i_true, i_miss)): 

        losses[w] = {k: losses[testset][k][i] for k in losses[testset]}

    for k in ('total', 'cross_x', 'kl'):

        for graph in ('hist', 'boxp'):
            f = os.path.join(root, f'losses-{k}-per-set-{graph}.tab')
        
            losses_distribution_graphs({s: losses[s][k] for s in losses},
                                       graph=graph, output=f, bins=100)

    for k in losses[testset]:
        losses_per_class = {}
        for c in range(net.num_labels):
            pred_is_c = torch.where(y_pred[testset] == c)[0]
            losses_per_class[f'{c}'] = losses[testset][k][pred_is_c]
        for graph in ('hist', 'boxp'):
            f = os.path.join(root, f'losses-{k}-per-class-{graph}.tab')

            losses_distribution_graphs(losses_per_class,
                                       graph=graph, output=f, bins=100)
            
    n_pred = {}
    for s in y_pred:
        n_pred[s] = [sum(y_pred[s] == c) for c in range(net.num_labels)]

    f = os.path.join(root, f'predicted-classes-per-set.tab')
    with open(f, 'w') as f:

        f.write(' '.join([f'{s:6}' for s in n_pred]) + '\n')
        for c in range(net.num_labels):
            f.write(' '.join([f'{n_pred[s][c]:6}' for s in n_pred]) + '\n')

            
def losses_distribution_graphs(dict_of_losses,
                               graph='histogram',  # or boxplot
                               output=sys.stdout,
                               **opt,):
    
    bins = opt.pop('bins', 10)
    
    alpha = opt.pop('quantiles', [0.05, 0.25, 0.5, 0.75, 0.95])

    if graph.startswith('hist'):
        plot, write, close = _create_output(output,
                                            pltf='bar',)
                                            # pltf='plot',)

        hist = {k: np.histogram(dict_of_losses[k], bins=bins) for k in dict_of_losses}        

        write(' '.join(['edge-{k:<8} num-{k:<7}'.format(k=k) for k in dict_of_losses]))
        write('\n')

        for b in range(bins - 1):
            write(' '.join(['{e:-13.6e} {v:-12g}'.format(e=hist[k][1][b],
                                                           v=hist[k][0][b])
                            for k in dict_of_losses]))
            write('\n')
            
        write(' '.join(['{e:-13.6e} {v:-12g}'.format(e=hist[k][1][-1],
                                                       v=0)
                        for k in dict_of_losses]))
        write('\n')

        for k in dict_of_losses:
            plot(hist[k][1][:-1], hist[k][0], label=k)  # , align='edge')
        
    if graph.startswith('box'):
        plot, write, close = _create_output(output, pltf='boxplot')

        quantiles = {k: np.quantile(dict_of_losses[k], alpha) for k in dict_of_losses}

        write('{:20} '.format('which') + ' '.join([f'{a:14}' for a in alpha]) + '\n')
        for k in quantiles:
            write(f'{k:20} ' + ' '.join([f'{q:-14.7e}' for q in quantiles[k]]) + '\n')
        
                
if __name__ == '__main__':

    root = 'results/%j/losses'
    root = '/tmp/%j/losses'    

    j = 112267
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', type=int,
                        nargs='+',
                        default=[j],)

    parser.add_argument('--bins', type=int, default=20)

    a = parser.parse_args()

    plt.clf()
    plt.close()

    output=None
    plot = False
    if plot:
        f, ax = plt.subplots()
        output = ax

    print('loadind net', end='... ', flush=True)
    nets = find_by_job_number('jobs', *a.jobs)
    print('done, found', len(nets))
    
    for net in nets:

        loss_comparisons(nets[net]['net'], root=root)
        
    # losses_distribution_graphs(dict_of_losses, bins=200, output=output)

    
