from utils.save_load import find_by_job_number, LossRecorder, job_to_str
import torch
import numpy as np
from matplotlib import pyplot as plt
import logging
import os
import errno
import sys
import argparse
import logging


def _create_output_plot(*outputs, pltf='plot'):

    close_functions = []
    write_functions = []
    plot_functions = []

    for o in outputs:
        if isinstance(o, type(sys.stdout)):

            def _w(*a, **kw):
                sys.stdout.write(*a)
                return

            write_functions.append(_w)

        elif isinstance(o, str):
            if not os.path.exists(os.path.dirname(o)):
                try:
                    os.makedirs(os.path.dirname(o))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            f = open(o, 'w')

            def _c():
                f.close()
                return

            close_functions.append(_c)
            
            def _w(*a, **kw):
                # print('writing on', f)
                f.write(*a, **kw)
                return

            write_functions.append(_w)
            
        elif isinstance(o, plt.Axes):

            def _p(*a, **kw):
                legend = kw.pop('legend', False)
                getattr(o, pltf)(*a, **kw)
                if legend:
                    o.legend()
                return

            plot_functions.append(_p)

    def plot_func(*a, **kw):
        for f in plot_functions:
            f(*a, **kw)

    def write_func(*a, **kw):
        for f in write_functions:
            f(*a, **kw)

    def close_func(*a, **kw):
        for f in close_functions:
            f(*a, **kw)
                
    return plot_func, write_func, close_func
    

def output_latent_distribution(mu_z, var_z, *outputs, result_type='hist_of_var',
                               **options):
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
            
        hrange = (data.min().item(), data.max().item()) if log_scale else (0, data.max().item()) 
        hist, bins = np.histogram(data, range=hrange, **options)
        if log_scale:
            bins = np.exp(bins)

        plot, write, close = _create_output_plot(*outputs, pltf='bar')
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

        plot, write, close = _create_output_plot(*outputs, pltf='scatter')
        plot(x, y)
        write(f'{x_title}   {y_title}\n')
        for a, b in zip(x, y):
            write(f'{a:-13.6e} {b:-12g}\n')
        close()

    
def loss_comparisons(net, root='results/%j/losses', plot=False, **kw):

    if plot == True:
        plot = 'all'
        
    sample_directory = os.path.join(net.saved_dir, 'samples', 'last')
    root = job_to_str(net.job_number, root)
    
    if not os.path.exists(sample_directory):
        logging.warning(f'Net #{net.job_number} has no recorded loss')
        return
    if not os.path.exists(root):
        os.makedirs(root)

    testset = net.training_parameters['set']
    datasets = [testset] + list(net.ood_results.keys())

    losses = {}
    logits = {}
    y_pred = {}

    recorders = LossRecorder.loadall(sample_directory, *datasets)
    
    for s in recorders:
        r = recorders[s]
        r.to('cpu')
        losses[s] = r._tensors
        logits[s] = losses[s].pop('logits').T
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
        logging.info('Distribution of %s', k)
        for graph in ('hist', 'boxp'):
            f_ = f'losses-{k}-per-set'
            f = os.path.join(root, f_ + f'-{graph}.tab')
            a = None
            if plot and (plot == 'all' or plot.startswith(graph)):
                a = plt.figure(f_ + str(net.job_number)).subplots(1)

            losses_distribution_graphs({s: losses[s][k] for s in losses},
                                       f, sys.stdout, a,
                                       graph=graph,
                                       **kw)

    for k in ('total', 'cross_x', 'kl'): # losses[testset]:
        logging.info('Distribution of %s per class', k)
        losses_per_class = {}
        for c in range(net.num_labels):
            pred_is_c = torch.where(y_pred[testset] == c)[0]
            losses_per_class[f'{c}'] = losses[testset][k][pred_is_c]
        for graph in ('hist', 'boxp'):
            f_ = f'losses-{k}-per-class.tab'
            f = os.path.join(root, f_ + f'-{graph}.tab')
            a = None
            if plot and (plot == 'all' or plot.startswith(graph)):
                a = plt.figure(f_ + str(net.job_number)).subplots(1)

            losses_distribution_graphs(losses_per_class,
                                       f, sys.stdout, a,
                                       graph=graph, **kw)
            
    n_pred = {}
    for s in y_pred:
        n_pred[s] = [sum(y_pred[s] == c) for c in range(net.num_labels)]

    f = os.path.join(root, 'predicted-classes-per-set.tab')
    with open(f, 'w') as f:

        f.write(' '.join([f'{s:6}' for s in n_pred]) + '\n')
        for c in range(net.num_labels):
            f.write(' '.join([f'{n_pred[s][c]:6}' for s in n_pred]) + '\n')

            
def losses_distribution_graphs(dict_of_losses,
                               *outputs,
                               graph='histogram',  # or boxplot
                               **opt,):

    bins = opt.pop('bins', 10)
    whis = opt.pop('whis', 1.5)
    
    alpha = opt.pop('quantiles', [0.05, 0.25, 0.5, 0.75, 0.95])

    if graph.startswith('hist'):
        plot, write, close = _create_output_plot(*outputs,
                                            pltf='plot',)
                                            # pltf='plot',)

        hist = {k: np.histogram(dict_of_losses[k], bins=bins, density=True)
                for k in dict_of_losses}        

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
            plot(hist[k][1][:-1], hist[k][0], label=k, legend=True)  # , align='edge')
        
    if graph.startswith('box'):
        plot, write, close = _create_output_plot(*outputs, pltf='boxplot')

        quantiles = {k: np.quantile(dict_of_losses[k], alpha) for k in dict_of_losses}

        write('{:20} '.format('which') + ' '.join([f'{a:14}' for a in alpha]) + '\n')
        for k in quantiles:
            write(f'{k:20} ' + ' '.join([f'{q:-14.7e}' for q in quantiles[k]]) + '\n')

        plot([l.numpy() for l in dict_of_losses.values()], labels=dict_of_losses.keys())


if __name__ == '__main__':

    root = 'results/%j/losses'

    j = 112267
    
    parser = argparse.ArgumentParser()

    parser.add_argument('jobs', type=int,
                        nargs='+')

    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('-D', '--dir', default=root)

    parser.add_argument('-p', '--plot', nargs='?', const='all')
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-v', action='count')

    args_from_file = ['112267', '-D', '/tmp/%j/losses']
    
    args = parser.parse_args(None if sys.argv[0] else args_from_file)

    if args.plot:
        plt.clf()
        plt.close()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.v:
        logging.getLogger().setLevel(logging.WARNING if args.v == 1 else logging.INFO)
            
    logging.info('loadind net...')
    nets = find_by_job_number(*args.jobs, force_dict=True)
    
    logging.info('done, found %d nets', len(nets))
    
    for net in nets:
        loss_comparisons(nets[net]['net'], root=args.dir, plot=args.plot, bins=args.bins)    

    if args.plot:
        plt.show(block=False)

    input()
