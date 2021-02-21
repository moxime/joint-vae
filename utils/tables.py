from cvae import ClassificationVariationalNetwork as Net
from utils.save_load import find_by_job_number
import sys
import os.path
import functools
from utils.save_load import create_file_for_job as create_file
from utils.print_log import texify
from utils.optimizers import Optimizer
import torch
import numpy as np

def printout(s='', file_id=None, std=True, end='\n'):
    if file_id:
        file_id.write(s + end)
    if std:
        sys.stdout.write(s + end)

        
def create_printout(file_id=None, std=True):
    return functools.partial(printout, file_id=file_id, std=std) 


def tex_architecture(net, filename='arch.tex', directory='results/%j', stdout=False,):

    f = create_file(net.job_number, directory, filename) if filename else None
    printout = create_printout(file_id=f, std=stdout)
    arch = net.architecture
    empty_optimizer = Optimizer([torch.nn.Parameter(torch.Tensor())], **net.training['optim'])
    
    exported_values = dict(
        oftype = net.architecture['type'],
        dataset = net.training['set'],
        epochs = net.train_history['epochs'],
        arch = net.print_architecture(excludes='type', sigma=True, sampling=True),
        option = net.option_vector('-', '--'),
        K = arch['latent_dim'],
        L = net.training['latent_sampling'],
        encoder = '-'.join(str(w) for w in arch['encoder']),
        decoder = '-'.join(str(w) for w in arch['decoder']),
        features = arch.get('features', {}).get('name', 'none'),
        sigma = '{:x}'.format(net.sigma),
        optimizer = '{:3x}'.format(empty_optimizer),
        )
        
    for cmd, k in exported_values.items():
        printout(f'\def\\net{cmd}{{{k}}}')

    history = net.train_history

    for _s in ('train', 'test'):
        for _w in ('loss', 'measures', 'accuracy'):
            _b = f'{_s}_{_w}' in history
            printout(f'\{_s}{_w}{_b}'.lower())
    
def texify_test_results(net,
                        directory='results/%j',
                        filename='res.tex',
                        which='all',
                        tpr=[0.95, 'auc'],
                        method='first',
                        stdout=False):
    """ 
    which: 'ood' or 'test' or 'all'
    method: 'first' or 'all' or a specific method (default, first)
    
    """
    def _pcf(x):
        if f is None:
            return '-'
        return f'{100 * x:5.2f}'

    if filename:
        f = create_file(net['job'], directory, filename)
    else: f = None
    
    printout = create_printout(file_id=f, std=stdout)

    show_ood = which in ('all', 'ood')
    show_test = which in ('all', 'test')
    all_methods = method == 'all'

    ood_methods = net['net'].ood_methods
    accuracies = net['accuracies']
    
    if not accuracies:
        printout('no result')
        return

    if not net['ood_fpr']:
        show_ood = False
    elif not list(net['ood_fpr'].values())[0]:
        show_ood = False
    
    
    header = dict()

    if show_test:
        header[net['set']] = len(accuracies) - 1 if all_methods else 1
    if show_ood:
        ood_sets = list(net['ood_fprs'])
        if not all_methods:
            ood_methods = ood_methods[:1]
        for dataset in net['ood_fprs']:
            fprs = net['ood_fprs'][dataset]
            header[dataset] = len(tpr) * ((len(fprs) - 1) if all_methods else 1)
            
    n_cols = sum(c for c in header.values())
    col_style = 'l'
    printout('\\begin{tabular}')
    printout(f'{{{col_style * n_cols}}}')
    printout('\\toprule')
    printout(' & '.join(f'\\multicolumn{cols}c{{{dataset}}}'
                      for dataset, cols in header.items()))
    printout('\\\\ \\midrule')
    if all_methods:
        if show_test:
            printout(' & '.join(list(accuracies)[:-1]), end='& ' if show_ood else '\n')
        if show_ood:
            printout(' & '.join(
                ' & '.join(f'\\multicolumn{len(tpr)}c{{{_}}}' for _ in ood_methods)
                           for s in ood_sets))
        printout('\\\\')
    if show_ood and len(tpr) > 1:
        printout('    &' * header[net['set']], end=' ')
        printout(' & '.join(' & '.join(' & '.join(str(t) for t in tpr)
                                       for _ in range(header[dataset] // len(tpr)))
                 for dataset in ood_sets))
        printout('\\\\ \\midrule')
    if show_test:
        acc = list(accuracies.values())[:-1] if all_methods else [accuracies['first']] 
        printout(' & '.join(_pcf(a) for a in acc), end=' & ' if show_ood else '\n')
    if show_ood:
        ood_ = []
        for dataset in net['ood_fprs']:
            if all_methods:
                fprs = list(net['ood_fprs'][dataset].values())[:-1]
            else:
                fprs = [net['ood_fprs'][dataset]['first']]
            ood_.append(' & '.join(' & '.join((_pcf(m[t]) if m is not None else '-')
                                              for t in tpr) for m in fprs))
        printout(' & '.join(ood_))

    printout('\\\\ \\bottomrule')
    printout('\\end{tabular}')


        
def export_losses(net, which='loss',
                  directory='results/%j',
                  filename='losses.tab',
                  col_width=0, stdout=False):
    """ which is either 'loss' or 'measures' or 'all'

    """
    f = create_file(net.job_number, directory, filename)
    printout = create_printout(file_id=f, std=stdout)

    history = net.train_history

    sets = ['train', 'test']

    if type(which) == str:
        which = ['loss', 'measures', 'accuracy'] if which=='all' else [which]

    entries = [f'{s}_{w}' for w in which for s in sets] 

    epochs = history['epochs']
    columns = {'epochs': [e + 1 for e in range(epochs)]}
    
    for entry in entries:
        if history.get(entry, []):
            for k in history[entry][0].keys():
                columns[f'{entry}_{k}'.replace('_', '-')] = [v[k] for v in history[entry]]
            
    col_width = max(col_width, 7)
    col_width = {c: max(len(c), col_width) for c in columns}
                    
    type_of_net = net.architecture['type']
    arch = net.print_architecture(excludes=['type'])
    training_set = net.training['set']
    printout(f'# {type_of_net} {arch} for {training_set}')

    for c in columns:
        printout(f'  {c:>{col_width[c]}}', end='')

    printout()

    for epoch in range(epochs):
        for c in columns:
            printout(f'  {columns[c][epoch]:{col_width[c]}.6g}', end='')

        printout()
        
    f.close()


def to_string_args(df, target=''):

    if target=='':
        return dict(na_rep='', float_format='{:.3g}'.format, sparsify=True)
    if target == 'tab':
        return dict(sparsify=False, index=False)
    return {}
    
def flatten(t):

    if type(t) == tuple:
        return '-'.join(str(_) for _ in t if _).replace('_', '-')
    return t
    
def format_df(df, style=''):

    df = df.fillna(np.nan)
    print('style:', style)
    if style=='tab':
        cols = df.columns
        df2 = df.copy()
        df2.columns = cols.to_flat_index()
        df2 = df2.reset_index()
        df2 = df2.applymap(lambda x: texify(x, space='-', num=True))
        if 'job' in df2.columns:
            return df2.set_index('job').reset_index().rename(columns=flatten)
        else:
            return df2.reset_index().rename(columns=flatten)
        
    return df

def output_df(df, *files, stdout=True, args_mapper=to_string_args):

    outputs = [dict(style='', f=sys.stdout)] if stdout else []
    for f in files:
        try:
            style = os.path.splitext(f)[-1].split('.')[-1]
            outputs.append(dict(style=style, f=open(f, 'w')))
        except FileNotFoundError as e:
            logging.error(f'{e.strerror}: {f}')

    for o in outputs:

        o['f'].write(format_df(df,o['style']).to_string(**args_mapper(df, o['style'])))
        o['f'].write('\n')
    
if __name__ == '__main__':

    from utils.save_load import collect_networks, test_results_df

    load = False
    load = True
    if load:
        nets = sum(collect_networks('jobs', load_state=False, load_net=False), [])

    df = test_results_df(nets, dataset='cifar10', best_net=False, first_method=False)
    output_df(df, '/tmp/tab.tab')
