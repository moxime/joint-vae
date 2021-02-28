from cvae import ClassificationVariationalNetwork as Net
from utils.save_load import find_by_job_number
import sys
from datetime import datetime
import string
import os.path
import functools
from utils.save_load import create_file_for_job as create_file
from utils.print_log import texify
from utils.optimizers import Optimizer
import torch
import numpy as np
import data.torch_load as torchdl
import pandas as pd
import hashlib

def printout(s='', file_id=None, std=True, end='\n'):
    if file_id:
        file_id.write(s + end)
    if std:
        sys.stdout.write(s + end)

        
def create_printout(file_id=None, std=True):
    return functools.partial(printout, file_id=file_id, std=std) 


def tex_architecture(net_dict, filename='arch.tex', directory='results/%j', stdout=False,):

    net = net_dict['net']
    f = create_file(net.job_number, directory, filename) if filename else None
    printout = create_printout(file_id=f, std=stdout)
    arch = net.architecture
    empty_optimizer = Optimizer([torch.nn.Parameter(torch.Tensor())], **net.training['optim'])

    net = net_dict['net']
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


        
def export_losses(net_dict, which='loss',
                  directory='results/%j',
                  filename='losses.tab',
                  col_width=0, stdout=False):
    """ which is either 'loss' or 'measures' or 'all'

    """
    net = net_dict['net']
    f = create_file(net.job_number, directory, filename)
    printout = create_printout(file_id=f, std=stdout)

    net = net_dict['net']
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


def infer_type(column, dataset, rate='my fixed', measures='my sci', string='my string'):

    datasets = torchdl.get_same_size_by_name(dataset)
    datasets.append(dataset)

    if column[0] in datasets:
        return rate

    if column[0] == 'measures':
        return measures

    return string


def texify_test_results_df(df, tex_file, tab_file):

    for c in df.columns:
        if c[-1] == 'rate':
            dataset = c[0]

    datasets = torchdl.get_same_size_by_name(dataset)
    datasets.append(dataset)
    
    replacement_dict = {'sigma': r'$\sigma$',
                        'optim_str': 'Optim',
                        'auc': r'\acron{auc}',
                        'measures': '',
                        'rmse': r'\acron{rmse}',
                        'rate': '',
    }
    
    def _r(w, macros=datasets, rdict=replacement_dict):

        if w in macros:
            return f'\\{w.rstrip(string.digits)}'

        try:
            float(w)
            return f'\\num{{{w}}}'
        except ValueError:
            pass
        return replacement_dict.get(w, w.replace('_', ' '))

    cols = df.columns

    tex_cols = pd.MultiIndex.from_tuples([tuple(_r(w) for w in c) for c in cols])

    tab_cols = ['-'.join([str(c) for c in col if c]).replace('_', '-') for col in cols] 
        
    # return tab_cols
    
    to_string_args = dict(sparsify=False, index=False)

    tab_df = df.copy()
    tab_df.columns = tab_cols
    tab_df = tab_df.reset_index()
    tab_df.columns = [texify(c, underscore='-') for c in tab_df.columns]
    tab_df = tab_df.applymap(lambda x: texify(x, space='-', num=True))
    
    if 'job' in tab_df.columns:
        tab_df = tab_df.set_index('job').reset_index()

    levels = df.columns.nlevels
    
    with open(tex_file, 'w') as f:

        f.write(f'% Generated on {datetime.now()}\n')
        f.write(f'\\def\\setname{{{dataset}}}\n')
        f.write(f'\\def\\testcolumn{{{dataset}-rate}}\n')

        total_epochs = tab_df['done'].sum()
        f.write(r'\def\totalepochs{')
        f.write(f'{total_epochs}')
        f.write(r'}')
        f.write('\n')

        file_code = hashlib.sha1(bytes(tab_file, 'utf-8')).hexdigest()[:6]
        f.write(r'\def\tabcode{')
        f.write(f'{file_code}')
        f.write(r'}')
        f.write('\n')

        unique_dict_vars = tab_df['dict-var'].unique()
        unique_dict_vars.sort()
        f.write(r'\def\tabdictvars{')
        f.write(','.join(str(a) for a in unique_dict_vars))
        f.write(r'}')
        f.write('\n')
        
        unique_sigmas = tab_df['sigma'].unique()
        unique_sigmas.sort()
        f.write(r'\def\tabsigmas{')
        f.write(','.join(str(a) for a in unique_sigmas))
        f.write(r'}')
        f.write('\n')
        
        
        f.write(f'\\pgfplotstableread{{{tab_file}}}{{\\testtab}}')
        f.write('\n')

        f.write(r'\def\typeset{\pgfplotstabletypeset[columns={')
        f.write(','.join(tab_cols))
        # f.write('job,type')
        f.write(r'}]{\testtab}}')
        f.write('\n')
        
    with open(tab_file, 'w') as f:
        tab_df.to_string(buf=f, **to_string_args)
        

def pgfplotstable_preambule(df, dataset, file, mode='a'):
    replacement_dict = {'rmse': 'RMSE'}
    def _r(s, f=string.capwords):
        return replacement_dict.get(s, f(s))

    oodsets = torchdl.get_same_size_by_name(dataset)
    
    with open(file, mode) as f:
        f.write('\pgfplotstableset{%\n')
        cols = {c: {} for c in df.columns}
        for c in df.columns:
            if c.startswith('measures'):
                cols[c] = {'style': 'sci num',
                           'name': ' '.join(_r(w) for w in  c.split('-')[1:])}
            elif c.startwith(dataset):           
                cols[c] = {'style': 'fixed num',
                           'name': '\\' + dataset.rstrip(string.digits)}
            elif c.startswith(tuple(oodsets)):
                w_ = c.split('-')
                w_[0] = '\\' + w[0]
                for i, w in enumerate(w_[1:]):
                    try:
                        float(w)
                        w_[i + 1] = '@FPR=' + w
                    except ValueError:
                        pass
                        
                cols[c] = {'style': 'fixed num',
                            'name': ' '.join(w_)}

        
if __name__ == '__main__':

    from utils.save_load import collect_networks, test_results_df

    load = True
    load = False
    if load:
        nets = sum(collect_networks('jobs', load_state=False, load_net=False), [])

    df_e = test_results_df(nets, dataset='cifar10', best_net=False, first_method=False)
    df = test_results_df(nets, dataset='cifar10', best_net=False, first_method=True)
    # output_df(df, '/tmp/tab.tab')
