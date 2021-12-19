import sys
from datetime import datetime
import string
import functools
from utils.save_load import create_file_for_job as create_file, find_by_job_number
from utils.print_log import texify
from module.optimizers import Optimizer
import torch
import numpy as np
import utils.torch_load as torchdl
import pandas as pd
import hashlib
import argparse
import logging


def printout(s='', file_id=None, std=True, end='\n'):
    if file_id:
        file_id.write(s + end)
    if std:
        sys.stdout.write(s + end)

        
def create_printout(file_id=None, std=True, end='\n'):
    return functools.partial(printout, file_id=file_id, std=std, end=end) 


def tex_architecture(net_dict, filename='arch.tex', directory='results/%j', stdout=False,):

    net = net_dict['net']
    epoch = net_dict['epoch']
    f = create_file(net.job_number, directory, filename) if filename else None
    printout = create_printout(file_id=f, std=stdout)
    arch = net.architecture
    empty_optimizer = Optimizer([torch.nn.Parameter(torch.Tensor())], **net.training_parameters['optim'])
    oftype = net.architecture['type']
    dict_var = net.training_parameters['dictionary_variance'] if oftype == 'cvae' else 0
    beta = net.training_parameters['beta']
    trainset = net.training_parameters['set']
    sigmabeta = r'\ensuremath\sigma=' +f'{net.sigma}'.upper()
    if net.sigma.is_rmse:
        sigmabeta += f' (\\ensuremath\\beta=\\num{{{beta}}})'

    parent_set, heldout = torchdl.get_heldout_classes_by_name(trainset)
    parent_classes = torchdl.set_dict[parent_set]['classes']
    classes = [c for (i, c) in enumerate(parent_classes) if i not in heldout]
    ood_results = net.ood_results.get(epoch, {})
    exported_values = dict(
        oftype=oftype,
        dataset=trainset,
        numclasses=arch['labels'],
        classes=','.join(classes),
        oodsets=','.join(ood_results.keys()),
        noodsets=len(ood_results),
        texoodsets=', '.join(['\\' + o.rstrip(string.digits) for o in ood_results.keys()]),
        epochs=net.train_history['epochs'],
        arch=net.print_architecture(excludes='type', sigma=True, sampling=True),
        archcode=net_dict['arch_code'],
        option=net.option_vector('-', '--'),
        K=arch['latent_dim'],
        L=net.training_parameters['latent_sampling'],
        encoder='-'.join(str(w) for w in arch['encoder']),
        encoderdepth=len(arch['encoder']),
        decoder='-'.join(str(w) for w in arch['decoder']),
        decoderdepth=len(arch['decoder']),
        features=arch.get('features', {}).get('name', 'none'),
        sigma='{:x}'.format(net.sigma),
        beta=beta,
        dictvar=dict_var,
        optimizer='{:3x}'.format(empty_optimizer),
        betasigma=sigmabeta,
        )
        
    for cmd, k in exported_values.items():
        printout(f'\def\\net{cmd}{{{k}}}')

    history = net.train_history

    for _s in ('train', 'test'):
        for _w in ('loss', 'measures', 'accuracy'):
            _b = f'{_s}_{_w}' in history
            printout(f'\{_s}{_w}{_b}'.lower())
    

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

    if 'validation_loss' in history:
        valid_w = 'validation'
    else:
        valid_w = 'test'
        
    sets = ['train', valid_w]

    if type(which) == str:
        which = ['loss', 'measures', 'accuracy'] if which == 'all' else [which]

    entries = [f'{s}_{w}' for w in which for s in sets] 

    epochs = history['epochs']
    columns = {'epochs': [e + 1 for e in range(epochs)]}
    
    for entry in entries:
        if history.get(entry, []):
            for k in history[entry][0].keys():
                columns[f'{entry}_{k}'.replace('_', '-')] = [v.get(k, np.nan)
                                                             for v in history[entry]]
            
    col_width = max(col_width, 7)
    col_width = {c: max(len(c), col_width) for c in columns}
                    
    type_of_net = net.architecture['type']
    arch = net.print_architecture(excludes=['type'])
    training_set = net.training_parameters['set']
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
        if x is None:
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
                fprs = [net['ood_fprs'][dataset].get('first', None)]
                # print('*** fprs', *fprs)
            ood_.append(' & '.join(' & '.join((_pcf(m.get(t, None)) if m is not None else '-')
                                              for t in tpr) for m in fprs))
        printout(' & '.join(ood_))

    printout('\\\\ \\bottomrule')
    printout('\\end{tabular}')


def agg_results(df_dict, kept_cols, kept_levels=['type'], tex_file=None, replacement_dict={}):
    """ 
    df_dict : dict of dataframe
    kept_cols: either a list or a dict (with the same keys as df_dict
    kept_levels: a list of kept indiex levels
    """

    if not isinstance(df_dict, dict):
        df_dict = {'main': df_dict}

    if not isinstance(kept_cols, dict):
        kept_cols = {k: kept_cols for k in df_dict}

    for k, df in df_dict.items():
        df.drop(columns=[_ for _ in df.columns if _[0] == 'measures'], inplace=True)

        df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna('NaN'))

        removed_index = [i for i, l in enumerate(df.index.levels) if len(l) < 2 and l.name not in kept_levels]
        df = df.droplevel(removed_index)

        kc = kept_cols[k]
        # print('*** 250')
        # print(df)
        if hasattr(kc, '__call__'):
            kc = [_  for _ in df.columns if kc(_)]            

        df = df[kc]
        level = df.index.nlevels - 1
        df = df.stack(level=0).unstack(level=level)

        # print('***259')
        # print(df)
        # print(df.index.names)
        if level:
            level_names = df.index.names
            level_names_ = [level_names[-1]] + level_names[:-1]
                                                
            df = df.reorder_levels(level_names).sort_index(0)
            # df.columns = df.columns.reorder_levels([2, 0, 1])
        # print(df)
        
        if df.index.nlevels > 1:
            removed_index = [i for i, l in enumerate(df.index.levels) if len(l) < 2 and l.name not in kept_levels]
            df = df.droplevel(removed_index)
        
        df_dict[k] = df
        
    large_df = pd.concat(df_dict.values(), axis=1)

    return large_df.reorder_levels(['metrics', 'type', 'method'], axis=1)

    
def texify_test_results_df(df, dataset, tex_file, tab_file):

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


    oodsets = [c[:-4] for c in tab_cols if c.endswith('-auc')]
    
    # print(cols, tab_cols)
    # return tab_cols
    
    to_string_args = dict(sparsify=False, index=False)

    tab_df = df.copy()
    tab_df.columns = tab_cols
    tab_df = tab_df.reset_index()
    # print('*** tab_df.cols', *tab_df.columns)
    tab_df.columns = [texify(c, underscore='-') for c in tab_df.columns]
    tab_df = tab_df.applymap(lambda x: texify(x, space='-', num=True))
    
    if 'job' in tab_df.columns:
        tab_df = tab_df.set_index('job').reset_index()

    levels = df.columns.nlevels

    if tex_file:
        with open(tex_file, 'w') as f:

            f.write(f'% Generated on {datetime.now()}\n')
            f.write(f'\\def\\setname{{{dataset}}}\n')
            f.write(f'\\def\\testcolumn{{{dataset}-rate}}\n')

            f.write(r'\def\oodsets{')
            f.write(','.join(oodsets))
            f.write(r'}')
            f.write('\n')

            if oodsets:
                f.write(r'\def\oodset{')
                f.write(oodsets[0])
                f.write(r'}')
                f.write('\n')                

            f.write(r'\def\noodsets{')
            f.write(str(len(oodsets)))
            f.write(r'}')
            f.write('\n')

            # colors = ['green', 'magenta', 'cyan']
            # f.write(r'\pgfplotscreateplotcyclelist{my colors}{')
            # f.write(','.join([f'{{color={c}}}' for c in colors[:len(oodsets)]]))
            # f.write(r'}')
            # f.write('\n')

            done_col = [c for c in tab_df if c.endswith('done')]
            if done_col:
                total_epochs = tab_df[done_col[0]].sum()
                f.write(r'\def\totalepochs{')
                f.write(f'{total_epochs}')
                f.write(r'}')
                f.write('\n')

            file_code = hashlib.sha1(bytes(tab_file, 'utf-8')).hexdigest()[:6]
            f.write(r'\def\tabcode{')
            f.write(f'{file_code}')
            f.write(r'}')
            f.write('\n')

            if 'measures-dict-var' in tab_df:
                unique_dict_vars = tab_df['measures-dict-var'].unique()
                unique_dict_vars.sort()
                f.write(r'\def\tabdictvars{')
                f.write(','.join(str(a) for a in unique_dict_vars))
                f.write(r'}')
                f.write('\n')

            unique_sigmas = sorted(tab_df['sigma'].unique(), key=lambda x: (str(type(x)), x))
            f.write(r'\def\tabsigmas{')
            f.write(','.join(str(a) for a in unique_sigmas))
            f.write(r'}')
            f.write('\n')

            f.write(f'\\pgfplotstableread{{{tab_file}}}{{\\testtab}}')
            f.write('\n')

            tab_cols = [_ for _ in tab_cols if not _.startswith('std-')]
            f.write(r'\def\typesetwithmeasures#1{\pgfplotstabletypeset[columns={')
            f.write(','.join(tab_cols))
            # f.write('job,type')
            f.write(r'},')
            f.write(r'#1')
            f.write(r']{\testtab}}')
            f.write('\n')

            tab_cols_wo_measures = [c for c in tab_cols if 'measures-' not in c]
            f.write(r'\def\typeset#1{\pgfplotstabletypeset[columns={')
            f.write(','.join(tab_cols_wo_measures))
            # f.write('job,type')
            f.write(r'},')
            f.write(r'#1')
            f.write(r']{\testtab}}')
            f.write('\n')

    if tab_file:
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

                
def digest_table(*jobs, 
                 tpr=0.95, precision=1,
                 tex_acron=r'\acron{%t}',
                 cols=['sigma'],
                 highlight=r'\bfseries',
                 empty= r'\text{--}',
                 stdout=True,
                 directory='./results/%j',
                 filename='row.tex',
                 **method_and_set_dict):

    ours = {'jvae': ['esty', 'max', 'iws'],
            'cvae': ['closest', 'max', 'iws', 'iws-2s']}

    replacement_dict = {'learned': r'$\nabla$',
                        'constant': r'--',
                        'coded': r'$c(x)$'}
    
    f = {j: create_file(j, directory, filename) if filename else None for j in jobs}
    printouts = {j: create_printout(file_id=f[j], std=stdout, end='') for j in jobs}
        
    models = find_by_job_number(*jobs, tpr_for_max=tpr, load_net=False, force_dict=True)
    for j in jobs:

        printout = printouts[j]
        logging.debug('Job # %d', j)
        #         printout(' \\\\  % job # {}\n'.format(j))
        printout('% job # {} @TPR {:2}\n'.format(j, 100 * tpr ))
        
        m = models[j]
        testset = m['set']
        mtype = m['type']
        test_results = m['net'].testing[m['epoch']]
        ood_results = m['ood_fprs']
        oodsets = method_and_set_dict[testset]
        methods = method_and_set_dict.get(mtype, [None, None])
        if mtype in ours:
            printout(highlight + ' ')
        printout(tex_acron.replace('%t', mtype) + ' & ')

        for c in cols:
            if c == 'sigma':
                s = replacement_dict[m['sigma_train']]
                printout(s + ' & ')
        
        predict_method = methods[0] if methods[0] != 'none' else None
        ood_methods = [_ if _.lower() != 'none' else None for _ in methods[1:]]
        
        if predict_method and predict_method in test_results:
            acc = 100 * test_results[predict_method]['accuracy']
            logging.debug('%s %.2f', predict_method, acc)
            if predict_method in ours.get(mtype, []):
                printout(f'{highlight} ')
            printout(f'{acc:.{precision}f} & ')
        else:
            printout(f'{empty} & ')
            logging.debug('No predict method')
            
        list_of_fpr = []
        for o in oodsets:

            _m = ','.join(ood_results[o])
            logging.debug('OOD methods for %s %s:',
                          o, _m)
            for m in ood_methods:
            
                highlighted = highlight + ' ' if m in ours.get(mtype, []) else ''
                if m and o in ood_results and m in ood_results[o]:
                    fpr_ = ood_results[o][m]
                    t_ = min((t for t in fpr_ if isinstance(t, float) and t >= tpr), default=None)
                    if t_:
                        fpr = highlighted + f'{100 * fpr_[t_]:.{precision}f}'
                    else:
                        fpr = empty
                else:
                    fpr = empty
                    
                list_of_fpr.append(fpr)

                logging.debug(' '.join([o, m, fpr]))
                
        printout(' & '.join(list_of_fpr))
        printout('\n')
            
              
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('jobs', nargs='+', type=int)

    parser.add_argument('--debug', '-d', action='store_true')
                       
    for k in ('sets', 'methods'):
        parser.add_argument('--' + k, action='append', nargs='+')

    parser.add_argument('--tpr', type=int, default=98)
    parser.add_argument('--empty', type=str, default=' ')
    parser.add_argument('--texfile', default='row')
    
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        
    method_and_set_dict = {}

    for s_ in args.sets:
        for s in s_:                       
            method_and_set_dict[s] = [_ for _ in s_ if _ != s]

    for m_ in args.methods:
        method_and_set_dict[m_[0]] = m_[1:]
                       
    logging.info('Jobs : ' + ', '.join(str(_) for _ in args.jobs))

    for k in method_and_set_dict:

        logging.info(k + ' : ' + ' - '.join(method_and_set_dict[k]))
    
    digest_table(*args.jobs, tpr=args.tpr / 100,
                 empty=f'\\text{{{args.empty}}}',
                 filename=args.texfile + '.tex',
                 **method_and_set_dict)
