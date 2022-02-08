import sys
import functools
from utils.save_load import create_file_for_job as create_file, find_by_job_number
from utils.print_log import harddebug, printdebug
import numpy as np
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


@printdebug(False)
def agg_results(df_dict, kept_cols, kept_levels=[], tex_file=None, replacement_dict={}, average=False):
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

        harddebug('*** index:', df.index.names, '\ndf:\n', df[df.columns[0:6]], '\n***')

        df = df[kept_cols[k]]

        # harddebug('*** kept cols', *df.columns, '\n', df)
        # harddebug('*** kept cols', kept_cols[k])

        harddebug('*** before stack\n', df)
        df = df.stack('set')
        harddebug('*** stack:\n', df)
        
        df_dict[k] = df.groupby(['set'] + kept_levels).agg('mean')

        harddebug(f'*** df[{k}]\n', df)

    large_df = pd.concat(df_dict, axis=1)

    large_df.columns.rename({None: 'which'}, inplace=True)

    # large_df = large_df.stack('which')

    # large_df = large_df.groupby(['which', 'set'] + kept_levels).agg('mean')
    large_df = large_df.groupby(['set'] + kept_levels).agg('mean')
    
    level = large_df.index.nlevels - 1

    if level:
        level_names = large_df.index.names
        level_names_ = [level_names[-1]] + level_names[:-1]

        harddebug('*** large_df\n', large_df)
        large_df = large_df.reorder_levels(level_names_)
        harddebug('*** reorder\n', large_df)

        removed_index = [i for i, l in enumerate(large_df.index.levels) if len(l) < 2]

        harddebug('index', *large_df.index.names)
        large_df = large_df.droplevel(removed_index)
        harddebug('removed index', *large_df.index.names)

    if average:
        large_df.loc[average] = large_df.mean()
    return large_df.reorder_levels(['metrics', 'which', 'method'], axis=1)

    
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
            

def format_df_index(df, float_format='{:.3g}', int_format='{}',
                    na_rep='-', na_reps=['NaN'],
                    inplace=False,
                    ):

    if not inplace:
        df_ = df.copy()
    else:
        df_ = df
    def return_as_type(x):
        if isinstance(x, str):
            if x in na_reps:
                return na_rep
            return x
        if isinstance(x, int):
            return int_format.format(x)
        if x is None:
            return na_rep
        if isinstance(x, float):
            return float_format.format(x)

    index_df = df.index.to_frame().applymap(return_as_type)
    
    # df.reset_index(inplace=True)
    # df_.index = index_.set_levels([idx.format(formatter=return_as_type) for idx in index_.levels])
    df_.index = pd.MultiIndex.from_frame(index_df)

    
    if not inplace:
        return df_

        
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
