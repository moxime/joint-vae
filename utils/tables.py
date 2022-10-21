import sys
import functools
from utils.save_load import create_file_for_job as create_file, find_by_job_number
from utils.print_log import harddebug, printdebug
import numpy as np
import pandas as pd
import hashlib
import argparse
import logging
from utils.parameters import DEFAULT_RESULTS_DIR
import os


def printout(s='', file_id=None, std=True, end='\n'):
    if file_id:
        file_id.write(s + end)
    if std:
        sys.stdout.write(s + end)


def create_printout(file_id=None, std=True, end='\n'):
    return functools.partial(printout, file_id=file_id, std=std, end=end)


def export_losses(net_dict, which='loss',
                  directory=os.path.join(DEFAULT_RESULTS_DIR, '%j'),
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


def test_results_df(nets,
                    predict_methods='first',
                    ood_methods='first',
                    ood={},
                    dataset=None, show_measures=True,
                    tpr=[0.95], tnr=False, sorting_keys=[]):
    """
    nets : list of dicts n
    n['net'] : the network
    n['sigma']
    n['arch']
    n['set']
    n['K']
    n['L']
    n['accuracies'] : {m: acc for m in methods}
    n['best_accuracy'] : best accuracy
    n['ood_fpr'] : '{s: {tpr : fpr}}' for best method
    n['ood_fprs'] : '{s: {m: {tpr: fpr} for m in methods}}
    n['options'] : vector of options
    n['optim_str'] : optimizer
    """

    if ood_methods is None:
        ood_methods = 'first'

    if predict_methods is None:
        predict_methods = 'first'

    if not dataset:
        testsets = {n['set'] for n in nets}
        return {s: test_results_df(nets,
                                   predict_methods=predict_methods,
                                   ood_methods=ood_methods,
                                   ood=ood.get(s),
                                   dataset=s,
                                   show_measures=show_measures,
                                   tpr=tpr, tnr=tnr,
                                   sorting_keys=sorting_keys) for s in testsets}

    arch_index = ['h/o'] if dataset.endswith('-?') else []
    arch_index += ['type',
                   'rep',
                   'depth',
                   'features',
                   'arch_code',
                   'K',
                   # 'dict_var',
                   ]

    train_index = [
        'options',
        'batch_norm',
        'optim_str',
        'latent',
        #        'forced_var',
        'L',
        'sigma_train',
        'sigma',
        # 'beta_sigma',
        'beta',
        'gamma',
        'job']

    indices = arch_index + train_index

    # acc_cols = ['best_accuracy', 'accuracies']
    # ood_cols = ['ood_fpr', 'ood_fprs']

    acc_cols = ['accuracies']
    ood_cols = ['ood_fprs']

    meas_cols = ['epoch', 'done', 'validation']

    if show_measures > 1:
        meas_cols += ['dict_var', 'beta_sigma', 'rmse',
                      'train_loss', 'test_loss',
                      'train_zdist', 'test_zdist']

    columns = indices + acc_cols + ood_cols + meas_cols
    df = pd.DataFrame.from_records([n for n in nets if n['set'] == dataset],
                                   columns=columns)

    df['batch_norm'] = df['batch_norm'].apply(lambda x: x[0] if x else x)

    df.set_index(indices, inplace=True)

    acc_df = pd.DataFrame(df['accuracies'].values.tolist(), index=df.index)
    acc_df.columns = pd.MultiIndex.from_product([acc_df.columns, ['rate']])
    ood_df = pd.DataFrame(df['ood_fprs'].values.tolist(), index=df.index)
    meas_df = df[meas_cols]
    # print(meas_df.columns)
    meas_df.columns = pd.MultiIndex.from_product([[''], meas_df.columns])

    # return acc_df
    # return ood_df
    d_ = {dataset: acc_df}

    # print('*** ood_df:', *ood_df, 'ood', ood)
    if ood is not None:
        ood_df = {s: ood_df[s] for s in ood}
    for s in ood_df:
        d_s = pd.DataFrame(ood_df[s].values.tolist(), index=df.index)
        d_s_ = {}
        for m in d_s:
            v_ = d_s[m].values.tolist()
            _v = []
            for v in v_:
                if type(v) is dict:
                    _v.append(v)
                else:
                    _v.append({})
            d_s_[m] = pd.DataFrame(_v, index=df.index)
        if d_s_:
            d_[s] = pd.concat(d_s_, axis=1)
            # print(d_[s].columns)
            # print('==')

            if tnr:
                cols_fpr = d_[s].columns[~d_[s].columns.isin(['auc'], level=-1)]
                d_[s][cols_fpr] = d_[s][cols_fpr].transform(lambda x: 1 - x)

        #d_[s] = pd.DataFrame(d_s.values.tolist(), index=df.index)

    for s in d_:
        show = predict_methods if s == dataset else ood_methods
        cols = d_[s].columns
        kept_columns = cols.isin(tpr + ['rate', 'auc'] + [str(_) for _ in tpr], level=1)
        first_method_columns = cols.isin(['first'], level=0)
        # print('*** 1st', *first_method_columns)

        if show == 'first':
            shown_columns = first_method_columns
        elif show == 'all':
            shown_columns = ~first_method_columns
        else:
            # print(show)
            if isinstance(show, str):
                show = [show]
            shown_columns = cols.isin(show, level=0)

        # print('*** kept', s, *shown_columns, '\n', *d_[s].columns)
        d_[s] = d_[s][cols[shown_columns * kept_columns]]

    if show_measures:
        d_['measures'] = meas_df

    df = pd.concat(d_, axis=1)

    df.columns.rename(['set', 'method', 'metrics'], inplace=True)

    cols = df.columns

    if False:
        df.columns = df.columns.droplevel(1)

    def _f(x, type='pc'):
        if type == 'pc':
            return 100 * x
        elif type == 'tuple':
            return '-'.join(str(_) for _ in x)
        return x

    col_format = {c: _f for c in df.columns}
    for c in df.columns[df.columns.isin(['measures'], level=0)]:
        col_format[c] = lambda x: _f(x, 'measures')

    index_format = {}
    index_format['heldout'] = lambda x: 'H'  # _f(x, 'tuple')

    sorting_index = []

    if sorting_keys:
        sorting_keys_ = [k.replace('-', '_') for k in sorting_keys]
        for k in sorting_keys_:
            if k in df.index.names:
                sorting_index.append(k)
                continue
            str_k_ = k.split('_')
            k_ = []
            for s_ in str_k_:
                try:
                    k_.append(float(s_))
                except ValueError:
                    k_.append(s_)
            tuple_k = (k_[0], '_'.join([str(_) for _ in k_[1:]]))
            if tuple_k in df.columns:
                sorting_index.append(tuple_k)
                continue
            tuple_k = ('_'.join([str(_) for _ in k_[:-1]]), k_[-1])
            if tuple_k in df.columns:
                sorting_index.append(tuple_k)
                continue

            logging.error(f'Key {k} not used for sorting')
            logging.error('Possible index keys: %s', '--'.join([_.replace('_', '-') for _ in df.index.names]))
            logging.error('Possible columns %s', '--'.join(['-'.join(str(k) for k in c) for c in df.columns]))

    if sorting_index:
        df = df.sort_values(sorting_index)

    return df.apply(col_format)


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
                 empty=r'\text{--}',
                 stdout=True,
                 directory=os.path.join(DEFAULT_RESULTS_DIR, '%j'),
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
        printout('% job # {} @TPR {:2}\n'.format(j, 100 * tpr))

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
                    indices_replacement={'batch_norm': 'bn',
                                         'latent': 'z',
                                         'sigma_train': 'sigma~',
                                         'arch_code': 'arch',
                                         'optim_str': 'optim'
                                         },
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

    df.index.names = [indices_replacement.get(_, _)for _ in df.index.names]

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
