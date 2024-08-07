import sys
import functools
from utils.save_load import create_file_for_job as create_file, find_by_job_number
from utils.print_log import harddebug, printdebug
from utils.misc import make_list
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


def results_dataframe(models,
                      predict_methods=None,
                      ood_methods=None,
                      misclass_methods='starred',
                      metrics='all',
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
    n['in_out_rate'] : '{s: {tpr : fpr}}' for best method
    n['ood_in_out_rates'] : '{s: {m: {tpr: fpr} for m in methods}}
    n['options'] : vector of options
    n['optim_str'] : optimizer
    """

    methods = {'ood': ood_methods, 'predict': predict_methods, 'misclass': misclass_methods}

    for _ in methods:
        # if methods[_] is None:
        #    methods[_] = 'first'
        if isinstance(methods[_], str):
            methods[_] = [methods[_]]

    metrics = make_list(metrics, ['acc', 'auc', 'fpr', 'P'])

    if show_measures:
        metrics.extend(['n', 'mean', 'std'])

    for m in ('fpr', 'P'):
        if m in metrics:
            metrics.remove(m)
            for _ in tpr:
                metrics.append(m + '@{:.0f}'.format(100 * _))

    if not dataset:
        testsets = {n['set'] for n in models}
        return {s: results_dataframe(models,
                                     predict_methods=predict_methods,
                                     ood_methods=ood_methods,
                                     misclass_methods=misclass_methods,
                                     metrics=metrics,
                                     ood=ood.get(s),
                                     dataset=s,
                                     show_measures=show_measures,
                                     tpr=tpr, tnr=tnr,
                                     sorting_keys=sorting_keys) for s in testsets}

    arch_index = ['h/o'] if dataset.endswith('-?') else []
    arch_index += ['type',
                   'output_distribution',
                   # 'rep',
                   'depth',
                   'features',
                   'upsampler',
                   'arch_code',
                   'activation_str',
                   'output_activation_str',
                   'K',
                   # 'dict_var',
                   ]

    train_index = [
        'options',
        'batch_norm',
        'optim_str',
        'encoder_forced_variance',
        'prior',
        'latent_prior_init_means',
        #
        #        'forced_var',
        'tilted_tau',
        'wim_prior',
        'wim_mean',
        'wim_from',
        'wim_sets',
        'wim_alpha',
        'wim_train_size',
        'wim_mix',
        'wim_augmentation_dataset',
        'wim_augmentation',
        'wim_augmentation_str',
        'wim_moving_size',
        'wim_array_size',
        'L',
        'l',
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
    in_out_cols = ['in_out_rates']

    meas_cols = []

    if show_measures:
        meas_cols = ['epoch', 'done', 'validation']

    if show_measures > 1:
        meas_cols += ['dB', 'nll', 'kl', 'dict_var', 'beta_sigma', 'rmse',
                      'train_loss', 'test_loss',
                      'train_zdist', 'test_zdist']

    columns = indices + acc_cols + in_out_cols + meas_cols
    df = pd.DataFrame.from_records([n for n in models if n['set'] == dataset],
                                   columns=columns)

    df['batch_norm'] = df['batch_norm'].apply(lambda x: x[0] if x else x)

    df.set_index(indices, inplace=True)

    remove_wim_from = all(df.index.to_frame()['job'] == df.index.to_frame()['wim_from'])

    if remove_wim_from:
        df = df.droplevel([_ for _ in df.index.names if _.startswith('wim')])

    col_names = ['set', 'method', 'metrics']

    acc_df = unfold_df_from_dict(df['accuracies'], depth=1, names=['method'])
    acc_df.columns = pd.MultiIndex.from_product([[dataset],
                                                 acc_df.columns,
                                                 ['acc']],
                                                names=col_names)

    # DEBUG
    # print('ACC\n', *acc_df.columns)
    in_out_df = unfold_df_from_dict(df['in_out_rates'], depth=3,
                                    names=col_names,
                                    keep={'method': ['starred']})

    # DEBUG
    # print('INOUT\n', *in_out_df.index.names)
    # print('INOUT\n', *in_out_df.columns)

    meas_df = df[meas_cols]
    meas_df.columns = pd.MultiIndex.from_product([['measures'], [''], meas_df.columns],
                                                 names=col_names)
    # DEBUG
    # print('MEAS\n', meas_df.index.names)

    df = pd.concat([acc_df, in_out_df, meas_df], axis=1)

    # DEBUG
    # print('CONCAT\n', df.index.names)
    # print('CONCAT\n', df.columns)

    cols = df.columns

    misclass_cols_set_level = set([_[0] for _ in cols if _[0].startswith('errors-')])
    in_out_cols_set_level = set([_[0] for _ in cols if _[-1] == 'auc'])
    ood_cols_set_level = set.difference(in_out_cols_set_level, misclass_cols_set_level)
    method_cols = {_: True for _ in methods}
    for _ in method_cols:
        if methods[_] is not None:
            # print('***', _, *methods[_])
            if 'all' in methods[_]:
                method_cols[_] = [not _ for _ in cols.isin(['first'], level='method')]
            else:
                method_cols[_] = cols.isin(methods[_], level='method')
        # print('***', _, *map(lambda s: '({})'.format(s), methods[_]), sum(method_cols[_]))
        # print('***', _, *map(lambda s: '({})'.format(s), set([c[1] for c in cols])))
    acc_cols = cols.isin(['acc'], level='metrics') & method_cols['predict']

    ood_cols = cols.isin(ood_cols_set_level, level='set') & method_cols['ood']

    if ood is not None:
        ood_cols = ood_cols & cols.isin(ood, level='set')

    misclass_cols_set_level = ['errors-' + _ for _ in methods['predict']]
    # print('***', cols[cols.isin(misclass_cols_set_level, level='set')])
    misclass_cols = cols.isin(misclass_cols_set_level, level='set') & method_cols['misclass']

    measures_cols = cols.isin(['measures'], level='set')
    metrics_cols = cols.isin(metrics, level='metrics')

    kept_cols = cols[(metrics_cols & (acc_cols | ood_cols | misclass_cols)) | measures_cols]

    # DEBUG
    # print('BEFORE:\n', *cols)
    # print('KEPT:\n', *kept_cols)
    df = df[kept_cols]

    if len(methods['predict']) == 1:
        df.rename(columns={'errors-{}'.format(methods['predict'][0]): 'errors'}, inplace=True)

    # Drop method level
    # dropped_levels = []
    # for _, level in enumerate(df.columns.names):
    #     levels = set(c[_] for c in df.columns if c[_] != '')
    #     if len(levels) == 1:
    #         dropped_levels.append(level)
    # for l_ in dropped_levels:
    #     df.columns = df.columns.droplevel(l_)

    def _f(x, type='%'):
        if type == '%':
            return 100 * x
        return x

    col_format = {c: _f for c in df.columns}
    for c in df.columns[df.columns.isin(['measures'], level=0)]:
        col_format[c] = lambda x: _f(x, 'measures')

    for c in df.columns[df.columns.isin(['n', 'mean', 'std'], level=-1)]:
        col_format[c] = lambda x: _f(x, 'measures')

    index_order = list(df.index.names)
    sorting_index_pre = []
    sorting_index_post = []

    if sorting_keys:
        try:
            i_sep = sorting_keys.index('!')

            if 'job' not in format_df_index(sorting_keys):
                sorting_keys.append('job')

        except ValueError:
            i_sep = len(index_order)

        sorting_keys_ = format_df_index([k.replace('-', '_') for k in sorting_keys], inverse_replace=True)
        for i, k in enumerate(sorting_keys_):
            if i == i_sep:
                continue
            if k in df.index.names:
                if i < i_sep:
                    index_order.remove(k)
                    sorting_index_pre.append(k)
                    continue

                index_order.remove(k)
                sorting_index_post.append(k)
                continue

            logging.error(f'Key {k} not used for sorting')
            logging.error('Possible index keys: %s', ' ; '.join([_.replace('_', '-') for _ in df.index.names]))

    index_order = [*sorting_index_pre, *index_order, *sorting_index_post]
    #    if index_order:
    #  df = df.sort_index(level=index_order)
    df = df.reset_index().set_index(index_order)
    # print('***', *index_order)
    # df = df.sort_index(level=['prior', 'sigma_train'])
    df = df.sort_index(level=index_order)
    #    print(df[df.columns[0]].to_string())

    return df.apply(col_format)


def auto_remove_index(df, keep=['job', 'type']):
    removed_index = [l.name for i, l in enumerate(df.index.levels)
                     if len(l) < 2 and l.name not in keep]

    auto_removed_index = {}

    for n in removed_index:
        i = df.index.names.index(n)
        auto_removed_index[n] = df.index[0][i]

    return auto_removed_index


@ printdebug(False)
def agg_results(df_dict, kept_cols=None, kept_levels=[], tex_file=None, replacement_dict={},
                average=False):
    """
    df_dict : dict of dataframe
    kept_cols: either a list or a dict (with the same keys as df_dict
    kept_levels: a list of kept indiex levels
    """

    if not isinstance(df_dict, dict):
        df_dict = {'main': df_dict}

    if not isinstance(kept_cols, dict):
        kept_cols = {k: kept_cols for k in df_dict}

    if 'sets' not in kept_levels:
        kept_levels = ['set'] + kept_levels

    for k, df in df_dict.items():
        harddebug('*** index:', df.index.names, '\ndf:\n', df[df.columns[0:6]], '\n***')

        if kept_cols[k] is not None:
            df = df[kept_cols[k]]

        # harddebug('*** kept cols', *df.columns, '\n', df)
        # harddebug('*** kept cols', kept_cols[k])

        harddebug('*** before stack\n', df)
        df = df.stack('set')
        harddebug('*** stack:\n', df)

        df_dict[k] = df.groupby(kept_levels).agg('mean')

        harddebug(f'*** df[{k}]\n', df)

    large_df = pd.concat(df_dict.values(), axis=1)

    # large_df = large_df.groupby(['set'] + kept_levels).agg('mean')

    level = large_df.index.nlevels - 1

    if level:

        harddebug('*** large_df\n', large_df)
        # large_df = large_df.reorder_levels(level_names_)
        harddebug('*** reorder\n', large_df)

        removed_index = [i for i, l in enumerate(large_df.index.levels) if len(l) < 2]

        harddebug('index', *large_df.index.names)
        large_df = large_df.droplevel(removed_index)
        harddebug('removed index', *large_df.index.names)

    unstack_stack = [_ for _ in kept_levels if _ != 'set']

    for _ in unstack_stack:
        large_df = large_df.unstack(_)

    # print('***Before average***')
    # print(large_df.to_string(float_format='{:2.1f}'.format), '\n\n')
    # print(large_df.index)

    average = average or {}

    average = {_: large_df.index.isin(average[_], level='set') for _ in average}

    # print(average)

    for a in average:
        large_df.loc[a] = large_df.loc[average[a]].mean()

    for _ in unstack_stack:
        large_df = large_df.stack(_)

    # if average:
    #     large_df.loc[average] = large_df.mean()
    #     acc_columns = large_df.columns[large_df.columns.isin(['acc'], level='metrics')]
    #     large_df.loc[average, acc_columns] = np.nan
    #     # large_df[acc_columns][average] = np.nan

    # print(large_df.to_string(float_format='{:2.1f}'.format), '\n\n')

    return large_df.reorder_levels(['metrics', 'method'], axis=1)


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

    models = find_by_job_number(*jobs, tpr_for_max=tpr, build_module=False, force_dict=True)
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
                    indices_replacement={'output_distribution': 'output',
                                         'batch_norm': 'bn',
                                         'latent': 'z',
                                         'sigma': '\u03c3',
                                         # 'sigma_train': 'sigma~',
                                         # 'sigma_train': '\u2207\u03c3',
                                         'sigma_train': '\u03c3~',
                                         'arch_code': 'arch',
                                         'optim_str': 'optim',
                                         # 'tilted_tau': 'tilt',
                                         'tilted_tau': '\u03c4',
                                         'wim_array_size': '[\u2207]',
                                         'wim_sets': '\u2207-sets',
                                         'wim_mean': '<\u2207>',
                                         'wim_prior': '\u2207-prior',
                                         'wim_alpha': '\u2207-\u03b1',
                                         'wim_train_size': '\u2207-N',
                                         'wim_moving_size': '\u2207-n',
                                         'wim_mix': '\u2207:',
                                         'wim_from': '\u2207#',
                                         'wim_augmentation_dataset': 'drop_wda',
                                         'wim_augmentation': 'drop_wa',
                                         'wim_augmentation_str': '\u2207+',
                                         'beta': '\u03b2',
                                         'gamma': '\u03b3',
                                         'latent_prior_init_means': '<m>',
                                         'encoder_forced_variance': 'fv',
                                         'depth': 'D',
                                         'features': 'feat',
                                         'upsampler': 'ups',
                                         'activation_str': 'act',
                                         'output_activation_str': 'out',
                                         },
                    inplace=False,
                    inverse_replace=False,
                    ):

    if isinstance(df, dict):
        replaced = {}
        for k, v in df.items():
            replaced[indices_replacement.get(k, k)] = v
        return replaced

    if isinstance(df, list):
        if inverse_replace:
            indices_replacement = {v: k for k, v in indices_replacement.items()}
        return [indices_replacement.get(k, k) for k in df]

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

    index_df = df.index.to_frame().map(return_as_type)

    # df.reset_index(inplace=True)
    # df_.index = index_.set_levels([idx.format(formatter=return_as_type) for idx in index_.levels])
    df_.index = pd.MultiIndex.from_frame(index_df)

    df_.index.names = [indices_replacement.get(_, _)for _ in df.index.names]

    df_.reset_index(level=[_ for _ in df_.index.names if _.startswith('drop')], drop=True, inplace=True)

    if not inplace:
        return df_


def unfold_df_from_dict(df, depth=1, names=None, keep=None):
    if not depth:
        if isinstance(df, pd.Series):
            return None if all(df.isna()) else df
        cols = [_ for _ in df if not all(df[_].isna())]
        return df[cols]

    if names is None:
        names = [_ for _ in range(depth)]

    if keep is None:
        keep = {_: None for _ in names}

    name = names[0]

    try:
        current_keeper = keep.get(name)
    except AttributeError:
        current_keeper = keep

    def keep_col(keeper, col, *cols):
        is_in = []
        startswith = []
        endswith = []
        str_cols = [_ for _ in cols if isinstance(_, str)]
        starred = [_ for _ in str_cols if _.endswith('*')]
        nonstarred = [_ for _ in str_cols if not any(_.startswith(s[:-1]) for s in starred)]
        approx = []
        if not keeper:
            return True
        for k in keeper:
            if isinstance(k, str):
                if '~' in k:
                    v, a = k.split('~')
                    v = float(v)
                    if not a:
                        a = v / 1000
                    else:
                        a = v * float(a)
                    approx.append((v, a))
                if k.startswith('?'):
                    endswith.append(k[1:])
                elif k.endswith('?'):
                    startswith.append(k[:-1])
                elif k == 'starred':
                    is_in.extend(starred)
                    is_in.extend(nonstarred)
                else:
                    is_in.append(k)
            else:
                is_in.append(k)

        if isinstance(col, str):
            is_kept = (col in is_in
                       or any(col.startswith(_) for _ in startswith)
                       or any(col.endswith(_) for _ in endswith))
            return is_kept
        elif isinstance(col, (float, int)):
            return col in is_in or any(abs(col - v) < a for (v, a) in approx)
        else:
            return col in is_in

    df_ = pd.DataFrame(df.values.tolist(), index=df.index)
    # print('*** in tables:551\n', df_.index.names)

    non_null_cols = [_ for _ in df_.columns if not all(df_[_].isnull())]

    def replace_floats(x):
        if isinstance(x, dict):
            return x
        n = x.name
        return {n[-1]: x}
    if depth > 1:
        df_ = df_[non_null_cols].map(lambda x: x if isinstance(x, dict) else {'val': x})
        # df_ = df_[non_null_cols].apply(replace_floats, axis=1, result_type='broadcast')
        # print('*** in tables:561\n', df_.index.names)
    unfolded = {_: unfold_df_from_dict(df_[_], depth=depth - 1, names=names[1:], keep=keep)
                for _ in df_.columns if keep_col(current_keeper, _, *df_.columns)}
    try:
        concatenated_df = pd.concat({_: unfolded[_] for _ in unfolded
                                     if unfolded[_] is not None and not unfolded[_].empty},
                                    axis=1, names=[names[0]])
        # print('*** in tables:570\n', *concatenated_df.index.names)

    except ValueError:
        # if all are empty
        return pd.DataFrame()
    return concatenated_df


if __name__ == '__main__':

    tpr = [0.95]
    from utils.save_load import fetch_models

    models = fetch_models('jobs', tpr=tpr)

    models = [m for m in models if m['set'] == 'mnist' and m['type'] == 'cvae']

    def test_unfold():
        df = pd.DataFrame.from_records(models, columns=['job', 'type', 'in_out_rates', 'accuracies'],
                                       index=['type', 'job'])

        which = 'acc'
        which = 'in_out'

        if which == 'acc':
            unf_df = unfold_df_from_dict(df['accuracies'], depth=1, names=['method'])
        else:
            unf_df = unfold_df_from_dict(df['in_out_rates'], depth=3,
                                         names=['set', 'method', 'metrics'],
                                         keep={'method': ['baseline?', 'starred']})

        cols = unf_df.columns
        print(unf_df[cols[:8]].to_string(float_format='{:.1%}'.format))
        print(unf_df[cols[-8:]].to_string(float_format='{:.1%}'.format))

    def test_results_data_frame():
        df = results_dataframe(models, metrics=['acc', 'fpr'], show_measures=False, ood={'mnist': ['average']})
        return df
