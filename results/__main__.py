from functools import partial
import argparse
import configparser
import sys
import os
import logging
import pandas as pd
import numpy as np
from utils.save_load import fetch_models, make_dict_from_model
from utils.filters import DictOfListsOfParamFilters, ParamFilter, get_filter_keys
from utils.tables import agg_results, results_dataframe
from pydoc import locate
import re
from utils.print_log import turnoff_debug
from utils.parameters import gethostname, DEFAULT_RESULTS_DIR, DEFAULT_JOBS_DIR
from utils.texify import TexTab

root = DEFAULT_RESULTS_DIR
job_dir = DEFAULT_JOBS_DIR

file_ini = None

args_from_file = ['-vv',
                  'jobs/results/manuscrit/tabs/mnist-params.ini',
                  '--keep-auc'
                  ]

tex_output = sys.stdout


def process_config_file(models, config_file, filter_keys, which=['all'], keep_auc=True, root=root):

    config = configparser.ConfigParser()
    config.read(config_file)

    if 'all' in which:
        which = list(config.keys())
        if 'DEFAULT' in which:
            which.remove('DEFAULT')
    else:
        which = [w for w in which if w in config]

    default_config = config['DEFAULT']
    dataset = default_config.get('dataset')
    kept_index = default_config.get('kept_index', '').split()

    kept_index_ = [_.split(':') for _ in kept_index]
    kept_index = [_[0] for _ in kept_index_]
    kept_index_format = [_[1] if len(_) == 2 else 'c' for _ in kept_index_]

    ini_file_name = os.path.splitext(os.path.split(config_file)[-1])[0]

    _auc = '-auc' if keep_auc else ''
    tab_file = default_config.get('file', ini_file_name + _auc + '-tab.tex')
    tab_file = os.path.join(root, tab_file)

    logging.info('Tab for {} will be saved in file {}'.format(dataset, tab_file))

    filters = {}

    logging.info('Keys in config file: %s', ' '.join(which))

    which_from_filters = [k for k in which if not config[k].get('from_csv')]
    which_from_csv = [k for k in which if config[k].get('from_csv')]

    for k in which_from_filters:

        logging.info('| key %s:', k)
        # logging.info(' '.join(['{}: {}'.format(_, config[k][_]) for _ in config[k]]))
        filters[k] = DictOfListsOfParamFilters()

        for _ in config[k]:
            if _ in filter_keys:
                dest = filter_keys[_]['dest']
                ftype = filter_keys[_]['type']
                filters[k].add(dest, ParamFilter.from_string(arg_str=config[k][_],
                                                             type=locate(ftype or 'str')))

    for k in filters:
        logging.debug('| filters for %s', k)
        f = filters[k]
        for _ in f:
            logging.debug('| | %s: %s', _, ' '.join(str(__) for __ in f[_]))

    models_by_type = {k: [] for k in filters}

    for n in models:
        for k, filter in filters.items():
            to_be_kept = filter.filter(n)
            d = n['dir']
            derailed = os.path.join(d, 'derailed')
            to_be_kept = to_be_kept and not os.path.exists(derailed)
            if to_be_kept:
                epoch_to_fetch = config[k].get('epoch', 'last')
                if epoch_to_fetch == 'min-loss':
                    epoch_to_fetch = 'early-min-loss'
                epoch = n['net'].training_parameters.get(epoch_to_fetch, 'last')
                logging.debug('Epoch for %s: %s = %s', n['job'], epoch_to_fetch, epoch)
                with turnoff_debug():
                    n = make_dict_from_model(n['net'], n['dir'], wanted_epoch=epoch)
                models_by_type[k].append(n)

    tpr_ = default_config['tpr']
    tpr = float(tpr_) / 100
    raw_df = {}
    job_list = {}
    for k in which_from_filters:
        job_list[k] = [_['job'] for _ in models_by_type[k]]
        job_list_str = ' '.join(str(_) for _ in job_list[k])
        logging.info('{} models for {}: {}'.format(len(models_by_type[k]), k, job_list_str))
        if not models_by_type[k]:
            logging.warning('Skipping {}'.format(k))
            continue
        df_ = results_dataframe(models_by_type[k],
                                predict_methods=config[k].get('acc_method', '').split(),
                                ood_methods=config[k].get('ood_method', '').split(),
                                tpr=[tpr])
        df = next(iter(df_.values()))
        df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna('NaN'))

        idx = list(df.index.names)
        if 'job' in idx:
            idx.remove('job')

        df_string = df[df.columns[:30]].to_string(float_format='{:.1f}'.format)
        df_width = len(df_string.split('\n')[0])
        print('\n{k:=^{w}s}'.format(k=k, w=df_width))
        print(df_string)
        print('{k:=^{w}s}\n'.format(k='', w=df_width))

        # print('****', df.columns.names)
        raw_df[k] = df.groupby(level=idx).agg('mean')
        raw_df[k].columns.rename(['set', 'method', 'metrics'], inplace=True)
        raw_df[k].rename(columns={tpr: 'rate'}, level='metrics', inplace=True)

    for k in which_from_csv:
        csv_file = config[k]['from_csv']
        logging.info('results for {} from csv file {}'.format(k, csv_file))
        index_col = [int(_) for _ in config[k]['index_col'].split()]
        header = [int(_) for _ in config[k]['header'].split()]

        df = pd.read_csv(csv_file, index_col=index_col, header=header)
        if df.index.nlevels > 1:
            df.index = df.index.set_levels([_.astype(str) for _ in df.index.levels])

        raw_df[k] = df.groupby(level=df.index.names).agg('mean')
        raw_df[k].columns.rename(['set', 'method', 'metrics'], inplace=True)

    average = default_config.get('average')

    kept_oods = set()

    what = ('acc', 'ood')

    col_idx = {}
    sets = {}

    agg_df = {}
    kept_methods = {}

    for k in raw_df:
        agg_df[k] = pd.DataFrame()
        kept_ood = config[k]['ood'].split()
        for o in kept_ood:
            kept_oods.add(o)

        sets['acc'] = [dataset]
        sets['ood'] = kept_ood

        kept_methods[k] = {w: config[k].get('{}_method'.format(w), '').strip() for w in what}

        for w in what:
            df = raw_df[k]
            cols = df.columns
            filtered_auc_metrics = True if keep_auc else ~cols.isin(['auc'], level='metrics')
            cols = cols[cols.isin(sets[w], level='set') & filtered_auc_metrics]
            df = df[cols]
            cols_m = cols[cols.isin([kept_methods[k][w]], level='method')]
            agg_df[k] = agg_df[k].append(df[cols_m].rename(columns={kept_methods[k][w]: k},
                                                           level='method'))

    results_df = agg_results(agg_df, kept_cols=None, kept_levels=kept_index, average=average)

    results_df = results_df.groupby(results_df.columns, axis=1).agg(np.max)

    results_df.columns = pd.MultiIndex.from_tuples(results_df.columns, names=['metrics', 'methods'])

    sorting_sets = {_: i for i, _ in enumerate(sum((sets[w] for w in what), []))}

    def key_(idx):
        return pd.Index([sorting_sets.get(_, -1) for _ in idx], name='set')

    results_df = results_df.sort_index(level='set', key=key_)

    acc_row_name = 'acc'
    results_df.rename({dataset: acc_row_name}, inplace=True)

    best_values = {}

    cols = []

    meta_cols = ('rate', 'auc') if keep_auc else ('rate',)
    for w in meta_cols:
        for k in raw_df:
            if (w, k) not in cols:
                cols.append((w, k))

    rate_cols = [_ for _ in cols if 'rate' in _]

    auc_cols = [_ for _ in cols if 'auc' in _]

    fpr_header = 'fpr@{}'.format(tpr_)

    results_df.rename(columns={'acc': 'rate', fpr_header: 'rate'}, level='metrics', inplace=True)

    results_df = results_df[cols]

    results_df = results_df.groupby(results_df.columns, axis=1).agg(np.max)[cols]
    results_df.columns = pd.MultiIndex.from_tuples(results_df.columns, names=['metrics', 'methods'])
    cols = results_df.columns

    print(dataset)
    print(results_df.to_string(float_format='{:2.1f}'.format))

    best_values['rate'] = results_df[rate_cols].min(axis=1)

    for idx in best_values['rate'].index:
        if not kept_index and idx == acc_row_name or kept_index and idx[0] == acc_row_name:
            best_values['rate'][idx] = results_df[rate_cols].loc[idx].max()

    best_values['auc'] = results_df[auc_cols].max(axis=1)

    n_methods = len(rate_cols)

    if not keep_auc:
        cols.droplevel('metrics')

    cols = results_df.columns

    renames = dict(**texify['datasets'], **texify['methods'], **texify['metrics'])
    results_df.rename(renames, inplace=True)

    for _ in best_values:
        best_values[_].rename(renames, inplace=True)
    results_df.rename(columns=renames, inplace=True)

    methods = [c[-1] for c in results_df.columns][:n_methods]

    col_fmt = ['l'] + kept_index_format
    tab_idx = [''] + [texify['parameters'].get(_, _) for _ in kept_index]
    for _ in meta_cols:
        col_fmt.extend(['s2.1'] * n_methods)

    tab = TexTab(*col_fmt, float_format='{:2.1f}', na_rep='--')

    # meta_headers = {'rate': r'\acron{{fpr}}@{}'.format(tpr) + ' ou acc.',
    #                 'auc': r'\acron{auc}'}
    meta_headers = {'rate': r'\acron{{fpr}}@{}'.format(tpr),
                    'auc': r'\acron{auc}'}

    if len(meta_cols) > 1:
        for _ in tab_idx:
            tab.append_cell('', row='meta_header')
            tab.append_cell(_, row='header')
        for i, _ in enumerate(meta_cols):
            tab.append_cell(meta_headers[_], row='meta_header', width=n_methods)
            for _i in range(len(tab_idx) + 1, n_methods + len(tab_idx)):
                tab.add_col_sep(i * n_methods + _i, sep='/')
        tab.append_cell('/'.join(methods), row='header', width=len(cols))

    else:
        for _ in tab_idx:
            tab.append_cell(_, row='header')
        for m in methods:
            tab.append_cell(m, row='header')

    no_multi_index = results_df.index.nlevels == 1

    last_acc_row = None
    for idx, r in results_df.iterrows():
        idx_ = (idx,) if no_multi_index else idx
        is_an_acc_row = idx_[0] == texify['datasets'][acc_row_name]
        if is_an_acc_row:
            last_acc_row = idx

        for ind in idx_:
            tab.append_cell(ind, row=idx)
        for i, c in enumerate(r):

            is_an_auc_col = 'auc' in cols[i]
            best = best_values['auc' if is_an_auc_col else 'rate'][idx]
            # print('***', *idx, '*** best: {:2.1f}'.format(best))
            face = 'bf' if abs(best - c) < 0.05 else None
            if not is_an_auc_col or not is_an_acc_row:
                if c > 99.95:
                    tab.append_cell(c, row=idx, face=face, multicol_format='l', formatter='{:.0f}')
                else:
                    tab.append_cell(c, row=idx, face=face)
        if 'auc' in meta_cols and is_an_acc_row:
            # tab.append_cell('', width=len(methods))
            pass

    tab.add_midrule(row='header', after=True)
    if last_acc_row is not None:
        tab.add_midrule(row=last_acc_row, after=True)

    for k in job_list:
        tab.comment('{:2} models for {:12}: {}'.format(len(job_list[k]), k,
                                                       ' '.join(str(_) for _ in job_list[k])))

    with open(tab_file, 'w') as f:
        tab.render(f)

    logging.info('{} done'.format(dataset))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--which', '-c', nargs='*', default=['all'])
    parser.add_argument('--job-dir', default=job_dir)
    parser.add_argument('--result-dir', default='/tmp', const=root, nargs='?')
    parser.add_argument('--texify', default='utils/texify.ini')
    parser.add_argument('--filters-file', default='utils/filters.ini')
    parser.add_argument('--tpr', default=95, type=int)
    parser.add_argument('--register', dest='flash', action='store_false')
    parser.add_argument('--auc', action='store_true')
    parser.add_argument('config_files', nargs='+', default=[file_ini])

    args = parser.parse_args(None if sys.argv[0] else args_from_file)

    root = args.result_dir

    if args.verbose > 0:
        logging.getLogger().setLevel(logging.WARNING)
    if args.verbose > 1:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.texify:
        texify = configparser.ConfigParser()
        texify.read(args.texify)

    else:
        texify = {}

    flash = args.flash

    registered_models_file = 'models-' + gethostname() + '.json'

    all_models = fetch_models(args.job_dir, registered_models_file, load_net=False, flash=flash)

    filter_keys = get_filter_keys(args.filters_file, by='key')

    for config_file in args.config_files:

        keep_auc = [False, True] if args.auc else [False]
        for auc in keep_auc:
            process_config_file(all_models, config_file, filter_keys, keep_auc=auc, root=root)
