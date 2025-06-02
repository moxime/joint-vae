import hashlib
import argparse
import configparser
import sys
import os
import logging
import pandas as pd
from utils.save_load import fetch_models, make_dict_from_model
from utils.filters import DictOfListsOfParamFilters, ParamFilter, get_filter_keys, MetaFilter
from utils.tables import agg_results, results_dataframe, format_df_index, auto_remove_index
from pydoc import locate
from utils.texify import TexTab


file_ini = None

args_from_file = ['-vv',
                  'jobs/results/manuscrit/tabs/mnist-params.ini',
                  '--keep-auc'
                  ]

tex_output = sys.stdout


def concat_df(**raw_df):

    average = default_config.get('average', '').split()

    kept_oods = set()

    what = ('acc', 'ood')

    col_idx = {}
    sets = {}

    agg_df = {}
    kept_methods = {}

    # reorder

    for k in which:
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
            agg_df[k] = pd.concat([agg_df[k],
                                   df[cols_m].rename(columns={kept_methods[k][w]: k}, level='method')])

    if len(average) == 1:
        average = {average[0]: kept_oods}

    elif len(average) > 1:
        average = {average[0]: average[1:]}

    # print('*** average', average)

    results_df = agg_results(agg_df, kept_cols=None, kept_levels=kept_index, average=average)

    # print(results_df.to_string(float_format='{:2.1f}'.format), '\n\n')

    results_df = results_df.T.groupby(results_df.columns).agg("max").T

    # print('***After agg***')
    # print(results_df.to_string(float_format='{:2.1f}'.format))

    results_df.columns = pd.MultiIndex.from_tuples(results_df.columns, names=['metrics', 'methods'])

    sorting_sets = {_: i for i, _ in enumerate(sum((sets[w] for w in what), []))}

    for a in average:
        sorting_sets[a] = max(sorting_sets[_] for _ in average[a]) + 0.5
    #    print(sorting_sets)

    def key_(idx):
        return pd.Index([sorting_sets.get(_, 10000) for _ in idx], name='set')

    results_df = results_df.sort_index(level='set', key=key_)

    # print('***After sort***')
    # print(results_df.to_string(float_format='{:2.1f}'.format), '\n\n')

    acc_row_name = 'acc'
    results_df.rename({dataset: acc_row_name}, inplace=True)

    best_values = {}

    cols = []

    meta_cols = ('rate', 'auc') if keep_auc else ('rate',)
    for w in meta_cols:
        for k in which:
            if (w, k) not in cols:
                cols.append((w, k))

    rate_cols = [_ for _ in cols if 'rate' in _]

    auc_cols = [_ for _ in cols if 'auc' in _]

    fpr_header = 'fpr@{}'.format(tpr_)

    results_df.rename(columns={'acc': 'rate', fpr_header: 'rate'}, level='metrics', inplace=True)

    results_df = results_df[cols]

    # print('*** Before agg 2 ***')
    # print(results_df.to_string(float_format='{:2.1f}'.format), '\n\n')

    results_df = results_df.T.groupby(results_df.columns).agg("max").T[cols]

    # print('*** After agg ***')
    # print(results_df.to_string(float_format='{:2.1f}'.format), '\n\n')

    results_df.columns = pd.MultiIndex.from_tuples(results_df.columns, names=['metrics', 'methods'])
    cols = results_df.columns

    print('\n\n\n\n' if show_dfs else '\n')
    print(dataset)
    print(results_df.to_string(float_format='{:2.1f}'.format))

    results_df.to_csv(tab_file)

    best_values['rate'] = results_df[rate_cols].min(axis=1)

    for idx in best_values['rate'].index:
        if not kept_index and idx == acc_row_name or kept_index and idx[0] == acc_row_name:
            best_values['rate'][idx] = results_df[rate_cols].loc[idx].max()

    best_values['auc'] = results_df[auc_cols].max(axis=1)

    n_methods = len(rate_cols)

    if not keep_auc:
        cols.droplevel('metrics')

    cols = results_df.columns

    texify_sets = {s: r'\dset{{{}}}'.format(s) for s in kept_oods}
    renames = dict(**texify_sets, **texify['methods'], **texify['metrics'])
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
        tab.append_cell(r'\smaller{' + '/'.join(methods) + '}', row='header', width=len(cols))

    else:
        for _ in tab_idx:
            tab.append_cell(_, row='header')
        for m in methods:
            tab.append_cell(m, row='header')

    no_multi_index = results_df.index.nlevels == 1

    last_acc_row = None
    average_row = []
    for idx, r in results_df.iterrows():
        idx_ = (idx,) if no_multi_index else idx
        is_an_acc_row = idx_[0] == texify['metrics'][acc_row_name]
        if is_an_acc_row:
            last_acc_row = idx

        if idx_[0] in average:
            average_row.append(idx)

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

    for r in average_row:
        tab.add_midrule(row=r, start=len(idx_), after=False)
        try:
            tab.add_midrule(row=r, after=True)
        except IndexError:
            pass

    for k in job_list:
        tab.comment('{:=^80}'.format(k.upper()))
        tab.comment('{:2} models for {:12}: {}'.format(len(job_list[k]), k,
                                                       ' '.join(str(_) for _ in job_list[k])))
        tab.comment('Index table:')
        for cl in df_string[k].split('\n'):
            tab.comment(cl)
        tab.comment('Archs:')
        for a in archs_by_type[k]:
            tab.comment('{}: {}'.format(hashlib.sha1(bytes(a, 'utf-8')).hexdigest()[:6], a))
        nans = []
        for _, v in format_df_index(removed_index[k]).items():
            if not _.startswith('drop'):
                if v != 'NaN':
                    tab.comment('{:8}: {}'.format(_, v))
                else:
                    nans.append(_)
        if nans:
            tab.comment('{:8}:'.format('NaNs'), ', '.join(nans))

        tab.comment('\n')
        tab.comment('\n')
        tab.comment('\n')

    with open(tex_file, 'w') as f:
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
    parser.add_argument('-q', action='store_false', dest='show_dfs')

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

    filter_keys = get_filter_keys(args.filters_file, by='key')

    for config_file in args.config_files:

        keep_auc = [False, True] if args.auc else [False]
        show_dfs = args.show_dfs
        for auc in keep_auc:
            process_config_file(config_file, filter_keys, keep_auc=auc, root=root,
                                show_dfs=show_dfs, flash=args.flash)
            show_dfs = False
