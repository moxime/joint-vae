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
from utils.tables import agg_results, test_results_df
from pydoc import locate
import re
from utils.print_log import turnoff_debug
from utils.parameters import gethostname, DEFAULT_RESULTS_DIR, DEFAULT_JOBS_DIR
from utils.texify import TexTab

def expand_row(row_format, *a, col_sep=' & '):

    reg = '{%(.*?)}'
    ks = re.findall(reg, row_format)

    return col_sep.join([col_sep.join([k + '_' + _ for _ in a]) for k in ks])


def bold_best_values(data, value, format_string='{:.1f}', prec=1, highlight='\\bfseries ', max_value=99.9):

    if round(data, prec) == round(value, prec):
        return highlight + format_string.format(min(data, max_value))
    return format_string.format(min(data, max_value))


root = DEFAULT_RESULTS_DIR
job_dir = DEFAULT_JOBS_DIR

file_ini = None

args_from_file = ['-vv', '--config',
                  'jobs/results/tabs/mnist-params.ini',
                  #                  '--keep-auc'
                  ]

tex_output = sys.stdout


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--config-files', nargs='+', default=[file_ini])
    parser.add_argument('--which', '-c', nargs='*', default=['all'])
    parser.add_argument('--job-dir', default=job_dir)
    parser.add_argument('--results-dir', default=root)
    parser.add_argument('--texify', default='utils/texify.ini')
    parser.add_argument('--filters', default='utils/filters.ini')
    parser.add_argument('--tpr', default=95, type=int)
    parser.add_argument('--register', dest='flash', action='store_false')
    parser.add_argument('--keep-auc', action='store_true')

    args = parser.parse_args(None if sys.argv[0] else args_from_file)

    root = args.results_dir

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

    filter_keys = get_filter_keys(args.filters, by='key')

    for config_file in args.config_files:

        config = configparser.ConfigParser()
        config.read(config_file)
        which = args.which
        if 'all' in which:
            which = list(config.keys())
            if 'DEFAULT' in which:
                which.remove('DEFAULT')
        else:
            which = [w for w in which if w in config]

        default_config = config['DEFAULT']
        dataset = default_config.get('dataset')
        kept_index = default_config.get('kept_index', '').split()

        ini_file_name = os.path.splitext(os.path.split(config_file)[-1])[0]

        tab_file = default_config.get('file', ini_file_name + '-tab.tex')
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

        for n in all_models:
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

        tpr = float(default_config['tpr']) / 100
        raw_df = {}
        job_list = {}
        for k in which_from_filters:
            job_list[k] = [_['job'] for _ in models_by_type[k]]
            job_list_str = ' '.join(str(_) for _ in job_list[k])
            logging.info('{} models for {}: {}'.format(len(models_by_type[k]), k, job_list_str))
            if not models_by_type[k]:
                logging.warning('Skipping {}'.format(k))
                continue
            df_ = test_results_df(models_by_type[k],
                                  predict_methods='all',
                                  ood_methods='all',
                                  tpr=[tpr])
            df = next(iter(df_.values()))
            df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna('NaN'))

            idx = list(df.index.names)
            if 'job' in idx:
                idx.remove('job')
            # print("***", k, '\n', df[df.columns[:3]].to_string())
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
                filtered_auc_metrics = True if args.keep_auc else ~cols.isin(['auc'], level='metrics')
                cols = cols[cols.isin(sets[w], level='set') & filtered_auc_metrics]
                df = df[cols]
                cols_m = cols[cols.isin([kept_methods[k][w]], level='method')]
                agg_df[k] = agg_df[k].append(df[cols_m].rename(columns={kept_methods[k][w]: k},
                                                               level='method'))

        results_df = agg_results(agg_df, kept_cols=None, kept_levels=kept_index, average=average)
        raise StopIteration

        def keep_number(df):

            for i, row in df.iterrows():
                assert len(row) - row.isna().sum() <= 1
            return np.max(df)

        results_df = results_df.groupby(results_df.columns, axis=1).agg(np.max)
        results_df.columns = pd.MultiIndex.from_tuples(results_df.columns, names=['metrics', 'methods'])

        results_df = results_df.reindex(sum((sets[w] for w in what), []))
        results_df.rename({dataset: 'acc'}, inplace=True)

        best_values = {}

        cols = []

        meta_cols = ('rate', 'auc') if args.keep_auc else ('rate',)
        for w in meta_cols:
            for k in raw_df:
                if (w, k) not in cols:
                    cols.append((w, k))

        rate_cols = [_ for _ in cols if 'rate' in _]
        auc_cols = [_ for _ in cols if 'auc' in _]

        results_df = results_df[cols]

        cols = results_df.columns

        print(dataset)
        print(results_df.to_string(float_format='{:2.1f}'.format))

        best_values['rate'] = results_df[rate_cols].min(axis=1)
        best_values['rate']['acc'] = results_df[rate_cols].loc['acc'].max()

        best_values['auc'] = results_df[auc_cols].max(axis=1)

        n_index = results_df.index.nlevels
        n_methods = len(rate_cols)

        if not args.keep_auc:
            cols.droplevel('metrics')

        cols = results_df.columns

        renames = dict(**texify['datasets'], **texify['methods'], **texify['metrics'])
        results_df.rename(renames, inplace=True)

        for _ in best_values:
            best_values[_].rename(renames, inplace=True)
        results_df.rename(columns=renames, inplace=True)

        methods = [c[-1] for c in results_df.columns][:n_methods]

        col_fmt = ['l']
        for _ in meta_cols:
            col_fmt.extend(['s2.1'] * n_methods)
        tab = TexTab(*col_fmt, float_format='{:2.1f}')

        meta_headers = {'rate': r'\acron{fpr}@' + default_config['tpr'] + ' ou acc.',
                        'auc': r'\acron{auc}'}
        
        if len(meta_cols) > 1:
            tab.append_cell('', row='meta_header')
            tab.append_cell('', row='header')
            for _ in meta_cols:
                tab.append_cell(meta_headers[_], row='meta_header', width=n_methods)
            tab.append_cell('/'.join(methods), row='header', width=len(cols))
        else:
            tab.append_cell('', row='header')
            for m in methods:
                tab.append_cell(m, row='header')

        tab.add_midrule(row='header')

        for idx, r in results_df.iterrows():

            tab.append_cell(idx, row=idx)
            for i, c in enumerate(r):
                best = best_values['auc' if 'auc' in cols[i] else 'rate'][idx]
                face = 'bf' if abs(best - c) < 0.05 else None
                tab.append_cell(c, row=idx, face=face)
            continue
            formatters = {c: partial(bold_best_values,
                                     value=best_values['rate' if 'rate' in c[0] else 'auc'][idx]) for c in cols}

        tab.add_midrule(row=texify['datasets']['acc'])

        for k in job_list:
            tab.comment('{:2} models for {:12}: {}'.format(len(job_list[k]), k,
                                                           ' '.join(str(_) for _ in job_list[k])))
        
        with open(tab_file, 'w') as f:
            tab.render(f)

        logging.info('{} done'.format(dataset))
