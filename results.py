from functools import partial
from cvae import ClassificationVariationalNetwork as Net
from utils.parameters import set_log
import argparse, configparser
import sys, os
import logging
import pandas as pd
from utils.save_load import LossRecorder, fetch_models, make_dict_from_model
from utils import torch_load as tl
from utils.sample import zsample, sample
from utils.inspection import loss_comparisons
import matplotlib.pyplot as plt
from utils.filters import DictOfListsOfParamFilters, ParamFilter, get_filter_keys
from utils.tables import agg_results, test_results_df
from pydoc import locate
import re
from utils.print_log import turnoff_debug
from utils.parameters import gethostname


def expand_row(row_format, *a, col_sep=' & '):

    reg = '{%(.*?)}'
    ks = re.findall(reg, row_format)

    return col_sep.join([col_sep.join([k + '_' + _ for _ in a]) for k in ks])


def bold_best_values(data, value, format_string='{:.1f}', prec=1, highlight = '\\bfseries ', max_value=99.9):

    if round(data, prec) == round(value, prec):
        return highlight + format_string.format(min(data, max_value))
    return format_string.format(min(data, max_value))


root = 'results/'
job_dir = 'jobs'

file_ini = None

args_from_file = ['-vv', '--config', 'cifar10.ini']

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
        ini_file_name = os.path.splitext(os.path.split(config_file)[-1])[0]
        tab_file = default_config.get('file', ini_file_name + '-ood-tab.tex')
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

        agg_df = {}
        for k in which_from_filters:
            logging.info('{} models for {}'.format(len(models_by_type[k]), k))
            df_ = test_results_df(models_by_type[k],
                                  predict_methods='all',
                                  ood_methods='all',
                                  tpr=[float(default_config['tpr']) / 100])
            df = next(iter(df_.values()))
            df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna('NaN'))

            idx = list(df.index.names)
            if 'job' in idx:
                idx.remove('job')
            # print("***", k, '\n', df[df.columns[:3]].to_string())
            agg_df[k] = df.groupby(level=idx).agg('mean')
            agg_df[k].columns.rename(['set', 'method', 'metrics'], inplace=True)
            
        for k in which_from_csv:
            csv_file = config[k]['from_csv']
            logging.info('results for {} from csv file {}'.format(k, csv_file))
            index_col = [int(_) for _ in config[k]['index_col'].split()]
            header = [int(_) for _ in config[k]['header'].split()]

            df = pd.read_csv(csv_file, index_col=index_col, header=header)
            if df.index.nlevels > 1:
                df.index = df.index.set_levels([_.astype(str) for _ in df.index.levels])

            agg_df[k] = df.groupby(level=df.index.names).agg('mean')
            agg_df[k].columns.rename(['set', 'method', 'metrics'], inplace=True)

        average = default_config.get('average')
                    
        kept_cols = {}
        kept_oods = []

        for k in which:
            kept_ood = config[k]['ood'].split()
            for o in kept_ood:
                if o not in kept_oods:
                    kept_oods.append(o)
            kept_methods_to_be_splited = config[k]['ood_methods'].split()

            kept_methods = {_.split(':')[0].strip(): _.split(':')[-1].strip()
                            for _ in kept_methods_to_be_splited}

            agg_df[k].rename(columns=kept_methods, level='method', inplace=True)
            
            kept_index = config[k].get('kept_index', '').split()

            cols = agg_df[k].columns

            kept_cols[k] = cols[cols.isin(kept_ood, level='set') &
                                cols.isin(kept_methods.values(), level='method')]
            # print('*** cols', k, ':', *kept_cols[k])

        results_df = agg_results(agg_df, kept_cols=kept_cols, kept_levels=kept_index, average=average)

        best_values = {}
        for tpr in [_ / 100 for _ in range(100)]:
            results_df.rename(columns={tpr: 'fpr'}, inplace=True)
            results_df.rename(columns={str(tpr): 'fpr'}, inplace=True)

        cols = []

        for w in ('fpr', 'auc'):
            for k in which:
                for m in config[k]['ood_methods'].split():
                    cols.append((w, k, m.split(':')[-1].strip()))
        
        fpr_cols = [_ for _ in cols if 'fpr' in _]
        auc_cols = [_ for _ in cols if 'auc' in _]

        # results_df = pd.concat([results_df[fpr_cols], results_df[auc_cols]], axis=1)
        # print('*** cols', *cols)

        # print('*** results cols:', results_df.columns)
        # print('*** new cols:', cols)

        # print(*cols)
        # print('*** results_df cols', *results_df.columns)  # 
        results_df = results_df[cols]

        cols = results_df.columns
        
        print(dataset)
        print(results_df.to_string())
        multi_index = results_df.index.nlevels > 1
        # print('*** index:\n', results_df.index)
        # print('***', *kept_oods)
        if 'set' in results_df.index.names:
            results_df = results_df.reindex(kept_oods, level='set' if multi_index else None) 
        # print('*** index:\n', results_df.index)
        best_values['fpr'] = results_df[fpr_cols].min(axis=1)
        best_values['auc'] = results_df[auc_cols].max(axis=1)

        n_index = results_df.index.nlevels
        n_methods = len(cols) // 2
        column_format = ('@{}' + 'l' * n_index + '%\n'
                         + '@{/}'.join(['S[table-format=2.1]%\n'] * n_methods) * 2 + '@{}')

        cols = cols.droplevel('which')

        renames = dict(**texify['datasets'], **texify['methods'], **texify['metrics'])
        results_df.rename(renames, inplace=True)

        for _ in best_values:
            best_values[_].rename(renames, inplace=True)
        results_df.rename(columns=renames, inplace=True)
        # cols = cols.set_levels(['\\acron{fpr}', '\\acron{auroc}'], level=0)
        # cols = cols.set_levels([_f(_) for _ in cols.levels[-1]], level=-1)

        n_methods = len(cols) // 2
        methods = [c[-1] for c in results_df.columns] [:n_methods]
        _row = '&\\multicolumn{{{n}}}c{{{fpr}}} & \\multicolumn{{{n}}}c{{{auc}}} \\\\\n'
        tex_header = _row.format(n=n_methods,
                                 fpr='\\text{\\acron{fpr}@' + default_config['tpr'] + '}',
                                 auc='\\text{\\acron{auroc}}')
        tex_header += '\\midrule'
        tex_header += '\gls{{ood}}&\\multicolumn{{{n}}}c{{{methods}}} \\\\'.format(n=n_methods*2,
                                                                                   methods= '/'.join(methods))
        header = True

        cols = results_df.columns

        f_ = open(tab_file, 'w')
        for i, r in results_df.iterrows():

            f_.write('%%%%%%% {}\n'.format(i))
            
            formatters = {c: partial(bold_best_values,
                                     value=best_values['fpr' if 'fpr' in c[0] else 'auc'][i]) for c in cols}

            tex_code = results_df.loc[[i]].to_latex(formatters=formatters,
                                                    header=False,
                                                    escape=False,
                                                    index=False,
                                                    multicolumn=False,
                                                    na_rep='\\text{--}',
                                                    column_format=None)
            i0 = 2
            i1 = -3
            tex_code_ = tex_code.split('\n')
            f_.write('%%%%%% FIRST LINE\n')
            if header:
                f_.write('\\begin{tabular}{%\n')
                f_.write(column_format)
                f_.write('}\n')
                f_.write('\\toprule\n')
                f_.write(tex_header + '\n')
            f_.write('\\midrule\n')

            if n_index == 1:
                i = (i,)
                
            tex_code = '\n'.join([('{} & ' * n_index).format(*i) + r for r in tex_code_[i0:i1]])
            f_.write(tex_code)
            last_lines = '\n'.join(tex_code_[i1:])
            column_format = None
            header = False
        f_.write('%%%%% END OF TAB\n')
        f_.write(last_lines)

        f_.close()
        logging.info('{} done'.format(dataset))


    
