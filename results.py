from functools import partial
from cvae import ClassificationVariationalNetwork as Net
from utils.parameters import get_filters_args, set_log, get_args_for_test
import argparse, configparser
import sys, os
import logging
import pandas as pd
from utils.save_load import LossRecorder, collect_networks, test_results_df
from utils import torch_load as tl
from utils.sample import zsample, sample
from utils.inspection import loss_comparisons
import matplotlib.pyplot as plt
from utils.filters import match_filters
from utils.tables import agg_results

import re

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

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--debug', action='store_true')
    conf_parser.add_argument('--verbose', '-v', action='count', default=0)
    conf_parser.add_argument('--config-files', nargs='+', default=[file_ini])
    conf_parser.add_argument('--which', '-c', nargs='*', default=['all'])
    conf_parser.add_argument('--job_dir', default=job_dir)
    conf_parser.add_argument('--results_dir', default=root)
    conf_parser.add_argument('--texify', default='utils/texify.ini')
    
    parser = argparse.ArgumentParser(parents=[conf_parser])
    parser.add_argument('--tpr', default=95, type=int)
    
    conf_args, remaining_args = conf_parser.parse_known_args(None if sys.argv[0] else args_from_file)
    
    if conf_args.verbose > 0:
        logging.getLogger().setLevel(logging.WARNING)
    if conf_args.verbose > 1:
        logging.getLogger().setLevel(logging.INFO)
    if conf_args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if conf_args.texify:
        texify = configparser.ConfigParser()
        texify.read(conf_args.texify)
            
    else:
        texify = {}
        
    all_models = sum(collect_networks(conf_args.job_dir, load_net=False), [])
    for config_file in conf_args.config_files:
        config = configparser.ConfigParser()
        config.read(config_file)
        which = conf_args.which
        if 'all' in which:
            which = list(config.keys())
            if 'DEFAULT' in which:
                which.remove('DEFAULT')
        else:
            which = [w for w  in which if w in config]

        default_config = config['DEFAULT']
        dataset = default_config.get('dataset')
        tab_file = default_config.get('file', dataset + '-ood-tab.tex')
        tab_file = os.path.join(root, tab_file)
        
        logging.info('Tab for {} will be saved in file {}'.format(dataset, tab_file))
        
        filters = {}
        args = {}
        logging.info('Keys in config file: %s', ' '.join(which))

        for k in which:
            logging.info('*** %s: ***', k)
            # logging.info(' '.join(['{}: {}'.format(_, config[k][_]) for _ in config[k]]))
            argv = sum([['--' + _.replace('_', '-'), v] for _, v in config[k].items()], [])
            a, ra = get_filters_args(argv)
            filters[k] = a.filters
            for _ in a.filters:
                logging.info(f'{_:16}: ' + ','.join(str(__) for __ in a.filters[_])) 
            logging.info('Remaining args: %s', ' '.join(ra))
            args[k], ra = parser.parse_known_args(ra)

        for k in filters:
            logging.debug(k)
            f = filters[k]
            for _ in f:
                logging.debug('%s: %s', _, ' '.join(str(__) for __ in f[_]))


        models = {k: [] for k in filters}

        for n in all_models:
            for k, f_ in filters.items():
                filter_results = sum([[f.filter(n[d]) for f in f_[d]] for d in f_], [])
                to_be_kept = all(filter_results)
                d = n['dir']
                derailed = os.path.join(d, 'derailed')
                to_be_kept = to_be_kept and not os.path.exists(derailed) and not n['is_resumed']
                if to_be_kept:
                    models[k].append(n)

        agg_df = {}
        for k in models:
            logging.info('{} models for {}'.format(len(models[k]), k))
            df_ = test_results_df(models[k], nets_to_show='all',
                                  first_method=False,
                                  ood={},
                                  show_measures=False,  # True,
                                  tnr=False,
                                  tpr=[args[k].tpr / 100],
                                  sorting_keys=[])
            df = next(iter(df_.values()))
            df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna('NaN'))

            idx = list(df.index.names)
            if 'job' in idx:
                idx.remove('job')
            agg_df[k] = df.groupby(level=idx).agg('mean')
            agg_df[k].columns.rename(['set', 'method', 'metrics'], inplace=True)

        def kc_(o, m):
            def kc(c):
                return c[0] in o and c[1] in m 
            return kc


        kept_cols = {}
        for k in models:
            kept_ood = config[k]['ood'].split() 
            kept_methods = config[k]['ood_methods'].split() 
            kept_cols[k] = kc_(kept_ood, kept_methods)

        results_df = agg_results(agg_df, kept_cols=kept_cols)


        best_values = {}
        results_df.rename(columns={0.95: 'fpr'}, inplace=True)
        cols = results_df.columns

        fpr_cols = [_ for _ in cols if 'fpr' in _]
        auc_cols = [_ for _ in cols if 'auc' in _]

        results_df = pd.concat([results_df[fpr_cols], results_df[auc_cols]], axis=1)

        cols = results_df.columns

        best_values['fpr'] = results_df[fpr_cols].min(axis=1)
        best_values['auc'] = results_df[auc_cols].max(axis=1)

        n_index = results_df.index.nlevels
        n_methods = len(cols) // 2
        column_format = '@{}' + 'l' * n_index + '%\n' + '@{/}'.join(['S[table-format=2.1]%\n'] * n_methods) * 2 + '@{}'

        cols = cols.droplevel('type')

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
        tex_header += '\gls{{ood}}&\\multicolumn{{{n}}}c{{{methods}}} \\\\'.format(n=n_methods*2, methods= '/'.join(methods))
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
            i_ = texify['datasets'].get(i, i)
            tex_code = '\n'.join(['{} & '.format(i) + r for r in tex_code_[i0:i1]])
            f_.write(tex_code)
            last_lines = '\n'.join(tex_code_[i1:])
            column_format = None
            header = False
        f_.write('%%%%% END OF TAB\n')
        f_.write(last_lines)

        f_.close()
        logging.info('{} done'.format(dataset))


    
