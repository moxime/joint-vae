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


def bold_best_values(data, value, format_string='{:.1f}', prec=1, highlight = '\\bfseries '):

    if round(data, prec) == round(value, prec):
        return highlight + format_string.format(data)
    return format_string.format(data)


root = 'results/'
job_dir = 'jobs'

file_ini = None

args_from_file = ['-vv', '--config', 'cifar10-ho.ini']

tex_output = sys.stdout

if __name__ == '__main__':

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--debug', action='store_true')
    conf_parser.add_argument('--verbose', '-v', action='count', default=0)
    conf_parser.add_argument('--config-file', default=file_ini)
    conf_parser.add_argument('--which', '-c', nargs='*', default=['all'])
    conf_parser.add_argument('--job_dir', default=job_dir)
    
    parser = argparse.ArgumentParser(parents=[conf_parser])
    parser.add_argument('--tpr', default=95, type=int)
    
    conf_args, remaining_args = conf_parser.parse_known_args(None if sys.argv[0] else args_from_file)
    
    if conf_args.verbose > 0:
        logging.getLogger().setLevel(logging.WARNING)
    if conf_args.verbose > 1:
        logging.getLogger().setLevel(logging.INFO)
    if conf_args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if conf_args.config_file:
        config = configparser.ConfigParser()
        config.read(conf_args.config_file)
        which = conf_args.which
        if 'all' in which:
            which = list(config.keys())
            default_config = which.remove('DEFAULT')
        else:
            which = [w for w  in which if w in config]

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

    else:
        conf_args, _ = get_filters_args(remaining_args)
        for k in sorted(conf_args.__dict__):
            v = conf_args.__dict__[k]
            logging.debug('{:10}: {}'.format(k, v))

        filters = {'only': conf_args.filters}

    for k in filters:
        logging.debug(k)
        f = filters[k]
        for _ in f:
            logging.debug('%s: %s', _, ' '.join(str(__) for __ in f[_]))

    all_models = sum(collect_networks(conf_args.job_dir, load_net=False), [])

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


    agg_df = {k: models[k]. for k in models}
    agg_results(models, [])
                
    default_config = config['DEFAULT']

    results = {_: {} for _ in ('fpr', 'auc')}
    ood = default_config['ood'].split()
    best_values = {}
    
    for k in models:
        logging.info('{} models for {}'.format(len(models[k]), k))
        df_ = test_results_df(models[k], nets_to_show='all',
                              first_method=False,
                              ood={},
                              show_measures=False,  # True,
                              tnr=False,
                              tpr=[args[k].tpr / 100],
                              sorting_keys=[])

        # assert len(df_) == 1, ' '.join(df_)
        df = next(iter(df_.values()))
        df.drop(columns=[_ for _ in df.columns if _[0] == 'measures'], inplace=True)
        df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna('NaN'))

        kept_levels = ['type']

        removed_index = [i for i, l in enumerate(df.index.levels) if len(l) < 2 and l.name not in kept_levels]
        df = df.droplevel(removed_index)

        
        idx = list(df.index.names)
        if 'job' in idx:
            idx.remove('job')
        gb = df.groupby(level=idx)

        # print(f'{k:=^80}')

        agg_df = {agg: gb.agg(agg) for agg in ('count', 'mean', 'std')}
        agg_df['count'] = agg_df['count'][agg_df['count'].columns[0]].to_frame()

        for agg in agg_df:
            level = agg_df[agg].index.nlevels - 1
            agg_df[agg] = agg_df[agg].stack(level=0).unstack(level=level)
            if level:
                agg_df[agg] = agg_df[agg].reorder_levels([-1, *list(range(level))]).sort_index(0)
            agg_df[agg].columns = agg_df[agg].columns.reorder_levels([2, 0, 1])

        # results[k] = pd.concat(agg_df, axis=1)
        # results[k].columns = results[k].columns.reorder_levels([1, 2, 3, 0])

        all_cols = agg_df['mean'].columns
        kept_cols = {'auc': [], 'fpr': []}
        
        for m in config[k]['ood_methods'].split():
            kept_cols['fpr'] += [c for c in all_cols if c == (k, m, args[k].tpr / 100)]
            kept_cols['auc'] += [c for c in all_cols if c == (k, m, 'auc')]

        for _ in ('fpr', 'auc'):
            results[_][k] = agg_df['mean'][kept_cols[_]].loc[ood]
        
    best_values['fpr'] = pd.concat(results['fpr'], axis=1).min(axis=1)
    best_values['auc'] = pd.concat(results['auc'], axis=1).max(axis=1)

    results_df = pd.concat([pd.concat(results[_].values(), axis=1) for _ in results], axis=1)
    results_df.columns = results_df.columns.reorder_levels([-1, 0, 1])

    cols = results_df.columns

    n_methods = len(results_df.columns) // 2
    column_format = '@{}l%\n' + '@{/}'.join(['S[table-format=2.1]%\n'] * n_methods) * 2 + '@{}'

    cols = cols.droplevel(1)
    def _f(s):
        rd = {'iws-2s': '\\acron{iws}-2s',
              'baseline': 'baseline',
              'logits': 'logits'}

        _ = rd.get(s, '\\acron{{{}}}'.format(s))
        return '\\text{{{}}}'.format(_)
        
    cols = cols.set_levels(['\\acron{fpr}', '\\acron{auroc}'], level=0)
    cols = cols.set_levels([_f(_) for _ in cols.levels[-1]], level=-1)
    results_df.columns = cols

    n_methods = len(cols) // 2
    methods = [c[-1] for c in cols] [:n_methods]
    _row = '&\\multicolumn{{{n}}}c{{{fpr}}} & \\multicolumn{{{n}}}c{{{auc}}} \\\\\n'
    tex_header = _row.format(n=n_methods,
                             fpr='\\text{\\acron{fpr}@' + default_config['tpr'] + '}',
                             auc='\\text{\\acron{auroc}}')
    tex_header += '\\midrule'
    tex_header += '\gls{{ood}}&\\multicolumn{{{n}}}c{{{methods}}} \\\\'.format(n=n_methods*2, methods= '/'.join(methods))
    header = True

    for i, r in results_df.iterrows():
        print('%%%%%%%', i)
        formatters = {c: partial(bold_best_values, value=best_values['auc' if c[0] == 'auc' else 'fpr'][i]) for c in cols}
        tex_code = results_df.loc[[i]].to_latex(formatters=formatters,
                                                header=False,
                                                escape=False,
                                                index=False,
                                                multicolumn=False,
                                                column_format=None)
        i0 = 2
        i1 = -3
        tex_code_ = tex_code.split('\n')
        print('%%%%%% FIRST LINE')
        if header:
            print('\\begin{tabular}{%')
            print(column_format)
            print('}')
            print('\\toprule')
            print(tex_header)
        print('\\midrule')
        tex_code = '\n'.join(['\\makecommand{{{}}} & '.format(i) + r for r in tex_code_[i0:i1]])
        print(tex_code)
        last_lines = '\n'.join(tex_code_[i1:])
        column_format = None
        header = False
    print('%%%%% END OF TAB')
    print(last_lines)
                               
        
    # print(results[k].to_string(float_format='{:.3g}'.format))
    """df_ = pd.concat(results.values(), axis=1)
    
        df_class = df_.loc[[default_config['dataset']]]
        row_format = default_config['row']
    """
    


    
