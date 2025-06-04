import os
import numpy as np
import pandas as pd
import logging
import configparser
from utils.print_log import turnoff_debug
from utils.parameters import gethostname, DEFAULT_RESULTS_DIR, DEFAULT_JOBS_DIR
from utils.filters import DictOfListsOfParamFilters, ParamFilter, MetaFilter
from utils.save_load import fetch_models, make_dict_from_model
from utils.tables import results_dataframe, format_df_index, auto_remove_index
from pydoc import locate
from utils.texify import TexTab


def _process_csv(csv_file, header=2, index_col=1):

    df = pd.read_csv(csv_file, header=[*range(header)], index_col=[*range(index_col)])

    df.columns.rename({'measures': 'metrics'}, inplace=True)
    i_names = set(df.columns.names) | set(df.index.names)

    col_names = ['set', 'method', 'metrics']
    assert set(col_names) <= i_names
    assert set(df.columns.names) <= set(col_names)

    for _ in df.index.names:
        if _ in col_names:
            df = df.unstack(_)

    df = df.reorder_levels(col_names, axis='columns')
    return df


def parse_config(config_file, which=['all'], root=DEFAULT_RESULTS_DIR, texify_file=None):

    config_dir = os.path.dirname(config_file)
    raw_config = configparser.ConfigParser()
    raw_config.read(config_file)

    texify = {'conf': raw_config.pop('texify', {}), 'file': {}}

    if texify_file:
        texify_conf = configparser.ConfigParser()
        texify_conf.read(texify_file)
        texify['file'] = texify_conf

    if 'all' in which:
        which = list(raw_config.keys())
        if 'DEFAULT' in which:
            which.remove('DEFAULT')
    else:
        which = [w for w in which if w in raw_config]

    default_config = raw_config['DEFAULT']

    config = {'texify': texify}

    config['config_dir'] = config_dir
    config['job_dir'] = {k: raw_config[k].get('jobs') or DEFAULT_JOBS_DIR for k in which}

    config['dataset'] = default_config.get('dataset')
    config['oodsets'] = default_config.get('ood').split()

    config['tpr'] = default_config.get('tpr', 95)

    config['ood_metrics'] = default_config.get('ood_metrics', '').split()
    config['acc_metrics'] = default_config.get('acc_metrics', '').split()

    kept_index = default_config.get('kept_index', '').split()

    average = default_config.get('average', '').split()
    if len(average) == 1:
        average = {average[0]: config['oodsets']}

    elif len(average) > 1:
        average = {average[0]: average[1:]}

    config['average'] = average

    kept_index_ = [_.split(':') for _ in kept_index]
    kept_index = [_[0] for _ in kept_index_]
    kept_index_format = [_[1] if len(_) == 2 else 'c' for _ in kept_index_]

    ini_file_name = os.path.splitext(os.path.split(config_file)[-1])[0]

    _suf = '{}'

    tex_file = default_config.get('file', ini_file_name + _suf + '-tab.tex')
    config['tex_file'] = os.path.join(root, tex_file)
    tab_file = default_config.get('file', ini_file_name + _suf + '-tab.tab')
    config['tab_file'] = os.path.join(root, tab_file)

    logging.info('Keys in config file: %s', ' '.join(which))

    config['from_filters'] = [k for k in which if not raw_config[k].get('from_csv')]
    config['from_csv'] = [k for k in which if raw_config[k].get('from_csv')]

    config['tabs'] = {k: raw_config[k] for k in which}

    return config


def _concat_df(df_dict, col_names=['set', 'metrics'], index_rename={}, return_best=False):

    df = pd.concat(df_dict, axis=1, names='method')

    s = df.stack()  # series with index = ['set', 'metrics', 'method']

    if not return_best:
        return s.rename(index=index_rename).unstack(col_names, sort=True)

    higher_is_better = ['auc', 'acc']

    df = s.unstack(['set', 'metrics'])

    s_best = df.min()
    s_max = df.max()

    max_i = s_best.index.isin(higher_is_better, level='metrics')

    s_best[max_i] = s_max[max_i]

    return s_best.rename(index=index_rename)


def make_tables(config, filter_keys,
                acc_metrics=['acc'],
                ood_metrics=['fpr', 'auc'],
                show_dfs=True, flash=True):

    config_dir = config['config_dir']
    job_dir = config['job_dir']
    dataset = config['dataset']
    oodsets = config['oodsets']
    which_from_filters = config['from_filters']
    which_from_csv = config['from_csv']
    average = config['average']

    tab_comments = []

    acc_metrics = config['acc_metrics'] or acc_metrics
    ood_metrics = config['ood_metrics'] or ood_metrics

    registered_models_file = 'models-' + gethostname() + '.json'

    filters = {}

    tab_config = config['tabs']

    for k in which_from_filters:

        logging.info('| key %s:', k)
        logging.info(' -- '.join(['{}: {}'.format(_, tab_config[k][_]) for _ in tab_config[k]]))
        filters[k] = DictOfListsOfParamFilters()
        print(k, filters[k])
        for _ in tab_config[k]:
            if _ in filter_keys:
                dest = filter_keys[_]['dest']
                ftype = filter_keys[_]['type']
                filters[k].add(dest, ParamFilter.from_string(arg_str=tab_config[k][_],
                                                             type=locate(ftype or 'str')))

    global_filters = MetaFilter(operator='or', **filters)

    models = {k: fetch_models(job_dir[k], registered_models_file,
                              filter=filters[k], build_module=False,
                              flash=flash) for k in filters}

    logging.info('Fetched {} models'.format(len(models)))

    for k in filters:
        logging.debug('| filters for %s', k)
        f = filters[k]
        for _ in f:
            logging.debug('| | %s: %s', _, ' '.join(str(__) for __ in f[_]))

    models_by_type = {k: [] for k in filters}
    archs_by_type = {k: set() for k in filters}

    for k, filter in filters.items():
        for n in models[k]:
            to_be_kept = filter.filter(n)
            d = n['dir']
            derailed = os.path.join(d, 'derailed')
            to_be_kept = to_be_kept and not os.path.exists(derailed)

            if to_be_kept:
                epoch_to_fetch = tab_config[k].get('epoch', 'last')
                if epoch_to_fetch == 'min-loss':
                    epoch_to_fetch = 'early-min-loss'
                epoch = n['net'].training_parameters.get(epoch_to_fetch, 'last')
                logging.debug('Epoch for %s: %s = %s', n['job'], epoch_to_fetch, epoch)
                with turnoff_debug():
                    n = make_dict_from_model(n['net'], n['dir'], wanted_epoch=epoch)
                models_by_type[k].append(n)
                archs_by_type[k].add(n['arch'])

    tpr_ = config['tpr']
    try:
        i_fpr = ood_metrics.index('fpr')
        ood_metrics[i_fpr] = 'fpr@{}'.format(tpr_)
    except ValueError:
        pass

    sorting_index = {}
    sorting_index['set'] = {_: i for i, _ in enumerate([dataset, *oodsets])}
    sorting_index['metrics'] = {_: i for i, _ in enumerate(['rate', *acc_metrics, *ood_metrics])}
    sorting_index['method'] = {_: i for i, _ in enumerate(tab_config)}

    for a in average:
        sorting_index['set'][a] = max([sorting_index['set'][_] for _ in average[a]]) + 0.5

    tpr = float(tpr_) / 100
    raw_df = {}
    job_list = {}
    df_string = {}
    removed_index = {}
    ood_methods = {}
    acc_methods = {}
    for k in which_from_filters:
        job_list[k] = [_['job'] for _ in models_by_type[k]]
        job_list_str = ' '.join(str(_) for _ in job_list[k])
        _s = '{} models for {}: {}'.format(len(models_by_type[k]), k, job_list_str)
        tab_comments.append(_s)
        logging.info(_s)
        if not models_by_type[k]:
            logging.warning('Skipping {}'.format(k))
            continue
        ood_methods[k] = tab_config[k].get('ood_method', '')
        acc_methods[k] = tab_config[k].get('acc_method', '')
        df_ = results_dataframe(models_by_type[k],
                                predict_methods=tab_config[k].get('acc_method', '').split(),
                                ood_methods=[ood_methods[k]],
                                tpr=[tpr])
        df = next(iter(df_.values()))
        df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna('NaN'))

        idx = list(df.index.names)
        if 'job' in idx:
            idx.remove('job')

        removed_index[k] = auto_remove_index(df)
        #  removed_index = {}      #
        df_short = format_df_index(df.droplevel(list(removed_index[k])))
        cols = df_short.columns
        showed_cols = cols.isin(['acc'], level='metrics')
        showed_cols |= (cols.isin(tab_config[k]['ood'].split(), level='set')
                        & ~cols.isin(['n', 'mean', 'std'], level='metrics'))
        df_string[k] = df_short[cols[showed_cols]].to_string(float_format='{:.1f}'.format)
        df_width = len(df_string[k].split('\n')[0])

        """ tab comments """
        tab_comments.append('{k:=^{w}s}\n'.format(k=k, w=df_width))
        _s = '{:2} models for {:12}: {}'
        tab_comments.append(_s.format(len(job_list[k]), k,
                                      ' '.join(str(_) for _ in job_list[k])))

        tab_comments.extend(df_string[k].split('\n'))
        tab_comments.append('{k:=^{w}s}'.format(k='', w=df_width))
        tab_comments.append('Common values')
        nans = []
        for _, v in format_df_index(removed_index[k]).items():
            if not _.startswith('drop'):
                if v != 'NaN':
                    tab_comments.append('{:8}: {}'.format(_, v))
                else:
                    nans.append(_)
        if nans:
            tab_comments.append('{:8}: '.format('NaNs') + ', '.join(nans))
        """ """

        raw_df[k] = df.groupby(level=idx).agg('mean')
        # print('****', k, raw_df[k].index.names)
        raw_df[k].columns.rename(['set', 'method', 'metrics'], inplace=True)

    for k in which_from_csv:
        csv_file = tab_config[k]['from_csv']
        if not os.path.exists(csv_file):
            csv_file = os.path.join(config_dir, csv_file)
            logging.info('Loaded {}'.format(csv_file))
        logging.info('results for {} from csv file {}'.format(k, csv_file))
        index_col = int(tab_config[k]['index_col'])
        header = int(tab_config[k]['header'])

        df = _process_csv(csv_file, index_col=index_col, header=header)

        df.rename(columns={'fpr': 'fpr@95', 'accuracy': 'acc'}, inplace=True)
        ood_methods[k] = tab_config[k].get('ood_method', '')
        acc_methods[k] = tab_config[k].get('acc_method', '')

        raw_df[k] = df.groupby(level=df.index.names).agg('mean')

        """ tab_comments """
        c = raw_df[k].columns
        c_ood = c.isin(oodsets, level='set') & c.isin([ood_methods[k]], level='method')
        c_acc = c.isin([dataset], level='set') & c.isin([acc_methods[k]], level='method')
        tab_comments.append('{k:=^{w}s} {f}\n'.format(k=k, w=20, f=csv_file))
        _df_s = raw_df[k][c[c_ood | c_acc]].to_string(float_format='{:.1f}'.format)
        tab_comments.extend(_df_s.split('\n'))
        """ """

    if show_dfs:
        print('\n'.join(tab_comments))

    config['tab_comments'] = tab_comments

    for k in raw_df:
        df = raw_df[k]
        c = df.columns
        ood_cols = (c.isin(oodsets, level='set') &
                    c.isin(ood_metrics, level='metrics') &
                    c.isin([ood_methods[k]], level='method'))

        acc_cols = (c.isin([dataset], level='set') &
                    c.isin(acc_metrics, level='metrics') &
                    c.isin([acc_methods[k]], level='method'))

        series = df[c[ood_cols | acc_cols]].mean()

        series.index = series.index.droplevel('method')

        """ average computation """
        avg_rows = {}
        for a in average:
            # print(series)
            i = series.index.isin(average[a], level='set')
            avg_rows[a] = series[i].groupby('metrics').agg('mean')
            # print('***', a, '***')
            # print(avg_rows[a])
            avg_rows[a].index = pd.MultiIndex.from_product([[a], avg_rows[a].index],
                                                           names=['set', 'metrics'])
            series = pd.concat([series, avg_rows[a]])
            # print(series)

        def sorter(s):
            if s.name in sorting_index:
                return s.map({k: v for k, v in sorting_index[s.name].items()})
            return s

        raw_df[k] = series.sort_index(key=sorter)

    best_values_t = _concat_df(raw_df, return_best=True)
    result_df_t = _concat_df(raw_df)
    best_values = _concat_df(raw_df, col_names=['metrics', 'method'],
                             index_rename={'acc': 'rate', 'fpr@95': 'rate'},
                             return_best=True)

    result_df = _concat_df(raw_df, col_names=['metrics', 'method'],
                           index_rename={'acc': 'rate', 'fpr@95': 'rate'})

    result_df = result_df.sort_index(key=sorter)
    result_df = result_df.T.sort_index(key=sorter).T

    result_df.rename(index={dataset: 'acc'}, inplace=True)
    best_values.rename(index={dataset: 'acc'}, inplace=True)

    _suf = ['']
    if 'auc' in ood_metrics:
        _suf.append('auc')
    result_df.to_csv(config['tab_file'].format('-'.join(_suf)))

    return result_df, result_df_t, best_values, best_values_t


def make_tex(config, df, best=None):

    texify_dict = config['texify']

    def texify_text(v, where=None):
        if where == 'set':
            return r'\dset{{{}}}'.format(v)
        if v in texify_dict['conf']:
            return texify_dict['conf'][v]

        if where in texify_dict['file'] and v in texify_dict['file'][where]:
            return texify_dict['file'].get(where, v)
        return '{}'.format(v)

    def header_row(tab, name, *cols, index_length=1):
        tab.append_cell('', width=index_length, row=name)
        unique_cols_period = {v: np.diff([i for i, _ in enumerate(cols) if _ == v])
                              for v in set(cols)}
        # print(unique_cols_period)
        is_periodic = False
        period = 0
        for v in unique_cols_period:
            if not len(np.unique(unique_cols_period[v]) == 1):
                break
            if unique_cols_period[v][0] <= 1:
                break
            if period and unique_cols_period[v][0] != period:
                break
            period = unique_cols_period[v][0]
        else:
            is_periodic = True

        if is_periodic:
            cell_content = '/'.join([texify_text(c, where=name) for c in cols])
            tab.append_cell(texify_text(cell_content), row=name,
                            width=len(cols), multicol_format='c')
        else:
            previous_col = None
            multicol = 0
            for c in cols:
                if c == previous_col:
                    multicol += 1
                else:
                    if multicol:
                        tab.append_cell(texify_text(previous_col, where=name), row=header,
                                        width=multicol, multicol_format='c')
                    multicol = 1
                    previous_col = c
            tab.append_cell(texify_text(previous_col, where=name), row=header,
                            width=multicol, multicol_format='c')

    def val_row(tab, name, row, columns, what='set'):
        tab.append_cell(texify_text(name, where=what), row=name)
        for val, col in zip(row, columns):
            best_val = None
            if best is not None:
                current = {}
                for k in ('set', 'metrics'):
                    if k not in columns.names:
                        current[k] = name
                    else:
                        current[k] = col[columns.names.index(k)]
                # print(current)
                try:
                    best_val = best.loc[(current['set'], current['metrics'])]
                except KeyError:
                    pass
                # print(best_val)
            face = None
            if best_val is not None:
                face = 'bf' if abs(best_val - val) < 0.05 else None
            if val > 99.95:
                tab.append_cell(val, row=name, face=face, multicol_format='l', formatter='{:.0f}')
            else:
                tab.append_cell(val, row=name, face=face)

    col_fmt = ['l'] * df.index.nlevels + ['s2.1'] * df.shape[1]
    tab = TexTab(*col_fmt, float_format='{:2.1f}', na_rep='--')

    for comment in config['tab_comments']:
        tab.comment(comment)

    for row, header in enumerate(df.columns.names):
        cols = [_[row] for _ in df.columns]
        header_row(tab, header, *cols, index_length=df.index.nlevels)

    for i in df.index:
        what = df.index.name
        if i == 'acc' and what == 'set':
            what = 'metrics'
        val_row(tab, i, df.loc[i], df.columns, what=what)
    tab.add_midrule(row=header, after=True)

    if 'acc' in tab:
        tab.add_midrule(row='acc', after=True)

    transpose = df.index.name == 'method'
    dataset = config['dataset']
    tex_file = config['tex_file']
    _suf = ['']
    if 'auc' in config['ood_metrics']:
        _suf.append('auc')

    if transpose:
        _suf.append('t')

    tex_file = tex_file.format('-'.join(_suf))
    logging.info('Tab for {} will be saved in file {}'.format(dataset, tex_file))
    with open(tex_file, 'w') as f:
        tab.render(f, robustify=True)

    return tab


if __name__ == '__main__':

    from utils.filters import get_filter_keys

    filter_keys = get_filter_keys('./utils/filters.ini', by='key')

    logging.getLogger().setLevel(20)

    config_file = '/tmp/tab.ini'
    texify_file = '/tmp/texify.ini'

    config = parse_config(config_file, root='/tmp', texify_file=texify_file)
    df, df_t, best_vals, best_vals_t = make_tables(config, filter_keys,
                                                   ood_metrics=['fpr', 'auc'], show_dfs=True)

    tab = make_tex(config, df, best=best_vals)
    tab_t = make_tex(config, df_t, best=best_vals_t)

    # df1 = concat_df(raw_df)

    # print(df1.to_string(float_format='{:.1f}'.format))

    # df2 = concat_df(raw_df, col_names=['metrics', 'method'],
    #                 index_rename={'acc': 'rate', 'fpr@95': 'rate'})
    # print('\n====\n')
    # print(df2.to_string(float_format='{:.1f}'.format))
