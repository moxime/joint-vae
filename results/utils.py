import os
import pandas as pd
import logging
import configparser
from utils.print_log import turnoff_debug
from utils.parameters import gethostname, DEFAULT_RESULTS_DIR, DEFAULT_JOBS_DIR
from utils.filters import DictOfListsOfParamFilters, ParamFilter, MetaFilter
from utils.save_load import fetch_models, make_dict_from_model
from utils.tables import results_dataframe, format_df_index, auto_remove_index, agg_results
from pydoc import locate


def process_csv(csv_file, header=2, index_col=1):

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


def process_config_file(config_file, filter_keys, which=['all'],
                        acc_metrics=['acc'],
                        ood_metrics=['fpr', 'auc'],
                        keep_auc=True,
                        root=DEFAULT_RESULTS_DIR, show_dfs=True, flash=True):

    config_dir = os.path.dirname(config_file)
    config = configparser.ConfigParser()
    config.read(config_file)

    if 'all' in which:
        which = list(config.keys())
        if 'DEFAULT' in which:
            which.remove('DEFAULT')
    else:
        which = [w for w in which if w in config]

    default_config = config['DEFAULT']

    job_dir = default_config.get('jobs', DEFAULT_JOBS_DIR)

    registered_models_file = 'models-' + gethostname() + '.json'

    dataset = default_config.get('dataset')
    oodsets = default_config.get('ood').split()
    kept_index = default_config.get('kept_index', '').split()

    average = default_config.get('average', '').split()
    if len(average) == 1:
        average = {average[0]: oodsets}

    elif len(average) > 1:
        average = {average[0]: average[1:]}

    kept_index_ = [_.split(':') for _ in kept_index]
    kept_index = [_[0] for _ in kept_index_]
    kept_index_format = [_[1] if len(_) == 2 else 'c' for _ in kept_index_]

    ini_file_name = os.path.splitext(os.path.split(config_file)[-1])[0]

    _auc = '-auc' if keep_auc else ''
    tex_file = default_config.get('file', ini_file_name + _auc + '-tab.tex')
    tex_file = os.path.join(root, tex_file)
    tab_file = default_config.get('file', ini_file_name + _auc + '-tab.tab')
    tab_file = os.path.join(root, tab_file)

    logging.info('Tab for {} will be saved in file {}'.format(dataset, tex_file))

    filters = {}

    logging.info('Keys in config file: %s', ' '.join(which))

    which_from_filters = [k for k in which if not config[k].get('from_csv')]
    which_from_csv = [k for k in which if config[k].get('from_csv')]

    for k in which_from_filters:

        logging.info('| key %s:', k)
        logging.info(' -- '.join(['{}: {}'.format(_, config[k][_]) for _ in config[k]]))
        filters[k] = DictOfListsOfParamFilters()

        for _ in config[k]:
            if _ in filter_keys:
                dest = filter_keys[_]['dest']
                ftype = filter_keys[_]['type']
                filters[k].add(dest, ParamFilter.from_string(arg_str=config[k][_],
                                                             type=locate(ftype or 'str')))

    global_filters = MetaFilter(operator='or', **filters)

    models = fetch_models(job_dir, registered_models_file, filter=global_filters, build_module=False,
                          flash=flash)

    logging.info('Fetched {} models'.format(len(models)))

    for k in filters:
        logging.debug('| filters for %s', k)
        f = filters[k]
        for _ in f:
            logging.debug('| | %s: %s', _, ' '.join(str(__) for __ in f[_]))

    models_by_type = {k: [] for k in filters}
    archs_by_type = {k: set() for k in filters}

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
                archs_by_type[k].add(n['arch'])

    tpr_ = default_config['tpr']
    try:
        i_fpr = ood_metrics.index('fpr')
        ood_metrics[i_fpr] = 'fpr@{}'.format(tpr_)
    except ValueError:
        pass

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
        logging.info('{} models for {}: {}'.format(len(models_by_type[k]), k, job_list_str))
        if not models_by_type[k]:
            logging.warning('Skipping {}'.format(k))
            continue
        ood_methods[k] = config[k].get('ood_method', '')
        acc_methods[k] = config[k].get('acc_method', '')
        df_ = results_dataframe(models_by_type[k],
                                predict_methods=config[k].get('acc_method', '').split(),
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
        showed_cols |= (cols.isin(config[k]['ood'].split(), level='set')
                        & ~cols.isin(['n', 'mean', 'std'], level='metrics'))
        df_string[k] = df_short[cols[showed_cols]].to_string(float_format='{:.1f}'.format)
        df_width = len(df_string[k].split('\n')[0])
        if show_dfs:
            print('\n{k:=^{w}s}'.format(k=k, w=df_width))
            print(df_string[k])
            print('{k:=^{w}s}\n'.format(k='', w=df_width))
            print('Common values')
            nans = []
            for _, v in format_df_index(removed_index[k]).items():
                if not _.startswith('drop'):
                    if v != 'NaN':
                        print('{:8}: {}'.format(_, v))
                    else:
                        nans.append(_)
            if nans:
                print('{:8}:'.format('NaNs'), ', '.join(nans))

        raw_df[k] = df.groupby(level=idx).agg('mean')
        # print('****', k, raw_df[k].index.names)
        raw_df[k].columns.rename(['set', 'method', 'metrics'], inplace=True)
        raw_df[k].rename(columns={tpr: 'rate'}, level='metrics', inplace=True)

    for k in which_from_csv:
        csv_file = config[k]['from_csv']
        if not os.path.exists(csv_file):
            csv_file = os.path.join(config_dir, csv_file)
            logging.info('Loaded {}'.format(csv_file))
        logging.info('results for {} from csv file {}'.format(k, csv_file))
        index_col = int(config[k]['index_col'])
        header = int(config[k]['header'])

        df = process_csv(csv_file, index_col=index_col, header=header)

        df.rename(columns={'fpr': 'fpr@95', 'accuracy': 'acc'}, inplace=True)
        ood_methods[k] = config[k].get('ood_method', '')
        acc_methods[k] = config[k].get('acc_method', '')

        # if df.index.nlevels > 1:
        #    df.index = df.index.set_levels([_.astype(str) for _ in df.index.levels])
        raw_df[k] = df.groupby(level=df.index.names).agg('mean')
        # raw_df[k].columns.rename(['set', 'method', 'metrics'], inplace=True)

    for k in raw_df:
        df = raw_df[k]
        c = df.columns
        ood_cols = c.isin(oodsets, level='set') & \
            c.isin(ood_metrics, level='metrics') & \
            c.isin([ood_methods[k]], level='method')

        acc_cols = c.isin([dataset], level='set') & \
            c.isin(acc_metrics, level='metrics') & \
            c.isin([acc_methods[k]], level='method')

        raw_df[k] = df[c[ood_cols | acc_cols]]

        raw_df[k].columns = raw_df[k].columns.droplevel('method')

        c = raw_df[k].columns

        raw_df[k] = raw_df[k].squeeze()

    return pd.concat(raw_df, axis=1).T


def format_df(df):

    return df


if __name__ == '__main__':

    from utils.filters import get_filter_keys

    filter_keys = get_filter_keys('./utils/filters.ini', by='key')

    logging.getLogger().setLevel(200)

    raw_df = process_config_file('/tmp/tab.ini', filter_keys, root='/tmp')

    for k in raw_df:
        print(raw_df[k])
