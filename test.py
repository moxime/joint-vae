from __future__ import print_function

import torch
from cvae import ClassificationVariationalNetwork as CVNet
import os
import sys
import hashlib
import logging
import pandas as pd
from utils.parameters import get_args, set_log, gethostname
from utils.print_log import EpochOutput, turnoff_debug
from utils.save_load import make_dict_from_model, available_results, save_json
from utils.save_load import fetch_models
from utils.tables import export_losses
from utils.texify import tex_architecture, texify_test_results, texify_test_results_df
from utils.tables import results_dataframe, format_df_index
from utils.testing import early_stopping


if __name__ == '__main__':

    hostname = gethostname()
    args = get_args('test')

    debug = args.debug
    verbose = args.verbose

    log_dir = os.path.join(args.job_dir, 'log')
    log = set_log(verbose, debug, log_dir, name='test', job_number=args.job_id)
    logging.debug('$ ' + ' '.join(sys.argv))

    job_dir = args.job_dir

    output_file = os.path.join(job_dir, f'test-{args.job_id:06d}.out')

    log.debug(f'Outputs registered in {output_file}')
    outputs = EpochOutput()
    outputs.add_file(output_file)

    batch_size = args.batch_size
    load_dir = args.load_dir
    flash = args.flash

    test_sample_size = args.test_sample_size
    ood_sample_size = args.ood
    min_test_sample_size = args.min_test_sample_size

    filters = args.filters
    filter_str = str(filters)

    logging.debug('Filters: %s', filter_str)

    oodsets = {}
    if args.sets:
        for s_ in args.sets:
            oodsets[s_[0]] = s_[1:]
            # rotate = False
            # if '90' in s_:
            #     s_.remove('90')
            #     rotate = True
            # for s in s_:
            #     oodsets[s] = [_ for _ in s_ if _ != s]
            #     if rotate:
            #         oodsets[s].append(s + '90')
    for s in oodsets:
        logging.info('OOD sets kept for %s: %s', s, ' - '.join(oodsets[s]))

    latex_formatting = args.latex

    sort = args.sort

    early_stopping_method = tuple(args.early_stopping.split('-')) if args.early_stopping else None

    for k, v in vars(args).items():
        logging.debug('%s: %s', k, str(v))

    search_dir = load_dir if load_dir else job_dir

    registered_models_file = 'models-' + gethostname() + '.json'

    list_of_networks = fetch_models(search_dir, registered_models_file, filter=filters,
                                    flash=flash,
                                    tpr=args.tpr[0] / 100,
                                    build_module=False,
                                    load_state=False)

    total = len(list_of_networks)

    log.debug('{} models found'.format(total))
    log.info('Is kept')
    log.info('| Results')
    log.info('| are fully available')
    log.info('| | can (*: all, x: partially) be extracted from recorders')
    log.info('| | | have to be computed')
    log.info('| | | | job #')
    log.info('| | | | |                      epoch #')
    # log.info('|||')

    n_trained = 0
    n_tested = 0
    n_to_be_tested = 0
    n_ood_computed = 0
    n_ood_to_be_computed = 0
    testsets = set()
    archs = {}
    n_epochs_to_be_computed = 0

    models_to_be_kept = []
    models_to_be_computed = {k: [] for k in ('json', 'recorders', 'compute', 'anywhere')}

    if not args.compute:
        where = ('json',)
    elif args.compute == 'recorder':
        where = ('json', 'recorders')
    elif args.compute == 're':
        where = ('recorders',)
    elif args.compute == 'hard':
        where = ('json', 'recorders', 'compute')
    elif args.compute == 'rehard':
        where = ('compute',)
    else:
        where = ('json',)

    # print('***', args.compute, *where)

    for n in list_of_networks:
        to_be_kept = filters.filter(n)
        # if to_be_kept:
        # for k in filters:
        #     logging.debug(k, n.get(k), 'in', ' and '.join(filters[k]))
        if to_be_kept:
            d = n['dir']
            derailed = os.path.join(d, 'derailed')
            to_be_kept = not os.path.exists(derailed)
            if args.cautious and to_be_kept:
                log.warning('Cautious verifications to be implemented')
                try:
                    # log.debug('Evaluation of one sample...')
                    # net.evaluate(torch.randn(1, *net.input_shape))
                    # log.debug('...done')
                    pass
                except ValueError:
                    open(derailed, 'a').close()
                    log.debug(f'Net in {d} has been marked as derailed')
            to_be_kept = not n['is_resumed']

            if n['set'] in archs:
                archs[n['set']].add(n['arch'])
            else:
                archs[n['set']] = {n['arch']}

            wanted_epochs = []
            if early_stopping_method:
                _k = 'early-' + '-'.join(early_stopping_method)
                if args.compute == 'early' and _k in n['net'].training_parameters:
                    n['net'].training_parameters.pop(_k)
                if _k in n['net'].training_parameters or 'json' not in where:
                    wanted_epochs.append(n['net'].training_parameters[_k])
                else:
                    early_stopping_epoch = early_stopping(n, strategy=early_stopping_method[0],
                                                          which=early_stopping_method[1])
                    if early_stopping_epoch is not None:
                        wanted_epochs.append(early_stopping_epoch)
                    if wanted_epochs:
                        n['net'].training_parameters[_k] = int(wanted_epochs[-1])
                    if not args.dry_run:
                        save_json(n['net'].training_parameters, n['dir'], 'train_params.json')

                # print('***', n['set'], wanted_epoch)

            if not wanted_epochs:
                wanted_epochs = ['last'] if not args.all_epochs else [*available_results(n, epoch_tolerance=1e9)]

            kept_epochs = []
            for wanted_epoch in wanted_epochs:
                logging.debug('Looking for wanted epodh {}'.format(wanted_epoch))
                to_be_kept = False
                epoch_tolerance = 0
                while not to_be_kept and epoch_tolerance <= 10:

                    available = available_results(n, wanted_epoch=wanted_epoch,
                                                  where=where, epoch_tolerance=epoch_tolerance)

                    total_available_by_epoch = {_: available[_]['all_sets']['anywhere'] for _ in available}
                    if total_available_by_epoch:
                        result_epoch = max(total_available_by_epoch, key=total_available_by_epoch.get)
                        a__ = available[result_epoch]
                        a_ = a__['all_sets']

                        if total_available_by_epoch[result_epoch]:
                            if result_epoch in kept_epochs:
                                break
                            kept_epochs.append(result_epoch)
                            oodsets_n = oodsets.get(n['set'])
                            # print('*** test.py:200', oodsets_n)  #
                            logging.debug('Making dict')
                            models_to_be_kept.append(dict(model=make_dict_from_model(n['net'],
                                                                                     directory=n['dir'],
                                                                                     oodsets=oodsets_n,
                                                                                     wanted_epoch=result_epoch),
                                                          epoch=result_epoch,
                                                          plan=a_))
                            logging.debug('Dict done')
                            to_be_kept = True

                    epoch_tolerance += 5
                if to_be_kept:
                    a_everywhere = available_results(n, wanted_epoch=result_epoch,
                                                     epoch_tolerance=0)[result_epoch]['all_sets']

                    is_a = a_.get('json', 0)
                    is_r = a_.get('recorders', 0)
                    is_c = a_.get('compute', 0)

                    # print('***', is_a, is_r, is_c, '***', a_everywhere)
                    n_epochs_to_be_computed += is_c

                    if not is_a:
                        _a = '|'
                    elif is_a == a_everywhere['anywhere']:
                        _a = '*'
                    else:
                        _a = 'x'

                    if a_everywhere['compute']:
                        _r = 'x' if is_r else 'o'
                        _c = '*' if is_c else 'o'
                    else:
                        _r = '*' if is_r else '|'
                        _c = '|'

                else:
                    _a = _r = _c = '|'
                    result_epoch = 'n/a'

                _s = '{x} {a} {r} {c} {j:6} {s:8} {t:5} {e:4}/{d:4} {arch:80.80}'
                logging.info(_s.format(x='*' if to_be_kept else '|',
                                       a=_a, r=_r, c=_c, j=n['job'],
                                       s=n['set'], t=n['type'], arch=n['arch'],
                                       e=result_epoch, d=n['done']))

            # for d in filters:
            #   print(d, n[d])
            #   for f in filters[d]:
            #       print(f, f.filter(n[d]))
        else:
            # logging.debug('| | | | {:6d}'.format(n['job']))
            logging.debug('| | | | {:6d} {:8} {:5} {:80.80}'. format(n['job'], n['set'], n['type'], n['arch']))

    logging.info('|     {:d} epochs to be computed'.format(n_epochs_to_be_computed))
    logging.info('{:d} models kept'.format(len(models_to_be_kept)))

    models_to_be_kept.sort(key=lambda d: d['model'].get('job', 0))
    models_to_be_kept = models_to_be_kept[-args.last:]

    if args.list_jobs_and_quit:
        for m in models_to_be_kept:
            print(m['model'].get('job', 0))

        logging.info('%d mdoels', len(models_to_be_kept))
        sys.exit(0)

    for s in archs:
        archs[s] = {n['model']['arch'] for n in models_to_be_kept if n['model']['set'] == s}

    for m_ in models_to_be_kept:
        m = m_['model']
        epoch = m_['epoch']
        plan = m_['plan']
        if plan['recorders'] or plan['compute']:
            print('Computing rates of job {} of type {} at epoch {}'.format(m['job'], m['type'], epoch))
            logging.debug('Plan for {}; {}'.format(m['job'], plan))
            model = CVNet.load(m['dir'], build_module=True, load_state=plan['compute'] and True)
            if plan['compute']:
                device = args.device or 'cuda'
            else:
                device = args.device or 'cpu'

            logging.debug('Will work on {}'.format(device))
            model.to(device)
            with torch.no_grad():
                model.test_loss = {}
                model.test_measures = {}
                sample_dirs = [os.path.join(m['dir'], 'samples', '{:04d}'.format(epoch))]
                print('OOD')
                if epoch not in model.train_history:
                    model.train_history[epoch] = {}
                history_checkpoint = model.train_history[epoch]
                model.ood_detection_rates(epoch=epoch,
                                          from_where=where,
                                          sample_dirs=sample_dirs,
                                          outputs=outputs,
                                          print_result='OFR' if not plan['compute'] else 'OFM')
                if model.predict_methods or True:
                    print('Acc', *model.predict_methods, 'in', *where)
                    test_accuracy = model.accuracy(epoch=epoch,
                                                   from_where=where,
                                                   sample_dirs=sample_dirs,
                                                   outputs=outputs,
                                                   print_result='TFR' if not plan['compute'] else 'TFM')
                    print('Misclassification')
                    model.misclassification_detection_rates(epoch=epoch,
                                                            from_where=where,
                                                            outputs=outputs,
                                                            print_result='MFR' if not plan['compute'] else 'MFM')
                    history_checkpoint['test_accuracy'] = test_accuracy
                test_loss = model.test_losses
                test_measures = model.test_measures

                if test_loss:
                    logging.info('updating test loss')
                    history_checkpoint['test_loss'] = test_loss
                if test_measures:
                    logging.info('updating test measures')
                    history_checkpoint['test_measures'] = test_measures

            if not args.dry_run:
                model.save(m['dir'])
            m.update(make_dict_from_model(model, m['dir'], wanted_epoch=epoch))

    models_to_be_kept = [_['model'] for _ in models_to_be_kept]
    for n in models_to_be_kept:
        tex_architecture(n)
        export_losses(n, which='all')
        # TK texify_test_results(n)

    all_methods = 'all' if args.expand > 1 else 'first'

    tpr = [t / 100 for t in args.tpr]

    print_sorting_keys = False
    if sort and 'print' in sort:
        sort.remove('print')
        print_sorting_keys = True

    df = results_dataframe(models_to_be_kept,
                           ood_methods=args.ood_methods or all_methods,
                           predict_methods=args.predict_methods or all_methods,
                           misclass_methods=args.misclass_methods or all_methods,
                           ood=oodsets,
                           show_measures=10,
                           tnr=args.tnr,
                           tpr=tpr,
                           sorting_keys=sort)

    _sep = ('_', ':', ',', ' ')
    tex_filter_str = filter_str
    for _ in _sep:
        tex_filter_str = tex_filter_str.replace(_, '-')

    tab_file, tex_file, agg_tab_file, agg_tex_file = None, None, None, None

    try:
        filters_file = '--or--'.join(os.path.splitext(os.path.basename(f))[0] for f in args.from_files)
        if str(args.filters['args']):
            filters_file += '--and--{}'.format(args.filters['args'])
    except TypeError:
        filters_file = None

    results_file_name = args.results_file or filters_file or tex_filter_str
    early_stopping_str = args.early_stopping or 'last'
    tab_code = hashlib.sha1(bytes(tex_filter_str + early_stopping_str, 'utf-8')).hexdigest()[:6]

    if len(df) == 1:
        _df = next(iter(df))
        tab_file = {_df: os.path.join(args.results_directory, results_file_name + '.tab')}
        tex_file = {_df: os.path.join(args.results_directory, results_file_name + '.tex')}
        agg_tab_file = {_df: os.path.join(args.results_directory, results_file_name + '-agg.tab')}
        agg_tex_file = {_df: os.path.join(args.results_directory, results_file_name + '-agg.tex')}
    elif len(df) > 1:
        tab_file = {_: os.path.join(args.results_directory, '{}-{}.tab'.format(_, results_file_name)) for _ in df}
        tex_file = {_: os.path.join(args.results_directory, '{}-{}.tex'.format(_, results_file_name)) for _ in df}
        agg_tab_file = {_: os.path.join(args.results_directory, '{}-{}-agg.tab'.format(_, results_file_name))
                        for _ in df}
        agg_tex_file = {_: os.path.join(args.results_directory, '{}-{}-agg.tex'.format(_, results_file_name))
                        for _ in df}

    log.info('')
    log.info('')
    log.info('')

    pd.set_option('max_colwidth', 15)

    try:
        args.remove_index.remove('auto')
        are_auto_removed_index = True
    except (ValueError, AttributeError):
        are_auto_removed_index = False
    first_set = True
    for s, d in df.items():

        auto_removed_index = {}

        texify_test_results_df(d, s, tex_file[s], tab_file[s], tab_code=tab_code)

        d.index = pd.MultiIndex.from_frame(d.index.to_frame().fillna('NaN'))

        remove_wim_from = all(d.index.to_frame()['job'] == d.index.to_frame()['wim_from'])

        if tex_file:
            with open(tex_file[s], 'a') as f:
                f.write('\def\joblist{')
                f.write(','.join(['{:06d}'.format(n['job']) for n in models_to_be_kept]))
                f.write('}\n')
        if args.remove_index is not None:
            non_removable_index = ['job', 'type']

            if are_auto_removed_index:
                removed_index = [i for i, l in enumerate(d.index.levels)
                                 if len(l) < 2 and l.name not in non_removable_index]
            else:
                removed_index = []
            for i in removed_index:
                auto_removed_index[d.index.names[i]] = d.index[0][i]
            unremoved_index = []
            for i in args.remove_index:
                if i.replace('-', '_') in d.index.names:
                    removed_index.append(i.replace('-', '_'))
                else:
                    unremoved_index.append(i)
            if unremoved_index:
                print(*d.index.names)
                logging.error('{} are not removed. Possible removable index: {}'.format(', '.join(unremoved_index),
                                                                                        ', '.join(d.index.names)))
            d = d.droplevel(removed_index)

            if remove_wim_from and 'wim_from' in d.index.names:
                pass  # d = d.droplevel('wim_from')

        idx = list(d.index.names)
        if 'job' in d.index.names:
            wj = d.index.get_level_values('job').to_series().apply(str).apply(len).max()
            idx.remove('job')

        gb = d.groupby(level=idx)

        d_mean = gb.agg('mean')
        d_std = gb.agg('std')
        d_count = gb.agg('count')

        if 'job' in d.index.names:
            _s = '{{:{}d}}'.format(wj - 1)
            d_count_s = d_count[d_count.columns[-1]].apply(_s.format)
            d_count_v = d_count_s.values
            # d_mean.reset_index(level='job', drop=True, inplace=True)
            # d_std.reset_index(level='job', drop=True, inplace=True)
            d_mean.insert(loc=0, column='count', value=d_count_v)

        if agg_tab_file:
            # last_names = d_mean.columns.levels[-1]
            # mean_last_names = [_ + 'mean' for _ in last_names]
            # std_last_names = [_ + 'std' for _ in last_names]
            # d_mean.columns = d_mean.columns.set_level(mean_last_names, level=-1)
            d_agg = pd.concat(dict(mean=d_mean, std=d_std), axis=1)
            texify_test_results_df(d_agg, s, agg_tex_file[s], agg_tab_file[s], tab_code=tab_code)

        col_show_levels = {_: 0 for _ in d.columns}
        col_show_levels.update({_: 4 for _ in d.columns if _[0] == 'measures'})
        col_show_levels.update({_: 3 for _ in d.columns if _[-1] in ['done']})
        col_show_levels.update({_: 2 for _ in d.columns if _[-1] in ['epoch', 'validation']})
        col_show_levels.update({_: 2 for _ in d.columns if _[-1] in ['n']})
        col_show_levels.update({_: 1 for _ in d.columns if _[-1] in ['dB', 'nll', 'kl']})

        drop_cols = [_ for _ in d.columns if col_show_levels[_] > args.show_measures]

        if drop_cols:
            for d_ in (d, d_mean, d_std):
                d_.drop(columns=drop_cols, inplace=True)
                format_df_index(d_, inplace=True)
                d_.rename(columns={'validation': 'val'}, inplace=True)

        d_str = d.to_string(na_rep='', float_format='{:.3g}'.format, sparsify=True)

        if not first_set:
            print('\n')
        else:
            first_set = False

        width = len(d_str.split('\n')[0])
        print(f'{s.upper():=^{width}}')

        if not args.only_average:
            print(d_str)

        if args.average or args.only_average:
            # d_mean.index = d_mean.index.format(formatter=_f)
            m_str = d_mean.to_string(na_rep='', float_format='{:.3g}'.format).split('\n')
            width = len(m_str[0])
            first_row = '{:-^{w}}'.format('AVERAGE', w=width)
            header = d.columns.nlevels
            # second_row = '\n'm_str[:header]
            if not args.only_average:
                print()
                print(first_row)
                print()
            # print(second_row)
            # print('\n'.join(m_str[header + 1:]))
            print('\n'.join(m_str))

        print('\nArchs')
        for a in archs[s]:
            arch_code = hashlib.sha1(bytes(a, 'utf-8')).hexdigest()[:6]
            print(arch_code, ':\n', a)
        if print_sorting_keys:
            print('Possible sorting keys :', *d.index.names)
        if auto_removed_index:
            print('\nCommon values')
        for k, v in format_df_index(auto_removed_index).items():
            print('{:8}: {}'.format(k, v))
        for _ in range(1):
            print('=' * width)
