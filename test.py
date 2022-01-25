from __future__ import print_function

import torch
from cvae import ClassificationVariationalNetwork as CVNet
import utils.torch_load as torchdl
import os
import sys
import hashlib
import logging

import pandas as pd
import numpy as np

from utils.parameters import get_args, set_log, gethostname
from utils.print_log import EpochOutput, turnoff_debug
from utils.save_load import collect_models, test_results_df, LossRecorder
from utils.save_load import make_dict_from_model, available_results, save_json, load_json
from utils.save_load import fast_collect_models, register_models
from utils.tables import export_losses, tex_architecture, texify_test_results, texify_test_results_df
from utils.tables import format_df_index
from utils.testing import early_stopping
from utils.filters import get_filter_keys

if __name__ == '__main__':

    hostname = gethostname()
    args = get_args('test')
    
    debug = args.debug
    verbose = args.verbose

    log_dir = os.path.join(args.job_dir, 'log')
    log = set_log(verbose, debug, log_dir, name='test', job_number=args.job_id)
    logging.debug('$ ' + ' '.join(sys.argv))
    if not args.force_cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f'Used device: {device} on {hostname}')
    else:
        device = torch.device('cpu')
        log.info(f'Used device: {device}')
        log.debug('CPU asked by user')

    job_dir = args.job_dir

    output_file = os.path.join(job_dir, f'test-{args.job_id:06d}.out')

    log.debug(f'Outputs registered in {output_file}')
    outputs = EpochOutput()
    outputs.add_file(output_file)

    batch_size = args.batch_size
    load_dir = args.load_dir
    dry_run = args.compute
    flash = args.flash

    test_sample_size = args.test_sample_size
    ood_sample_size = args.ood
    min_test_sample_size = args.min_test_sample_size
    
    filters = args.filters
    filter_str = '--'.join(f'{d}:{f}' for d, f in filters.items() if not f.always_true)

    logging.debug('Filters: %s', filter_str)

    oodsets = {}
    if args.sets:
        for s_ in args.sets:
            rotate = False
            if '90' in s_:
                s_.remove('90')
                rotate = True
            for s in s_:                       
                oodsets[s] = [_ for _ in s_ if _ != s]
                if rotate:
                    oodsets[s].append(s + '90')
                
    for s in oodsets:
        logging.info('OOD sets kept for %s: %s', s, ' - '.join(oodsets[s]))
                     
    latex_formatting = args.latex

    sort = args.sort
  
    early_stopping_method = tuple(args.early_stopping.split('-')) if args.early_stopping else None
    
    for k, v in vars(args).items():
        logging.debug('%s: %s', k, str(v))
    
    search_dir = load_dir if load_dir else job_dir
    
    filter_keys = get_filter_keys()
    registered_models_file = 'models-' + gethostname() + '.json'
    if flash:
        logging.debug('Flash collecting networks')
        try:
            rmodels = load_json(search_dir, registered_models_file)
            with turnoff_debug():
                list_of_networks = fast_collect_models(rmodels, filters, tpr_for_max=args.tpr[0] / 100,
                                                       load_net=False, load_state=False)
        except FileNotFoundError as e:
            logging.warning('%s, will recollect networks', e)
            flash = False
            
    if not flash:    
        logging.debug('Collecting networks')
        with turnoff_debug():
            list_of_networks = collect_models(search_dir,
                                              tpr_for_max=args.tpr[0] / 100,
                                              load_net=False,
                                              load_state=False)
            rmodels = register_models(list_of_networks, *filter_keys)
            save_json(rmodels, search_dir, registered_models_file)

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

            wanted_epoch = None
            if early_stopping_method:
                _k = 'early-' + '-'.join(early_stopping_method)
                if _k in n['net'].training_parameters:
                    wanted_epoch = n['net'].training_parameters[_k]
                else:
                    wanted_epoch = early_stopping(n, strategy=early_stopping_method[0],
                                                  which=early_stopping_method[1])
                    if wanted_epoch:
                        n['net'].training_parameters[_k] = int(wanted_epoch)
                    if not args.dry_run:
                        save_json(n['net'].training_parameters, n['dir'], 'train.json')

                # print('***', n['set'], wanted_epoch)
            if not wanted_epoch:
                wanted_epoch = 'last'

            # print('***', n['set'], n['dir'], wanted_epoch)

            to_be_kept = False
            epoch_tolerance = 0
            while not to_be_kept and epoch_tolerance <= 10:
                epoch_tolerance += 5
                available = available_results(n, misclass_methods=[], wanted_epoch=wanted_epoch,
                                              where=where, epoch_tolerance=epoch_tolerance)

                total_available_by_epoch = {_: available[_]['all_sets']['anywhere'] for _ in available}
                if total_available_by_epoch:
                    result_epoch = max(total_available_by_epoch, key=total_available_by_epoch.get)
                    a__ = available[result_epoch]
                    a_ = a__['all_sets']

                    if total_available_by_epoch[result_epoch]:
                        models_to_be_kept.append(dict(model=make_dict_from_model(n['net'],
                                                                                 directory=n['dir'],
                                                                                 wanted_epoch=result_epoch),
                                                    epoch=result_epoch,
                                                      plan=a_))
                        to_be_kept = True

            if to_be_kept:
                a_everywhere = available_results(n, misclass_methods=[], wanted_epoch=result_epoch,
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

            _s = '{x} {a} {r} {c} {j:6d} {s:8} {t:5} {e:4d}/{d:4d} {arch:80.80}'
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

    for s in archs:
        archs[s] = {n['model']['arch'] for n in models_to_be_kept if n['model']['set'] == s}
    
    for m_ in models_to_be_kept:
        m = m_['model']
        epoch = m_['epoch']
        plan = m_['plan']

        # print('*** test:258', plan)
        if plan['recorders'] or plan['compute']:
            print('Computing rates of job {} of type {} at epoch {}'.format(m['job'], m['type'], epoch)) 
            model = CVNet.load(m['dir'], load_state=plan['compute'])

            with torch.no_grad():
                model.ood_detection_rates(epoch=epoch,
                                          from_where=where,
                                          sample_dirs=[os.path.join(m['dir'], 'samples', '{:4d}'.format(epoch))],
                                          outputs=outputs,
                                          print_result='OFR' if not plan['compute'] else 'OFM')

                model.accuracy(epoch=epoch,
                               from_where=where,
                               sample_dirs=[os.path.join(m['dir'], 'samples', '{:4d}'.format(epoch))],
                               outputs=outputs,
                               print_result='TFR' if not plan['compute'] else 'TFM')

            if not args.dry_run:
                model.save(m['dir'])
            m.update(make_dict_from_model(model, m['dir'], wanted_epoch=epoch))                

    models_to_be_kept = [_['model'] for _ in models_to_be_kept]
    for n in models_to_be_kept:
        tex_architecture(n)
        export_losses(n, which='all')
        texify_test_results(n)
        
    all_methods = 'all' if args.expand > 1 else None

    tpr = [t/100 for t in args.tpr]

    print_sorting_keys = False
    if sort and 'print' in sort:
        sort.remove('print')
        print_sorting_keys = True
        
    df = test_results_df(models_to_be_kept, 
                         ood_methods=args.ood_methods or all_methods,
                         predict_methods=args.predict_methods or all_methods,
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

    results_file_name = args.results_file or tex_filter_str
    early_stopping_str = args.early_stopping or 'last'
    tab_code = hashlib.sha1(bytes(tex_filter_str + early_stopping_str, 'utf-8')).hexdigest()[:6]

    
    if len(df) == 1:
        tab_file = os.path.join(args.results_directory, results_file_name + '.tab') 
        tex_file = os.path.join(args.results_directory, results_file_name + '.tex')
        agg_tab_file = os.path.join(args.results_directory, results_file_name + '-agg.tab') 
        agg_tex_file = os.path.join(args.results_directory, results_file_name + '-agg.tex') 

    log.info('')
    log.info('')
    log.info('')
    
    pd.set_option('max_colwidth', 15)

    for s, d in df.items():

        # print('**********', d.reset_index().columns)  # 
        texify_test_results_df(d, s, tex_file, tab_file, tab_code=tab_code)

        
        # d.index = d.index.droplevel(('sigma_train', 'beta_sigma', 'features'))
        # d.index = d.index.droplevel(('sigma_train', 'sigma', 'features'))
        # d.index = d.index.droplevel(('sigma', 'features'))
        # d.index = d.index.droplevel(('features'))
        d.index = pd.MultiIndex.from_frame(d.index.to_frame().fillna('NaN'))
        
        if tex_file:
            with open(tex_file, 'a') as f:
                f.write('\def\joblist{')
                f.write(','.join(['{:06d}'.format(n['job']) for n in models_to_be_kept]))
                f.write('}\n')

        if args.remove_index is not None:
            removable_index = ['L', 'sigma_train', 'sigma', 'beta', 'gamma', 'forced_var']
            if 'auto' in args.remove_index:
                args.remove_index.remove('auto')
                removed_index = [i for i, l in enumerate(d.index.levels)
                                 if len(l) < 2 and l.name in removable_index]
            else:
                removed_index = []
            unremoved_index = []
            for i in args.remove_index:
                if i.replace('-', '_') in d.index.names:
                    removed_index.append(i.replace('-', '_'))
                else:
                    unremoved_index.append(i)
            if unremoved_index:
                logging.error('{} are not removed. Possible removable index: {}'.format(', '.join(unremoved_index),
                                                                                        ', '.join(d.index.names)))
            d = d.droplevel(removed_index)

        idx = list(d.index.names)
        if 'job' in d.index.names:
            wj = d.index.get_level_values('job').to_series().apply(str).apply(len).max()
            idx.remove('job')

        gb = d.groupby(level=idx)

        d_mean = gb.agg('mean')
        d_std = gb.agg('std')
        d_count = gb.agg('count')

        if 'job' in d.index.names:
            _s = '{{:{}d}}'.format(wj-1)
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
            texify_test_results_df(d_agg, s, agg_tex_file, agg_tab_file, tab_code=tab_code)

        col_show_levels = {_: 0 for _ in d.columns}
        col_show_levels.update({_: 2 for _ in d.columns if _[0] == 'measures'})
        col_show_levels.update({_: 1 for _ in d.columns if _[-1] in ['done', 'epoch']})

        drop_cols = [_ for _ in d.columns if col_show_levels[_] > args.show_measures]

        if drop_cols:
            d.drop(columns=drop_cols, inplace=True)
            d_mean.drop(columns=drop_cols, inplace=True)

        format_df_index(d, inplace=True)
        format_df_index(d_mean, inplace=True)
        
        d_str = d.to_string(na_rep='', float_format='{:.3g}'.format, sparsify=True)

        width = len(d_str.split('\n')[0])
        print(f'{s.upper():=^{width}}')

        if not args.only_average:
            print(d_str)

        if args.average:
            # d_mean.index = d_mean.index.format(formatter=_f)        
            m_str = d_mean.to_string(na_rep='', float_format='{:.3g}'.format).split('\n')
            width = len(m_str[0])
            first_row = '{:-^{w}}'.format('AVERAGE', w=width)
            header = d.columns.nlevels
            second_row = m_str[header-1]
            if not args.only_average:
                print()
                print(first_row)
                print()
            print(second_row)
            print('\n'.join(m_str[header+1:]))
        
        for a in archs[s]:
            arch_code = hashlib.sha1(bytes(a, 'utf-8')).hexdigest()[:6]
            print(arch_code,':\n', a)
        if print_sorting_keys:
            print('Possible sorting keys :', *d.index.names)

        for _ in range(0):
            print('=' * width)
