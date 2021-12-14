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
from utils.save_load import collect_networks, test_results_df, LossRecorder, make_dict_from_model, available_results
from utils.tables import export_losses, tex_architecture, texify_test_results, texify_test_results_df
from utils.testing import early_stopping


if __name__ == '__main__':

    hostname = gethostname()
    args = get_args('test')
    
    debug = args.debug
    verbose = args.verbose

    log = set_log(verbose, debug, 'jobs/log', name='test', job_number=args.job_id)
    logging.debug('$ ' + ' '.join(sys.argv))
    if not args.force_cpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f'Used device: {device} on {hostname}')
    else:
        device = torch.device('cpu')
        log.info(f'Used device: {device}')
        log.debug('CPU asked by user')

    batch_size = args.batch_size
    job_dir = args.job_dir
    load_dir = args.load_dir
    dry_run = args.compute
    flash = args.flash

    test_sample_size = args.test_sample_size
    ood_sample_size = args.ood
    min_test_sample_size = args.min_test_sample_size
    
    filters = args.filters
    _comma = ','
    filter_str = '--'.join(f'{d}:{_comma.join([str(_) for _ in f])}' for d, f in filters.items())

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

    logging.debug('Collecting networks')
    list_of_networks = collect_networks(search_dir,
                                        tpr_for_max=args.tpr[0] / 100,
                                        load_net=False,
                                        load_state=False)
        
    total = sum(map(len, list_of_networks))
    log.debug(f'{total} networks in {len(list_of_networks)} lists collected:')

    for (i, l) in enumerate(list_of_networks):
        a = l[0]['arch']  # ['net'].print_architecture(sampling=True)
        w = 'networks' if len(l) > 1 else 'network '
        log.debug('|')
        log.debug(f'|_{len(l)} {w} of type {a}')

    log.debug('{} models found'.format(sum([len(l) for l in list_of_networks])))
    log.info('Is kept')
    log.info('| Results')
    log.info('| are fully available')
    log.info('| | can (*: all, x: partially) be extracted from recorders')
    log.info('| | | have to be computed')
    log.info('| | | | job #')
    log.info('| | | | |     # trained epochs')
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
    else:
        where = ('json', 'recorders')
    for n in sum(list_of_networks, []):
        filter_results = sum([[f.filter(n[d]) for f in filters[d]] for d in filters], [])
        to_be_kept = all(filter_results)
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
        if to_be_kept:
            if n['set'] in archs:
                archs[n['set']].add(n['arch'])
            else:
                archs[n['set']] = {n['arch']}

            if early_stopping_method:
                _k = 'early-' + '-'.join(early_stopping_method)
                if _k in n['net'].training_parameters:
                    wanted_epoch = n['net'].training_parameters[_k]
                else:
                    wanted_epoch = early_stopping(n, strategy=early_stopping_method[0],
                                                  which = early_stopping_method[1])
            else:
                wanted_epoch = 'last'

            available = available_results(n, misclass_methods=[], wanted_epoch=wanted_epoch,
                                          where=where, epoch_tolerance=5)

            total_available_by_epoch = {_: available[_]['all_sets']['anywhere'] for _ in available}
            result_epoch = max(total_available_by_epoch, key=total_available_by_epoch.get)

            a_ = available[result_epoch]['all_sets']
            print(a_)
            if total_available_by_epoch[result_epoch]:
                models_to_be_kept.append(dict(model=n, epoch=result_epoch,
                                              plan=a_))

            a_everywhere = available_results(n, misclass_methods=[], wanted_epoch=result_epoch,
                                             epoch_tolerance=0)[result_epoch]['all_sets']

            is_a = a_['json']
            is_r = a_['recorders']
            is_c = a_['compute']
            n_epochs_to_be_computed += is_c

            if is_a == a_everywhere['anywhere']:
                _a = '*'
            elif is_a:
                _a = 'x'
            else:
                _a = '|'
            if a_everywhere['compute']:
                _r = 'x' if is_r else 'o'
                _c = '*' if is_c else 'o'
            else:
                _r = '*' if is_r else '|'
                _c = '|'

            logging.info('* {} {} {} {:6d} {:8} {:5} {:80.80}'. format(_a, _r,
                                                             _c, n['job'], n['set'], n['type'], n['arch']))

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
    
    for m_ in models_to_be_computed['recorders']:
        m = m_['model']
        epoch = m_['epoch']
        plan = m['plan']

        if plan['recorders'] or plan['compute']:
            print('Computing rates of job {} of type {}'.format(m['job'], m['type'])) 
            model = CVNet.load(m['dir'], load_state=plan['compute'])
            model.ood_detection_rates(epoch=epoch,
                                      from_where=where,
                                      sample_dirs=[os.path.join(m['dir'], 'samples', '{:4d}'.format(epoch))],
                                      print_result='OFR' if not plan['compute'] else 'OFM')

            model.accuracy(epoch=epoch,
                           from_where=where,
                           sample_dirs=[os.path.join(m['dir'], 'samples', '{:4d}'.format(epoch))],
                           print_result='TFR' if not plan['compute'] else 'TFM')

            if not args.dry_run:
                model.save(m['dir'])
            m.update(make_dict_from_model(model, m['dir'], wanted_epoch=epoch))                

    for n in models_to_be_kept:
        tex_architecture(n)
        export_losses(n, which='all')
        texify_test_results(n)
        
    first_method = args.expand < 2
    nets_to_show = 'all' if args.expand >= 1 else 'mean' 

    tpr = [t/100 for t in args.tpr]

    print_sorting_keys = False
    if sort and 'print' in sort:
        sort.remove('print')
        print_sorting_keys = True
    df = test_results_df(models_to_be_kept, nets_to_show=nets_to_show,
                         first_method=first_method,
                         ood=oodsets,
                         show_measures=args.show_measures,
                         tnr=args.tnr,
                         tpr=tpr,
                         sorting_keys=sort)

    _sep = ('_', ':', ',', ' ')
    tex_filter_str = filter_str
    for _ in _sep:
        tex_filter_str = tex_filter_str.replace(_, '-')

    tab_file, tex_file, agg_tab_file, agg_tex_file = None, None, None, None

    if len(df) == 1:
        tab_file = os.path.join('results', tex_filter_str + '.tab') 
        tex_file = os.path.join('results', tex_filter_str + '.tex')
        agg_tab_file = os.path.join('results', tex_filter_str + '-agg.tab') 
        agg_tex_file = os.path.join('results', tex_filter_str + '-agg.tex') 

    log.info('')
    log.info('')
    log.info('')
    
    pd.set_option('max_colwidth', 15)

    for s, d in df.items():

        # print('**********', d.reset_index().columns)  # 
        texify_test_results_df(d, s, tex_file, tab_file)

        
        # d.index = d.index.droplevel(('sigma_train', 'beta_sigma', 'features'))
        # d.index = d.index.droplevel(('sigma_train', 'sigma', 'features'))
        # d.index = d.index.droplevel(('sigma', 'features'))
        d.index = d.index.droplevel(('features'))
        d.index = pd.MultiIndex.from_frame(d.index.to_frame().fillna('NaN'))
        
        if tex_file:
            with open(tex_file, 'a') as f:
                f.write('\def\joblist{')
                f.write(','.join(['{:06d}'.format(n['job']) for n in models_to_be_kept]))
                f.write('}\n')

        if args.remove_index is not None:
            removable_index = ['sigma_train', 'sigma', 'beta', 'gamma']
            removed_index = [i for i, l in enumerate(d.index.levels) if len(l) < 2 and l.name in removable_index]
            removed_index += args.remove_index
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
            _s = '{{:{}d}}'.format(wj)
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
            texify_test_results_df(d_agg, s, agg_tex_file, agg_tab_file)

        if not args.show_measures:
            d.drop(columns=[_ for _ in d.columns if _[0] == 'measures'], inplace=True)
            d_mean.drop(columns=[_ for _ in d_mean.columns if _[0] == 'measures'], inplace=True)

        d_str = d.to_string(na_rep='', float_format='{:.3g}'.format, sparsify=True)

        width = len(d_str.split('\n')[0])
        print(f'{s.upper():=^{width}}')
        print(d_str)

        # d_mean.index = d_mean.index.format(formatter=_f)        
        m_str = d_mean.to_string(na_rep='', float_format='{:.3g}'.format).split('\n')
        width = len(m_str[0])
        first_row = '{:=^{w}}'.format('AVERAGE', w=width)
        header = d.columns.nlevels
        second_row = m_str[header-1]

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

        for _ in range(2):
            print('=' * width)
