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
from utils.save_load import collect_networks, test_results_df, LossRecorder, make_dict_from_model
from utils.tables import export_losses, tex_architecture, texify_test_results, texify_test_results_df
from utils.testing import worth_computing, early_stopping


def test_accuracy_if(jvae=None,
                     directory=None,
                     testset=None,
                     test_sample_size='all',
                     batch_size=100,
                     unfinished=False,
                     dry_run=False,
                     min_test_sample_size=1000,
                     tolerance=10,
                     dict_of_sets={},
                     sample_dirs=[],
                     recorder=None,
                     **kw,):

    assert jvae or directory

    num_batch = test_sample_size
    if type(test_sample_size) is int:
        num_batch = test_sample_size // batch_size
        min_test_sample_size = min(test_sample_size, min_test_sample_size)
    
    if not jvae:
        try:
            jvae = CVNet.load(directory, load_state=not dry_run, load_net=not dry_run)
        except FileNotFoundError:
            logging.warning(f'Has been asked to load net in {directory}'
                            'none found')

    # deleting old testing methods
    jvae.testing = {m: jvae.testing[m] for m in jvae.predict_methods if m in jvae.testing}
        
    is_trained = jvae.trained >= jvae.training_parameters['epochs']

    if not is_trained and not unfinished:
        return None

    enough_samples = True
    tested_at_last_epoch = True

    for m in jvae.predict_methods:
        t = jvae.testing.get(m, {'n': 0, 'epochs': 0})
        enough_samples = enough_samples and t['n'] >= min_test_sample_size 
        tested_at_last_epoch = tested_at_last_epoch and t['epochs'] >= jvae.trained - tolerance
    
    has_been_tested = tested_at_last_epoch and enough_samples

    if dry_run:
        return has_been_tested

    if not has_been_tested:

        # print('*** test.py:74', jvae.training_parameters['set'],
        # jvae.training_parameters['transformer'])

        if not testset:
            t_p = jvae.training_parameters 
            _, testset = torchdl.get_dataset_from_dict(dict_of_sets,
                                                       t_p['set'],
                                                       transformer=t_p['transformer'])

        with torch.no_grad():
            jvae.accuracy(testset,
                          batch_size=batch_size,
                          num_batch=num_batch,
                          recorder=recorder,
                          sample_dirs=sample_dirs,
                          print_result='TEST',
                          **kw)

    return jvae.testing


def test_ood_if(jvae=None,
                directory=None,
                testset=None,
                oodsets=[],
                test_sample_size='all',
                batch_size=100,
                unfinished=False,
                dry_run=False,
                tolerance=10,
                min_test_sample_size=1000,
                dict_of_sets={},
                recorders=None,
                sample_dirs=[],
                **kw, ):

    assert jvae or directory

    num_batch = test_sample_size 
                             
    if type(test_sample_size) is int:
        num_batch = test_sample_size // batch_size
        min_test_sample_size = min(test_sample_size, min_test_sample_size)

    if not jvae:
        try:
            jvae = CVNet.load(directory, load_state=not dry_run, load_net=not dry_run)
        except FileNotFoundError:
            logging.warning(f'Has been asked to load net in {directory}'
                            'none found')
            return {}

    desc = 'in ' + directory if directory else jvae.print_architecture()

    if not jvae.ood_methods:
        logging.debug(f'Net {desc} has no ood methods')
        return {}
    
    assert jvae.training_parameters['set']

    is_trained = jvae.trained >= jvae.training_parameters['epochs']

    if not is_trained and not unfinished:
        logging.debug(f'Net {desc} training not ended, will not be tested')
        return None

    transformer = jvae.training_parameters['transformer']

    if testset:
        testset_name = testset.name
        dict_of_sets[testset_name] = {transformer: (None, testset)}
    else:
        testset_name = jvae.training_parameters['set']

    if oodsets:
        oodset_names = [o.name for o in oodsets]
        dict_of_sets.update({o.name: {transformer: (None, o)} for o in oodsets})

    else:
        oodset_names = torchdl.get_same_size_by_name(testset_name)
                             
    enough_tested_samples = {o: True for o in oodset_names}
    tested_at_last_epoch = {o: True for o in oodset_names}
    
    zero = {'epochs': 0, 'n': 0}
    zeros = {m: zero for m in jvae.ood_methods}
    oodsets_to_be_tested = []

    has_been_tested = {}
    
    for o in oodset_names:

        r = jvae.ood_results.get(o, {})
        for m in jvae.ood_methods:
            n = r.get(m, zero)['n']
            enough_tested_samples[o] = enough_tested_samples[o] and n >= min_test_sample_size
            ep = r.get(m, zero)['epochs']
            tested_at_last_epoch[o] = (tested_at_last_epoch[o]
                                       and ep >= jvae.trained - tolerance)
    
        has_been_tested[o] = enough_tested_samples[o] and tested_at_last_epoch[o]
        
        _w = '' if has_been_tested[o] else 'not ' 
        logging.debug(f'ood rate has {_w}been computed with enough samples for {o}')

    if not dry_run:
        # print('**** test.py:175', testset_name)

        for o in [n for n in has_been_tested if not has_been_tested[o]]:
            # print('**** test.py:180', n)
            _, oodset = torchdl.get_dataset_from_dict(dict_of_sets, o,
                                                      transformer=transformer)
            oodsets_to_be_tested.append(oodset)

        if oodsets_to_be_tested:
            _, testset = torchdl.get_dataset_from_dict(dict_of_sets,
                                                       testset_name,
                                                       transformer=transformer)

            _o = ' - '.join([o.name for o in oodsets_to_be_tested])
            logging.debug(f'OOD sets that will be tested: {_o}')
            jvae.ood_detection_rates(oodsets_to_be_tested, testset,
                                     batch_size=batch_size,
                                     num_batch=num_batch,
                                     print_result='*',
                                     recorders=recorders,
                                     sample_dirs=sample_dirs,
                                     **kw)

    else:
        return has_been_tested

    return jvae.ood_results
        
    
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

    early_stopping_method = tuple(args.early_stopping.split('-'))
    
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
    models_to_be_computed = {k: [] for k in ('json', 'recorders', 'compute')}
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
            to_be_computed = worth_computing(n, from_which='all', misclass_methods=[], wanted_epoch=wanted_epoch)
            
            models_to_be_kept.append((n, wanted_epoch))
            
            for k in to_be_computed:
                if to_be_computed[k]:
                    models_to_be_computed[k].append(dict(model=n, epoch=wanted_epoch))
            is_a = to_be_computed['json']
            is_r = to_be_computed['recorders']
            is_c = to_be_computed['compute']
            n_epochs_to_be_computed += is_c

            _a = '*' if is_a else '|'
            if is_r:
                _r = 'x' if is_c else '*'
            else:
                _r = '|'
            _c = is_c if is_c else '|'

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
    
    models_to_be_kept.sort(key=lambda d: d['model'].get('job',0))
    models_to_be_kept = models_to_be_kept[-args.last:]

    for s in archs:
        archs[s] = {n['model']['arch'] for n in models_to_be_kept if n['model']['set'] == s}
    
    if args.compute:
        for m in models_to_be_computed['recorders']:
            print('Computing rates of job {} of type {}'.format(m['model']['job'], m['model']['type'])) 
            model = CVNet.load(m['model']['dir'], load_state=False)
            model.accuracy(wygiwyu=True, print_result='TFR')
            model.ood_detection_rates(wygiwyu=True, print_result='OFR')
            if not args.dry_run:
                model.save(m['dir'])
            m.update(make_dict_from_model(model, m['dir']))                

    if args.compute == 'hard':

        batch_sizes = {}
        dict_of_sets = {}

        for n in enough_trained:
            try:
                n['net'] = CVNet.load(n['dir'])
            except RuntimeError as e:
                directory = n['dir']
                logging.warning(f'Load error in {directory} see log file')
                logging.debug(f'Load error: {e}')
                continue
    
            trained_set = n['net'].training_parameters['set']
            transformer = n['net'].training_parameters['transformer']
            n['net'].to(device)

            # set_of_trained_sets.add(trained_set)
            
            arch = n['net'].print_architecture(sampling=True)

            max_batch_size = batch_size
            batch_size = min(batch_sizes.get(arch, 0),
                             max_batch_size)
            if not batch_size:

                n['net'].compute_max_batch_size(max_batch_size,
                                                which='test')
                batch_size = n['net'].max_batch_sizes['test']
                
                batch_sizes[arch] = batch_size
            
            log.info('Test %s with %s and batch size %s',
                     n['dir'], trained_set, batch_size)

            _, testset = torchdl.get_dataset_from_dict(dict_of_sets,
                                                       trained_set,
                                                       transformer)

            sample_dir = os.path.join(n['dir'], 'samples')
            if not os.path.exists(sample_dir):
                os.mkdir(sample_dir)

            sample_dirs = [os.path.join(sample_dir, d)
                           for d in ('last', '{:04d}'.format(n['done']))]

            for d in sample_dirs:
                if not os.path.exists(d):
                    os.mkdir(d)
                
            samples = {int(_): _ for _ in os.listdir(sample_dir) if _.isnumeric()}
            samples[-1 - train_tolerance] = None
            
            last_sample = max(samples)
            
            # TO BE CONTINUED

            recorders = {}
            all_sets = [testset.name] + testset.same_size
            recorders = {s: LossRecorder(batch_size) for s in all_sets}
            
            if last_sample >= n['net'].trained - train_tolerance:
                for s in all_sets:
                    try:
                        f = os.path.join(sample_dir,
                                         samples[last_sample],
                                         f'record-{s}.pth')
                        recorders[s] = LossRecorder.load(f)
                        _l = len(recorders[s]) * recorders[s].batch_size
                        log.debug(f'Recorder loaded for {s} of length {_l}')
                    except FileNotFoundError:
                        log.debug(f'Recorder not found for {s}')

            test_accuracy_if(jvae=n['net'],
                             testset=testset,
                             unfinished=unfinished_training,
                             min_test_sample_size=min_test_sample_size,
                             batch_size=batch_size,
                             dict_of_sets=dict_of_sets,
                             recorder=recorders[testset.name],
                             tolerance=train_tolerance,
                             sample_dirs=sample_dirs,
                             method='all')

            if ood_sample_size:
                oodsets = [torchdl.get_dataset_from_dict(dict_of_sets,
                                                         n, transformer)[1]
                           for n in testset.same_size]
                test_ood_if(jvae=n['net'],
                            testset=testset,
                            oodsets=oodsets,
                            unfinished=unfinished_training,
                            test_sample_size=ood_sample_size,
                            min_test_sample_size=min_test_sample_size,
                            batch_size=batch_size,
                            dict_of_sets=dict_of_sets,
                            tolerance=train_tolerance,
                            recorders=recorders,
                            sample_dirs=sample_dirs,
                            method='all')
            
            n['net'].save(n['dir'])

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
