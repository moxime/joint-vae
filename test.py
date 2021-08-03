from __future__ import print_function

import torch
from cvae import ClassificationVariationalNetwork as CVNet
import utils.torch_load as torchdl
import os
import sys
import hashlib
import logging

import pandas as pd

from utils.parameters import get_args, set_log, gethostname
from utils.save_load import collect_networks, test_results_df, LossRecorder
from utils.tables import export_losses, tex_architecture, texify_test_results, texify_test_results_df


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
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        log.info(f'Used device: {device} on {hostname}')
    else:
        device = torch.device('cpu')
        log.info(f'Used device: {device}')
        log.debug('CPU asked by user')

    batch_size = args.batch_size
    job_dir = args.job_dir
    load_dir = args.load_dir
    dry_run = args.dry_run
    flash = args.flash

    test_sample_size = args.test_sample_size
    ood_sample_size = args.ood
    min_test_sample_size = args.min_test_sample_size
    
    filters = args.filters
    _comma = ','
    filter_str = '--'.join(f'{d}:{_comma.join([str(_) for _ in f])}' for d, f in filters.items())
    logging.debug('Filters: %s', filter_str)
    
    latex_formatting = args.latex

    sort = args.sort
    
    for k, v in vars(args).items():
        logging.debug('%s: %s', k, str(v))
    
    search_dir = load_dir if load_dir else job_dir

    logging.debug('Collecting networks')
    list_of_networks = collect_networks(search_dir,
                                        load_net=False,
                                        load_state=False)
        
    total = sum(map(len, list_of_networks))
    log.debug(f'{total} networks in {len(list_of_networks)} lists collected:')

    for (i, l) in enumerate(list_of_networks):
        a = l[0]['arch']  # ['net'].print_architecture(sampling=True)
        w = 'networks' if len(l) > 1 else 'network '
        log.debug('|')
        log.debug(f'|_{len(l)} {w} of type {a}')

    log.info('Is keeped')
    log.info('| Results')
    log.info('| are available')
    log.info('| | can be extracted from recorders (x:partially)')
    log.info('| | | have to be computed')
    log.info('| | | job #')
    log.info('| | |        # trained epochs')
    # log.info('|||')
    enough_trained = []
    n_trained = 0
    n_tested = 0
    n_to_be_tested = 0
    n_ood_computed = 0
    n_ood_to_be_computed = 0
    testsets = set()
    archs = {}

    networks_to_be_studied = []
    for n in sum(list_of_networks, []):
        filter_results = sum([[f.filter(n[d]) for f in filters[d]] for d in filters], [])
        to_be_studied = all(filter_results)

        if to_be_studied:
            networks_to_be_studied.append(n)
            # for d in filters:
            #   print(d, n[d])
            #   for f in filters[d]:
            #       print(f, f.filter(n[d]))

    for n in networks_to_be_studied:

        net = n.get('net', None)

        is_tested = test_accuracy_if(jvae=net,
                                     directory=n['dir'],
                                     dry_run=True,
                                     min_test_sample_size=min_test_sample_size,
                                     batch_size=batch_size,
                                     unfinished=unfinished_training)
        
        is_enough_trained = is_tested is not None
        will_be_tested = is_enough_trained and not is_tested

        ood_are_tested = test_ood_if(jvae=net,
                                     directory=n['dir'],
                                     dry_run=True,
                                     min_test_sample_size=min_test_sample_size,
                                     batch_size=batch_size,
                                     unfinished=unfinished_training,)

        if is_enough_trained:
            ood_will_be_computed = sum([not v for v in ood_are_tested.values()])
        else:
            ood_will_be_computed = 0

        is_derailed = False

        if is_enough_trained:
            d = n['dir']
            derailed = os.path.join(d, 'derailed')
            if args.cautious:
                log.warning('Cautious verifications to be implemented')
                try:
                    pass
                    # log.debug('Evaluation of one sample...')
                    # net.evaluate(torch.randn(1, *net.input_shape))
                    # log.debug('...done')
                except ValueError:
                    open(derailed, 'a').close()
                    log.debug(f'Net in {d} has been marked as derailed')

            is_derailed = os.path.exists(derailed)
            if not is_derailed:
                if not n['is_resumed']:
                    enough_trained.append(n)
                    testsets.add(n['set'])

                    if n['set'] in archs:
                        archs[n['set']].add(n['arch'])
                    else:
                        archs[n['set']] = {n['arch']} 
                else:
                    is_enough_trained = False
                    will_be_tested = False
                    ood_will_be_computed = 0

        if is_derailed:
            train_mark = '+'
            ood_mark = '+'
            log.info('++ Derailed net in %s',
                     n['dir'])
        else:
            if not is_enough_trained:
                train_mark = '|'
                ood_mark = '|'
            else:
                train_mark = '*' if is_tested else '.'
                ood_mark = '*' if not ood_will_be_computed else ood_will_be_computed

            _dir = n['dir'][:130]
            _dir2 = n['dir'][130:]
            log.info('%s%s %3d %s', 
                     train_mark,
                     ood_mark,
                     n['done'],
                     _dir)
            if _dir2:
                log.info('||' +
                         '     ' +
                         '_' * (130 - len(_dir2)) +
                         _dir2)
        
        n_trained += is_enough_trained
        n_tested += (is_tested is True)
        n_to_be_tested += will_be_tested
        n_ood_computed += (ood_are_tested is True)
        n_ood_to_be_computed += ood_will_be_computed

    log.info('||')
    log.info('|%s ood to be computed', n_ood_to_be_computed)
    log.info('%s tested nets (%s tests to be done)',
             n_trained,
             n_to_be_tested)

    if not dry_run:

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

    for n in enough_trained:
        tex_architecture(n)
        export_losses(n, which='all')
        texify_test_results(n)
        
    first_method = args.expand < 2
    show_best = args.expand < 1 

    tpr = [t/100 for t in args.tpr]
        
    df = test_results_df(enough_trained, best_net=show_best,
                         first_method=first_method,
                         ood=True,
                         tnr=args.tnr,
                         tpr=tpr,
                         sorting_keys=sort)

    _sep = ('_', ':', ',', ' ')
    tex_filter_str = filter_str
    for _ in _sep:
        tex_filter_str = tex_filter_str.replace(_, '-')
    
    tab_file = os.path.join('results', tex_filter_str + '.tab') if len(df) == 1 else None
    tex_file = os.path.join('results', tex_filter_str + '.tex') if len(df) == 1 else None

    log.info('')
    log.info('')
    log.info('')
    
    pd.set_option('max_colwidth', 15)

    sep = ['\n' * 2 + '=' * 180 for _ in df]
    if sep:
        sep[0] = ''

    sep_ = iter(sep)
    for s, d in df.items():

        texify_test_results_df(d, s, tex_file, tab_file)

        # d.index = d.index.droplevel(('sigma_train', 'beta_sigma', 'features'))
        # d.index = d.index.droplevel(('sigma_train', 'sigma', 'features'))
        d.index = d.index.droplevel(('sigma', 'features'))
        
        print(next(sep_))
        print(f'Results for {s}')
        print(d.to_string(na_rep='', float_format='{:.3g}'.format, sparsify=True))

        if tex_file:
            with open(tex_file, 'a') as f:
                f.write('\def\joblist{')
                f.write(','.join(['{:06d}'.format(n['job']) for n in enough_trained]))
                f.write('}\n')

        for a in archs[s]:
            arch_code = hashlib.sha1(bytes(a, 'utf-8')).hexdigest()[:6]
            print(arch_code,':\n', a)
            
    # print(df.to_string())
    
    # if latex_formatting:
    """
    with open('test.tex', 'w') as f:
    f.write(d.to_latex(na_rep='',
                            float_format='%.2f',
                            decimal=',',
                            formatters=formats))

    """
