from __future__ import print_function

from itertools import groupby
import numpy as np
import torch
from cvae import ClassificationVariationalNetwork as CVNet
import data.torch_load as torchdl
import os
import sys
import argparse
import logging

import pandas as pd

from utils.parameters import alphanum, list_of_alphanums, get_args, set_log
from utils.save_load import collect_networks, data_frame_results


def test_net_if(jvae=None,
                directory=None,
                testset=None,
                test_sample_size='all',
                unfinished=False,
                dry_run=False,
                min_epochs=0,
                min_test_sample_size=1000,
                **kw,
                      ):

    assert jvae or directory

    if not jvae:
        try:
            jvae = CVNet.load(directory)
        except FileNotFoundError:
            logging.warning(f'Has been asked to load lent in {directory}'
                            'none found')
            
    is_trained = jvae.trained >= jvae.training['epochs']
    enough_trained_epochs = jvae.trained >= min_epochs

    min_tested_epochs = min(d['epochs'] for d in jvae.testing.values())
    min_tested_sample_size = min(d['n'] for d in jvae.testing.values())
    enough_samples = min_tested_sample_size >= min_test_sample_size
    enough_epochs = min_tested_epochs >= min_epochs
    
    if not is_trained and not unfinished:
        logging.debug(f'Net in {directory} not trained, will not be tested')
        return None

    if not enough_trained_epochs:
        logging.debug(f'Net in {directory} not trained enough, will not be tested')
        return None
    
    has_been_tested = enough_epochs and enough_samples

    if dry_run:
        return has_been_tested

    if not has_been_tested:

        if not testset:
            _, testset = torchdl.get_dataset(jvae.training['set'])
        if test_sample_size == 'all':
            num_batch = 'all'
        else:
            num_batch = test_sample_size // batch_sizez
        with torch.no_grad():
            jvae.accuracy(testset,
                          **kw)

    return jvae.testing
        
    
if __name__ == '__main__':

    list_of_args = get_args('test')

    args = list_of_args[0]
    
    debug = args.debug
    verbose = args.verbose

    log = set_log(verbose, debug, name='test')
    log.debug('$ ' + ' '.join(sys.argv))
    if not args.force_cpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        log.info(f'Used device: {device}')
    else:
        device = torch.device('cpu')
        log.info(f'Used device: {device}')
        log.debug(f'CPU asked by user')

    batch_size = args.batch_size
    job_dir = args.job_dir
    load_dir = args.load_dir
    dry_run = args.dry_run
    epochs = args.epochs
    min_test_sample_size = args.min_test_sample_size
    unfinished_training = args.unfinished
    
    latex_formatting = args.latex
    
    for k in vars(args).items():
        log.debug('%s: %s', *k)
    
    search_dir = load_dir if load_dir else job_dir

    l_o_l_o_d_o_n = []
    collect_networks(search_dir, l_o_l_o_d_o_n) #, like=dummy_jvae)
    total = sum(map(len, l_o_l_o_d_o_n))
    log.debug(f'{total} networks in {len(l_o_l_o_d_o_n)} lists collected:')

    for (i, l) in enumerate(l_o_l_o_d_o_n):
        a = l[0]['net'].print_architecture(sampling=True)
        w = 'networks' if len(l) > 1 else 'network '
        log.debug('|')
        log.debug(f'|_{len(l)} {w} of type {a}')
        betas, num = np.unique([n['beta'] for n in l], return_counts=True)

        beta_s = ' '.join([f'{beta:.3e} ({n})'
                           for (beta, n) in zip(betas, num)])
        log.debug(f'| |_ beta={beta_s}')

    log.info('Is trained')
    log.info('|Is tested')
    log.info('||Will be tested')
    log.info('|||')
    enough_trained = []
    n_trained = 0
    n_tested = 0
    n_to_be_tested = 0
    testsets = set()
    betas = set()
    archs = set()
    
    for n in sum(l_o_l_o_d_o_n, []):

        # log.debug('Cuda me: %s', torch.cuda.memory_allocated())
        net = n['net']
        is_tested = test_net_if(jvae=net,
                                dry_run=True,
                                min_test_sample_size=min_test_sample_size,
                                unfinished=unfinished_training,
                                min_epochs=epochs)
        
        is_enough_trained = is_tested is not None
        will_be_tested = is_enough_trained and not is_tested

        if is_enough_trained:
            try:
                net.evaluate(torch.randn(1, *net.input_shape))
                enough_trained.append(n)
                betas.add(n['beta'])
                testsets.add(n['set'])
                archs.add(n['arch'])
            except ValueError:
                d = n['dir']
                is_enough_trained = False
                will_be_tested = False
                log.warning(f'Better get rid of net in {d}')

        log.info('%s%s%s %3d epochs for %s', 
                 '*' if is_enough_trained else '|',
                 '*' if is_tested else '|',
                 '*' if will_be_tested else '|',
                 net.trained,
                 n['dir'])
        
        n_tested = n_tested + (is_tested is True)
        n_trained = n_trained + is_enough_trained
        n_to_be_tested = n_to_be_tested + will_be_tested

    log.info('|||')
    log.info('||%s to be tested', n_to_be_tested)
    log.info('|%s tested', n_tested)
    log.info('%s trained (enough)', n_trained)

    dict_of_sets = dict()
    for s in testsets:
        log.debug('Get %s dataset', s)
        _, testset = torchdl.get_dataset(s)
        dict_of_sets[s] = testset
        log.debug(testset)

    for n in enough_trained:
        
        trained_set = n['net'].training['set']
        log.info('Test %s with %s', n['dir'], trained_set)

        test_net_if(n['net'],
                    testset=dict_of_sets[trained_set],
                    unfinished=unfinished_training,
                    min_epochs=epochs,
                    min_test_sample_size=min_test_sample_size,
                    batch_size=batch_size,
                    print_result=True,
                    device=device,
                    method='all')

        n['net'].save(n['dir'])

    df = data_frame_results(enough_trained)

    formats = []

    def finite(u, f):
        if np.isnan(u):
            return ''
        if np.isinf(u):
            return 'inf'
        return f.format(u)

    def f_pc(u):
        return finite(100 * u, '{:.1f}')

    def f_db(u):
        return finite(u, '{:.1f}')
    
    for (b, m) in df.columns:

        formats.append(f_db if m == 'snr' else f_pc)
    
    if verbose:
        print('\n' * 2)
    pd.set_option('max_colwidth', 15)
    print(df.to_string(na_rep='', decimal=',', formatters=formats))

    for a in archs:
        print(hex(hash(a))[2:10],':\n', a)
    #print(df.to_string())
