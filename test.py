from __future__ import print_function

from itertools import groupby
import numpy as np
import torch
from cvae import ClassificationVariationalNetwork as CVNet
import data.torch_load as torchdl
import os
import sys
import argparse

from utils.parameters import alphanum, list_of_alphanums, get_args, set_log
from utils.save_load import collect_networks


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
        log.debug(f'|_{len(l)} {w} of type {a}')
        betas, num = np.unique([n['beta'] for n in l], return_counts=True)

        beta_s = ' '.join([f'{beta:.3e} ({n})'
                           for (beta, n) in zip(betas, num)])
        log.debug(f'| |_ beta={beta_s}')

    log.info('Is trained')
    log.info('|Is tested')
    log.info('||')
    to_be_tested = []
    n_trained = 0
    n_tested = 0
    n_to_be_tested = 0
    testsets = set()
    betas = set()

    for n in sum(l_o_l_o_d_o_n, []):

        # log.debug('Cuda me: %s', torch.cuda.memory_allocated())
        net = n['net']
        is_trained = net.trained >= net.training['epochs']
        is_tested = False
        will_be_tested = False
        enough_epochs = net.trained >= epochs
        if is_trained or unfinished_training:
            to_be_tested.append(n)
            trained_set = net.training['set']
            n['set'] = trained_set
            betas.add(n['beta'])

            testsets.add(trained_set)
            testings_by_method = net.testing.get(trained_set,
                                        {None: {'epochs': 0, 'n':0}})
            enough_samples = True
            is_tested = True
            for m in testings_by_method:
                enough_samples = (enough_samples and
                                  testings_by_method[m]['n'] > min_test_sample_size)
                is_tested = is_tested and testings_by_method[m]['epochs'] == net.trained
                # log.debug('Tested at %s epochs (trained with %s) for %s',
                #           testings_by_method[m]['epochs'],
                #           net.trained,
                #           m)
            will_be_tested = enough_epochs and not is_tested
        log.info('%s%s%s %3d epochs for %s', 
                 '*' if is_trained else '|',
                 '*' if is_tested else '|',
                 '*' if will_be_tested else '|',
                 net.trained,
                 n['dir'])
        
        n_tested = n_tested + is_tested
        n_trained = n_trained + is_trained
        n_to_be_tested = n_to_be_tested + will_be_tested

    log.info('|||')
    log.info('|%s tested', n_tested)
    log.info('%s trained', n_trained)

    dict_of_sets = dict()
    for s in testsets:
        log.debug('Get %s dataset', s)
        _, testset = torchdl.get_dataset(s)
        dict_of_sets[s] = testset
        log.debug(testset)

    method = 'loss'

    for n in to_be_tested:
        
        trained_set = n['net'].training['set']
        log.info('Test %s with %s', n['dir'], trained_set)
        with torch.no_grad():
            n['net'].accuracy(dict_of_sets[trained_set],
                              print_result=True,
                              batch_size=batch_size, device=device,
                              method='all')
        n['net'].save(n['dir'])
        n['acc'] = n['net'].testing[trained_set][method]['accuracy']
        n['arch'] = n['net'].print_architecture(excludes=['latent_dim'])
        n['K'] = n['net'].latent_dim
        n['L'] = n['net'].latent_sampling
        
    sorting_key = lambda n: (n['set'],
                             n['net'].depth,
                             n['net'].width,
                             n['arch'],
                             n['K'],
                             n['L'],
                             n['beta'],
                             n['acc'])
    
    to_be_tested.sort(key=sorting_key)

    grouped_by_set = groupby(to_be_tested, key=lambda n: n['set'])

    for s, group in grouped_by_set:

        string = f'Networks trained for {s}'
        print(f'\n{string:_^120}')

        grouped_by_arch = groupby(group, key=lambda n: n['arch'])

        for a, arch_group in grouped_by_arch:

            print(f'{a:=^120}')

            header = '   K   L '
            print(f'{header:<15}', end='')
            for beta in sorted(betas):
                print(f'{beta:^12.2e}', end='')
            print('\n') #, 120*'_')

            grouped_by_k_l = groupby(arch_group,
                                     key = lambda n: (n['K'], n['L']))
            
            for (K, L), k_l_group in grouped_by_k_l: 

                # lg = list(k_l_group)
                grouped_by_beta = groupby(k_l_group,
                                          key=lambda n: n['beta'])
                
                header = f'{K:4} {L:4} '
                print(f'{header:<15}', end='')

                d_beta_acc = dict()
                acc_max = 0
                for beta, beta_group in grouped_by_beta:
                    acc_max_beta = max(n['acc'] for n in beta_group)
                    d_beta_acc[beta] = acc_max_beta
                    if acc_max_beta > acc_max:
                        acc_max = acc_max_beta
                        d_beta_acc['max'] = beta

                for beta in sorted(betas):

                    if beta in d_beta_acc:
                        if d_beta_acc['max'] == beta:
                            fo, fc = '\033[1m', '\033[0m'
                            # fo, fc = '*', '*'
                        else:
                            fo, fc = '', ''
                        print(f'  {fo}{d_beta_acc[beta]:7.1%}{fc}   ', end='')
                    else:
                        print(' ' * 12, end='')
                print()
            print(f'{"":_^120}')

