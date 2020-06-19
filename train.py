from __future__ import print_function

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
    
    list_of_args = get_args()
    
    common_args = list_of_args[0]
    
    debug = common_args.debug
    verbose = common_args.verbose
    repeat = common_args.repeat
    log = set_log(verbose, debug)

    log.debug('$ ' + ' '.join(sys.argv))

    for k in common_args.__dict__.items():
        log.debug('%s: %s', *k)

    job_dir = common_args.job_dir

    if not common_args.force_cpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        log.info(f'Used device: {device}')
    else:
        device = torch.device('cpu')
        log.info(f'Used device: {device}')
        log.debug(f'CPU asked by user')

    batch_size = common_args.batch_size
    test_sample_size = common_args.test_sample_size
    dry_run = common_args.dry_run    
    load_dir = common_args.load_dir    
    find_and_finish = (common_args.finish or repeat > 1) and not load_dir

    for args in list_of_args:
        args.already_trained = []

    if load_dir:

        repeat = 1
        list_of_args = []
        nets_in_dir = []

        collect_networks(load_dir,
                         nets_in_dir)

        for n in sum(nets_in_dir, []):
            a_ = argparse.Namespace(**vars(common_args))
            a_ = argparse.Namespace()
            a_.epochs = common_args.epochs
            a_.batch_size = common_args.batch_size
            to_be_finished = {'dir': n['dir'], 'done': n['done']}

            if to_be_finished['done'] < a_.epochs:
                log.debug('Will finish training of net in %s (%s epochs done',
                          n['dir'], n['done'])
                a_.dataset = n['set']
                a_.transformer = n['net'].training.get('transformer', 'default')
                a_.load_dir = n['dir']
                a_.already_trained = [to_be_finished]
                list_of_args.append(a_)

    elif find_and_finish:

        nets_in_dir = []
        collect_networks(job_dir,
                         nets_in_dir)

        num_of_nets = sum(map(len, nets_in_dir))
        num_of_archs = len(nets_in_dir)
        log.debug(f'{num_of_nets} networks with {num_of_archs}'
                  'different architectures:')

        for (i, nets_of_arch) in enumerate(nets_in_dir):
            arch = nets_of_arch[0]['net'].print_architecture(sampling=True, short=True)
            w = 'networks' if len(nets_of_arch) > 1 else 'network '
            log.debug(f'|_{len(l)} {w} of type {arch}')
            betas, num = np.unique([n['beta'] for n in l], return_counts=True)
            beta_s = ' '.join([f'{beta:.3e} ({n})'
                               for (beta, n) in zip(betas, num)])
            log.debug(f'| |_ beta={beta_s}')

        for a in list_of_args:

            input_shape, num_labels = torchdl.get_shape_by_name(a.dataset)
            
            log.debug('Building dummy network for comparison')
            dummy_jvae = CVNet(input_shape, num_labels,
                               features=a.features,
                               type_of_net=a.type,
                               features_channels=a.features_channels,
                               conv_padding=a.conv_padding,
                               pretrained_features=a.pretrained_features,
                               encoder_layer_sizes=a.encoder,
                               latent_dim=a.latent_dim,
                               latent_sampling=a.latent_sampling,
                               decoder_layer_sizes=a.decoder,
                               upsampler_channels=a.upsampler,
                               classifier_layer_sizes=a.classifier,
                               beta=a.beta,
                               output_activation=a.output_activation)

            dummy_jvae.training['set'] = a.dataset
            log.debug('Built %s with training params %s',
                      dummy_jvae.print_architecture(),
                      dummy_jvae.print_training())

            log.debug('input shape: %s, labels: %s',
                      ' '.join(str(i) for i in input_shape),
                      num_labels)

            for i, n in enumerate(sum(nets_in_dir, [])):                    
                same_arch = dummy_jvae.has_same_architecture(n['net'])
                same_train = dummy_jvae.has_same_training(n['net'])
                log.debug('(%2s) %s %s %s', i,
                          dummy_jvae.print_training(),
                          # dummy_jvae.training,
                          '==' if same_train else '!=',
                          n['net'].print_training())
                          # n['net'].training)
                if same_arch and same_train:
                    s = 'Found alreay trained '
                    beta = n['beta']
                    epochs = n['done']
                    s += f'{beta:1.3e} ({epochs} epochs) '
                    log.debug(s)
                    a.already_trained.append({'dir': n['dir'], 'done': n['done']})

            a.already_trained.sort(key=lambda i: i['done'], reverse=True)
            log.info(f'{len(args.already_trained)} already trainend (%s)',
                     ' '.join([str(i['done']) for i in args.already_trained]))
            a.already_trained = args.already_trained[:repeat]

    for args in list_of_args:
        while len(args.already_trained) < repeat:
            args.already_trained.append({'dir': None,
                                         'done': 0,})
            
    args_to_be_done = []

    for _ in range(repeat):
        for a in list_of_args:

            to_be_finished = a.already_trained.pop(0)
            a_ = argparse.Namespace(**vars(a))
            a_.load_dir = to_be_finished['dir']
            a_.epochs_to_be_done = max(0, args.epochs - to_be_finished['done'])
            args_to_be_done.append(a_)

    total_epochs = sum([a.epochs_to_be_done for a in args_to_be_done])
    to_be_finished = sum([a.epochs_to_be_done > 0 for a in args_to_be_done])
    log.info('%s total nets (%s epochs) to be trained', to_be_finished,
             total_epochs)
    
    for a in args_to_be_done:

        save_dir = a.load_dir

        for k in a.__dict__.items():
            log.debug('%s: %s', *k)

        trainset, testset = torchdl.get_dataset(a.dataset, transformer=a.transformer)

        log.debug(f'{trainset.name} dataset loaded')
        
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=a.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)

        testloader = torch.utils.data.DataLoader(testset, batch_size=a.batch_size,
                                                 shuffle=True, num_workers=0)

        input_shape, num_labels = torchdl.get_shape(trainset)
        
        rebuild = a.load_dir is None

        if not rebuild:
            try:
                log.info('Loading network in %s', a.load_dir)
                jvae = CVNet.load(a.load_dir, load_state=True)
                log.debug(f'Network loaded')
                done_epochs = jvae.trained
                if done_epochs == 0:
                    verb = 'will start from scratch.'
                elif done_epochs < a.epochs:
                    verb = f'will resume from {done_epochs}.'
                else:
                    verb = 'is already done.'
                log.info(f'Training {verb}')
            except(FileNotFoundError, NameError) as err:
                log.warning(f'NETWORK NOT LOADED -- REBUILDING bc of {err}')
                rebuild = True

        if rebuild:
            log.info('Building network...')
            jvae = CVNet(input_shape, num_labels,
                         type_of_net=a.type,
                         features=a.features,
                         features_channels=a.features_channels,
                         conv_padding=a.conv_padding,
                         pretrained_features=a.pretrained_features,
                         encoder_layer_sizes=a.encoder,
                         latent_dim=a.latent_dim,
                         latent_sampling=a.latent_sampling,
                         decoder_layer_sizes=a.decoder,
                         upsampler_channels=a.upsampler,
                         classifier_layer_sizes=a.classifier,
                         beta=a.beta,
                         output_activation=a.output_activation)

        if not save_dir:

            save_dir_root = os.path.join(job_dir, a.dataset,
                                         jvae.print_architecture(sampling=False),
                                         f'beta={a.beta:1.2e}' +
                                         f'--sampling={a.latent_sampling}')
            i = 0
            save_dir = os.path.join(save_dir_root, f'{i:02d}')

            while os.path.exists(save_dir):
                log.debug(f'{save_dir} exists')
                i += 1
                save_dir = os.path.join(save_dir_root, f'{i:02d}')

            log.info('Network built, will be saved in')
            log.info(save_dir)

        log.debug('%s: %s', 'Network architecture',
                  jvae.print_architecture(True, True))

        jvae.to(device)

        x, y = torchdl.get_batch(trainset, device=device)
        if debug:
            log.debug('Trying a first pass')
            outs = jvae(x, y)
            log.debug([u.shape for u in outs])

        if not dry_run:
            if jvae.trained < a.epochs:
                log.info('Training of %s', jvae.print_architecture())

                jvae.train(trainset, epochs=a.epochs,
                           batch_size=batch_size,
                           device=device,
                           testset=testset,
                           sample_size=test_sample_size,  # 10000,
                           mse_loss_weight=None,
                           x_loss_weight=None,
                           kl_loss_weight=None,
                           save_dir=save_dir)
                log.info('Done training')
            else:
                log.info('No need to train %s', jvae.print_architecture())
        else:
            log.info('Dry-run %s', jvae.print_training(epochs=epochs, set=trainset.name))
            

        if save_dir is not None and not dry_run:
            jvae.save(save_dir)
