from __future__ import print_function

import numpy as np
import torch
from cvae import ClassificationVariationalNetwork as CVNet
import data.torch_load as torchdl
import os
import sys

from utils.parameters import alphanum, list_of_alphanums, get_args, set_log
from utils.save_load import collect_networks

list_of_args = get_args()

if __name__ == '__main__':

    
    
    args = list_of_args[0]
    
    debug = args.debug
    verbose = args.verbose
    log = set_log(verbose, debug)

    log.debug('$ ' + ' '.join(sys.argv))

    job_dir = args.job_dir

    find_and_finish = args.finish
    dry_run = args.dry_run


    if find_and_finish:
        
        for args in list_of_args:

            beta = args.beta

            latent_sampling = args.latent_sampling
            latent_dim = args.latent_dim

            features = args.features

            encoder = args.encoder
            decoder = args.decoder
            upsampler = args.upsampler
            conv_padding = args.conv_padding
            features_channels = args.features_channels

            output_activation = args.output_activation

            train_vae = args.vae
            classifier = args.classifier if not train_vae else []

            dataset = args.dataset
            transformer = args.transformer

            input_shape, num_labels = torchdl.get_shape_by_name(dataset)
            
            log.debug('Building dummy network for comparison')
            dummy_jvae = CVNet(input_shape, num_labels,
                 features=features,
                 features_channels=features_channels,
                 conv_padding=conv_padding,
                 # pretrained_features='vgg11.pth',
                 encoder_layer_sizes=encoder,
                 latent_dim=latent_dim,
                 latent_sampling=latent_sampling,
                 decoder_layer_sizes=decoder,
                 upsampler_channels=upsampler,
                 classifier_layer_sizes=classifier,
                 beta=beta,
                 output_activation=output_activation)

            log.debug('%s %s', 'built',
                      dummy_jvae.print_training())

            log.debug('%s: %s', 'input_shape',
                      ' '.join(str(i) for i in input_shape))
            log.debug('%s %s', num_labels, 'labels')


            l_o_n = [[{'net': dummy_jvae}]]
            collect_networks(job_dir, l_o_n) #, like=dummy_jvae)
            log.debug(f'{len(l_o_n)} list of networks collected:')
            for l in l_o_n:
                log.debug(f'{len(l)} networks')

            log.debug(f'I found {len(l_o_n[0])} networks')
            for i, net in enumerate(l_o_n[0]):
                arch = net['net'].print_architecture()
                log.debug(f'{i:2d}: {arch}')
    
    for args in list_of_args:

        epochs = args.epochs
        batch_size = args.batch_size
        test_sample_size = args.test_sample_size

        beta = args.beta

        latent_sampling = args.latent_sampling
        latent_dim = args.latent_dim

        features = args.features

        encoder = args.encoder
        decoder = args.decoder
        upsampler = args.upsampler
        conv_padding = args.conv_padding
        features_channels = args.features_channels

        output_activation = args.output_activation

        train_vae= args.vae
        classifier = args.classifier if not train_vae else []

        dataset = args.dataset
        transformer = args.transformer

        dry_run = args.dry_run

        refit = args.refit
        load_dir = args.load_dir
        save_dir = load_dir if not refit else None

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for k in args.__dict__.items():
            log.debug('%s: %s', *k)

        log.info(f'Used device: {device}')

        trainset, testset = torchdl.get_dataset(dataset, transformer=transformer)

        log.info(f'{trainset.name} dataset loaded')
        
        if train_vae:
            for the_set in (trainset, testset):
                new_labels = np.zeros(len(the_set), dtype=int)
                if hasattr(the_set, 'targets'):
                    the_set.targets = new_labels
                elif hasattr(the_set, 'labels'):
                    the_set.labels = new_labels
                else:
                    raise AttributeError(f'labels or targets is not an attribute of {the_set.name}')

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True, num_workers=0)

        input_shape, num_labels = torchdl.get_shape(trainset)
        



        rebuild = load_dir is None

        if not rebuild:
            try:
                log.info('Loading network')
                jvae = CVNet.load(load_dir, load_state=not refit)
                log.info(f'Network loaded')
                done_epochs = jvae.train_history['epochs']
                verb = 'Will resume' if done_epochs else 'Will start'
                log.info(f'{verb} training since epoch {done_epochs}')
            except(FileNotFoundError, NameError) as err:
                log.warning(f'NETWORK NOT LOADED -- REBUILDING bc of {err}')
                rebuild = True

        if rebuild:
            log.info('Building network...')
            jvae = CVNet(input_shape, num_labels,
                         features=features,
                         features_channels=features_channels,
                         conv_padding=conv_padding,
                         # pretrained_features='vgg11.pth',
                         encoder_layer_sizes=encoder,
                         latent_dim=latent_dim,
                         latent_sampling=latent_sampling,
                         decoder_layer_sizes=decoder,
                         upsampler_channels=upsampler,
                         classifier_layer_sizes=classifier,
                         beta=beta,
                         output_activation=output_activation)

        if not save_dir:

            beta_ = 'vae-beta=' if train_vae else 'beta='
            save_dir_root = os.path.join(job_dir, dataset,
                                         jvae.print_architecture(sampling=False),
                                         beta_ + f'{jvae.beta:1.2e}' +
                                         f'--sampling={latent_sampling}')

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
            log.info('Starting training')

            jvae.train(trainset, epochs=epochs,
                       batch_size=batch_size,
                       device=device,
                       testset=testset,
                       sample_size=test_sample_size,  # 10000,
                       mse_loss_weight=None,
                       x_loss_weight= 0 if train_vae else None,
                       kl_loss_weight=None,
                       save_dir=save_dir)
            log.info('Done training')
        else:
            log.info(jvae.print_training(epochs=epochs, set=trainset.name))


        if save_dir is not None and not dry_run:
            jvae.save(save_dir)
