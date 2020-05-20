from __future__ import print_function

import numpy as np
import torch
from cvae import ClassificationVariationalNetwork as CVNet
import data.torch_load as torchdl
import os
import sys

from utils.parameters import alphanum, list_of_alphanums, get_args

import logging 

args = get_args()

debug = args.debug
verbose = args.verbose
dry_run = args.dry_run

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
job_dir = args.job_dir


if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    log = logging.getLogger('')
    log.setLevel(0)
    if (log.hasHandlers()):
        log.handlers.clear()

    h_formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    formatter = logging.Formatter('[%(levelname).1s] %(message)s')
    stream_handler = logging.StreamHandler()
    log_level = logging.WARNING
    if verbose:
        log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG

    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    log.addHandler(stream_handler)

    log.info('Verbose is on')
    log.debug(f'Debug is on')
    
    for k in args.__dict__.items():
        log.debug('%s: %s', *k)
        
    log.info(f'Used device: {device}')
    
    trainset, testset = torchdl.get_dataset(dataset, transformer=transformer)

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

    
    test_batch = next(iter(testloader))
    x, y = test_batch[0].to(device), test_batch[1].to(device)

    input_shape = x.shape[1:]
    num_labels = len(torch.unique(y))

    log.debug('%s: %s', 'input_shape',
              ' '.join(str(i) for i in input_shape))
    log.debug('%s %s', num_labels, 'labels')

    
        
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
