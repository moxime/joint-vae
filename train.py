from __future__ import print_function

import numpy as np
import torch
from cvae import ClassificationVariationalNetwork as CVNet
import data.torch_load as torchdl
import os
import sys

from utils.parameters import alphanum, list_of_alphanums, get_args


args = get_args()

debug = args.debug
verbose = args.verbose

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

refit = args.refit
load_dir = args.load_dir
save_dir = load_dir if not refit else None
job_dir = args.job_dir


if __name__ == '__main__':
    
    if debug:

        for k in args.__dict__.items():
            print(*k)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Used device:', device)
    
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
    if debug:
        print('input_shape', *input_shape, num_labels)
        
    rebuild = load_dir is None
    
    if not rebuild:
        print('Loading...', end=' ')
        try:
            jvae = CVNet.load(load_dir, load_state=not refit)
            print(f'done', end=' ')
            done_epochs = jvae.train_history['epochs']
            verb = 'resuming' if done_epochs else 'starting'
            print(f'{verb} training since epoch {done_epochs}')
        except(FileNotFoundError, NameError) as err:
            print(f'*** NETWORK NOT LOADED -- REBUILDING bc of {err} ***')
            rebuild = True

    if rebuild:
        print('Building network...', end=' ')
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
        if debug:
            print(save_dir, end=' ')
            if os.path.exists(save_dir):
                print('exists')
            else:
                print('does not esist')
        while os.path.exists(save_dir):
            i += 1
            save_dir = os.path.join(save_dir_root, f'{i:02d}')

    print('done.', 'Will be saved in\n' + save_dir)

    print(jvae.print_architecture(True, True))

    jvae.to(device)

    if debug:
        outs = jvae(x, y)
        print([u.shape for u in outs])
        
    print('\nTraining\n')

    jvae.train(trainset, epochs=epochs,
               batch_size=batch_size,
               device=device,
               testset=testset,
               sample_size=test_sample_size,  # 10000,
               mse_loss_weight=None,
               x_loss_weight= 0 if train_vae else None,
               kl_loss_weight=None,
               save_dir=save_dir)

    if save_dir is not None:
        jvae.save(save_dir)

