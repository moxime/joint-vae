from __future__ import print_function

import torch
from cvae import ClassificationVariationalNetwork as CVNet
import data.torch_load as torchdl
import os
import sys

import argparse
import configparser
from utils.parameters import alphanum, list_of_alphanums

conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument('--debug', action='store_true')
conf_parser.add_argument('--verbose', '-v', action='count')
conf_parser.add_argument('--config-file', default='config.ini')
conf_parser.add_argument('--config', '-c', default='DEFAULT')

conf_args, remaining_args = conf_parser.parse_known_args()
config = configparser.ConfigParser()
config.read(conf_args.config_file)

config_params = config[conf_args.config]

defaults = {'batch_size': 100,
            'epochs':100,
            'test_sample_size': 1000, 'job_dir': './jobs'}

defaults.update(config_params)

for k in ('encoder', 'features_channels',
          'decoder', 'upsampler',
          'classifier'):
     p = defaults.get(k, '')
     defaults[k] = list_of_alphanums(p)


parser = argparse.ArgumentParser(parents=[conf_parser])
# description=__doc__)

parser.set_defaults(**defaults)


parser.add_argument('--epochs', type=int)
parser.add_argument('-m', '--batch-size', type=int, metavar='M')

help = 'Num of samples to compute test accuracy'
parser.add_argument('-t', '--test_sample_size', type=int,
                    metavar='N',
                    help=help)

parser.add_argument('-b', '--beta', type=float, metavar='ß')

parser.add_argument('-K', '--latent_dim', metavar='K', type=int)
parser.add_argument('-L', '--latent_sampling', metavar='L', type=int)

parser.add_argument('--features', metavar='NAME',
                    choices=['vgg11', 'vgg16', 'conv', 'none'])

parser.add_argument('--no-features', action='store_true')

parser.add_argument('--encoder', type=int, metavar='H', nargs='*')
parser.add_argument('--features_channels', type=int, nargs='*')
parser.add_argument('--decoder', type=int, nargs='*')
parser.add_argument('--upsampler', type=int, nargs='*')
parser.add_argument('--classifier', type=int, nargs='*')

parser.add_argument('--dataset', 
                    choices=['fashion', 'mnist', 'svhn', 'cifar10'])

parser.add_argument('--transformer',
                    choices=['simple', 'normal', 'default'],
                    help='transform data, simple : 0--1, normal 0 +/- 1')


help = 'Force refit of a net with same architecture'
# help += '(may have a different beta)'
parser.add_argument('-F', '--refit', action='store_true')

help = 'save train(ing|ed) network in job-r/<architecture/i>'
help += 'unless load_dir is specified'
parser.add_argument('-j', '--job_dir',
                    help=help)

help = 'where to load the network'
help += ' (overrides all other parameters)'
parser.add_argument('load_dir',
                    help=help,
                    nargs='?', default=None)

args = parser.parse_args(remaining_args)

epochs = args.epochs
batch_size = args.batch_size
test_sample_size = args.test_sample_size
beta = args.beta

latent_sampling = args.latent_sampling
latent_dim = args.latent_dim

features = args.features
if features.lower() == 'none' or args.no_features:
    features=None

encoder = args.encoder
decoder = args.decoder
upsampler = args.upsampler
features_channels = args.features_channels

output_activation = args.output_activation

classifier = args.classifier

dataset = args.dataset
transformer = args.transformer


refit = args.refit
load_dir = args.load_dir
save_dir = load_dir if not refit else None
job_dir = args.job_dir
debug = conf_args.debug


if __name__ == '__main__':
    
    if debug:

        print('bs', batch_size)
        print('epochs', epochs)
        print('test', test_sample_size)
        print('L', latent_sampling)
        print('K', latent_dim)
        print('ß', beta)
        print('feat', features)
        print('features channels', *features_channels)
        print('encoder', *encoder)
        print('decoder', *decoder)
        print('upsampler', *upsampler)
        print('output_activation', output_activation)
        print('classifier', *classifier)
        print('data', dataset)
        print('transformer', transformer)
        print('refit', refit)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Used device:', device)
    
    trainset, testset = torchdl.get_dataset(dataset, transformer=transformer)

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

        save_dir_root = os.path.join(job_dir, dataset,
                                     jvae.print_architecture(sampling=True),
                                     f'{jvae.beta:1.2e}')

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
               x_loss_weight=None,
               kl_loss_weight=None,
               save_dir=save_dir)

    if save_dir is not None:
        jvae.save(save_dir)

