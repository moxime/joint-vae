from __future__ import print_function

import torch
from cvae import ClassificationVariationalNetwork as CVNet
import data.torch_load as torchdl
import os

import argparse, configparser


def get_param(default_parameters, args, key, type=str, list=False):

    args_dict = vars(args)
    arg = args_dict.get(key, None)
    if arg is not None:
        return arg
    if type is bool:
        return default_parameters.getboolean(key)
    if not list:
        return type(default_parameters.get(key))
    else:
        return [type(s) for s in default_parameters.get(key).split()]

    
default_job_dir = './jobs'

config = configparser.ConfigParser()
parser = argparse.ArgumentParser(description="train a network")

parser.add_argument('--debug', action='store_true')

parser.add_argument('--config-file', default='config.ini')
parser.add_argument('--config', default='DEFAULT',
                    help='Which config to choose in config file')

parser.add_argument('--epochs', type=int)
parser.add_argument('-m', '--batch-size', type=int)

help = 'Num of samples to compute test accuracy'
parser.add_argument('-t', '--test_sample_size',
                    metavar='N',
                    help=help)

parser.add_argument('-b', '--beta', type=float,
                    metavar='ß')

parser.add_argument('-K', '--latent_dim', metavar='K', type=int)

parser.add_argument('-L', '--latent_sampling', type=int,
                    metavar='L')

parser.add_argument('--features',
                    choices=['vgg11'])
parser.add_argument('--encoder', type=int, nargs='+')
parser.add_argument('--decoder', type=int, nargs='+')    

parser.add_argument('--dataset', 
                    choices=['fashion', 'mnist', 'svhn', 'cifar10'])

parser.add_argument('--transformer',
                    choices=['simple', 'normal', 'default'],
                    help='simple : 0--1, normal 0 +/- 1')

help = 'Force refit of a net with same architecture'
help += '(may have a different beta)'
parser.add_argument('-F', '--refit', action='store_true')

help = 'save train(ing|ed) network in job-r/<architecture/i>'
help += 'unless load_dir is specified'
parser.add_argument('-j', '--job_dir',
                    default=default_job_dir,
                    help=help)

help = 'where to load the network'
help += ' (overrides all other parameters)'
parser.add_argument('load_dir',
                    help=help,
                    nargs='?', default=None)

args = parser.parse_args()
config.read(args.config_file)
default_parameters = config[args.config if args.config in config
                            else 'DEFAULT']

batch_size = get_param(default_parameters, args, 'batch_size', type=int)
epochs = get_param(default_parameters, args, 'epochs', type=int)
test_sample_size = get_param(default_parameters, args, 'test_sample_size', type=int)

beta = get_param(default_parameters, args, 'beta', type=float)

latent_sampling = get_param(default_parameters, args, 'latent_sampling',
                            type=int)
latent_dim = get_param(default_parameters, args, 'latent_dim',
                       type=int)

features = get_param(default_parameters, args, 'features')

encoder = get_param(default_parameters, args, 'encoder',
                    type=int, list=True)
decoder = get_param(default_parameters, args, 'decoder',
                    type=int, list=True)
output_activation = get_param(default_parameters, args, 'output_activation')
classifier = get_param(default_parameters, args, 'classifier',
                       type=int, list=True)

dataset = get_param(default_parameters, args, 'dataset')
transformer = get_param(default_parameters, args, 'transformer')

refit = args.refit
load_dir = args.load_dir
save_dir = load_dir if not refit else None
job_dir = args.job_dir
debug = args.debug


if __name__ == '__main__':
    
    if debug:

        print('bs', batch_size)
        print('epochs', epochs)
        print('test', test_sample_size)
        print('L', latent_sampling)
        print('K', latent_dim)
        print('ß', beta)
        print('feat', features)
        print('encoder', *encoder)
        print('decoder', *decoder)
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
                     features='vgg11',
                     # pretrained_features='vgg11.pth',
                     encoder_layer_sizes=encoder,
                     latent_dim=latent_dim,
                     latent_sampling=latent_sampling,
                     decoder_layer_sizes=decoder,
                     classifier_layer_sizes=classifier,
                     beta=beta,
                     output_activation=output_activation)

    if not save_dir:

        save_dir_root = os.path.join(job_dir, dataset,
                                     jvae.print_architecture(),
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

    print(jvae.print_architecture())

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

