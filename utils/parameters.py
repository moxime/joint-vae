import argparse
import configparser

def alphanum(x):
    try:
        return int(x)
    except(ValueError):
        try:
            return float(x)
        except(ValueError):
            return x


def list_of_alphanums(string):

    return [alphanum(a) for a in string.split()]


def get_args(argv=None):

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--debug', action='store_true')
    conf_parser.add_argument('--verbose', '-v', action='count')
    conf_parser.add_argument('--config-file', default='config.ini')
    conf_parser.add_argument('--config', '-c', default='DEFAULT')

    conf_args, remaining_args = conf_parser.parse_known_args(argv)
    config = configparser.ConfigParser()
    config.read(conf_args.config_file)

    config_params = config[conf_args.config]

    defaults = {'batch_size': 100,
                'epochs':100,
                'test_sample_size': 1000, 'job_dir': './jobs'}

    defaults.update(config_params)

    alphanum_keys = ('encoder',
                     'features_channels',
                     'decoder',
                     'upsampler',
                     'classifier')
    
    for k in alphanum_keys:
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

    parser.add_argument('--force_cpu', action='store_true')

    parser.add_argument('-b', '--beta', type=float, metavar='ß')

    parser.add_argument('-K', '--latent_dim', metavar='K', type=int)
    parser.add_argument('-L', '--latent_sampling', metavar='L', type=int)

    parser.add_argument('--features', metavar='NAME',
                        choices=['vgg11', 'vgg16', 'conv', 'none'])

    parser.add_argument('--no-features', action='store_true')

    parser.add_argument('--encoder', type=alphanum, metavar='W', nargs='*')
    parser.add_argument('--features_channels', type=alphanum, nargs='*')
    parser.add_argument('--conv_padding', type=alphanum)
    parser.add_argument('--decoder', type=alphanum, nargs='*')
    parser.add_argument('--upsampler', type=alphanum, nargs='*')
    parser.add_argument('--classifier', type=alphanum, nargs='*')
    parser.add_argument('--vae', action='store_true')

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

    args.debug = conf_args.debug
    args.verbose = conf_args.verbose
    args.config_file = conf_args.config_file
    args.config = conf_args.config

    if args.features.lower() == 'none' or args.no_features:
        args.features=None

    return args
    


if __name__ == '__main__':

    args = get_args()
    for k in args.__dict__.items():
        print(*k)
