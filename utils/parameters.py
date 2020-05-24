import argparse
import configparser
import logging 
from logging.handlers import RotatingFileHandler

def set_log(verbose, debug):
    
    log = logging.getLogger('')
    log.setLevel(0)
    if (log.hasHandlers()):
        log.handlers.clear()

    h_formatter = logging.Formatter('%(asctime)s [%(levelname).1s] %(message)s')
    formatter = logging.Formatter('[%(levelname).1s] %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler('./log/train-py.log',
                                       maxBytes=500000,
                                       backupCount=10)

    file_handler.doRollover()
    dump_file_handler = RotatingFileHandler('./log/dump.log',
                                            maxBytes=1, backupCount=20)
    log_level = logging.ERROR
    if verbose == 1:
        log_level = logging.WARNING
    if verbose > 1:
        log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG

    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    log.addHandler(stream_handler)

    file_handler.setFormatter(h_formatter)
    file_handler.setLevel(logging.DEBUG)
    log.addHandler(file_handler)

    dump_file_handler.setFormatter(h_formatter)
    dump_file_handler.setLevel(logging.DEBUG)
    log.addHandler(dump_file_handler)

    def dump_filter(record):
        r = record.getMessage().startswith('DUMPED')
        # print('dump filter', r)
        # if r:
        #    logging.error('An error has been dumped somewhere')  
        return r

    def no_dump_filter(record):
        r = record.getMessage().startswith('DUMPED')
        # print('no dump filter', not r)
        return not r

    for h in (stream_handler, file_handler):
        h.addFilter(no_dump_filter)
        pass #

    dump_file_handler.addFilter(dump_filter)
    

    log.error('not an error, just showing what logs look like')
    log.info('Verbose is on')
    log.debug(f'Debug is on')
    
    return log


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

def get_args(what_for='train', argv=None):

    for_train = what_for == 'train'
    for_test = not for_train

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--debug', action='store_true')
    conf_parser.add_argument('--verbose', '-v', action='count', default=0)
    conf_parser.add_argument('--config-file', default='config.ini')
    conf_parser.add_argument('--config', '-c', default='DEFAULT')

    if for_train:
        conf_parser.add_argument('--grid-file', default='grid.ini')

    conf_args, remaining_args = conf_parser.parse_known_args(argv)
    config = configparser.ConfigParser()
    config.read(conf_args.config_file)

    config_params = config[conf_args.config]

    default_test_sample_size = 1000 if for_train else 10000
    defaults = {'batch_size': 100,
                'epochs':100,
                'test_sample_size': default_test_sample_size,
                'job_dir': './jobs'}
    
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

    help = 'epochs for training' if for_train else 'min epochs for testing'

    parser.add_argument('--epochs', type=int, help=help)

    parser.add_argument('-m', '--batch-size', type=int, metavar='M')

    help = 'Num of samples to compute test accuracy'
    parser.add_argument('-t', '--test_sample_size', type=int,
                        metavar='N',
                        help=help)

    if for_test:
        help = 'Minimum accepted before retesting'
        parser.add_argument('-T', '--min-test-samplesize',
                            type=int, metavar='N0', default=0)

    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='will show you what it would do')

    parser.add_argument('-b', '--beta',
                        type = float,
                        # type=alphanum,
                        metavar='ß') #,
                        # nargs='*',
                        # help='several values can be provided for several trainings')

    parser.add_argument('-K', '--latent_dim', metavar='K',
                        type=int)
                        # type=alphanum, nargs='*',
                        # help='several dims can be provided for several trainings')
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

    if for_train:
        help = 'Force refit of a net with same architecture (NOT IMPLEMENTED)'
        # help += '(may have a different beta)'
        parser.add_argument('--refit', action='store_true')

        help = 'Find and finish begun trainings'
        parser.add_argument('-F', '--finish', default=None, const=1,
                            nargs='?',
                            type=int, help=help) 

        parser.add_argument('-R', '--repeat', default=1, type=int)
    
    help = 'save train(ing|ed) network in DIR/<architecture/i>'
    help += 'unless load_dir is specified'
    parser.add_argument('-j', '--job_dir', metavar='DIR',
                        help=help)

    help = 'where to load the network'
    help += ' (overrides all other parameters)'
    parser.add_argument('load_dir',
                        help=help,
                        nargs='?', default=None)

    if for_train:
        parser.add_argument('--grid-config', default=None)
    
    args = parser.parse_args(remaining_args)

    if for_test:
        args.grid_config = False
    
    args.debug = conf_args.debug
    args.verbose = conf_args.verbose
    args.config_file = conf_args.config_file
    args.config = conf_args.config

    if args.features.lower() == 'none' or args.no_features:
        args.features=None

    if args.grid_config:
        config = configparser.ConfigParser()
        config.read(args.grid_file)

        grid_params = config[args.grid_config]

        args.repeat = grid_params.getint('repeat', 1)

        list_of_args = [args]
        for param_name in grid_params:
            final_list_of_args = []
            # print(param_name)
            param_values = list_of_alphanums(grid_params[param_name])
            # print(param_values)
            for args in list_of_args:
                for value in param_values:
                    d = vars(args)
                    d[param_name] = value
                    final_list_of_args.append(argparse.Namespace(**d))
            list_of_args = final_list_of_args.copy()
        # creates a list of args

        return list_of_args
    
    return [args]

def same_dict(d1, d2):

    result = True
    for k in tuple(d1.keys()) + tuple(d2.keys()):
        if not d1.get(k, 0):
            if d2.get(k, 0):
                return False
        elif not d2.get(k, 0):
            return False
        elif d1[k] != d2[k]:
            return False

    return True
        
if __name__ == '__main__':

    list_of_args = get_args()
    for args in list_of_args:
        for k in args.__dict__.items():
            print(*k)

            
