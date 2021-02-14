import argparse
import configparser
import logging
from logging import FileHandler
from logging.handlers import RotatingFileHandler
import re, numpy as np
from socket import gethostname as getrawhostname


def gethostname():

    raw_host = getrawhostname()

    if 'lab-ia' in raw_host:
        return 'labia'
    if raw_host == 'DESKTOP-DUHAMEL':
        return 'lss'
    if raw_host == 'astrov':
        return 'home'

    return raw_host.split('.')[0]
    

def set_log(verbose, debug, name='train', job_number=0):
    
    log = logging.getLogger('')
    log.setLevel(0)
    if (log.hasHandlers()):
        log.handlers.clear()

    h_formatter = logging.Formatter('%(asctime)s [%(levelname).1s] %(message)s')
    formatter = logging.Formatter('[%(levelname).1s] %(message)s')
    stream_handler = logging.StreamHandler()

    if job_number:
        file_handler = FileHandler(f'./log/{name}.log.{job_number}')
                                           
    else:
        file_handler = RotatingFileHandler(f'./log/{name}.log',
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
    

    # log.error('not an error, just showing what logs look like')
    # log.info('Verbose is on')
    # log.debug(f'Debug is on')
    
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


def get_args(what_for='train', *a ,**kw):

    if what_for=='train':
        return get_args_for_train(*a, **kw)

    return get_args_for_test(*a, **kw)


def get_args_for_train(argv=None):

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--debug', action='store_true')
    conf_parser.add_argument('--verbose', '-v', action='count', default=0)
    conf_parser.add_argument('--config-file', default='config.ini')
    conf_parser.add_argument('--config', '-c', default='DEFAULT')
    
    conf_args, remaining_args = conf_parser.parse_known_args(argv)
    config = configparser.ConfigParser()
    config.read(conf_args.config_file)

    config_params = config[conf_args.config]

    defaults = {'batch_size': 100,
                'epochs':100, 
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

    logging.debug('Defaults:')
    for k in defaults:
        # print('****', k, defaults[k])
        logging.debug(k, defaults[k])

    help = f'epochs for training (default is {defaults["epochs"]})'
    parser.add_argument('--epochs', type=int, help=help)

    parser.add_argument('-m', '--batch-size', type=int, metavar='M')
    parser.add_argument('--test-batch-size', type=int, metavar='M')

    help = 'Num of samples to compute test accuracy'
    parser.add_argument('-t', '--test-sample-size', type=int,
                        metavar='N',
                        help=help)

    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='will show you what it would do')

    parser.add_argument('--type', choices=['jvae', 'cvae', 'vib', 'vae', 'xvae'])
    
    parser.add_argument('-s', '--sigma',
                        type = float,
                        # type=alphanum,
                        metavar='S') #,
                        # nargs='*',
                        # help='several values can be provided for several trainings')

    parser.add_argument('--sigma-reach',
                        type=float,
                        nargs='?',
                        default=1.,
                        const=4)

    parser.add_argument('--sigma-decay',
                        type=float,
                        default=0.)

    parser.add_argument('--sigma-max-step',
                        type=float,
                        default=None)
    
    parser.add_argument('--sigma-learned',
                        action='store_true',)

    parser.add_argument('--sigma-is-rmse',
                        action='store_true',)

    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta-(C)VAE not implemented',
                        metavar='ß')

    parser.add_argument('--dictionary-variance',
                        type=float,
                        default=1)
    
    parser.add_argument('--learned-coder',
                        action='store_true')

    parser.add_argument('--dict-min-distance', type=float)

    parser.add_argument('--no-dict-distance-regularization', action='store_false',
                        dest='dict_distance_regularization')
    
    parser.add_argument('-K', '--latent-dim', metavar='K',
                        type=int)

    parser.add_argument('-L', '--latent-sampling', metavar='L', type=int)

    parser.add_argument('--latent-prior-variance', type=float, default=1.)

    parser.add_argument('--features', metavar='NAME',
                        choices=['vgg11', 'vgg16', 'vgg19', 'conv', 'none'])

    parser.add_argument('--pretrained-features', metavar='feat.pth')
    parser.add_argument('--no-features', action='store_true')

    parser.add_argument('--pretrained-upsampler', metavar='upsampler.pth')

    parser.add_argument('--fine-tuning', action='store_true')

    parser.add_argument('--encoder', type=alphanum, metavar='W', nargs='*')
    parser.add_argument('--features-channels', type=alphanum, metavar='C', nargs='*')
    parser.add_argument('--conv-padding', type=alphanum, metavar='P')
    parser.add_argument('--decoder', type=alphanum, nargs='*', metavar='W')
    parser.add_argument('--upsampler', type=alphanum, nargs='*', metavar='C')
    parser.add_argument('--classifier', type=alphanum, nargs='*', metavar='W')

    parser.add_argument('--forced-encoder-variance', type=float, default=False, nargs='?', const=1.0)
    
    parser.add_argument('--dataset', 
                        choices=['fashion', 'mnist', 'fashion32', 'svhn', 'cifar10'])

    parser.add_argument('--transformer',
                        choices=['simple', 'normal', 'default'],
                        help='transform data, simple : 0--1, normal 0 +/- 1')

    parser.add_argument('--data-augmentation',
                        choices=['flip', 'crop'],
                        nargs='*')

    parser.add_argument('--force-cross-y', type=float, nargs='?', const=1.0, default=0.)
    
    parser.add_argument('--batch-norm',
                        choices=['encoder', 'both'],
                        nargs='?',
                        const='encoder')

    parser.add_argument('--optimizer', choices=('sgd', 'adam'))
    parser.add_argument('--lr', default=0, type=float)
    parser.add_argument('--lr-decay', default=0, type=float)
    
    help = 'Find by job number and resume begun training'
    parser.add_argument('-R', '--resume', default=None,
                        help=help, metavar='#') 
    
    help = 'save train(ing|ed) network in DIR/<architecture/#>'

    parser.add_argument('--job-dir', metavar='DIR/',
                        help=help)

    parser.add_argument('-j', '--job-number',
                        type=int,
                        metavar='#',
                        default=0)

    parser.add_argument('--output-dir', metavar='DIR/')

    parser.add_argument('--show', action='store_true',
                        help='Shiw network structure and exit')
    
    parser.add_argument('--where', action='store_true',
                        help='Print saving dir and exit')
    
    args = parser.parse_args(remaining_args)
        
    args.debug = conf_args.debug
    args.verbose = conf_args.verbose
    args.config_file = conf_args.config_file
    args.config = conf_args.config
    
    if args.features.lower() == 'none' or args.no_features:
        args.features=None

    return args


def get_args_for_test():

    parser = argparse.ArgumentParser() #(add_help=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    defaults = {'batch_size': 100,
                'epochs': 0,
                'test_sample_size': 10000,
                'job_dir': './jobs'}

    parser.set_defaults(**defaults)

    logging.debug('Defaults:')
    for k in defaults:
        logging.debug(k, defaults[k])

    parser.add_argument('load_dir',
                        nargs='?', default=None)

    parser.add_argument('-m', '--batch-size', type=int, metavar='M')

    help = 'Num of samples to compute test accuracy'
    parser.add_argument('-t', '--test-sample-size', type=int,
                        metavar='N',
                        help=help)

    help = 'Minimum accepted before retesting'
    parser.add_argument('-T', '--min-test-sample-size',
                        type=int, metavar='N0', default=0)
    parser.add_argument('-F', '--only-finished', action='store_false',
                        dest='unfinished',
                        help='Only finished training')

    help = 'Number of sample to comute OOD FPR'
    parser.add_argument('-o', '--ood', type=int, nargs='?',
                        const=-1, default=0,
                        help=help)

    parser.add_argument('--cautious', action='store_true')

    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('--compute', action='store_false',
                        dest='dry_run',
                        help='will compute missing rates')
    parser.add_argument('--flash', action='store_true')

    parser.add_argument('--show', action='store_true')
    
    parser.add_argument('--latex', action='store_true')

    parser.add_argument('--expand', '-e' , action='count', default=0)

    parser.add_argument('--tpr', nargs='*', default=[95], type=int)
    parser.add_argument('--tnr', action='store_true', help='Show TNR instead of FPR')
    
    is_true = ParamFilter()

    parser.add_argument('--epochs',
                        dest='done',
                        action=FilterAction,
                        of_type=int,)
    
    parser.add_argument('--finished',
                        nargs='?',
                        action=FilterAction,
                        of_type=bool)

    parser.add_argument('--train-batch-size',
                        nargs='+',
                        action=FilterAction,
                        of_type=int)

    parser.add_argument('--best-accuracy',
                        nargs='+',
                        action=FilterAction,
                        of_type=float)


    parser.add_argument('--optimizer',
                        dest='optim',
                        nargs='+',
                        action=FilterAction,
                        of_type=str)
    
    parser.add_argument('--type',
                        action=FilterAction,
                        nargs='+')

    parser.add_argument('-s', '--sigma',
                        nargs='+',
                        action=FilterAction,
                        of_type=float,
                        metavar='S')

    parser.add_argument('-K', '--latent-dim', metavar='K',
                        dest='K',
                        nargs='+',
                        action=FilterAction,
                        of_type=int)

    parser.add_argument('-L', '--latent-sampling', metavar='L',
                        dest='L',
                        nargs='+',
                        action=FilterAction,
                        of_type=int)

    parser.add_argument('--depth',
                        nargs='+',
                        action=FilterAction,
                        of_type=int)
    
    parser.add_argument('--dataset',
                        dest='set',
                        nargs='+',
                        action=FilterAction)

    parser.add_argument('--job-number',
                        dest='job',
                        of_type=int,
                        nargs='+',
                        action=FilterAction)
    
    args = parser.parse_args()

    if not hasattr(args, 'filters'):
        args.filters = {}
    
    return args


class ParamFilter():

    def __init__(self, arg_str='',
                 arg_type=int,
                 neg=False,
                 always_true=False,
    ):

        self.arg_str = arg_str
        self.arg_type = arg_type
        self.always_true = always_true
        self.neg = neg
        
        interval_regex = '\.{2,}'
        self.is_interval = re.search(re.compile(interval_regex),
                                arg_str)

        list_regex = '[\s\,]+\s*'
        self.is_list = re.search(re.compile(list_regex),
                            arg_str)

        if self.is_interval:

            endpoints = re.split(interval_regex, arg_str)
            self.interval = [-np.inf, np.inf]
            for i in (0, -1):
                try:
                    self.interval[i] = arg_type(endpoints[i])
                except ValueError:
                    pass

        if self.is_list:

            _values = re.split(list_regex, arg_str)
            self.values = [arg_type(v) for v in _values]

    def __str__(self):

        pre = 'not ' if self.neg else ''

        if self.is_interval:
            return pre + '..'.join([str(a) for a in self.interval])

        if self.is_list:
            return pre + ' or '.join([str(v) for v in self.values])

        if self.arg_type is bool:
            return 'False' if self.neg else 'True'

        else:
            return pre + (self.arg_str if self.arg_str else 'any')
        
    def filter(self, value):

        neg = self.neg
        if self.always_true:
            return False if neg else True
        
        if not self.arg_str:
            return not value if neg else value
        
        # if not value:
        #    return True
        
        if self.is_interval:
            if value is None:
                return False

            try:
                a, b = self.interval
                in_ =  a <= value <= b
                return not in_ if neg else in_
            except TypeError as e:
                print(a, type(a), b, type(b), value, type(value), self)
                raise(e)
            
        if self.is_list:
            in_ = value in self.values
            return not in_ if neg else in_

        # else
        the_value = self.arg_type(self.arg_str)
        in_ = value == the_value
        return not in_ if neg else in_
    
class FilterAction(argparse.Action):

    def __init__(self, option_strings, dest, of_type=str, neg=False, **kwargs):
        super(FilterAction, self).__init__(option_strings, dest, **kwargs)

        self._type=of_type
        self._neg=neg
        self.default=ParamFilter()

    def __call__(self, parser, namespace, values, option_string=None):

        if type(values) is not list:
            values= [values]
            
        if values and values[0].lower() == 'not':
            self._neg = True
            values.pop(0)

        if values:
            arg_str = ' '.join(str(v) for v in values)
        else:
            arg_str = ''

        filter = ParamFilter(arg_str,
                             arg_type=self._type,
                             neg=self._neg)
        # print(filter)
        if not hasattr(namespace, 'filters'):
            setattr(namespace, 'filters', {})
        namespace.filters[self.dest] = filter


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

    arg = get_args_for_test()

    for k in arg.filters.items():
        print(*k)
