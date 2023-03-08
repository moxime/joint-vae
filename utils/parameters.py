import argparse
import configparser
import logging
from pydoc import locate
from logging import FileHandler
from logging.handlers import RotatingFileHandler
import re
import numpy as np
from socket import gethostname as getrawhostname
from utils.filters import ParamFilter, FilterAction, DictOfListsOfParamFilters, get_filter_keys
import os

DEFAULT_JOBS_DIR = 'jobs'
DEFAULT_RESULTS_DIR = 'jobs/results'


def gethostname():

    raw_host = getrawhostname()

    if 'lab-ia' in raw_host:
        return 'labia'
    if raw_host == 'DESKTOP-DUHAMEL':
        return 'lss'
    if raw_host == 'astrov':
        return 'home'

    return raw_host.split('.')[0]


def in_list_with_starred(k, list_with_starred):

    for k_ in list_with_starred:
        if k_.endswith('*') and k.startswith(k_[:-1]):
            return True
        elif k_ == k:
            return True
    return False


def set_log(verbose, debug, log_dir, name='train', job_number=0, tmp_dir='/tmp'):

    log = logging.getLogger('')
    log.setLevel(0)
    if (log.hasHandlers()):
        log.handlers.clear()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.isdir(log_dir):
        log_dir = '/tmp'

    h_formatter = logging.Formatter('%(asctime)s [%(levelname).1s] %(message)s')
    formatter = logging.Formatter('[%(levelname).1s] %(message)s')
    stream_handler = logging.StreamHandler()

    if job_number:
        fn = os.path.join(log_dir, f'{name}.log.{job_number}')
        file_handler = FileHandler(fn)

    else:
        fn = os.path.join(log_dir, f'{name}.log')
        file_handler = RotatingFileHandler(fn,
                                           maxBytes=5000000,
                                           backupCount=10)
        file_handler.doRollover()

    fn = os.path.join(log_dir, 'dump.log')
    dump_file_handler = RotatingFileHandler(fn,
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

    if log_dir == tmp_dir:
        logging.warning('will log in %s',  tmp_dir)

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
        pass

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


def str2bool(s):

    return s.lower() in ['true', 'yes', 't', '1']


def list_of_alphanums(string):

    return [alphanum(a) for a in string.split()]


def get_args(what_for='train', *a, **kw):

    if what_for == 'train':
        return get_args_for_train(*a, **kw)

    return get_args_for_results(*a, **kw)


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

    defaults = {'batch_size': 128,
                'test_batch_size': 512,
                'test_sample_size': 1024,
                'validation': 8192,
                'features': 'none',
                'epochs': 100,
                'job_dir': DEFAULT_JOBS_DIR}

    defaults.update(config_params)

    alphanum_keys = ('encoder',
                     'data_augmentation',
                     'features_channels',
                     'decoder',
                     'upsampler',
                     'classifier')

    bool_keys = ('learned_prior_means',)

    for k in alphanum_keys:
        p = defaults.get(k, '')
        defaults[k] = list_of_alphanums(p)

    for k in bool_keys:
        p = defaults.get(k, '')
        defaults[k] = str2bool(p)

    parser = argparse.ArgumentParser(parents=[conf_parser],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # description=__doc__)

    logging.debug('Defaults:')
    for k in defaults:
        # print('****', k, defaults[k])
        logging.debug(k, defaults[k])

    help = f'epochs for training (default is {defaults["epochs"]})'
    parser.add_argument('--epochs', type=int, help=help)

    parser.add_argument('-M', '--batch-size', type=int, metavar='m')
    parser.add_argument('-m', '--test-batch-size', type=int, metavar='M', default=1024)

    help = 'Num of samples to compute test accuracy at each epoch'
    parser.add_argument('-t', '--test-sample-size', type=int,
                        metavar='N',
                        help=help)

    parser.add_argument('-V', '--validation', type=int, default=4096,
                        help='Number of validation samples taken away from the training set')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='will show you what it would do')

    parser.add_argument('--type', choices=['jvae', 'cvae', 'vib', 'vae', 'xvae'])

    parser.add_argument('--sigma', '-s',
                        # type = float,
                        type=alphanum,
                        metavar='S',
                        help='Value of sigma (float) or in [\'learned\', \'rmse\', \'coded\']'
                        )  # ,
    # nargs='*',
    # help='several values can be provided for several trainings')

    # parser.add_argument('--sigma-reach',
    #                     type=float,
    #                     nargs='?',
    #                     default=1.,
    #                     const=4)

    # parser.add_argument('--sigma-decay',
    #                     type=float,
    #                     default=0.)

    # parser.add_argument('--sigma-max-step',
    #                     type=float,
    #                     default=None)

    # parser.add_argument('--sigma-learned',
    #                     action='store_true',)

    # parser.add_argument('--sigma-is-rmse',
    #                     action='store_true',)

    parser.add_argument('--sigma-per-dim', action='store_true')

    # parser.add_argument('--sigma-coded', action='store_true')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta-(C)VAE',
                        metavar='ß')

    parser.add_argument('--gamma', type=float,
                        default=0.)

    parser.add_argument('--prior', choices=['gaussian', 'tilted'], default='gaussian')

    parser.add_argument('--tilted-tau', default=25., type=float)

    parser.add_argument('--prior-means',
                        type=alphanum,
                        default=0,
                        help='For CVAE, std of latent prior means (or \'onehot\')'
                        )

    parser.add_argument('--learned-prior-means',
                        action='store_true')
    parser.add_argument('--static-prior-means',
                        dest='learned_prior_means',
                        action='store_false')

    parser.add_argument('--prior-variance',
                        choices=['scalar', 'diag', 'full'],
                        default='scalar')

    parser.add_argument('-K', '--latent-dim', metavar='K',
                        type=int)

    parser.add_argument('-L', '--latent-sampling', metavar='L', type=int)
    parser.add_argument('-l', '--test-latent-sampling', metavar='l', type=int)

    parser.add_argument('--hsv', action='store_true')
    parser.add_argument('--representation')

    parser.add_argument('--features', metavar='NAME',)
    # choices=['vgg11', 'vgg16', 'vgg19', 'conv', 'none',])

    parser.add_argument('--pretrained-features', metavar='feat.pth', nargs='?', const='online')
    parser.add_argument('--no-features', action='store_true')

    parser.add_argument('--pretrained-upsampler', metavar='upsampler.pth')

    parser.add_argument('--fine-tuning', action='store_true')
    parser.add_argument('--warmup', type=float, default=[0], nargs='+')

    parser.add_argument('--encoder', type=alphanum, metavar='W', nargs='*')
    parser.add_argument('--features-channels', type=alphanum, metavar='C', nargs='*')
    parser.add_argument('--conv-padding', type=alphanum, metavar='P')
    parser.add_argument('--decoder', type=alphanum, nargs='*', metavar='W')
    parser.add_argument('--upsampler', type=alphanum, metavar='CxK-CxK+P...')
    parser.add_argument('--classifier', type=alphanum, nargs='*', metavar='W')

    parser.add_argument('--encoder-forced-variance', type=float, default=False, nargs='?', const=1.0)

    parser.add_argument('--dataset',)
    # choices=['fashion', 'mnist', 'fashion32', 'svhn', 'cifar10', 'letters'])

    parser.add_argument('--transformer',
                        choices=['simple', 'normal', 'default', 'crop', 'pad'],
                        help='transform data, simple : 0--1, normal 0 +/- 1')

    parser.add_argument('--data-augmentation',
                        choices=['flip', 'crop'],
                        type=str,
                        nargs='*')

    parser.add_argument('--force-cross-y', type=float, nargs='?', const=1.0, default=0.)

    parser.add_argument('--batch-norm',
                        choices=['encoder', 'both', 'none'],
                        nargs='?',
                        const='encoder')

    parser.add_argument('--optimizer', choices=('sgd', 'adam'))
    parser.add_argument('--lr', default=0, type=float)
    parser.add_argument('--wd', default=0, type=float, dest='weight_decay')

    parser.add_argument('--lr-decay', default=0, type=float)

    parser.add_argument('--grad-clipping', type=float)

    help = 'Find by job number and resume begun training'
    parser.add_argument('-R', '--resume', default=None,
                        help=help, metavar='#')

    help = 'save train(ing|ed) network in DIR/<architecture/#>'

    parser.add_argument('--full-test-every', type=int, default=10)

    parser.add_argument('--job-dir', metavar='DIR/',
                        help=help)

    parser.add_argument('-j', '--job-number',
                        type=int,
                        metavar='#',
                        default=0)

    parser.add_argument('--output-dir', metavar='DIR/')

    parser.add_argument('--show', action='store_true',
                        help='Show network structure and exit')

    parser.add_argument('--where', action='store_true',
                        help='Print saving dir and exit')

    parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_args)

    args.debug = conf_args.debug
    args.verbose = conf_args.verbose
    args.config_file = conf_args.config_file
    args.config = conf_args.config

    if args.features.lower() == 'none' or args.no_features:
        args.features = None

    return args


def get_args_for_results(argv=None):

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    defaults = {'batch_size': 128,
                'epochs': 0,
                'test_sample_size': 1024,
                'job_dir': DEFAULT_JOBS_DIR}

    parser.set_defaults(**defaults)

    logging.debug('Defaults:')
    for k in defaults:
        logging.debug(k, defaults[k])

    parser.add_argument('--load-dir',
                        default=None)

    parser.add_argument('-J', '--job-dir', default=DEFAULT_JOBS_DIR)

    parser.add_argument('-M', '--batch-size', type=int, metavar='M')

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

    parser.add_argument('--early-stopping')

    parser.add_argument('--device')
    parser.add_argument('--compute',
                        nargs='?',
                        default=False,
                        const='recorder')

    parser.add_argument('--register', dest='flash', action='store_false')

    parser.add_argument('--dry-run', action='store_true',
                        help='will show you what it would do')

    parser.add_argument('--list-jobs-and-quit', action='store_true')

    parser.add_argument('--results-file')
    parser.add_argument('--results-directory', default=DEFAULT_RESULTS_DIR)

    parser.add_argument('--show', action='store_true')

    parser.add_argument('--latex', action='store_true')

    parser.add_argument('--expand', '-x', action='count', default=1)

    parser.add_argument('-e', dest='show_measures', action='count', default=0)

    parser.add_argument('--tpr', nargs='*', default=[95], type=int)
    parser.add_argument('--tnr', action='store_true', help='Show TNR instead of FPR')

    parser.add_argument('--sort', nargs='+')

    parser.add_argument('--hide-average', action='store_false', dest='average')
    parser.add_argument('--only-average', action='store_true')

    parser.add_argument('--job-id', type=int, default=0)

    parser.add_argument('--sets', action='append', nargs='+')

    parser.add_argument('--last', nargs='?', const=10, default=0, type=int)

    parser.add_argument('--classification-methods', action=NewEntryDictofLists, nargs='+', default={})

    parser.add_argument('--ood-methods', nargs='*', default=None)

    parser.add_argument('--predict-methods', nargs='*', default=None)

    parser.add_argument('--remove-index', nargs='*')

    args, ra = parser.parse_known_args(argv)

    args.filters = DictOfListsOfParamFilters()

    filter_parser = create_filter_parser(parents=[parser])

    filter_args = filter_parser.parse_args(ra)

    filter_keys = get_filter_keys()

    for _ in filter_keys:
        args.filters.add(_, filter_args.__dict__[_])

    return args


def create_filter_parser(default_ini_file='utils/filters.ini', **kw):

    parser = argparse.ArgumentParser(**kw)

    config = configparser.ConfigParser()

    config.read(default_ini_file)

    types = config['type']

    dests = config['dest']

    abbrs = config['abbr']

    metavars = config['metavar']

    defaults = config['default']

    argnames = {}
    for k in types:

        argname = ['--' + k.replace('_', '-')]
        if k in abbrs:
            argname.append('-' + abbrs[k])
        argnames[k] = argname

    for k in types:

        if not types.get(k):
            argtype = str
        else:
            argtype = locate(types[k])
        nargs = '*'
        metavar = None
        # if argtype is bool:
        #     metavar = 'not'
        #     nargs = '?'
        if k in metavars:
            metavar = metavars[k]

        parser.add_argument(*argnames[k], dest=dests.get(k),
                            default=defaults.get(k),
                            type=argtype, nargs=nargs,
                            metavar=metavar, action=FilterAction)

    return parser


def add_filters_to_parsed_args(parser, args, ra):

    filter_parser_for_help = create_filter_parser(parents=[parser])
    filter_parser = create_filter_parser(add_help=False)

    filter_parser_for_help.parse_args()
    filter_args = filter_parser.parse_args(ra)

    args.filters = DictOfListsOfParamFilters()
    for _ in filter_args.__dict__:
        args.filters.add(_, filter_args.__dict__[_])

    return args.filters


class NewEntryDictofLists(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):

        if type(values) is not list:
            values = [values]

        if getattr(namespace, self.dest) == None:

            setattr(namespace, self.dest, {})

        d = getattr(namespace, self.dest)

        d[values[0]] = d.get(values[0], []) + values[1:]


if __name__ == '__main__':

    cli = '--upsampler vgg19 4 5'.split()
    arg = get_args_for_train(cli)

    print(arg.upsampler)

    # arg = get_args_for_test()
    # for k in arg.filters:
    # if not arg.filters[k].always_true:
    # print('{:20} {} of type {}'.format(k, arg.filters[k], arg.filters[k].type.__name__))

    # m = {'done': 4, 'job': 45, 'batch_norm': ['decoder', 'encoder'], 'type': 'cvae'}

    # for k, v in arg.filters.items():

    #     print(k, *v, *[f.filter(m[k]) for f in v])

    # print(match_filters(m, arg.filters))
