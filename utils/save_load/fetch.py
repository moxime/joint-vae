import os
from filelock import FileLock
import logging
import torch
import functools
from utils.print_log import turnoff_debug
from utils.filters import get_filter_keys, ParamFilter, DictOfListsOfParamFilters
from utils.parameters import gethostname
from utils.torch_load import get_same_size_by_name

from .misc import load_json, save_json
from .exceptions import NoModelError, StateFileNotFoundError
from .dictify import make_dict_from_model


class NoLock(object):

    lock_file = 'nolock'

    def __enter__(self):
        logging.info('Already lock, doing nothing')

    def __exit__(self, *args):
        pass


# def lock_models_file_in(arg):

#     def lock_models_file(func):

#         def modified_func(*a, **kw):
#             dir_path = a[arg]
#             lock = FileLock(os.path.join(dir_path, 'lock'))
#             logging.info('Acquiring lock in {}'.format(dir_path))
#             with lock:
#                 logging.info('Acquired lock in {}'.format(dir_path))
#                 return func(*a, **kw)

#         return modified_func

#     return lock_models_file


def iterable_over_subdirs(arg, iterate_over_subdirs=False, keep_none=False,
                          iterate_over_subdirs_if_found=False):
    def iterate_over_subdirs_wrapper(func):
        @functools.wraps(func)
        def iterated_func(*a, keep_none=keep_none, **kw):
            if isinstance(arg, str):
                directory = kw.get(arg)
            else:
                directory = a[arg]
            out = func(*a, **kw)

            if out is not None or keep_none:
                yield out
            try:
                rel_paths = os.listdir(directory)
                paths = [os.path.join(directory, p) for p in rel_paths]
                dirs = [d for d in paths if os.path.isdir(d)]
            except PermissionError:
                dirs = []
            if out is None or iterate_over_subdirs_if_found:
                for d in dirs:
                    if isinstance(arg, str):
                        kw[arg] = d
                    else:
                        a = list(a)
                        a[arg] = d
                    yield from iterated_func(*a, **kw)

        @functools.wraps(func)
        def wrapped_func(*a, iterate_over_subdirs=iterate_over_subdirs, **kw):

            if not iterate_over_subdirs:
                try:
                    return next(iter(iterated_func(*a, keep_none=True, **kw)))
                except StopIteration:
                    return
            if iterate_over_subdirs:
                return iterated_func(*a, **kw)
            else:
                return iterate_over_subdirs(iterated_func(*a, **kw))
        return wrapped_func

    return iterate_over_subdirs_wrapper


def _register_models(models, *keys):
    """Register the models in a dictionary that will be later
    recorded in a json file

    """
    d = {}
    for m in models:
        d[m['dir']] = {_: m[_] for _ in keys}

    return d


def load_model(d, **kw):
    """Load a model that can be either a CVAE, a WIM_Job or a wim_array

    """

    from cvae import ClassificationVariationalNetwork as M
    from module.wim import WIMJob as WJ
    from module.wim.array import WIMArray as WA

    if WA.is_wim_array(d):
        return WA.load(d, **kw)

    if WJ.is_wim(d):
        return WJ.load(d, **kw)

    return M.load(d, **kw)


def _collect_models(search_dir, registered_models_file=None):
    from cvae import ClassificationVariationalNetwork as M
    from module.wim import WIMJob as W

    lock = FileLock(os.path.join(search_dir, 'lock'))

    if not registered_models_file:
        registered_models_file = 'models-{}.json'.format(gethostname())

    try:
        rmodels = load_json(search_dir, registered_models_file)
    except FileNotFoundError:
        _ws = '{} not found, this will take time to register models'
        logging.warning(_ws.format(registered_models_file))
        rmodels = {}

    models_to_be_deleted = list(rmodels)
    models_to_be_registered = []

    n_models = 0

    for directory, _, files in os.walk(search_dir, followlinks=True):

        if 'params.json' in files and 'deleted' not in files:
            n_models += 1
            if directory in models_to_be_deleted:
                models_to_be_deleted.remove(directory)
            else:
                logging.debug(f'Loading net in: {directory}')
                if W.is_wim(directory):
                    model = W.load(directory, build_module=False, load_state=False)
                else:
                    model = M.load(directory, build_module=False, load_state=False)

                models_to_be_registered.append(make_dict_from_model(model, directory))

    logging.log(logging.INFO if models_to_be_deleted else logging.DEBUG,
                '{} models seem to have been deleted sincde last time'.format(len(models_to_be_deleted)))
    logging.log(logging.INFO if models_to_be_registered else logging.DEBUG,
                '{} models have to be registered'.format(len(models_to_be_registered)))

    for m in models_to_be_deleted:
        rmodels.pop(m)

    rkeys = get_filter_keys()

    rmodels.update(_register_models(models_to_be_registered, *rkeys))

    save_json(rmodels, search_dir, registered_models_file)

    return rmodels


def fetch_models(search_dir, registered_models_file=None, filter=None, flash=True,
                 light=False,
                 tpr=0.95,
                 build_module=False,
                 show_debug=False,
                 lock_file=True,
                 **kw):
    """Fetches models matching filter.

    Params:

    -- flash: if True, takes the models from registered_models_file.

    -- light: does not remake dictionay for models (faster)

    -- kw: args pushed to load function(eg. load_state)

    """

    logging.debug('Fetching models from {} (flash={})'.format(search_dir, flash))

    if lock_file:
        lock = FileLock(os.path.join(search_dir, 'lock-rmodels'))
    else:
        lock = NoLock()
    with lock:

        logging.info('Acquired lock on {}'.format(lock.lock_file))
        if not registered_models_file:
            registered_models_file = 'models-{}.json'.format(gethostname())
        if flash:
            logging.debug('Flash collecting networks in {}'.format(search_dir))
            try:
                rmodels = load_json(search_dir, registered_models_file)
                # logging.info('Json loaded from {}'.format(registered_models_file))
                with turnoff_debug(turnoff=not show_debug):
                    mlist = _gather_registered_models(rmodels, filter,
                                                      tpr=tpr, build_module=build_module,
                                                      light=light, **kw)
                logging.debug('Gathered {} models'.format(len(mlist)))
                logging.info('Releasing lock on {}'.format(lock.lock_file))
                return mlist

            except StateFileNotFoundError as e:
                raise e

            except FileNotFoundError as e:
                # except (FileNotFoundError, NoModelError) as e:
                logging.warning('{} not found, will recollect networks'.format(e.filename))
                flash = False

        if not flash:
            # logging.debug('Collecting networks in {}'.format(search_dir))
            with turnoff_debug(turnoff=not show_debug):
                rmodels = _collect_models(search_dir, registered_models_file)
                # logging.info('Collected {} models'.format(len(rmodels)))

            logging.info('Releasing lock on {}'.format(lock.lock_file))
            return fetch_models(search_dir, registered_models_file,
                                filter=filter, flash=True, light=light,
                                tpr=tpr, build_module=build_module, lock_file=False, **kw)


def _gather_registered_models(mdict, filter, tpr=0.95, wanted_epoch='last', light=False, **kw):
    from cvae import ClassificationVariationalNetwork as M
    from module.wim import WIMJob as W

    mlist = []
    n = 0
    for d in mdict:
        #        print('*** ***', mdict[d]['job'], mdict[d].get('wim_hash'))
        if filter is None or filter.filter(mdict[d]):
            n += 1
            logging.debug('Keeping {}'.format(d[-100:]))
            if not light:
                m = load_model(d, **kw)
                mlist.append(make_dict_from_model(m, d, tpr=tpr, wanted_epoch=wanted_epoch))
            else:
                mdict[d]['dir'] = d
                mlist.append(mdict[d])
            # if not n % 200:
            #     logging.info('Gathered {} models... (light={})'.format(n, light))
            # print('Keeping ({:8}) {:4} {}'.format(sys.getsizeof(mlist), n, d[-100:]))
        else:
            logging.debug('Not keeping {}'.format(d[-100:]))
    logging.debug('Gathered {} models'.format(len(mlist)))
    return mlist


def is_derailed(model, load_model_for_check=False):
    from cvae import ClassificationVariationalNetwork

    if isinstance(model, dict):
        directory = model['dir']

    elif isinstance(model, str):
        directory = model

    else:
        directory = model.saved_dir

    if os.path.exists(os.path.join(directory, 'derailed')):
        return True

    elif load_model_for_check:
        try:
            model = ClassificationVariationalNetwork.load(directory)
            if torch.cuda.is_available():
                model.to('cuda')
            x = torch.zeros(1, *model.input_shape, device=model.device)
            model.evaluate(x)
        except ValueError:
            return True

    return False


def find_by_job_number(*job_numbers, job_dir='jobs',
                       force_dict=False, **kw):

    job_filter = ParamFilter.from_string(' '.join(str(_) for _ in job_numbers), type=int)
    filter = DictOfListsOfParamFilters()
    filter.add('job', job_filter)

    d = {}
    models = fetch_models(job_dir, filter=filter, **kw)
    for m in models:
        d[m['job']] = m

    return d if len(job_numbers) > 1 or force_dict else d.get(job_numbers[0])


def needed_remote_files(*mdirs, epoch='last', which_rec='all',
                        state=False,
                        optimizer=False,
                        missing_file_stream=None):
    r""" list missing recorders to be fetched on a remote

    -- mdirs: list of directories

    -- epoch: last or min - loss or int

    -- which_rec: either 'none' 'ind' or 'all'

    -- state: wehter to include state.pth

    returns generator of needed files paths

    """

    assert not state or epoch == 'last'

    from cvae import ClassificationVariationalNetwork as M
    from module.wim import WIMJob as W

    for d in mdirs:

        logging.debug('Inspecting {}'.format(d))

        is_wim = W.is_wim(d)
        m = (W if is_wim else M).load(d, build_module=False)
        epoch_ = epoch
        if epoch_ == 'min-loss':
            epoch_ = m.training_parameters.get('early-min-loss', 'last')
        if epoch_ == 'last':
            epoch_ = max(m.testing) if m.predict_methods else max(m.ood_results or [0])

        if isinstance(epoch_, int):
            epoch_ = '{:04d}'.format(epoch_)

        testset = m.training_parameters['set']

        sets = []

        recs_to_exclude = which_rec.split('-')[1:]
        which_rec_ = which_rec.split('-')[0]

        if which_rec_ in ('all', 'ind'):
            sets.append(testset)
            if which_rec_ == 'all':
                if is_wim:
                    for s in m.wim_params['sets']:
                        sets.append(s)
                else:
                    sets += get_same_size_by_name(testset)
                    for _ in [_ for _ in recs_to_exclude if _ in sets]:
                        sets.remove(_)
        sub_dirs = ['']
        if is_wim:
            sub_dirs.append('init')
        for s in sets:
            for sub in sub_dirs:
                sfile = os.path.join(d, 'samples', epoch_, sub, 'record-{}.pth'.format(s))
                logging.debug('Looking for {}'.format(sfile))
                if not os.path.exists(sfile):
                    if missing_file_stream:
                        missing_file_stream.write(sfile + '\n')
                        yield d, sfile

        if state:
            sfile = os.path.join(d, 'state.pth')
            logging.debug('Looking for {}'.format(sfile))
            if not os.path.exists(sfile):
                if missing_file_stream:
                    missing_file_stream.write(sfile + '\n')
                yield d, sfile

        if optimizer:
            sfile = os.path.join(d, 'optimizer.pth')
            logging.debug('Looking for {}'.format(sfile))
            if not os.path.exists(sfile):
                if missing_file_stream:
                    missing_file_stream.write(sfile + '\n')
                yield d, sfile


def get_submodule(model, sub='features', job_dir='jobs', name=None, **kw):

    if isinstance(model, int):
        model_number = model
        logging.debug('Will find model {} in {}'.format(model_number, job_dir))
        model = find_by_job_number(model_number, job_dir=job_dir,
                                   build_module=True, load_state=True, **kw)['net']
        logging.debug('Had to search {} found model of type {}'.format(model_number, model.type))
        return get_submodule(model, sub=sub, job_dir=job_dir, name='job-{}'.format(model.job_number), **kw)

    elif isinstance(model, str) and model.startswith('job-'):
        number = int(model.split('-')[1])
        return get_submodule(number, sub=sub, job_dir=job_dir, **kw)

    elif isinstance(model, str) and os.path.exists(model):
        s = torch.load(model)
        s.name = name
        return s

    if not hasattr(model, sub):
        logging.error('Prb with model {}'.format(str(model)))
        raise AttributeError('model {} does not seem to have {}'.format(name or str(model), sub))

    logging.debug('Found {}.{}'.format(name, sub))

    s = getattr(model, sub).state_dict()
    s.name = name

    return s
