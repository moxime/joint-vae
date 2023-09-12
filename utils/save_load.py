import os
import json
import logging
import pandas as pd
import hashlib
import utils.torch_load as torchdl
import numpy as np
import random
import torch
from module.optimizers import Optimizer
import re
from utils.misc import make_list
from utils.torch_load import get_same_size_by_name, get_shape_by_name
from utils.roc_curves import fpr_at_tpr
from contextlib import contextmanager
import functools
from utils.print_log import turnoff_debug
from utils.filters import get_filter_keys, ParamFilter, DictOfListsOfParamFilters
from utils.parameters import gethostname


class NoModelError(Exception):
    pass


class StateFileNotFoundError(FileNotFoundError):
    pass


class DeletedModelError(NoModelError):
    pass


class MissingKeys(Exception):
    def __str__(self):
        return 'MissingKeys({})'.format(', '.join(e.args[-1]))


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
            if iterate_over_subdirs == True:
                return iterated_func(*a, **kw)
            else:
                return iterate_over_subdirs(iterated_func(*a, **kw))
        return wrapped_func

    return iterate_over_subdirs_wrapper


def get_path(dir_name, file_name, create_dir=True):

    dir_path = os.path.realpath(dir_name)
    if not os.path.exists(dir_path) and create_dir:
        os.makedirs(dir_path)

    return os.path.join(dir_name, file_name)


def job_to_str(number, string, formats={int: '{:06d}'}):
    job_format = formats.get(type(number), '{}')
    return string.replace('%j', job_format.format(number))


def create_file_for_job(number, directory, filename, mode='w'):

    directory = job_to_str(number, directory)

    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)

    return open(filepath, mode)


def save_json(d, dir_name, file_name, create_dir=True):

    p = get_path(dir_name, file_name, create_dir)

    with open(p, 'w') as f:
        json.dump(d, f)


def load_json(dir_name, file_name, presumed_type=str):

    p = get_path(dir_name, file_name, create_dir=False)

    # logging.debug('*** %s', p)
    with open(p, 'rb') as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError:
            logging.error('Corrupted file\n%s', p)
            with open('/tmp/corrupted', 'a') as f:
                f.write(p + '\n')
            return {}
    d_ = {}
    for k in d:
        try:
            k_ = presumed_type(k)
        except ValueError:
            k_ = k
        d_[k_] = d[k]

    return d_


def shorten_path(path, max_length=30):

    if len(path) > max_length:
        return (path[:max_length // 2 - 2] +
                '...' + path[-max_length // 2 + 2:])

    return path


def get_path_from_input(dir_path=os.getcwd(), count_nets=True):

    rel_paths = os.listdir(dir_path)
    abs_paths = [os.path.join(dir_path, d) for d in rel_paths]
    sub_dirs_rel_paths = [rel_paths[i] for i, d in enumerate(abs_paths) if os.path.isdir(d)]
    print(f'<enter>: choose {dir_path}', end='')
    if count_nets:
        list_of_nets = collect_models(dir_path, load_net=False)
        num_of_nets = len(list_of_nets)
        print(f' ({num_of_nets} networks)')
    else:
        print()

    for i, d in enumerate(sub_dirs_rel_paths):
        print(f'{i+1:2d}: enter {d}', end='')
        if count_nets:
            list_of_nets = collect_models(os.path.join(dir_path, d), load_net=False)
            num_of_nets = len(list_of_nets)
            print(f' ({num_of_nets} networks)')
        else:
            print()

    print(' p: return to ..')
    input_string = input('Your choice: ')
    try:
        i = int(input_string)
        is_int = True
    except ValueError:
        i = input_string
        is_int = False

    if is_int:
        if 0 < i < len(sub_dirs_rel_paths) + 1:
            return get_path_from_input(dir_path=os.path.join(dir_path,
                                                             sub_dirs_rel_paths[i - 1]))
        else:
            return get_path_from_input(dir_path)
    elif i == '':
        return dir_path
    elif i == 'p':
        path = os.path.join(dir_path, os.pardir)
        path = os.path.abspath(path)
        return get_path_from_input(path)
    else:
        return get_path_from_input(dir_path)


def model_directory(model, *subdirs):

    if isinstance(model, str):
        directory = model

    elif isinstance(model, dict):
        directory = model['dir']

    else:
        directory = model.saved_dir

    return os.path.join(directory, *subdirs)


class ObjFromDict:

    def __init__(self, d, **defaults):
        for k, v in defaults.items():
            setattr(self, k, v)
        for k, v in d.items():
            setattr(self, k, v)


def print_architecture(o, sigma=False, sampling=False,
                       excludes=[], short=False):

    arch = ObjFromDict(o.architecture, features=None)
    training = ObjFromDict(o.training_parameters)

    # for d, _d in zip((o.architecture, o.training_parameters), ('ARCH', 'TRAIN')):
    #     print('\n\n***', _d, '***')
    #     print('\n'.join('*** {:18} : {}'.format(*_) for _ in d.items()))

    def _l2s(l, c='-', empty='.'):
        if l:
            return c.join(str(_) for _ in l)
        return empty

    def s_(s):
        return s[0] if short else s

    if arch.features:
        features = arch.features
    s = ''
    if 'type' not in excludes:

        s += s_('type') + f'={arch.type}--'
    if 'activation' not in excludes:
        if arch.type != 'vib':
            s += s_('output') + f'={arch.output_activation}--'
        s += s_('activation') + f'={arch.activation}--'
    if 'latent_dim' not in excludes:
        s += s_('latent-dim') + f'={arch.latent_dim}--'
    # if sampling:
    #    s += f'sampling={self.latent_sampling}--'
    if arch.features:
        s += s_('features') + f'={features}--'
    if 'batch_norm' not in excludes:
        w = '-' + arch.batch_norm if arch.batch_norm else ''
        s += f'batch-norm{w}--' if arch. batch_norm else ''

    s += s_('encoder') + f'={_l2s(arch.encoder)}--'
    if 'decoder' not in excludes:
        s += s_('decoder') + f'={_l2s(arch.decoder)}--'
        if arch.upsampler:
            s += s_('upsampler') + f'={arch.upsampler}--'
    s += s_('classifier') + f'={_l2s(arch.classifier)}--'

    # TK s += s_('variance') + f'={arch.latent_prior_variance:.1f}'

    if sigma and 'sigma' not in excludes:
        s += '--' + s_('sigma') + f'={o.sigma}'

    if sampling and 'sampling' not in excludes:
        s += '--'
        s += s_('sampling')
        s += f'={training.latent_sampling}'

    return s


def option_vector(o, empty=' ', space=' '):

    arch = ObjFromDict(o.architecture, features=None)
    training = ObjFromDict(o.training_parameters, transformer='default', warmup_gamma=(0, 0))
    v_ = []
    if arch.features:
        w = ''
        w += 'p:'
        if training.pretrained_features:
            w += 'f'
        else:
            w += empty

        if arch.upsampler:
            if training.pretrained_upsampler:
                w += 'u'
            else:
                w += empty
        v_.append(w)

    w = 't:' + training.transformer[0]
    v_.append(w)

    # w = 'bn:'
    # if not arch.batch_norm:
    #     c = empty
    # else:
    #     # print('****', self.batch_norm)
    #     c = arch.batch_norm[0]
    # w += c
    # v_.append(w)

    w = 'a:'
    for m in ('flip', 'crop'):
        if m in training.data_augmentation:
            w += m[0]
        else:
            w += empty
    v_.append(w)

    w = 'w:'
    if training.warmup[-1]:
        w += f'{training.warmup[0]:02.0f}--{training.warmup[1]:02.0f}'
    else:
        w += 2 * empty

    if training.warmup_gamma[-1]:
        w += '-{}:{:.0f}--{:.0f}'.format(chr(947), *training.warmup_gamma)

    v_.append(w)

    # w = 'p:'
    # if arch.prior.get('learned_means'):
    #     w += 'l'
    # elif arch.prior.get('init_mean') == 'onehot':
    #     w += '1'
    # elif arch.type in ('cvae', 'xvae'):
    #     w += 'r'

    # v_.append(w)

    return space.join(v_)


class Shell:

    print_architecture = print_architecture
    option_vector = option_vector


class LossRecorder:

    def __init__(self,
                 batch_size,
                 num_batch=0,
                 device=None,
                 **tensors):

        self.last_batch_size = None
        self._seed = None

        self._num_batch = 0
        self._samples = 0

        self.batch_size = batch_size
        self.reset()

        self._tensors = {}

        self.device = device

        if tensors:
            self._create_tensors(num_batch, device=device, **tensors)

    def _create_tensors(self, num_batch, device=None, **tensors):

        assert not self._tensors
        self._num_batch = num_batch
        self._samples = num_batch * self.batch_size

        if not device and not self.device:
            device = next(iter(tensors.values())).device

        self.device = device

        for k, t in tensors.items():
            shape = t.shape[:-1] + (self._samples,)
            self._tensors[k] = torch.zeros(shape,
                                           dtype=t.dtype,
                                           device=self.device)
        self.last_batch_size = self.batch_size

    def to(self, device):

        for t in self._tensors:
            self._tensors[t] = self._tensors[t].to(device)

    def reset(self, seed=False):

        self._recorded_batches = 0
        if self._seed is None or seed:
            self._seed = np.random.randint(1, int(1e8))
        self.last_batch_size = self.batch_size
        return

    def init_seed_for_dataloader(self):

        self._initial_seed = torch.seed()
        seed = self._seed
        torch.manual_seed(seed)

    def restore_seed(self):
        torch.manual_seed(self._initial_seed)

    def keys(self):
        return self._tensors.keys()

    def __len__(self):
        return self._recorded_batches

    def __repr__(self):
        return ('Recorder for '
                + ' '.join([str(k) for k in self.keys()]))

    def __getitem__(self, k):

        return self._tensors[k]

    def __iter__(self):

        return iter(self._tensors)

    def save(self, file_path, cut=True):
        """dict_ = self.__dict__.copy()
        tensors = dict.pop('_tensors')
        """

        if cut:
            self.num_batch = len(self)
            t = self._tensors
            for k in t:
                end = (self.num_batch - 1) * self.batch_size + self.last_batch_size
                t[k] = t[k][..., 0:end]

        torch.save(self.__dict__, file_path)

    @classmethod
    def load(cls, file_path, device=None, **kw):

        if 'map_location' not in kw and not torch.cuda.is_available():
            kw['map_location'] = torch.device('cpu')

        dict_of_params = torch.load(file_path, **kw)
        num_batch = dict_of_params['_num_batch']
        batch_size = dict_of_params['batch_size']
        tensors = dict_of_params['_tensors']

        r = LossRecorder(batch_size, num_batch, **tensors)

        for k in ('_seed', '_tensors', '_recorded_batches'):
            setattr(r, k, dict_of_params[k])

        for k in dict_of_params:
            if not k.startswith('_'):
                setattr(r, k, dict_of_params[k])

        # retro-compatibility
        if 'last_batch_size' in dict_of_params and isinstance(r.last_batch_size, dict):
            last_batch_size = next(iter(r.last_batch_size.values()))
            r.last_batch_size = last_batch_size

        if device:
            for k in r._tensors:
                if r._tensors[k].device != device:
                    r._tensors[k] = r._tensors[k].to('cpu')
        return r

    @classmethod
    def loadall(cls, dir_path, *w, file_name='record-{w}.pth', output='recorders', **kw):
        r"""
        If w is empty will find all recorders in directory
        if output is 'recorders' return recorders, if 'paths' return full paths

        """

        def outputs(p): return LossRecorder.load(path, **kw) if output.startswith('record') else p

        r = {}

        if not w:

            pattern = file_name.replace('.', '\.')
            pattern = pattern.replace('{w}', '(?P<name>.+)')

            for f in os.listdir(dir_path):
                regexp_match = re.match(pattern, f)
                if regexp_match:
                    path = os.path.join(dir_path, f)
                    r[regexp_match.group('name')] = outputs(path)

        for word in w:
            f = file_name.format(w=word)
            path = os.path.join(dir_path, f)
            if os.path.exists(path):
                r[word] = outputs(path)
            else:
                logging.warning(f'{f} not found')

        return r

    @property
    def num_batch(self):
        return self._num_batch

    @num_batch.setter
    def num_batch(self, n):

        if not self._tensors:
            return

        first_tensor = next(iter(self._tensors.values()))
        height = first_tensor.shape[-1]
        n_sample = n * self.batch_size

        if n_sample > height:
            d_h = n_sample - height
            for k in self._tensors:

                t = self._tensors[k]
                # print('sl353:', 'rec', self.device, k, t.device)
                z = torch.zeros(t.shape[:-1] + (d_h,),
                                dtype=t.dtype,
                                device=self.device)
                self._tensors[k] = torch.cat([t, z], axis=-1)

        self._num_batch = n
        self._samples = n * self.batch_size
        self._recorded_batches = min(n, self._recorded_batches)

    def has_batch(self, number, only_full=False):
        r""" number starts at 0
        """
        if number == self._recorded_batches - 1:
            return not only_full or self.last_batch_size == self.batch_size
        return number < self._recorded_batches

    def get_batch(self, i, *which, device=None):

        if not which:
            return self.get_batch(i, *self.keys())

        if len(which) > 1:
            return {w: self.get_batch(i, w) for w in which}

        if not self.has_batch(i):
            raise IndexError(f'{i} >= {len(self)}')

        start = i * self.batch_size

        w = which[0]
        if i == len(self) - 1:
            end = start + self.last_batch_size
        else:
            end = start + self.batch_size

        t = self._tensors[w]
        if device:
            t = t.to(device)

        return t[..., start:end]

    def append_batch(self, extend=True, **tensors):

        if not self._tensors:
            self._create_tensors(1, **tensors)

        start = self._recorded_batches * self.batch_size
        end = start + self.batch_size

        if end > self._samples:
            if extend:
                self.num_batch *= 2
            else:
                raise IndexError

        batch_sizes = set(tensors[k].shape[-1] for k in tensors)
        assert len(batch_sizes) == 1, 'all batches have to be of same size'
        batch_size = batch_sizes.pop()
        if batch_size < self.batch_size:
            assert self.last_batch_size == self.batch_size
            self.last_batch_size = batch_size
        end = start + batch_size

        for k in tensors:
            if k not in self.keys():
                raise KeyError(k)
            self._tensors[k][..., start:end] = tensors[k]

        self._recorded_batches += 1


def last_samples(model):

    directory = model_directory(model, 'samples')

    samples = [int(d) for d in os.listdir(directory) if d.isnumeric()]

    return max(samples)


def average_ood_results(ood_results, *oodsets):

    ood = [s for s in ood_results if not s.endswith('90')]
    if oodsets:
        ood = [s for s in ood if s in oodsets]

    mean_keys = {'auc': 'val', 'fpr': 'list'}
    min_keys = {'epochs': 'val', 'n': 'val'}
    same_keys = {'tpr', 'thresholds'}

    all_methods = [set(ood_results[s].keys()) for s in ood]
    if all_methods:
        methods = set.intersection(*[set(ood_results[s].keys()) for s in ood])

    else:
        return None

    avge_res = {m: {} for m in methods}

    for m in methods:
        for k in mean_keys:
            if mean_keys[k] == 'val':
                vals = [ood_results[s][m][k] for s in ood]
                avge_res[m][k] = np.mean(vals)
            else:
                lists = [ood_results[s][m][k] for s in ood]
                n = min(len(l) for l in lists)
                avge_res[m][k] = [np.mean([l_[i] for l_ in lists]) for i in range(n)]

        for k in min_keys:
            avge_res[m][k] = min(ood_results[s][m][k] for s in ood)

        for k in same_keys:
            avge_res[m][k] = ood_results[ood[0]][m][k]

    return avge_res


def clean_results(results, methods, **zeros):

    trimmed = {k: results[k] for k in results if k in methods}
    completed = {k: dict(n=0, epochs=0, **zeros) for k in methods}
    completed.update(trimmed)
    return completed


def develop_starred_methods(methods, methods_params, inplace=True):

    if not inplace:
        methods = methods.copy()
    starred_methods = []
    for m in methods:
        if m.endswith('*'):
            methods += methods_params.get(m[:-1], [])
            starred_methods.append(m)

    for m in starred_methods:
        methods.remove(m)
        pass

    return methods


def needed_components(*methods):

    total = ('loss', 'logpx', 'sum', 'max', 'mag', 'std', 'mean')
    ncd = {'iws': ('iws',),
           'softiws': ('iws',),
           'closest': ('zdist',),
           'kl': ('kl',),
           'soft': ('kl',),
           'mse': ('cross_x',)}

    ncd.update({_: (_,) for _ in ('kl', 'fisher_rao', 'mahala', 'kl_rec')})
    ncd.update({'soft' + _: (_,) for _ in ('kl', 'mahala', 'zdist')})

    for k in total:
        ncd[k] = ('total',)

    methods_ = [_.split('-')[0] for _ in methods]
    #    for m in methods:
    # if m.endswith('-2s'):
    #     methods_.append(m[:-3])
    # elif '-a-' in m:
    #     methods_.append(m.split('-')[0])
    # else:
    #     methods_.append(m)
    return sum((ncd.get(m, ()) for m in methods_), ())


def available_results(model,
                      testset='trained',
                      min_samples_by_class=200,
                      samples_available_by_class=800,
                      predict_methods='all',
                      misclass_methods='all',
                      oodsets='all',
                      wanted_epoch='last',
                      epoch_tolerance=5,
                      where='all',
                      ood_methods='all'):

    if isinstance(model, dict):
        model = model['net']

    ood_results = model.ood_results
    test_results = model.testing
    if wanted_epoch == 'min-loss':
        wanted_epoch = model.training_parameters.get('early-min-loss', 'last')
    if wanted_epoch == 'last':
        wanted_epoch = max(model.testing) if model.predict_methods else max(model.ood_results or [0])
    predict_methods = make_list(predict_methods, model.predict_methods)
    ood_methods = make_list(ood_methods, model.ood_methods)
    misclass_methods = make_list(misclass_methods, model.misclass_methods)

    anywhere = ('json', 'recorders', 'compute')
    where = make_list(where, anywhere)

    for _l in (predict_methods, ood_methods, misclass_methods):
        develop_starred_methods(_l, model.methods_params)

    if testset == 'trained':
        testset = model.training_parameters['set']
    # print('***', testset)
    # print('*** testset', testset)
    all_ood_sets = get_same_size_by_name(testset)

    if ood_methods:
        oodsets = make_list(oodsets, all_ood_sets)
    else:
        oodsets = []

    sets = [testset] + oodsets

    min_samples = {}
    samples_available_by_compute = {}

    for s in sets:
        C = get_shape_by_name(s)[-1]
        if not C:
            C = model.architecture['num_labels']
        min_samples[s] = C * min_samples_by_class
        samples_available_by_compute[s] = C * samples_available_by_class

    # print(*min_samples.values())
    # print(*samples_available_by_compute.values())

    methods = {testset: [(m,) for m in predict_methods]}
    methods[testset] += [(pm, mm) for mm in misclass_methods for pm in predict_methods]
    methods[testset] += [(m, ) for m in ood_methods]
    methods.update({s: [(m,) for m in ood_methods] for s in oodsets})

    sample_dir = os.path.join(model.saved_dir, 'samples')

    if os.path.isdir(sample_dir):
        sample_sub_dirs = {int(_): _ for _ in os.listdir(sample_dir) if _.isnumeric()}
    else:
        sample_sub_dirs = {}

    epochs = set(sample_sub_dirs)

    epochs.add(model.trained)
    # print('****', *epochs, '/', *test_results, '/', *ood_results)
    epochs = sorted(set.union(epochs, set(test_results), set(ood_results)))

    if wanted_epoch:
        epochs = [_ for _ in epochs if abs(_ - wanted_epoch) <= epoch_tolerance]
    test_results = {_: clean_results(test_results.get(_, {}), predict_methods) for _ in epochs}

    results = {}

    for e in sorted(epochs):
        pm_ = list(test_results[e].keys())
        results[e] = {s: clean_results(ood_results.get(e, {}).get(s, {}), ood_methods) for s in sets}
        for pm in pm_:
            misclass_results = clean_results(test_results[e][pm], misclass_methods)
            test_results[e].update({pm + '-' + m: misclass_results[m] for m in misclass_results})
        results[e][testset].update({m: test_results[e][m] for m in test_results[e]})

    available = {e: {s: {'json': {m: results[e][s][m]['n']
                                  for m in results[e][s]}}
                     for s in results[e]}
                 for e in results}

    # print(available['json'])

    for e in available:
        for s in available[e]:
            for w in ('recorders', 'compute'):
                available[e][s][w] = {'-'.join(m): 0 for m in methods[s]}

    for epoch in results:
        rec_dir = os.path.join(sample_dir, sample_sub_dirs.get(epoch, 'false_dir'))
        if os.path.isdir(rec_dir):
            recorders = LossRecorder.loadall(rec_dir, map_location='cpu')
            # epoch = last_samples(model)
            for s, r in recorders.items():
                # print('***', s)
                if s not in sets:
                    continue
                n = len(r) * r.batch_size
                for m in methods[s]:
                    all_components = all(c in r.keys() for c in needed_components(*m))
                    if all_components:
                        available[epoch][s]['recorders']['-'.join(m)] = n
                        available[epoch]['rec_dir'] = rec_dir

    if abs(wanted_epoch - model.trained) <= epoch_tolerance:
        for s in sets:
            for m in methods[s]:
                available[model.trained][s]['compute']['-'.join(m)] = samples_available_by_compute[s]

    # return available

    wheres = [w for w in ['compute', 'recorders', 'json'] if w in where]
    wheres.append('zeros')
    for epoch in available:
        for dset in sets:
            a_ = available[epoch][dset]
            a_['where'] = {w: 0 for w in anywhere}
            a_['zeros'] = {'-'.join(m): 0 for m in methods[dset]}
            # print(epoch, dset) # a_['json'])
            for i, w in enumerate(wheres[:-1]):
                gain = {'-'.join(m): 0 for m in methods[dset]}
                others = {'-'.join(m): 0 for m in methods[dset]}
                for m in gain:
                    others[m] = max(a_[_].get(m, 0) for _ in wheres[i + 1:])
                    gain[m] += a_[w].get(m, 0) - others[m] > min_samples[dset]
                    # gain[m] *= (gain[m] > 0)
                available[epoch][dset]['where'][w] = sum(gain.values())
            a_.pop('zeros')

    for epoch in available:
        available[epoch]['all_sets'] = {w: sum(available[epoch][s]['where'][w] for s in sets) for w in anywhere}
        available[epoch]['all_sets']['anywhere'] = sum(available[epoch]['all_sets'][w] for w in anywhere)
    return available


def make_dict_from_model(model, directory, tpr=0.95, wanted_epoch='last', misclass_on_method='first',
                         oodsets=None,
                         **kw):

    try:
        iter(tpr)
    except TypeError:
        tpr = [tpr]

    architecture = ObjFromDict(model.architecture, features=None)
    # training = ObjFromDict(model.training_parameters)
    training = ObjFromDict(model.training_parameters, transformer='default', warmup_gamma=(0, 0))

    logging.debug(f'net found in {shorten_path(directory)}')
    arch = model.print_architecture(excludes=('latent_dim', 'batch_norm'))
    arch_code = hashlib.sha1(bytes(arch, 'utf-8')).hexdigest()[:6]
    # arch_code = hex(hash(arch))[2:10]
    pretrained_features = (None if not architecture.features
                           else training.pretrained_features)
    pretrained_upsampler = training.pretrained_upsampler
    batch_size = training.batch_size
    if not batch_size:
        train_batch_size = training.max_batch_sizes['train']
    else:
        train_batch_size = batch_size

    model.testing[-1] = {}
    if wanted_epoch == 'min-loss':
        if 'early-min-loss' in model.training_parameters:
            wanted_epoch = model.training_parameters['early-min-loss']
        else:
            logging.warning('Min loss epoch had not been computed for %s. Will fecth last', model.trained)
            wanted_epoch = 'last'

    if wanted_epoch == 'last':
        wanted_epoch = max(model.testing) if model.predict_methods else max(model.ood_results or [0])

    testing_results = clean_results(model.testing.get(wanted_epoch, {}), model.predict_methods, accuracy=0.)
    accuracies = {m: testing_results[m]['accuracy'] for m in testing_results}
    ood_results = model.ood_results.get(wanted_epoch, {}).copy()
    training_set = model.training_parameters['set']

    encoder_forced_variance = architecture.encoder_forced_variance
    if not encoder_forced_variance:
        encoder_forced_variance = None

    if training_set in ood_results:
        ood_results.pop(training_set)

    if model.testing.get(wanted_epoch) and model.predict_methods:
        # print('*** model.testing', *model.testing.keys())
        # print('*** model.predict_methods', model.architecture['type'], *model.predict_methods)
        accuracies['first'] = accuracies[model.predict_methods[0]]
        best_accuracy = max(testing_results[m]['accuracy'] for m in testing_results)
        tested_epoch = min(testing_results[m]['epochs'] for m in testing_results)
        n_tested = min(testing_results[m]['n'] for m in testing_results)
    else:
        best_accuracy = accuracies['first'] = None
        tested_epoch = n_tested = 0

    parent_set, heldout = torchdl.get_heldout_classes_by_name(training_set)

    if heldout:
        # print('***', *heldout, '***', *model.ood_results)
        matching_ood_sets = [k for k in ood_results if k.startswith(parent_set)]
        if matching_ood_sets:
            ood_results[parent_set + '+?'] = ood_results.pop(matching_ood_sets[0])
        all_ood_sets = [parent_set + '+?']

    else:
        all_ood_sets = torchdl.get_same_size_by_name(training_set)

    heldout = tuple(sorted(heldout))

    average_ood = average_ood_results(ood_results, *all_ood_sets)
    if average_ood:
        ood_results['average*'] = average_ood

    if oodsets:
        oodsets = [_ for _ in oodsets if 'average' not in _]

        average_ood = average_ood_results(ood_results, *oodsets)
        if average_ood:
            ood_results['average'] = average_ood

    all_ood_sets.append('average')
    all_ood_sets.append('average*')

    tested_ood_sets = [s for s in ood_results if s in all_ood_sets]

    methods_for_in_out_rates = {s: model.ood_methods.copy() for s in tested_ood_sets}
    in_out_results = {_: ood_results[_] for _ in tested_ood_sets}

    if model.misclass_methods:
        for pm in accuracies:
            pm_ = pm
            if pm == 'first':
                pm_ = model.predict_methods[0]
            prefix = 'errors-'

            if pm_ in model.testing.get(wanted_epoch, {}):
                in_out_results[prefix + pm] = model.testing.get(wanted_epoch, {}).get(pm_, None)
                in_out_results[prefix + pm]['acc'] = accuracies[pm]
                methods_for_in_out_rates[prefix + pm] = model.misclass_methods.copy()

    # TO MERGE WITH MISCLASS
    # res_fmt: {'fpr': {0.9:.., 0.91:...}, 'P': {0.9:.., 0.91:...}, 'auc': 0.9}
    in_out_rates = {s: {} for s in in_out_results}
    in_out_rate = {s: None for s in in_out_results}
    best_auc = {s: None for s in in_out_results}
    best_method = {s: None for s in in_out_results}
    n_in_out = {s: 0 for s in in_out_results}
    epochs_in_out = {s: 0 for s in in_out_results}

    for s in in_out_results:
        res_by_set = {}

        starred_methods = [m for m in methods_for_in_out_rates[s] if m.endswith('*')]
        first_method = methods_for_in_out_rates[s][0]
        develop_starred_methods(methods_for_in_out_rates[s], model.methods_params)

        in_out_results_s = clean_results(in_out_results[s],
                                         methods_for_in_out_rates[s] + starred_methods,
                                         fpr=[], tpr=[], precision=[], auc=None, acc=None)
        _r = in_out_results[s]
        for m in starred_methods:
            methods_to_be_maxed = {m_: fpr_at_tpr(_r[m_]['fpr'], _r[m_]['tpr'], tpr[0])
                                   for m_ in _r if m_.startswith(m[:-1]) and _r[m_]['auc']}
            params_max_auc = min(methods_to_be_maxed, key=methods_to_be_maxed.get, default=None)

            if params_max_auc is not None:
                in_out_results_s[m] = _r[params_max_auc].copy()
                in_out_results_s[m]['params'] = params_max_auc

        for m in in_out_results_s:
            res_by_method = {}
            fpr_ = in_out_results_s[m]['fpr']
            tpr_ = in_out_results_s[m]['tpr']
            P_ = in_out_results_s[m].get('precision', [None for _ in tpr_])
            auc = in_out_results_s[m]['auc']

            if auc and (not best_auc[s] or auc > best_auc[s]):
                best_auc[s] = auc
                best_method[s] = m

            for target_tpr in tpr:
                for (the_tpr, fpr, P) in zip(tpr_, fpr_, P_):
                    if abs(the_tpr - target_tpr) < 1e-4:
                        break
                else:
                    the_tpr = None

                if the_tpr:
                    suffix = '@{:.0f}'.format(100 * target_tpr)
                    res_by_method['fpr' + suffix] = fpr
                    res_by_method['auc'] = auc
                    if P is not None:
                        res_by_method['P' + suffix] = P
                        # print('***', in_out_results_s[m].keys())
                        # res_by_method['dP'] = P - in_out_results_s[m]['acc']
                    # if params := in_out_results_s[m].get('params'):
                    #     res_by_method['params'] = params
            res_by_set[m] = res_by_method

        res_by_set['first'] = res_by_set[first_method]
        in_out_rates[s] = res_by_set
        if best_method[s]:
            in_out_rate[s] = res_by_set[best_method[s]]

        epochs_in_out[s] = min(in_out_results_s[m]['epochs'] for m in in_out_results_s)
        n_in_out[s] = min(in_out_results_s[m]['n'] for m in in_out_results_s)

    history = model.train_history.get(wanted_epoch, {})
    if history.get('test_measures', {}):
        mse = history['test_measures'].get('mse', np.nan)
        rmse = np.sqrt(mse)
        dB = history['test_measures'].get('dB', np.nan)
    else:
        rmse = np.nan
        dB = np.nan

    loss_ = {}
    for s in ('train', 'test'):
        loss_[s] = {_: np.nan for _ in ('zdist', 'total', 'iws', 'kl')}
        loss_[s].update(history.get(s + '_loss', {}))

    num_dims = np.prod(model.architecture['input_shape'])
    nll = -loss_['test']['iws'] / np.log(2) / num_dims

    kl = loss_['test']['kl']

    if architecture.type in ('cvae', 'xvae'):
        C = model.architecture['num_labels']
        nll += np.log2(C) / num_dims

    has_validation = 'validation_loss' in history
    validation = model.training_parameters.get('validation', 0)
    sigma = model.sigma
    beta = model.training_parameters['beta']
    if sigma.learned and not sigma.coded:
        sigma_train = 'learned'
        beta_sigma = sigma.value * np.sqrt(beta)
    elif sigma.coded:
        sigma_train = 'coded'
        beta_sigma = sigma.value * np.sqrt(beta)
    elif sigma.is_rmse:
        sigma_train = 'rmse'
        beta_sigma = rmse * np.sqrt(beta)
    elif sigma.decay:
        sigma_train = 'decay'
        beta_sigma = rmse * np.sqrt(beta)
    else:
        sigma_train = 'constant'
        beta_sigma = sigma.value

    sigma_size = 'S' if sigma.sdim == 1 else 'M'

    prior_params = architecture.prior
    latent_prior_distribution = prior_params['distribution']

    latent_prior_variance = prior_params['var_dim']

    latent_prior = latent_prior_distribution[:4] + '-'

    if architecture.type in ('cvae', 'xvae'):
        learned_prior_means = prior_params['learned_means']
        latent_means = prior_params['init_mean']
        if latent_means == 'onehot':
            latent_prior += '1'
            latent_init_means = 1
        elif learned_prior_means:
            latent_init_means = latent_means
            latent_means = 'learned'
            latent_prior += 'l'
        else:
            latent_init_means = latent_means
            latent_means = 'random'
            latent_prior += 'r'
        latent_prior += '-'
    else:
        latent_means = None
        learned_prior_means = False
        latent_init_means = 0.

    latent_prior += latent_prior_variance[0]

    empty_optimizer = Optimizer([torch.nn.Parameter()], **training.optimizer)

    try:
        class_width = sum(architecture.classifier)
        class_type = 'linear'
    except TypeError:
        class_width = 0
        class_type = 'softmax'

    width = (architecture.latent_dim +
             sum(architecture.encoder) +
             sum(architecture.decoder) +
             class_width)

    depth = (1 + len(architecture.encoder)
             + len(architecture.decoder)
             + len(architecture.classifier) if class_type == 'linear' else 0)

    # print('TBR', architecture.type, model.job_number, *loss_['test'].keys())

    rec_dir = os.path.join(directory, 'samples', 'last')
    if os.path.exists(rec_dir):
        recorders = LossRecorder.loadall(rec_dir,
                                         output='paths')
    else:
        recorders = {}
    if recorders:
        recorded_epoch = last_samples(directory)
    else:
        recorded_epoch = None

    try:
        wim = model.wim_params
    except AttributeError:
        wim = {}

    wim_sets = '-'.join(sorted(wim['sets'])) if wim.get('sets') else None
    wim_prior = wim.get('distribution')
    wim_from = wim.get('from')

    finished = model.train_history['epochs'] >= model.training_parameters['epochs']
    return {'net': model,
            'job': model.job_number,
            'is_resumed': model.is_resumed,
            'type': architecture.type,
            'arch': arch,
            'output_distribution': architecture.output_distribution,
            'activation': architecture.activation,
            'activation_str': architecture.activation[:4],
            'output_activation': architecture.output_activation,
            'output_activation_str': architecture.output_activation[:3],
            'prior_distribution': latent_prior_distribution,
            'tilted_tau': architecture.prior['tau'] if latent_prior_distribution == 'tilted' else None,
            'learned_prior_means': learned_prior_means,
            'latent_prior_variance': latent_prior_variance,
            'latent_prior_means': latent_means,
            'latent_prior_init_means': latent_init_means,
            'prior': latent_prior,
            'encoder_forced_variance': encoder_forced_variance,
            'gamma': model.training_parameters['gamma'],
            'arch_code': arch_code,
            'features': architecture.features or 'none',
            'upsampler': architecture.upsampler or 'none',
            'dir': directory,
            'heldout': heldout,  # tuple(sorted(heldout)),
            'h/o': ','.join(str(_) for _ in heldout),
            'set': parent_set + ('-?' if heldout else ''),
            'rep': architecture.representation,
            # 'parent_set': parent_set,
            'data_augmentation': training.data_augmentation,
            'transformer': training.transformer,
            'train_batch_size': train_batch_size,
            'sigma': sigma.value if sigma_train == 'constant' else np.nan,
            'beta_sigma': beta_sigma,
            'sigma_train': sigma_train,  # [:5],
            'beta': beta,
            'done': model.train_history['epochs'],
            'epochs': model.training_parameters['epochs'],
            'has_validation': has_validation,
            'validation': validation,
            'trained': model.train_history['epochs'] / model.training_parameters['epochs'],
            'full_test_every': model.training_parameters['full_test_every'],
            'finished': finished,
            'n_tested': n_tested,
            'epoch': wanted_epoch,
            'accuracies': accuracies,
            'best_accuracy': best_accuracy,
            'n_in_out': n_in_out,
            'in_out_rates': in_out_rates,
            'in_out_rate': in_out_rate,
            'recorders': recorders,
            'recorded_epoch': recorded_epoch,
            'nll': nll,
            'dB': dB,
            'kl': kl,
            'rmse': rmse,
            'test_loss': loss_['test']['total'],
            'train_loss': loss_['train']['total'],
            'test_zdist': np.sqrt(loss_['test']['zdist']),
            'train_zdist': np.sqrt(loss_['train']['zdist']),
            'K': architecture.latent_dim,
            'L': training.latent_sampling,
            'l': architecture.test_latent_sampling,
            'warmup': training.warmup[-1],
            'warmup_gamma': training.warmup_gamma[-1],
            'wim_sets': wim_sets,
            'wim_prior': wim_prior,
            'wim_alpha': wim.get('alpha'),
            'wim_epochs': wim.get('epochs'),
            'wim_from': wim.get('from'),
            'pretrained_features': str(pretrained_features),
            'pretrained_upsampler': str(pretrained_upsampler),
            'batch_norm': architecture.batch_norm or None,
            'depth': depth,
            'width': width,
            'classif_type': class_type,
            'options': model.option_vector(),
            'optim_str': f'{empty_optimizer:3}',
            'optim': empty_optimizer.kind,
            'lr': empty_optimizer.init_lr,
            'version': architecture.version
            }


def _register_models(models, *keys):
    """
    Register the models in a dictionary that will be later recorded in a json file

    """
    d = {}
    for m in models:
        d[m['dir']] = {_: m[_] for _ in keys}

    return d


def fetch_models(search_dir, registered_models_file=None, filter=None, flash=True,
                 tpr=0.95,
                 load_net=False,
                 show_debug=False,
                 **kw):
    """Fetches models matching filter.

    Params:

    -- flash: if True, takes the models from registered_models_file.
    -- kw: args pushed to load function (eg. load_state)

    """
    if not registered_models_file:
        registered_models_file = 'models-{}.json'.format(gethostname())
    if flash:
        logging.debug('Flash collecting networks')
        try:
            rmodels = load_json(search_dir, registered_models_file)
            with turnoff_debug(turnoff=not show_debug):
                return _gather_registered_models(rmodels, filter, tpr=tpr, load_net=load_net, **kw)

        except StateFileNotFoundError as e:
            raise e

        except FileNotFoundError as e:
            # except (FileNotFoundError, NoModelError) as e:
            logging.warning('{} not found, will recollect networks'.format(e.filename))
            flash = False

    if not flash:
        logging.debug('Collecting networks')
        with turnoff_debug(turnoff=not show_debug):
            list_of_networks = collect_models(search_dir,
                                              load_net=False,
                                              **kw)
        filter_keys = get_filter_keys()
        rmodels = _register_models(list_of_networks, *filter_keys)
        save_json(rmodels, search_dir, registered_models_file)
        return fetch_models(search_dir, registered_models_file, filter=filter, flash=True,
                            tpr=tpr, load_net=load_net, **kw)


def _gather_registered_models(mdict, filter, tpr=0.95, wanted_epoch='last', **kw):

    from cvae import ClassificationVariationalNetwork as M
    from module.wim import WIMVariationalNetwork as W

    mlist = []
    for d in mdict:
        if filter is None or filter.filter(mdict[d]):
            m = W.load(d, **kw) if W.is_wim(d) else M.load(d, **kw)
            mlist.append(make_dict_from_model(m, d, tpr=tpr, wanted_epoch=wanted_epoch))

    return mlist


@ iterable_over_subdirs(0, iterate_over_subdirs=list)
def collect_models(directory,
                   wanted_epoch='last',
                   load_state=True, tpr=0.95, **default_load_paramaters):

    from cvae import ClassificationVariationalNetwork as M
    from module.wim import WIMVariationalNetwork as W

    if 'dump' in directory:
        return

    assert wanted_epoch == 'last' or not load_state

    try:
        logging.debug(f'Loading net in: {directory}')
        if W.is_wim(directory):
            model = W.load(directory, load_state=load_state, **default_load_paramaters)
        else:
            model = M.load(directory, load_state=load_state, **default_load_paramaters)

        return make_dict_from_model(model, directory, tpr=tpr, wanted_epoch=wanted_epoch)

    except FileNotFoundError:
        return
    except PermissionError:
        return
    except NoModelError:
        return
    except StateFileNotFoundError:
        raise

    except RuntimeError as e:
        logging.warning(f'Load error in {directory} see log file')
        logging.debug(f'Load error: {e}')


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

    -- epoch: last or min-loss or int

    -- which_rec: either 'none' 'ind' or 'all'

    -- state: wehter to include state.pth

    returns generator of needed files paths

    """

    assert not state or epoch == 'last'

    from cvae import ClassificationVariationalNetwork as M

    for d in mdirs:

        logging.debug('Inspecting {}'.format(d))

        m = M.load(d, load_net=False)
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
                sets += get_same_size_by_name(testset)
                for _ in [_ for _ in recs_to_exclude if _ in sets]:
                    sets.remove(_)

        for s in sets:
            sdir = os.path.join(d, 'samples', epoch_, 'record-{}.pth'.format(s))
            logging.debug('Looking for {}'.format(sdir))
            if not os.path.exists(sdir):
                if missing_file_stream:
                    missing_file_stream.write(sdir + '\n')
                yield d, sdir

        if state:
            sdir = os.path.join(d, 'state.pth')
            logging.debug('Looking for {}'.format(sdir))
            if not os.path.exists(sdir):
                if missing_file_stream:
                    missing_file_stream.write(sdir + '\n')
                yield d, sdir

        if optimizer:
            sdir = os.path.join(d, 'optimizer.pth')
            logging.debug('Looking for {}'.format(sdir))
            if not os.path.exists(sdir):
                if missing_file_stream:
                    missing_file_stream.write(sdir + '\n')
                yield d, sdir


def get_submodule(model, sub='features', job_dir='jobs', name=None, **kw):

    if isinstance(model, int):
        model_number = model
        logging.debug('Will find model {} in {}'.format(model_number, job_dir))
        model = find_by_job_number(model_number, job_dir=job_dir, load_net=True, load_state=True, **kw)['net']
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


if __name__ == '__main__':
    import sys
    import tempfile
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('jobs', nargs='+')
    parser.add_argument('--job-dir', default='./jobs')
    parser.add_argument('--state', action='store_true')
    parser.add_argument('--optimizer', action='store_true')
    parser.add_argument('--output', default=os.path.join(tempfile.gettempdir(), 'files'))
    parser.add_argument('--rec-files', default='ind')
    parser.add_argument('--register', dest='flash', action='store_false')

    logging.getLogger().setLevel(logging.DEBUG)

    args = parser.parse_args()

    output_file = args.output

    job_dict = find_by_job_number(*args.jobs, job_dir=args.job_dir, force_dict=True, flash=args.flash)

    logging.info('Will recover jobs {}'.format(', '.join(str(_) for _ in job_dict)))

    mdirs = [job_dict[_]['dir'] for _ in job_dict]

    with open(output_file, 'w') as f:
        for _ in needed_remote_files(*mdirs, which_rec=args.rec_files,
                                     state=args.state, optimizer=args.optimizer,
                                     missing_file_stream=f):
            pass

    with open('/tmp/rsync-files', 'w') as f:
        f.write('rsync -avP --files-from={f} $1 .\n'.format(f=output_file))
