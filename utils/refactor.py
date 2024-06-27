import json
import os
from module.vae_layers import Sigma
from utils.save_load import find_by_job_number, load_json, fetch_models, save_json
from shutil import copyfile
import functools
from utils.save_load.fetch import iterable_over_subdirs
import numpy as np
from datetime import date
from utils.parameters import gethostname
from itertools import count


def delete_job(directory, msg=''):
    deleted_file = os.path.join(directory, 'deleted')
    with open(deleted_file, 'a') as f:
        f.write(str(msg) + '\n')


def print_dict(d, sep=' -- ', format='{}:{}'):

    print(sep.join(format.format(k, v) for k, v in d.items()))


def backup_json(d, n, fingerprint=None):
    ext = '.bak'
    if fingerprint not in (None, True):
        ext = ext + '.{}'.format(fingerprint)
    name = os.path.join(d, n)
    copyfile(name, name + ext)
    return name + ext


def restore_json(d, n, fingerprint=None):
    ext = '.bak'
    if fingerprint not in (None, True):
        ext = ext + '.{}'.format(fingerprint)
    name = os.path.join(d, n)
    if os.path.exists(name):
        copyfile(name + ext, name)

    return name + ext


@ iterable_over_subdirs(0, iterate_over_subdirs=list)
def modify_train_file_coder(directory, write_json=False, old_suffix='bak'):

    name = os.path.join(directory, 'train.json')
    # print(directory)

    if not os.path.exists(name):
        return

    #        print(name)
    with open(name, 'rb') as f:
        try:
            t = json.load(f)
        except json.JSONDecodeError:
            print('error with', name)
            t = dict()
            return

    if 'learned_coder' not in t:
        return
    if t.get('learned_coder'):
        t['coder_means'] = 'learned'
    else:
        t['coder_means'] = 'random'

    delete = False

    if t.get('dictionary_min_dist', 0) and t['coder_capacity_regularization']:
        delete = True

    if t.get('rho'):
        delete = True

    if delete and write_json:
        delete_job(directory, msg=date.today().strftime('%Y-%m-%d'))

    pre = 'D' if delete else ' '
    if t.get('rho') or True:  # and t.get('rho_temp') < np.inf:
        print('{pre} {dir} -- learned: {l} min dist:{d:4g} reg:{r}'
              ' rho:{rho:6g}, {rho_temp:4g}'
              '-> means:{m_:7} {pre}'.format(
                  pre=pre,
                  dir=directory[-6:],
                  rho=t.get('rho') or 0,
                  rho_temp=t.get('rho_temp', np.inf) or np.inf,
                  l='yes' if t['learned_coder'] else 'no ',
                  d=t.get('dictionary_min_dist', 0) or 0,
                  r='yes' if t.get('coder_capacity_regularization') else 'no ',
                  m_=t['coder_means'],
              ))

    for _ in ('learned_coder', 'coder_capacity_regularization', 'dictionary_min_dist', 'rho', 'rho_temp'):
        if _ in t:
            t.pop(_)

    if write_json:

        copyfile(name, name + '.' + old_suffix)
        print('w', name[-250:])
        for k in sorted(t):
            print('|_{:30}: {}'.format(k, t[k]))
        with open(name, 'w') as f:
            print('write')
            json.dump(t, f)


@ iterable_over_subdirs('directory')
def verify_has_valid(directory='jobs/'):
    try:
        train = load_json(directory, 'train.json')
        history = load_json(directory, 'history.json')
        samples = os.path.join(directory, 'samples', 'last')
    except FileNotFoundError:
        return
    if os.path.isdir(samples):
        files = os.listdir(samples)
        trainset = train['set']
        has_valid = 'validation_loss' in history
        if has_valid:
            if not any('valid' in _ for _ in files) or not any(trainset in _ for _ in files):
                print(directory.split('/')[-1])
                # print(*files)


def refactor_from_log_file(json_file,
                           key,
                           ktype=int,
                           log_directory='./jobs/log',
                           log_file_pre='train.log',
                           follow_resumed=True,
                           write_json=False):

    to_follow = {}

    log_files = [f for f in os.listdir(
        log_directory) if f.startswith(log_file_pre)]

    for f in log_files:

        with open(os.path.join(log_directory, f), 'r') as f_:
            # print(f_.name)
            lines = f_.readlines()
            vals = {key: None, 'job_dir': 'none'}
            types = {key: ktype, 'job_dir': str}
            directory = None
            for k in vals:
                for line in lines:
                    if k + ':' in line:
                        vals[k] = types[k](
                            line.strip().split(k + ':')[-1].strip())
            for line in lines:
                if '[I] ' + vals['job_dir'] in line:
                    directory = line.strip('\n').split('[I] ')[-1]
                    # print(directory)

            if vals[key] is not None and directory:
                # print(f_.name, directory, ':', key, '=', vals[key])
                load_and_save_json(directory, json_file, key,
                                   new_value=vals[key], recursive=False, write_json=write_json)
                f_r = os.path.join(directory, 'RESUMED')
                if os.path.exists(f_r):
                    with open(f_r, 'r') as f_r_:
                        is_resumed = int(f_r_.read())
                        to_follow[is_resumed] = vals[key]

    nets = find_by_job_number(*to_follow.keys(), build_module=False)

    print('==== RESUMED ====')

    for j in nets:

        directory = nets[j]['dir']
        val = to_follow[j]
        print('\n***', j, ':', val, '***')
        load_and_save_json(directory, json_file, key, new_value=val,
                           recursive=False, write_json=write_json)


def load_and_save_json(directory,
                       json_file,
                       key,
                       new_key=None,
                       to_list=False,
                       old_value=None,
                       new_value=None,
                       recursive=True,
                       suffix='',
                       write_json=False):

    assert new_key is None or old_value is None

    name = os.path.join(directory, json_file)

    if os.path.exists(name):
        #        print(name)
        with open(name, 'rb') as f:

            try:
                t = json.load(f)
            except json.JSONDecodeError:
                print('error with', name)
                t = dict()

        if key in t.keys() or (old_value is None and new_key is None):
            print(name[-80:], '\n---', key, ':', t.get(key, None), end='')

            if new_value is not None:
                if t.get(key, None) == old_value:
                    print(' ->', new_value, '*' if write_json else '')
                    t[key] = new_value
                else:
                    print()
                # print('r', write_json, name, '\n', t)
            if new_key:
                v = t.pop(key)
                if to_list:
                    t[new_key] = [v] if v else []
                    pass
                else:
                    t[new_key] = v
                print(' ->', new_key, ':',
                      t[new_key], '*' if write_json else '')
            if write_json:
                copyfile(name, name + '.bak')
                print('w', name, '\n', t)
                with open(name, 'w') as f:
                    json.dump(t, f)

    if recursive:
        rel_paths = os.listdir(directory)
        paths = [os.path.join(directory, p) for p in rel_paths]
        dirs = [d for d in paths if os.path.isdir(d)]

        for d in dirs:
            load_and_save_json(d, json_file, key,
                               new_key=new_key,
                               to_list=to_list,
                               old_value=old_value,
                               new_value=new_value,
                               recursive=recursive,
                               write_json=write_json)


def beta_to_dict(directory, write_json=False):

    def _f(f, k=6):
        if f is not None:
            return f'{f:{k}.3f}'
        return '.' * k

    file_paths = {n: os.path.join(directory, n + '.json')
                  for n in ('train', 'history')}

    t = {n: None for n in file_paths}
    for n in file_paths:
        if os.path.exists(file_paths[n]):
            with open(file_paths[n], 'rb') as f:
                try:
                    t[n] = json.load(f)
                except json.JSONDecodeError:
                    print(file_paths[n], 'not loaded')

    if t['train'] and t['history']:

        sigma = t['train']['sigma']
        reach = t['train'].get('sigma_reach', None)
        decay = t['train'].get('sigma_decay', None)
        sigma0 = t['train'].get('sigma0', None)

        if type(sigma) != dict:

            print(f'|   s: {_f(sigma)} '
                  f' r: {_f(reach)} '
                  f' d: {_f(decay)} '
                  f' 0: {_f(sigma0)}  |')

            if reach and not decay:
                decay = 0.1
            if not reach:
                reach = 1
                decay = 0

            if not decay and not sigma0:
                sigma0 = sigma

            if not sigma0:
                try:
                    sigma0 = t['history']['train_measures'][0]['sigma']
                except IndexError:
                    sigma0 = sigma

            print(f'|   v: {_f(sigma)} '
                  f' r: {_f(reach)} '
                  f' d: {_f(decay)} '
                  f' 0: {_f(sigma0)}  |')

            t_ = t['train'].copy()
            k_ = [k for k in t_.keys() if 'sigma' in k]
            for k in k_:
                t_.pop(k, None)
            t_['sigma'] = dict(value=sigma, reach=reach,
                               decay=decay, sigma0=sigma0)
            sigma = Sigma(**t_['sigma'])
            k_ = [k for k in t_.keys() if 'sigma' in k]
            print('| ', ' '.join(k_))
            print('|_', sigma, f'{sigma:f}')
            if write_json:
                with open(file_paths['train'], 'w') as f:
                    json.dump(t_, f)

        else:
            print(f'[ {Sigma(**sigma)} : {sigma} ]')

    rel_paths = os.listdir(directory)
    paths = [os.path.join(directory, p) for p in rel_paths]
    dirs = [d for d in paths if os.path.isdir(d)]

    for d in dirs:

        beta_to_dict(d, write_json=write_json)


def strip_json(directory, write_json=False):

    name = os.path.join(directory, 'test.json')

    if os.path.exists(name):
        with open(name, 'rb') as f:

            try:
                t = json.load(f)
                loaded = True
            except json.JSONDecodeError:
                print(name, 'not loaded')
                loaded = False

        if loaded:

            t_ = next(iter(t.values()))
            print('w', name, '\n', t, '\n', t_)
            if write_json:
                with open(name, 'w') as f:
                    json.dump(t_, f)

    rel_paths = os.listdir(directory)
    paths = [os.path.join(directory, p) for p in rel_paths]
    dirs = [d for d in paths if os.path.isdir(d)]

    for d in dirs:

        strip_json(d, write_json=write_json)


def json_pretrained_from_params_to_train(directory, write_json=False):

    params_json = os.path.join(directory, 'params.json')
    train_json = os.path.join(directory, 'train.json')

    if os.path.exists(params_json) and os.path.exists(train_json):

        with open(params_json, 'rb') as f:

            try:
                params = json.load(f)
                loaded = True
            except json.JSONDecodeError:
                print(params_json, 'not loaded')
                loaded = False

        with open(train_json, 'rb') as f:

            try:
                train = json.load(f)
            except json.JSONDecodeError:
                print(params_json, 'not loaded')
                loaded = False

        if loaded:

            if 'features' in params.keys():
                features = params['features'].pop('pretrained_features', None)
                upsampler = params.pop('pretrained_upsampler', None)
                train['pretrained_features'] = features
                train['pretrained_upsampler'] = upsampler
                print(directory, '\n', params, train)
                if write_json:

                    with open(train_json, 'w') as f:
                        json.dump(train, f)
                        print('w', train_json)
                    with open(params_json, 'w') as f:
                        json.dump(params, f)
                        print('w', params_json)

    rel_paths = os.listdir(directory)
    paths = [os.path.join(directory, p) for p in rel_paths]
    dirs = [d for d in paths if os.path.isdir(d)]

    for d in dirs:
        json_pretrained_from_params_to_train(d, write_json=write_json)


@ iterable_over_subdirs('directory', keep_none=False, iterate_over_subdirs=list)
def history_from_list_to_dict(directory='jobs', write_json=False):

    json_file = os.path.join(directory, 'history.json')
    json_file_bak = os.path.join(directory, 'history_.json')
    # print(json_file)
    if not os.path.exists(json_file) or os.path.exists(json_file_bak):
        return None
    print(directory.split('/')[-1])
    if write_json:
        copyfile(json_file, json_file_bak)
        print('backuped file')
    history = load_json(directory, 'history.json')

    params = load_json(directory, 'train_params.json')
    epochs_ = history['epochs']
    validation = params['validation']
    epochs = {}
    epochs['train'] = range(0, epochs_)
    full_test = params['full_test_every']
    epochs['test'] = range(full_test, epochs_, full_test)
    epochs['validation'] = range(0, epochs_) if validation else []
    kept_k = []
    new_history = {e: {} for e in epochs['train']}
    for k in [*history.keys()]:
        for prefix in ('test', 'train', 'validation'):
            if k.startswith(prefix):
                h_k = history.pop(k)
                _k = len(h_k)
                __ = len(epochs[prefix])
                assert __ == _k or not _k
                print(' ' if _k == __ else '*', k, _k, __)
                if _k:
                    for i, e in enumerate(epochs[prefix]):
                        new_history[e].update({k: h_k[i]})
                break
        else:
            kept_k.append(k)
    lr = history['lr']
    for i, e in enumerate(epochs['train']):
        new_history[e]['lr'] = lr[i]
    kept_k.remove('lr')
    print('kept', *kept_k)
    for k in kept_k:
        new_history[k] = history[k]
    if write_json:
        with open(json_file, 'w') as f:
            json.dump(new_history, f)
            print('w', json_file)

    if 10 in new_history:
        print(*new_history[10].keys())
    else:
        print('10 not in history')
    if 50 in new_history:
        print(*new_history[50].keys())
    else:
        print('50 not in history')
    print('train:', min(epochs['train']), max(epochs['train']))
    print('test:', *epochs['test'])
    return True


def add_default_values_to_registered_models(job_dir, write_json=False, **kw):

    rmodels_file = 'models-{}.json'.format(gethostname())
    rpath = os.path.join(job_dir, rmodels_file)

    if write_json:
        copyfile(rpath, rpath + '.bak')

    rmodels = load_json(job_dir, rmodels_file)

    print('{} models registered'.format(len(rmodels)))

    for i, d in enumerate(rmodels):
        for k, v in kw.items():
            if k not in rmodels[d]:
                rmodels[d][k] = v
                print('{} -- {}: {}'.format(d[-80:], k, v))
        if i > 100:
            pass

    if write_json:
        print('w', rpath, '\n')
        with open(rpath, 'w') as f:
            json.dump(rmodels, f)


def reset_wim_arrays(job_dir, do_it=False):

    from utils.filters import DictOfListsOfParamFilters, ParamFilter, get_filter_keys
    from module.wim.array import WIMArray as A
    wim_job_filter = DictOfListsOfParamFilters()
    wim_job_filter.add('wim_array_size', ParamFilter(type=int, interval=[1, np.inf]))

    models = fetch_models(job_dir, filter=wim_job_filter, flash=False)

    n = 0
    for m in models:

        wim_params = load_json(m['dir'], 'wim.json')
        s = wim_params.pop('array_size', None)

        if do_it:
            name = os.path.join(m['dir'], 'wim.json')
            copyfile(name, name + '.bak')
            with open(name, 'w') as f:
                json.dump(wim_params, f)

        n += 1

    print('***', n)


def print_prior_params(json_file, params_dict={}):

    if not params_dict:
        params_dict.update({'train_params': {}, 'params': {}})
    directory = os.path.dirname(json_file)
    for file in 'params.json', 'train_params.json':
        prior_params = params_dict[os.path.splitext(file)[0]]
        json_file = os.path.join(directory, file)
        if not os.path.exists(json_file):
            raise FileNotFoundError

        #        print(name)
        with open(json_file, 'rb') as f:
            try:
                t = json.load(f)
            except json.JSONDecodeError:
                print('error with', name)
                t = dict()
                return

            for k in t:
                if k == 'prior':
                    for k_ in t['prior']:
                        if k_ not in prior_params:
                            prior_params[k_] = set()
                        prior_params[k_].add(t['prior'][k_])

                elif 'prior' in k:
                    if k not in prior_params:
                        prior_params[k] = set()

                    prior_params[k].add(t[k])


def learned_variance(json_file):

    directory = os.path.dirname(json_file)
    t = {}
    for name in ('params.json', 'train_params.json'):
        json_file = os.path.join(directory, name)
        with open(json_file, 'rb') as f:
            t.update(json.load(f))

    if t.get('learned_latent_prior_variance'):
        print(' -- '.join('{}:{}'.format(k, t[k]) for k in t if 'prior' in k))


def prior_in_params(directory, write_json=False):

    json_files = ('params.json', 'train_params.json')

    original_params = {_: load_json(directory, _) for _ in json_files}

    original_prior_params = {}

    for k, v in original_params.items():

        original_prior_params.update({k: v.pop(k) for k in list(v.keys()) if 'prior' in k})

    prior = {}

    keys = {'learned_latent_prior_means': 'learned_means',
            'latent_prior_variance': 'var_dim',
            'latent_prior_means': 'init_mean'}

    for k, k_ in keys.items():
        prior[k_] = original_prior_params[k]

    prior['distribution'] = 'gaussian'
    params = original_params['params.json']
    mtype = params['type']
    prior['num_priors'] = 1 if mtype in ('vae', 'vib', 'jvae') else params['num_labels']

    if not write_json:
        print('=' * 80)
        for _ in original_params:
            print('-' * 20, _, '-' * 10)
            print_dict(original_params[_])
        print('---------- prior --------')
        print_dict(prior)

    else:
        for _ in json_files:
            bak = backup_json(directory, _, fingerprint=write_json)
            print(directory)
            print('-- {} -> {}'.format(_, os.path.basename(bak)))

        original_params['params.json']['prior'] = prior
        for _ in original_params:
            save_json(original_params[_], directory, _)
            print('-- {} updated'.format(_))


def key_in_json(directory, json_file, k):

    json_file = os.path.splitext(json_file)[0] + '.json'
    d = load_json(directory, json_file)

    d = {_: __ for _, __ in d.items() if k in _}

    if d:
        print_dict(d)
    else:
        print('--')


def change_json_key(directory, json_file, old_key, new_key, write_json=False):

    d = load_json(directory, json_file)

    v = d.pop(old_key)
    d[new_key] = v

    if not write_json:
        print('{}->{}:{}'.format(old_key, new_key, v))

    else:
        bak = backup_json(directory, json_file, fingerprint=write_json)
        print(directory)
        print('-- {} -> {}'.format(_, os.path.basename(bak)))
        save_json(d, directory, json_file)
        print('-- {} updated'.format(json_file))


def change_params_value(directory, json_file, key, func, write_json=False, on_miss='raise'):

    json_file = os.path.splitext(json_file)[0] + '.json'
    d = load_json(directory, json_file)
    try:
        v = d.pop(key)
    except KeyError:
        if on_miss == 'raise':
            raise
        return

    v_ = func(v)
    d[key] = v_

    if not write_json:
        print('{}:{}->{}'.format(key, v, v_))

    else:
        bak = backup_json(directory, json_file, fingerprint=write_json)
        print(directory)
        print('-- {} -> {}'.format(_, os.path.basename(bak)))
        save_json(d, directory, json_file)
        print('-- {} updated'.format(json_file))


def walk_json_files(directory, name):

    for d, _, files in os.walk(directory):
        if name + '.json' in files:
            yield d, name + '.json'


def refactor_prior_from_v1(job_dir='jobs-v1-refactored', write_json=False):

    prior_params = {}

    # for fp in walk_json_files(job_dir, 'params'):
    #     print_prior_params(fp, params_dict=prior_params)
    # for k in prior_params:
    #     print(k, prior_params[k])

    for d, f in walk_json_files(job_dir, 'params'):
        prior_in_params(d, write_json=write_json)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', default='jobs-v1-refactored')
    parser.add_argument('--write', '-x', nargs='?', const=True)
    parser.add_argument('--restore', nargs='?', const=True)
    parser.add_argument('--show', nargs=2)
    parser.add_argument('--feat', action='store_true')
    parser.add_argument('--upsampler', action='store_true')

    args = parser.parse_args()

    if args.restore:
        for d, _ in walk_json_files(args.job_dir, 'params'):
            print(d)
            print(restore_json(d, 'params.json', fingerprint=args.restore))
            print(restore_json(d, 'train_params.json', fingerprint=args.restore))

    elif args.show:
        param_file = os.path.splitext(args.show[1])[0]
        key = args.show[0]

        for d, _ in walk_json_files(args.job_dir, param_file):

            key_in_json(d, param_file, key)

    elif args.feat:
        for d, _ in walk_json_files(args.job_dir, 'params'):
            change_params_value(d, _, 'features', lambda x: x['features'],
                                on_miss='return',
                                write_json=args.write and 'feat')

    elif args.upsampler:
        def update_upsampler(u):
            if u == 'None':
                return u
            if isinstance(u, list):
                return '[x4:2+1]{}'.format('-'.join(map(str, u)))
        for d, _ in walk_json_files(args.job_dir, 'params'):
            change_params_value(d, _, 'upsampler', update_upsampler,
                                on_miss='return',
                                write_json=args.write and 'upsampler')
