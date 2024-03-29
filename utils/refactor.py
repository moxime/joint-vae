import json
import os
from module.vae_layers import Sigma
from utils.save_load import find_by_job_number, load_json
from shutil import copyfile
import functools
from utils.save_load import iterable_over_subdirs
import numpy as np
from datetime import date


def delete_job(directory, msg=''):
    deleted_file = os.path.join(directory, 'deleted')
    with open(deleted_file, 'a') as f:
        f.write(str(msg) + '\n')


@iterable_over_subdirs(0, iterate_over_subdirs=list)
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


@iterable_over_subdirs('directory')
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

    nets = find_by_job_number(*to_follow.keys(), load_net=False)

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

        if key in t.keys() or old_value is None:
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


@iterable_over_subdirs('directory', keep_none=False, iterate_over_subdirs=list)
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


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', default='jobs')
    parser.add_argument('--write', action='store_true')

    args = parser.parse_args()

    # print('Working on', args.job_dir)
    # load_and_save_json(args.job_dir, 'train.json', 'warmup',
    #                    old_value=0, new_value=[0, 0], suffix='-warmup')

    history_from_list_to_dict(directory=args.job_dir, write_json=args.write)
