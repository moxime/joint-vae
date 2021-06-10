import json
import os
from module.vae_layers import Sigma


def load_and_save_json(directory,
                       json_file,
                       key,
                       new_key=None,
                       old_value=None,
                       new_value=None,
                       recursive=True,
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

        if key in t.keys():
            print(name, '\n---', key, ':', t[key], end='')

            if new_value:
                if t[key] == old_value:
                    print(' ->', new_value, '*' if write_json else '')
                    t[key] = new_value
                else: print()
                # print('r', write_json, name, '\n', t)
            if new_key:
                v = t.pop(key)
                t[new_key] = v
                print(' ->', new_key,':', t[new_key], '*' if write_json else '')
            if write_json:
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
    
    file_paths = {n: os.path.join(directory, n+'.json') for n in ('train', 'history')}

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
                t_['sigma'] = dict(value=sigma, reach=reach, decay=decay, sigma0=sigma0)
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

        strip_json(d,write_json=write_json)


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


if __name__ == '__main__':

    beta_to_dict('jobs', write_json=True)
