import os
import pickle
import json
import logging
import pandas as pd

# from cvae import ClassificationVariationalNetwork

def get_path(dir_name, file_name, create_dir=True):

    dir_path = os.path.realpath(dir_name)
    if not os.path.exists(dir_path) and create_dir:
        os.makedirs(dir_path)

    return full_path(dir_name, file_name)


def save_net(net, dir_name, file_name):

    dir_path = os.path.realpath(dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = full_path(dir_name, file_name)
    net.save(file_path)

    

def save_json(d, dir_name, file_name, create_dir=True):

    p = get_path(dir_name, file_name, create_dir)

    with open(p, 'w') as f:
        json.dump(d, f)


def load_json(dir_name, file_name):

    p = get_path(dir_name, file_name, create_dir=False)

    with open(p, 'rb') as f:
        return json.load(f)
    
    
def save_object(o, dir_name, file_name):
    dir_path = os.path.realpath(dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, file_name), 'wb') as f:
        pickle.dump(o, f)

            
def load_object(dir_path, file_name):

    file_path = full_path(dir_path, file_name)

    with open(file_path, 'rb') as f:
        return pickle.load(f)


def full_path(dir_name, file_name):

    return os.path.join(os.path.realpath(dir_name), file_name)


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
        lists_of_nets = []
        collect_networks(dir_path, lists_of_nets, only_trained=False)
        num_of_nets = len(sum(lists_of_nets, []))
        print(f' ({num_of_nets} networks)')
    else:
        print()

    for i, d in enumerate(sub_dirs_rel_paths):
        print(f'{i+1:2d}: enter {d}', end='')
        if count_nets:
            lists_of_nets = []
            collect_networks(os.path.join(dir_path, d),
                             lists_of_nets, only_trained=False)
            num_of_nets = len(sum(lists_of_nets, []))
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
                                                             sub_dirs_rel_paths[i-1]))
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


def collect_networks(directory,
                     list_of_vae_by_architectures,
                     **default_load_paramaters):

    from cvae import ClassificationVariationalNetwork
    from roc_curves import ood_roc, fpr_at_tpr, load_roc, save_roc
    
    def append_by_architecture(net_dict, list_of_lists):
        
        net = net_dict['net']

        added = False
        for list_of_nets in list_of_lists:
            if not added:
                if net.has_same_architecture(list_of_nets[0]['net']):
                    list_of_nets.append(net_dict)
                    added = True
        if not added:
            list_of_lists.append([net_dict])
            added = True

        return added

    try:
        # logging.debug(f'in {directory}')
        vae = ClassificationVariationalNetwork.load(directory,
                                                    **default_load_paramaters)
        logging.debug(f'net found in {shorten_path(directory)}')
        arch =  vae.print_architecture(excludes=('latent_dim'))
        arch_code = hex(hash(arch))[2:10]
        pretrained_features =  (None if not vae.features
                                else vae.training['pretrained_features'])
        pretrained_upsampler = vae.training.get('pretrained_upsampler', None)
        methods = vae.predict_methods
        vae_dict = {'net': vae,
                    'type': vae.type,
                    'arch': arch,
                    'arch_code': arch_code,
                    'dir': directory,
                    'set': vae.training['set'],
                    'sigma': vae.sigma,
                    'done': vae.trained,
                    'epochs': vae.training['epochs'],
                    'n_tested': min(vae.testing[m]['n'] for m in methods),
                    'epochs_tested': min(vae.testing[m]['epochs'] for m in methods),
                    'acc': {m: vae.testing[m]['accuracy'] for m in methods},
                    'K': vae.latent_dim,
                    'L': vae.latent_sampling,
                    'pretrained_features': str(pretrained_features),
                    'pretrained_upsampler': str(pretrained_upsampler),
                    'depth': vae.depth,
                    'width': vae.width,
        }
        append_by_architecture(vae_dict, list_of_vae_by_architectures)

    except FileNotFoundError:    
        pass
    except RuntimeError as e:
        logging.warning(f'Load error in {directory} see log file')
        logging.debug(f'Load error: {e}')
    
    list_dir = [os.path.join(directory, d) for d in os.listdir(directory)]
    sub_dirs = [e for e in list_dir if os.path.isdir(e)]
    
    for d in sub_dirs:
        collect_networks(d,
                         list_of_vae_by_architectures,
                         **default_load_paramaters)

    num_of_archs = len(list_of_vae_by_architectures)
    num_of_nets = len(sum(list_of_vae_by_architectures, []))

    logging.debug(f'{num_of_nets} nets in {num_of_archs} different architectures'
                  f'found in {shorten_path(directory)}')

        
def load_and_save(directory, output_directory=None, **kw):
    """ load the incomplete params (with default missing parameter
    that can be specified in **kw and save them
    """

    list_of_vae = []

    collect_networks(directory, list_of_vae, **kw)
    
    for l in list_of_vae:
        for d in l:
            # print(d['net'].print_architecture())
            saved_dir = d['dir']
            if output_directory is not None:
                saved_dir = os.path.join(output_directory, saved_dir)
                # print(saved_dir)
            d['net'].save(saved_dir)
            v = ClassificationVariationalNetwork.load(saved_dir)
            print('L:', d['net'].print_architecture())
            print('S:', v.print_architecture())
    return list_of_vae


def data_frame_results(nets):
    """
    nets : list of dicts n
    n['net'] : the network
    n['sigma']
    n['arch']
    n['set']
    n['K']
    n['L']
    n['acc'] : {m: acc for m in methods}
    """

    set_arch_index = ['set', 'type', 'arch_code',
                      'pretrained_features',
                      'pretrained_upsampler',
    ]

    K_L_index = ['K', 'L', 'sigma']
    columns = set_arch_index + K_L_index + ['acc']

    df = pd.DataFrame.from_records(nets, columns=columns)

    df = df.drop('acc', axis=1).join(pd.DataFrame(df.acc.values.tolist()))

    df.set_index(set_arch_index + K_L_index, inplace=True)

    # for pre in 'pretrained_features', 'pretrained_upsampler':
    #     for i, l in enumerate(df.index.names):
    #         if l==pre:
    #             level=i
    #     idx = df.index
    #     levels=idx.levels[level] #.astype(str)
    #     print(levels)
    #     idx.set_levels(level=level, levels=levels, inplace=True)
        
    df = df.groupby(level=set_arch_index + K_L_index)[df.columns].max()
    df = df.stack()

    df.index.rename('method', level=-1, inplace=True)
    df.index.rename('pre_feat', level='pretrained_features', inplace=True)
    df.index.rename('pre_up', level='pretrained_upsampler', inplace=True)

    df = df.unstack(level=('sigma', 'method'))
    # return df
    return df.reindex(sorted(df.columns), axis=1)


def load_and_save_json(directory, write_json=False):

    name = os.path.join(directory, 'train.json')
    if os.path.exists(name):
        #        print(name)
        with open(name, 'rb') as f:

            try:
                t = json.load(f)
            except json.JSONDecodeError:
                print(name)
                t = dict()

        if 'sampling' in t.keys():
            print(t['sampling'], t.get('latent_sampling', -1))
            t['latent_sampling'] = t.pop('sampling')
            print('r', write_json, name, '\n', t)

            if write_json:
                print('w', name, '\n', t)
                with open(name, 'w') as f:
                    json.dump(t, f)


    rel_paths = os.listdir(directory)
    paths = [os.path.join(directory, p) for p in rel_paths]

    dirs = [d for d in paths if os.path.isdir(d)]

    for d in dirs:
        load_and_save_json(d,write_json=write_json)

        
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

