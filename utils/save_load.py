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
        logging.debug(f'net found in {directory}')
        vae_dict = {'net': vae,
                    'type': vae.type,
                    'arch': vae.print_architecture(short=True, excludes=('latent_dim')),
                    'dir': directory,
                    'set': vae.training['set'],
                    'beta': vae.beta,
                    'epochs': vae.trained,
                    'n_tested': min(vae.testing[m]['n'] for m in vae.testing),
                    'epochs_tested': min(vae.testing[m]['epochs'] for m in vae.testing),
                    'acc': {m: vae.testing[m]['accuracy'] for m in vae.testing},
                    'K': vae.latent_dim,
                    'L': vae.latent_sampling,
                    'depth': vae.depth,
                    'width': vae.width,
        }
        append_by_architecture(vae_dict, list_of_vae_by_architectures)

    except FileNotFoundError:    
        pass
    except RuntimeError as e:
        logging.warning(f'Load error in {directory} see log file')

    
    list_dir = [os.path.join(directory, d) for d in os.listdir(directory)]
    sub_dirs = [e for e in list_dir if os.path.isdir(e)]
    
    for d in sub_dirs:
        collect_networks(d,
                         list_of_vae_by_architectures,
                         **default_load_paramaters)

    # logging.debug(f'{len(list_of_vae_by_architectures[0])} different architectures')
    # for l in list_of_vae_by_architectures:
    #     logging.debug(f'{len(l)} networks')

        
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
    n['beta']
    n['arch']
    n['set']
    n['K']
    n['L']
    n['acc'] : {m: acc for m in methods}
    """

    indices = ['set', 'type', 'arch', 'K', 'L', 'beta']
    columns = indices + ['acc']

    df = pd.DataFrame.from_records(nets, columns=columns)
    
    df2 = df.drop('acc', axis=1).join(pd.DataFrame(df.acc.values.tolist()))

    df2.set_index(indices, inplace=True)

    df = df2.groupby(level=indices)[df2.columns].max()
    sdf = df.stack().rename('method', level=-1)
    df = sdf.unstack(level='beta')

    return df


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
