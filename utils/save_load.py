import os
import pickle
import json
from tensorflow.keras.models import load_model
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

    
def load_net(dir_name, file_name):

    file_path = full_path(dir_name, file_name)
    return load_model(file_path)
    

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


def get_path_from_input(dir_path=os.getcwd()):

    rel_paths = os.listdir(dir_path)
    abs_path = [os.path.join(dir_path, d) for d in rel_paths]
    sub_dirs_rel_paths = [rel_paths[i] for i, d in enumerate(abs_path) if os.path.isdir(d)]
    print(f' 0: choose {dir_path}')
    for i, d in enumerate(sub_dirs_rel_paths):
        print(f'{i+1:2d}: enter {d}')

    try:
        i = int(input('Your choice: '))
    except ValueError:
        path = os.path.join(dir_path, os.pardir)
        path = os.path.abspath(path)
        return get_path_from_input(path)
    if i == 0:
        return dir_path
    elif 0 < i < len(sub_dirs_rel_paths) + 1:
        return get_path_from_input(dir_path=os.path.join(dir_path,
                                                         sub_dirs_rel_paths[i-1]))
    else:
        return get_path_from_input(dir_path)


def collect_networks(directory,
                     list_of_vae_by_architectures,
                     only_trained=True,
                     x_test=None,
                     y_test=None,
                     **default_load_paramaters):
    
    from cvae import ClassificationVariationalNetwork
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
        vae = ClassificationVariationalNetwork.load(directory,
                                                    **default_load_paramaters)
        vae_dict = {'net': vae}
        vae_dict['beta'] = vae.beta
        vae_dict['dir'] = directory
        if vae.trained or not only_trained:
            append_by_architecture(vae_dict, list_of_vae_by_architectures)

        results_path_file = os.path.join(directory, 'test_accuracy') 
        try:
            # get result from file
            with open(results_path_file, 'r') as f:
                vae_dict['acc'] = float(f.read())
        except FileNotFoundError:
            if x_test is not None:
                acc = vae.accuracy(x_test, y_test)
                with open(results_path_file, 'w+') as f:
                    f.write(str(acc) + '\n')            
                vae_dict['acc'] = acc
            else:
                vae_dict['acc'] = None

    except FileNotFoundError:
        pass
    
    list_dir = [os.path.join(directory, d) for d in os.listdir(directory)]
    sub_dirs = [e for e in list_dir if os.path.isdir(e)]
    
    for d in sub_dirs:
        collect_networks(d,
                         list_of_vae_by_architectures,
                         only_trained=only_trained,
                         x_test=x_test,
                         y_test=y_test,
                         **default_load_paramaters)


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
