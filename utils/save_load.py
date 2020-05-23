import os
import pickle
import json
import logging

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
                     like=None,
                     only_trained=True,
                     testset=None,
                     oodset=None,
                     true_pos_rates=[95, 98],
                     batch_size=100,
                     num_batch=5,
                     min_stats=None,
                     device=None,
                     verbose=0,
                     **default_load_paramaters):

    from cvae import ClassificationVariationalNetwork
    from roc_curves import ood_roc, fpr_at_tpr, load_roc, save_roc

    if like:
        list_of_vae_by_architectures.append([{'net': like}])
    
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

    if min_stats is None:
        min_stats = num_batch * batch_size

    min_stats = min(min_stats, num_batch * batch_size)

    try:
        # logging.debug(f'in {directory}')
        vae = ClassificationVariationalNetwork.load(directory,
                                                    **default_load_paramaters)
        logging.debug(f'net found in {directory}')
        vae_dict = {'net': vae}
        vae_dict['beta'] = vae.beta
        vae_dict['dir'] = directory
        if vae.trained or not only_trained:
            append_by_architecture(vae_dict, list_of_vae_by_architectures)

        vae_dict['acc'] = dict()
        compute_accuracies = False

        if vae.trained:
            for method in vae.predict_methods:
                results_path_file = os.path.join(directory,
                                                 'test_accuracy_' + method) 
                try: # get result from file
                    with open(results_path_file, 'r') as f:
                        vae_dict['acc'][method] = float(f.read())

                except FileNotFoundError:
                    compute_accuracies = True
                    vae_dict['acc'][method] = None

            if compute_accuracies and testset:
                acc = vae.accuracy(testset,
                                   batch_size=batch_size,
                                   num_batch='all',
                                   method='all',
                                   device=device)
                for method in vae.predict_methods:
                    results_path_file = os.path.join(directory,
                                                     'test_accuracy_' +
                                                     method) 
                    with open(results_path_file, 'w+') as f:
                        f.write(str(acc[method]) + '\n')            
                vae_dict['acc'] = acc

            compute_oodroc = False
            derailed_file = os.path.join(directory, 'derailed')
            vae_dict['fpr at tpr'] = {rate: None for rate in true_pos_rates}
            if oodset:
                for rate in true_pos_rates:
                    ood_file_name = f'ood_{oodset.name}_fpr_at_tpr={rate}'
                    # ood_file_name = 'ood_fpr_at_tpr.json'
                    results_path_file = os.path.join(directory, ood_file_name)
                    n, fp, tp = load_roc(results_path_file)
                    assert(tp == rate/100 or tp == 0.)
                    vae_dict['fpr at tpr'][rate] = fp
                    if n < min_stats:
                        print(f'{n} < {min_stats} for tpr {rate}')
                        compute_oodroc = True
                    is_derailed = os.path.exists(derailed_file)

            if compute_oodroc and not is_derailed:
                if verbose > 0:
                    print('Computing fprs for', vae_dict['dir'])
                try: 
                    fpr, tpr = ood_roc(vae, testset, oodset,
                                       batch_size=batch_size,
                                       num_batch=num_batch,
                                       device=device)
                    print(f'fprs: {len(fpr)}')
                    for rate in true_pos_rates:
                        ood_file_name = f'ood_{oodset.name}_fpr_at_tpr={rate}'
                        results_path_file = os.path.join(directory,
                                                         ood_file_name)
                        fp = save_roc(results_path_file, fpr, tpr,
                                      rate/100, num_batch * batch_size)
                        vae_dict['fpr at tpr'][rate] = fp
                except ValueError as err:
                    print('ValueError', err)
                    with open(derailed_file, 'w+') as f:
                        f.write('network has derailed')
                    for rate in true_pos_rates:
                        vae_dict['fpr at tpr'][rate] = None

    except FileNotFoundError:    
        pass
    except RuntimeError as e:
        logging.warning(f'Load error in {directory} see log file')

    
    list_dir = [os.path.join(directory, d) for d in os.listdir(directory)]
    sub_dirs = [e for e in list_dir if os.path.isdir(e)]
    
    for d in sub_dirs:
        collect_networks(d,
                         list_of_vae_by_architectures,
                         only_trained=only_trained,
                         testset=testset,
                         oodset=oodset,
                         true_pos_rates=true_pos_rates,
                         batch_size=batch_size,
                         num_batch=num_batch,
                         device=device,
                         verbose=verbose,
                         **default_load_paramaters)

    if like:
        # list_of_vae_by_architectures[0].pop(0)
        list_of_vae_by_architectures = [list_of_vae_by_architectures[0]]

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

