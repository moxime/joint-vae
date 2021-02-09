import os
import pickle
import json
import logging
import pandas as pd
import hashlib
import data.torch_load as torchdl
import numpy as np
import torch
from utils.optimizers import Optimizer


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


def load_json(dir_name, file_name):

    p = get_path(dir_name, file_name, create_dir=False)

    with open(p, 'rb') as f:
        return json.load(f)
    
    
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


class ObjFromDict:

    def __init__(self, d, **defaults):
        for k, v in defaults.items():
            setattr(self, k, v)
        for k, v in d.items():
            setattr(self, k, v)

            
def print_architecture(o, sigma=False, sampling=False, excludes=[], short=False):

    arch = ObjFromDict(o.architecture, features=None)
    training = ObjFromDict(o.training)
    
    def _l2s(l, c='-', empty='.'):
        if l:
            return c.join(str(_) for _ in l)
        return empty

    def s_(s):
        return s[0] if short else s

    if arch.features:
        features = arch.features['name']
    s = ''
    if 'type' not in excludes:

        s += s_('type') + f'={arch.type}--'
    if 'activation' not in excludes:
        if arch.type != 'vib':
            s += s_('output') + f'={arch.output}--'
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
            s += s_('upsampler') + f'={_l2s(arch.upsampler)}--'
    s += s_('classifier') + f'={_l2s(arch.classifier)}'

    if sigma and 'sigma' not in excludes:
        s += '--' + s_('sigma') + f'={o.sigma}'
    

    if sampling and 'sampling' not in excludes:
        s += '--'
        s += s_('sampling')
        s += f'={training.latent_sampling}'

    return s


def option_vector(o):

    arch = ObjFromDict(o.architecture, features=None)
    training = ObjFromDict(o.training, transformer='default')
    v_ = []
    if arch.features:
        w = ''
        w += 'p:'
        if training.pretrained_features:
            w+= 'f'
        else:
            w+= ' '

        if arch.upsampler:
            if training.pretrained_upsampler:
                w += 'u'
            else:
                w += ' '
        v_.append(w)

    w = 't:' + training.transformer[0]
    v_.append(w)

    w = 'bn:'
    if not arch.batch_norm:
        c = ' '
    else:
        # print('****', self.batch_norm)
        c = arch.batch_norm[0]
    w += c
    v_.append(w)

    w = 'a:'
    for m in ('flip', 'crop'):
        if m in training.data_augmentation:
            w += m[0]
        else: w += ' '
    v_.append(w)

    if arch.type == 'cvae':
        w = 'c:'
        if training.learned_coder:
                w += 'l'
        else:
            w += 'r'
        _md = training.dictionary_min_dist
        if _md:
            w += f'>{_md:.1f}'
        else:
            _dv = training.dictionary_variance
            w += f'={_dv:5.2f}'

        v_.append(w)

    return ' '.join(v_)


class Shell:
    print_architecture = print_architecture
    option_vector = option_vector


def collect_networks(directory,
                     list_of_vae_by_architectures=None,
                     load_state=True,
                     **default_load_paramaters):

    from cvae import ClassificationVariationalNetwork
    from roc_curves import ood_roc, fpr_at_tpr, load_roc, save_roc
        
    if list_of_vae_by_architectures is None:
        l = []
        collect_networks(directory, l, load_state, **default_load_paramaters)
        return l
    
    def append_by_architecture(net_dict, list_of_lists):
        
        arch = net_dict['arch']
        
        added = False
        for list_of_nets in list_of_lists:
            if not added:
                if arch == list_of_nets[0]['arch']:
                    list_of_nets.append(net_dict)
                    added = True
        if not added:
            list_of_lists.append([net_dict])
            added = True

        return added

    try:
        logging.debug(f'Loading net in: {directory}')
        vae = ClassificationVariationalNetwork.load(directory,
                                                    load_state=load_state,
                                                    **default_load_paramaters)
        architecture = ObjFromDict(vae.architecture, features=None)
        training = ObjFromDict(vae.training,
                               transformer='default',
                               pretrained_upsampler=None)
        
        logging.debug(f'net found in {shorten_path(directory)}')
        arch =  vae.print_architecture(excludes=('latent_dim', 'batch_norm'))
        arch_code = hashlib.sha1(bytes(arch, 'utf-8')).hexdigest()[:6]
        # arch_code = hex(hash(arch))[2:10]
        pretrained_features =  (None if not architecture.features
                                else training.pretrained_features)
        pretrained_upsampler = training.pretrained_upsampler
        predict_methods =  vae.predict_methods
        ood_methods = vae.ood_methods

        batch_size = training.batch_size
        if not batch_size:
            train_batch_size = vae.training.max_batch_sizes
        else:
            train_batch_size = batch_size

        accuracies = {m: vae.testing[m]['accuracy'] for m in predict_methods}
        if predict_methods:
            accuracies['first'] = accuracies.get(predict_methods[0], None) 
            best_accuracy = max(vae.testing[m]['accuracy'] for m in predict_methods)
            epochs_tested = min(vae.testing[m]['epochs'] for m in predict_methods)
            n_tested = min(vae.testing[m]['n'] for m in predict_methods)
        else:
            best_accuracy = epochs_tested = n_tested = None

        ood_sets = torchdl.get_same_size_by_name(vae.training['set'])
        if ood_methods:
            ood_fprs = {s: {} for s in ood_sets}
            ood_fpr = {s: None for s in ood_sets}
            best_auc = {s: 0 for s in ood_sets}
            best_method = {s: None for s in ood_sets}
            n_ood = {s: 0 for s in vae.ood_results}
            for s in ood_sets:
                res_by_set = {}
                for m in vae.ood_results.get(s, {}):
                    if m in ood_methods:
                        fpr_ = vae.ood_results[s][m]['fpr']
                        tpr_ = vae.ood_results[s][m]['tpr']
                        auc = vae.ood_results[s][m]['auc']
                        if not best_method[s] or auc > best_auc[s]:
                            best_auc[s] = auc
                            best_method[s] = m
                        n = vae.ood_results[s][m]['n']
                        if n_ood[s] == 0 or n < n_ood[s]:
                            n_ood[s] = n
                        res_by_method = {tpr: fpr for tpr, fpr in zip(tpr_, fpr_)}
                        res_by_method['auc'] = auc
                        res_by_set[m] = res_by_method
                res_by_set['first'] = res_by_set.get(ood_methods[0], None)
                ood_fprs[s] = res_by_set
                ood_fpr[s] = res_by_set.get(best_method[s], None)
                
        else:
            ood_fprs = {s: {} for s in ood_sets}
            ood_fpr = {s: None for s in ood_sets}
            n_ood = {}


        history = vae.train_history
        if history.get('test_measures', {}):
            mse = vae.train_history['test_measures'][-1].get('mse', np.nan)
            rmse = np.sqrt(mse)
        else:
            rmse = np.nan
        nans = {'total': np.nan, 'zdist':np.nan}
        loss_ = {s: ([nans] + history.get(s + '_loss', [nans]))[-1]
                 for s in ('train', 'test')}        

        empty_optimizer = Optimizer([torch.nn.Parameter()], **training.optim)
        depth = (1 + len(architecture.encoder)
                 + len(architecture.decoder) 
                 + len(architecture.classifier))
        
        width = (architecture.latent_dim +
                 sum(architecture.encoder) +
                 sum(architecture.decoder) +
                 sum(architecture.classifier)) 

        vae_dict = {'net': vae,
                    'job': vae.job_number,
                    'type': architecture.type,
                    'arch': arch,
                    'arch_code': arch_code,
                    'dir': directory,
                    'set': training.set,
                    'train_batch_size': train_batch_size,
                    'sigma': f'{vae.sigma}',
                    'done': vae.train_history['epochs'],
                    'epochs': vae.training['epochs'],
                    'finished': vae.train_history['epochs'] >= vae.training['epochs'],
                    'n_tested': n_tested,
                    'epochs_tested': epochs_tested,
                    'accuracies': accuracies,
                    'best_accuracy': best_accuracy,
                    'n_ood': n_ood,
                    'ood_fprs': ood_fprs,
                    'ood_fpr': ood_fpr,
                    'rmse': rmse,
                    'test_loss': loss_['test']['total'],
                    'train_loss': loss_['train']['total'],
                    'test_zdist': np.sqrt(loss_['test']['zdist']),
                    'train_zdist': np.sqrt(loss_['train']['zdist']),
                    'K': architecture.latent_dim,
                    'L': training.latent_sampling,
                    'pretrained_features': str(pretrained_features),
                    'pretrained_upsampler': str(pretrained_upsampler),
                    'depth': depth,
                    'width': width,
                    'options': vae.option_vector(),
                    'optim_str': f'{empty_optimizer:3}',
                    'optim': empty_optimizer.kind,
                    'lr': empty_optimizer.init_lr,
        }
        append_by_architecture(vae_dict, list_of_vae_by_architectures)

    except RuntimeError as e:
        logging.warning(f'Load error in {directory} see log file')
        logging.debug(f'Load error: {e}')
    
    except FileNotFoundError:    
        pass
    list_dir = [os.path.join(directory, d) for d in os.listdir(directory)]
    sub_dirs = [e for e in list_dir if os.path.isdir(e)]
    
    for d in sub_dirs:
        collect_networks(d,
                         list_of_vae_by_architectures,
                         load_state=load_state,
                         **default_load_paramaters)

    num_of_archs = len(list_of_vae_by_architectures)
    num_of_nets = len(sum(list_of_vae_by_architectures, []))

    logging.debug(f'{num_of_nets} nets in {num_of_archs} different architectures'
                  f'found in {shorten_path(directory)}')


def find_by_job_number(dir, *numbers, json_file=None, **kw):

    d = {}
    if json_file:
        try:
            networks_dict = load_json(dir, json_file)
            l = sum(networks_dict.values(), [])
            for n in l:
                for num in numbers:
                    if n['job'] == num:
                        d[num] = n
            return d

        except FileNotFoundError:
            print('File Error')
            pass

    v_ = sum(collect_networks(dir, **kw), [])
    for number in numbers:
        for v in v_:
            if v['job'] == number:
                d[number] = v

    return d


def save_features_upsampler(net, dir='.', name=''):

    try:
        feats = net.encoder.features.state_dict()
    except:
        feats = None
    try:
        ups = None
    except:
        pass
        

def test_results_df(nets, best_net=True, first_method=True, ood=True,
                    dataset=None,
                    tpr=[0.95], tnr=False):
    """
    nets : list of dicts n
    n['net'] : the network
    n['sigma']
    n['arch']
    n['set']
    n['K']
    n['L']
    n['accuracies'] : {m: acc for m in methods}
    n['best_accuracy'] : best accuracy
    n['ood_fpr'] : '{s: {tpr : fpr}}' for best method
    n['ood_fprs'] : '{s: {m: {tpr: fpr} for m in methods}}
    n['options'] : vector of options
    n['optim_str'] : optimizer
    """

    if not dataset:
        testsets = {n['set'] for n in nets}
        return {s: test_results_df(nets, best_net, first_method, ood, s, tpr, tnr) for s in testsets}

    arch_index = ['type',
                  'depth',
                  'arch_code',
                  'K',
    ]

    all_nets = [] if best_net else ['job', 'done']

    train_index = [
        'options',
        'optim_str',
        'L',
        'sigma',
    ] + all_nets

    indices = arch_index + train_index
    
    #acc_cols = ['best_accuracy', 'accuracies']
    #ood_cols = ['ood_fpr', 'ood_fprs']

    acc_cols = ['accuracies']
    ood_cols = ['ood_fprs']
    meas_cols = ['rmse', 'train_loss', 'test_loss', 'train_zdist', 'test_zdist']
    
    columns = indices + acc_cols + ood_cols + meas_cols

    df = pd.DataFrame.from_records([n for n in nets if n['set'] == dataset],
                                    columns=columns)

    df.set_index(indices, inplace=True)
    
    acc_df = pd.DataFrame(df['accuracies'].values.tolist(), index=df.index)
    acc_df.columns = pd.MultiIndex.from_product([acc_df.columns, ['rate']])
    ood_df = pd.DataFrame(df['ood_fprs'].values.tolist(), index=df.index)
    meas_df = df[meas_cols]
    # print(meas_df.columns)
    meas_df.columns = pd.MultiIndex.from_product([[''], meas_df.columns])
    
    # return acc_df
    # return ood_df
    d_ = {dataset: acc_df}
    for s in ood_df:
        d_s = pd.DataFrame(ood_df[s].values.tolist(), index=df.index)
        d_s_ = {}
        for m in d_s:
            v_ = d_s[m].values.tolist()
            _v = []
            for v in v_:
                if type(v) is dict:
                    _v.append(v)
                else: _v.append({})
            d_s_[m] = pd.DataFrame(_v, index=df.index)
        if d_s_:
            d_[s] = pd.concat(d_s_, axis=1)
            # print(d_[s].columns)
            # print('==')

            if tnr:
                cols_fpr = d_[s].columns[~d_[s].columns.isin(['auc'], level=-1)]
                d_[s][cols_fpr] = d_[s][cols_fpr].transform(lambda x: 1 - x)

        #d_[s] = pd.DataFrame(d_s.values.tolist(), index=df.index)

    d_['measures'] = meas_df
    df = pd.concat(d_, axis=1)

    cols = df.columns
    # print('*** save_load:379', [type(c[-1]) for c in cols])
    keeped_columns = cols.isin(tpr + ['rate', 'auc'] + [str(_) for _ in tpr], level=2)
    # tpr_columns = True
    method_columns = cols.isin(['first'], level=1)
    if not first_method: method_columns = ~method_columns

    # print(cols)
    measures_columns = cols.isin(meas_cols, level=2)
    # print(measures_columns)

    df = df[cols[(keeped_columns * method_columns) + measures_columns]]

    if first_method:
        df.columns = df.columns.droplevel(1)

    def _f(x, type='pc'):
        if np.isnan(x):
            return ''
        if np.isinf(x):
            return 'inf'

        if type == 'pc':
            return f'{100*x:4.1f}'
        return f'{x:.1e}'
        
    col_format = {c: _f for c in df.columns}
    for c in df.columns[df.columns.isin(['measures'], level=0)]:
        col_format[c] = lambda x: _f(x, 'measures')

    return df.sort_index().apply(col_format)
        
    if not best_method:
        
        df = df.drop('accuracies', axis=1).join(pd.DataFrame(df.accuracies.values.tolist()))
        # print('\n\n**** 341 *** dict of accuracies \n', df.head(), '\n***************')
    return df

    if ood:
        # if best_method:
        ood_df = pd.DataFrame(df.ood_fpr.values.tolist())
        oodsets = ood_df.columns
        print(*oodsets)

        ood_df_ = []
        for s in oodsets:
            l = ood_df[s].tolist()
            l_ = []
            for fpr in l:
                try:
                    l_.append(fpr[tpr])
                except TypeError:
                    l_.append(fpr)
            ood_df_.append(pd.DataFrame(l_))
        df = df.drop('ood_fpr', axis=1)
        df = df.join(ood_df_[1])
        return df
        for d in ood_df_:
            df = df.join(d)
        return df
        
    df.set_index(arch_index + train_index, inplace=True)
    print('\n\n**** 343 *** Index set\n', df.head(), '\n***************')

    
    # for pre in 'pretrained_features', 'pretrained_upsampler':
    #     for i, l in enumerate(df.index.names):
    #         if l==pre:
    #             level=i
    #     idx = df.index
    #     levels=idx.levels[level] #.astype(str)
    #     print(levels)
    #     idx.set_levels(level=level, levels=levels, inplace=True)

    if best_net:
        df = df.groupby(level=arch_index + train_index)[df.columns].max()
        print('\n\n**** 359 *** groupby\n', df.head(), '\n***************')
        df = df.stack()
        print('\n\n**** 361 *** stack\n', df.head(), '\n***************')
        
        df.index.rename('method', level=-1, inplace=True)
        print('\n\n**** 364 *** rename\n', df.head(), '\n***************')
        df = df.unstack(level=('sigma', 'method'))
        print('\n\n**** 366 *** unstack\n', df.head(), '\n***************')
    # return df
    
    return df.reindex(sorted(df.columns), axis=1)


def save_list_of_networks(list_of, dir_name, file_name='nets.json'):

    d = {}

    for n in list_of:
        n_ = n.copy()
        n_.pop('net')
        d[n_['dir']] = n_

    save_json(d, dir_name, file_name)


def load_list_of_networks(dir_name, file_name='nets.json'):
    
    return list(load_json(dir_name, file_name).values())

if __name__ == '__main__':

    import numpy as np
    import logging

    logging.getLogger().setLevel(logging.DEBUG)

    dir = 'old-jobs/saved-jobs/fashion32'
    dir = './jobs/'
    dir ='./jobs/fashion'

    reload = True
    reload = False
    if reload:
        l = sum(collect_networks(dir, load_state=False), [])

    testsets = ('cifar10',  'fashion', 'mnist')
    
    df_ = test_results_df(l) # [n for n in l if n['set']==s])


    def finite(u, f):
        if np.isnan(u):
            return ''
        if np.isinf(u):
            return 'inf'
        return f.format(u)

    def f_pc(u):
        return finite(100 * u, '{:.1f}')
    
    def f_db(u):
        return finite(u, '{:.1f}')

    formats = {s: [] for s in testsets}
    
    for s, df in df_.items():
        for _ in df.columns:
            formats[s].append(f_pc)

    """
    for s, df in df_.items():
        print('=' * 80)
        print(f'Results for {s}')
        print(df.to_string(na_rep='', decimal=',', formatters=formats[s]))
    """
        # for a in archs[s]:
        #     arch_code = hashlib.sha1(bytes(a, 'utf-8')).hexdigest()[:6]
        #     print(arch_code,':\n', a)
