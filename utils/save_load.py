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

            
def print_architecture(o, sigma=False, sampling=False,
                       excludes=[], short=False):

    arch = ObjFromDict(o.architecture, features=None)
    training = ObjFromDict(o.training_parameters)
    
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
    s += s_('classifier') + f'={_l2s(arch.classifier)}--'

    s += s_('variance') + f'={arch.latent_prior_variance:.1f}'
    
    if sigma and 'sigma' not in excludes:
        s += '--' + s_('sigma') + f'={o.sigma}'
    
    if sampling and 'sampling' not in excludes:
        s += '--'
        s += s_('sampling')
        s += f'={training.latent_sampling}'

    return s


def option_vector(o, empty=' ', space=' '): 

    arch = ObjFromDict(o.architecture, features=None)
    training = ObjFromDict(o.training_parameters, transformer='default')
    v_ = []
    if arch.features:
        w = ''
        w += 'p:'
        if training.pretrained_features:
            w+= 'f'
        else:
            w+= empty

        if arch.upsampler:
            if training.pretrained_upsampler:
                w += 'u'
            else:
                w += empty
        v_.append(w)

    w = 't:' + training.transformer[0]
    v_.append(w)

    w = 'bn:'
    if not arch.batch_norm:
        c = empty
    else:
        # print('****', self.batch_norm)
        c = arch.batch_norm[0]
    w += c
    v_.append(w)

    w = 'a:'
    for m in ('flip', 'crop'):
        if m in training.data_augmentation:
            w += m[0]
        else: w += empty
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

        self.reset()

        self._num_batch = 0
        self._samples = 0
        
        self.batch_size = batch_size

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

    def to(self, device):

        for t in self._tensors:
            self._tensors[t] = self._tensors[t].to(device)

    def reset(self):

        self._recorded_batches = 0
        self._seed = np.random.randint(1, int(1e8))
        return

    def init_seed_for_dataloader(self):
    
        seed = self._seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
    def keys(self):
        return self._tensors.keys()
    
    def __len__(self):
        return self._recorded_batches

    def __repr__(self):
        return ('Recorder for '
                + ' '.join([str(k) for k in self.keys()]))

    def save(self, file_path, cut=True):

        """dict_ = self.__dict__.copy()
        tensors = dict.pop('_tensors')
        """

        if cut:
            self.num_batch = len(self)
            t = self._tensors
            end = self.num_batch * self.batch_size
            for k in t:
                t[k] = t[k][..., 0:end]

        torch.save(self.__dict__, file_path)

    @classmethod
    def load(cls, file_path):

        dict_of_params = torch.load(file_path)
        num_batch = dict_of_params['_num_batch']
        batch_size = dict_of_params['batch_size']
        tensors = dict_of_params['_tensors']
        
        r = LossRecorder(batch_size, num_batch, **tensors)

        for k in ('_seed', '_tensors', '_recorded_batches'):
            setattr(r, k, dict_of_params[k])

        for k in dict_of_params:
            if not k.startswith('_'):
                setattr(r, k, dict_of_params[k])
            
        return r

    @classmethod
    def loadall(cls, dir_path, *w, file_name='record-{w}.pth'):

        r = {}

        for word in w:
            try:
                f = file_name.format(w=word)
                r[word] = LossRecorder.load(os.path.join(dir_path, f))
            except FileNotFoundError:
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
                
    def has_batch(self, number):
        r""" number starts at 0
        """

        return number < self._recorded_batches
    
    def get_batch(self, i, *which):
        
        if not which:
            return self.get_batch(i, *self.keys())
            
        if len(which) > 1:
            return {w: self.get_batch(i, w) for w in which}

        if not self.has_batch(i):
            raise IndexError(f'{i} >= {len(self)}')
        
        start = i * self.batch_size
        end = start + self.batch_size
        
        w = which[0]

        t = self._tensors[w]
        
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
        
        for k in tensors:
            if k not in self.keys():
                raise KeyError(k)
            # print('sl:426', 'rec', k, *tensors[k].shape)  # 
            self._tensors[k][..., start:end] = tensors[k]
                                                    
        self._recorded_batches += 1
    

def collect_networks(directory,
                     list_of_vae_by_architectures=None,
                     load_state=True,
                     **default_load_paramaters):


    from cvae import ClassificationVariationalNetwork
    
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
        training = ObjFromDict(vae.training_parameters,
                               transformer='default',
                               max_batch_sizes={'train': 8, 'test': 8},
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
            train_batch_size = training.max_batch_sizes['train']
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

        ood_sets = torchdl.get_same_size_by_name(vae.training_parameters['set'])
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

        nans = {'total': np.nan, 'zdist': np.nan}
        loss_ = {}
        for s in ('train', 'test'):
            last_loss = ([nans] + history.get(s + '_loss', [nans]))[-1]
            loss_[s] = nans.copy()
            loss_[s].update(last_loss)
        
        sigma = vae.sigma
        beta = vae.training_parameters['beta']
        if sigma.learned:
            sigma_train = 'learned'
            beta_sigma = rmse * np.sqrt(beta)
        elif sigma.is_rmse:
            sigma_train = 'rmse'
            beta_sigma = rmse * np.sqrt(beta)
        elif sigma.decay:
            sigma_train = 'decay'
            beta_sigma = rmse * np.sqrt(beta)
        else:
            sigma_train = 'constant'
            beta_sigma = sigma.value

        if architecture.type == 'cvae':
            if vae.training_parameters['learned_coder']:
                coder_dict = 'learned'
                if history['train_measures']:
                    # print('sl:366', rmse, *history.keys(), *[v for v in history.values()])
                    dict_var = history['train_measures'][-1]['ld-norm']
                else:
                    dict_var = vae.training_parameters['dictionary_variance']
            else:
                coder_dict = 'constant'
                dict_var = vae.training_parameters['dictionary_variance']
        else:
            coder_dict = None
            dict_var = 0.
            
        empty_optimizer = Optimizer([torch.nn.Parameter()], **training.optim)
        depth = (1 + len(architecture.encoder)
                 + len(architecture.decoder))
                 # + len(architecture.classifier))
        
        width = (architecture.latent_dim +
                 sum(architecture.encoder) +
                 sum(architecture.decoder) +
                 sum(architecture.classifier)) 

        # print('TBR', architecture.type, vae.job_number, *loss_['test'].keys())
        vae_dict = {'net': vae,
                    'job': vae.job_number,
                    'type': architecture.type,
                    'arch': arch,
                    'dict_var': dict_var,
                    'coder_dict': coder_dict,
                    'gamma': vae.training_parameters['gamma'],
                    'arch_code': arch_code,
                    'features': architecture.features['name'] if architecture.features else 'none',
                    'dir': directory,
                    'set': training.set,
                    'data_augmentation': training.data_augmentation,
                    'train_batch_size': train_batch_size,
                    'sigma': f'{sigma}',
                    'beta_sigma': beta_sigma,
                    'sigma_train': sigma_train,
                    'beta': beta,
                    'done': vae.train_history['epochs'],
                    'epochs': vae.training_parameters['epochs'],
                    'finished': vae.train_history['epochs'] >= vae.training_parameters['epochs'],
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
                    'batch_norm': architecture.batch_norm,
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


def find_by_job_number(*job_numbers, dir='jobs', load_net=True, force_dict=False, **kw):

    from cvae import ClassificationVariationalNetwork
    d = {}

    v_ = sum(collect_networks(dir, load_net=False, **kw), [])
    for number in job_numbers:
        for v in v_:
            if v['job'] == number:
                d[number] = v
                if load_net:
                    d[number]['net'] = ClassificationVariationalNetwork.load(v['dir'], **kw)

    return d if len(job_numbers) > 1 or force_dict else d[job_numbers[0]]


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
                    tpr=[0.95], tnr=False, sorting_keys=[]):
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
        return {s: test_results_df(nets,
                                   best_net, first_method, ood,
                                   s,
                                   tpr, tnr,
                                   sorting_keys) for s in testsets}

    arch_index = ['type',
                  'depth',
                  'features',
                  'arch_code',
                  'K',
                  'dict_var',
    ]

    all_nets = [] if best_net else ['job', 'done']

    train_index = [
        'options',
        'optim_str',
        'L',
        'sigma',
        'sigma_train',
        'beta_sigma',
        'beta',
    ] + all_nets

    indices = arch_index + train_index
    
    # acc_cols = ['best_accuracy', 'accuracies']
    # ood_cols = ['ood_fpr', 'ood_fprs']

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
        if type == 'pc':
            return 100 * x
        return x
        
    col_format = {c: _f for c in df.columns}
    for c in df.columns[df.columns.isin(['measures'], level=0)]:
        col_format[c] = lambda x: _f(x, 'measures')

    sorting_levels = list(df.index.names)
    sorting_levels.remove('sigma')
    sorting_levels.remove('sigma_train')

    if sorting_keys:
        sorting_keys_ = [k.replace('-', '_') for k in sorting_keys]
        for k in sorting_keys_:
            sorting_levels.remove(k)
        sorting_levels = sorting_keys_ + sorting_levels
    
    return df.sort_index(level=sorting_levels).apply(col_format)
        
    if not best_method:
        
        df = df.drop('accuracies', axis=1).join(pd.DataFrame(df.accuracies.values.tolist()))
        # print('\n\n**** 341 *** dict of accuracies \n', df.head(), '\n***************')
    return df.fillna(np.nan)

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
    
    return df.reindex(sorted(df.columns), axis=1).fillna(np.nan)


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

    dim = {'A': (10,), 'B': (1,), 'I': (3, 32, 32)}
    batch_size = 512
    device = 'cuda'
    
    tensors = {k: torch.randn(*dim[k], 7, device=device) for k in dim}
    
    r = LossRecorder(batch_size)  # , **tensors)    
    r.num_batch = 4
    r.epochs = 10
    
    for _ in range(3):
        r.append_batch(**{k: torch.randn(*dim[k], batch_size) for k in dim})

    r.save('/tmp/r.pth')

    r_ = LossRecorder.load('/tmp/r.pth')
