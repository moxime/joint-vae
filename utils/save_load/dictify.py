import os
import logging
import hashlib

import numpy as np
import torch

from module.optimizers import Optimizer
import utils.torch_load as torchdl
from utils.misc import make_list
from utils.torch_load import get_same_size_by_name, get_shape_by_name
from utils.roc_curves import fpr_at_tpr

from .recorders import LossRecorder


def shorten_path(path, max_length=30):

    if len(path) > max_length:
        return (path[:max_length // 2 - 2] +
                '...' + path[-max_length // 2 + 2:])

    return path


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

    def _l2s(l_, c='-', empty='.'):
        if l_:
            return c.join(str(_) for _ in l_)
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


def model_subdir(model, *subdirs):

    if isinstance(model, str):
        directory = model

    elif isinstance(model, dict):
        try:
            directory = model['dir']
        except KeyError as e:
            logging.error('Keys: {}'.format(' ; '.join(sorted(model))))
            raise e

    else:
        directory = model.saved_dir

    return os.path.join(directory, *subdirs)


def last_samples(model):

    directory = model_subdir(model, 'samples')

    samples = [int(d) for d in os.listdir(directory) if d.isnumeric()]

    return max(samples)


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


def available_results(model,
                      testset='trained',
                      min_samples_by_class=10,
                      samples_available_by_class=800,
                      predict_methods='all',
                      misclass_methods='all',
                      oodsets='all',
                      wanted_epoch='last',
                      epoch_tolerance=5,
                      where='all',
                      ood_methods='all'):

    def debug_step():
        pass
        # caller = getframeinfo(stack()[1][0])
        # logging.debug('Step {}'.format(caller.lineno))

    logging.debug('Retreiving availale results for {}'.format(wanted_epoch))

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

    debug_step()

    for s in sets:
        # debug_step()
        C = get_shape_by_name(s)[-1]
        # debug_step()

        if not C:
            C = model.architecture['num_labels']
        min_samples[s] = C * min_samples_by_class
        samples_available_by_compute[s] = C * samples_available_by_class

    debug_step()

    # print(min_samples)
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

    debug_step()

    epochs.add(model.trained)
    # print('****', *epochs, '/', *test_results, '/', *ood_results)
    epochs = sorted(set.union(epochs, set(test_results), set(ood_results)))

    if wanted_epoch:
        epochs = [_ for _ in epochs if abs(_ - wanted_epoch) <= epoch_tolerance]
    test_results = {_: clean_results(test_results.get(_, {}), predict_methods) for _ in epochs}

    results = {}

    debug_step()
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

    debug_step()

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

    logging.debug('Availale results retrieved')

    return available


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
                n = min(len(l_) for l_ in lists)
                avge_res[m][k] = [np.mean([l_[i] for l_ in lists]) for i in range(n)]

        for k in min_keys:
            avge_res[m][k] = min(ood_results[s][m][k] for s in ood)

        for k in same_keys:
            avge_res[m][k] = ood_results[ood[0]][m][k]

    return avge_res


def needed_components(*methods):

    total = ('loss', 'logpx', 'sum', 'max', 'mag', 'std', 'mean')
    ncd = {'iws': ('iws',),
           'softiws': ('iws',),
           'closest': ('zdist',),
           'zdist': ('zdist',),
           'zdist~': ('zdist~',),
           'softzdist~': ('softzdist~',),
           'already': ('y_est_already',),
           'kl~': ('kl~',),
           'softkl~': ('softkl~',),
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
    wim_from = wim.get('from', model.job_number)
    wim_mean = wim.get('mean_shift') or wim.get('init_mean')
    wim_mix = wim.get('mix')
    if isinstance(wim_mix, (list, tuple)):
        wim_mix = wim_mix[1] / sum(wim_mix)

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
            'wim_mean': wim_mean,
            'wim_mix': wim_mix,
            'wim_alpha': wim.get('alpha'),
            'wim_train_size': wim.get('train_size'),
            'wim_moving_size': wim.get('moving_size'),
            'wim_from': wim_from,
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
