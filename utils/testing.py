from utils.torch_load import get_same_size_by_name
from utils.save_load import find_by_job_number, available_results
from utils.misc import make_list
import logging
import numpy as np


def testing_plan(model, wanted_epoch='last', min_samples=1000, epoch_tolerance=5,
                 available_by_compute=10000,
                 predict_methods='all', ood_sets='all', ood_methods='all', misclass_methods='all'):

    available = available_results(model, min_samples=min_samples,
                                  samples_available_by_compute=available_by_compute,
                                  predict_methods=predict_methods,
                                  misclass_methods=misclass_methods,
                                  ood_sets=ood_sets,
                                  ood_methods=ood_methods,
                                  wanted_epoch=wanted_epoch, epoch_tolerance=epoch_tolerance,
    )
    
    if isinstance(model, str):
        model = Model.load(model, load_state=False)
    elif isinstance(model, dict):
        model = model['net']

    if wanted_epoch == 'last':
        wanted_epoch = model.trained

    predict_methods = make_list(predict_methods, model.predict_methods)
    ood_methods = make_list(ood_methods, model.ood_methods)

    testset = model.training_parameters['set']
    all_ood_sets = get_same_size_by_name(testset)
    ood_sets = make_list(ood_sets, all_ood_sets)

    sets = [testset] + ood_sets
    
    from_recorder = {s: {} for s in sets}
    from_compute = {s: {} for s in sets}
    from_json = {s: {} for s in sets}

    methods = {testset: predict_methods}
    methods.update({s: ood_methods for s in ood_sets})

    # return available, None
    epochs_to_look_for = [e for e in available['recorders'] if abs(e - wanted_epoch) <= epoch_tolerance]
    n = {}
    """
    which:
    json >= min_samples, recorder >= min_samples, compute >= min_samples, json >= recorder
    """
    # which = {(True, _, __): 'json' for _ in (False, True) for __ in (False, True)}
    # which.update({(False, True, _): 'recorder' for _ in (True, False)})
    # which.update({(False, False, False): 'compute' if available_by_compute else None})
    # which.update({(False, False, True): 'compute' if available_by_compute else 'recorder'})
    if epochs_to_look_for:
        for s in sets:
            for m in available['recorders'][epochs_to_look_for[0]][s]:
                max_n = {w: {'n': 0, 'delta_epoch': None} for w in ('json', 'recorders', 'compute')}
                for e in epochs_to_look_for:
                    for w in max_n:
                        n = available[w][e][s][m]['n']
                        if n >= max_n[w]['n']:
                            max_n[w] = {'n': n, 'delta_epoch': e - wanted_epoch}
                            # if w == 'recorders': print('***', e, w, n, max_n[w])
                if max_n['recorders']['n'] > max_n['json']['n']:
                    # print('***', e, max_n['recorders']['n'], max_n['json']['n'])
                    if max_n['recorders']['n'] >= min_samples or max_n['compute'] < 2 * max_n['recorders']['n']:
                        from_recorder[s][m] = max_n['recorders']
                        from_recorder['rec_sub_dir'] = available['recorders'][e][s][m].get('rec_sub_dir')
                    else:
                        from_compute[s][m] = max_n['compute']
                elif max_n['json']['n'] >= min_samples:
                        from_json[s][m] = max_n['json']
                elif max_n['recorders']['n'] < min_samples and max_n['compute']['n'] >=  min_samples:
                    from_compute[s][m] = max_n['compute']
                
    return {'json': from_json, 'recorders': from_recorder, 'compute': from_compute}


def worth_computing(model, from_which='recorders', **kw):

    from_which = make_list(from_which, ('recorders', 'compute', 'json')) 
    froms = testing_plan(model, **kw)

    resd = {}
    for w in from_which:
        resd[w] = sum(1 if froms[w][k] else 0 for k in froms[w])
        if not list(froms[w])[0] and resd[w]:
            resd[w] += 1
        
    if len(froms) > 1:
        return {f: resd[f] for f in from_which}

    return resd[from_which]


def early_stopping(model, strategy='min', which='loss', full_valid=10):
    r""" Returns the epoch at which it should be stopped"""

    if isinstance(model, dict):
        model = model['net']
    mtype = model.type
    history = model.train_history
    has_validation = 'validation_loss' in history

    valid_k = 'validation'
    if not has_validation:
        logging.warning('No validation has been produced for {}'.format(model.job_number))
        valid_k = 'test'
        
    measures = history[valid_k + '_measures']
    losses = history[valid_k + '_loss']

    metrics = {}
    
    kl = np.array([_['kl'] for _ in history[valid_k + '_loss']])
    metrics['loss'] = np.array([_['total'] for _ in history[valid_k + '_loss']])
    if mtype in ('cvae', 'vae'):
        sigma = np.array([_['sigma'] for _ in history[valid_k + '_measures']])
        metrics['mse'] = np.array([_['mse'] for _ in history[valid_k + '_measures']])

    validation = metrics[which]
    epoch = {} 

    epoch['min'] = validation[::full_valid].argmin() * full_valid

    return epoch[strategy]
                        

if __name__ == '__main__':

    j = 137540
    j = 111076
    j = 169
    
    model = find_by_job_number(j, load_net=False)

    print('Loaded')
    froms = testing_plan(model)
    
    
