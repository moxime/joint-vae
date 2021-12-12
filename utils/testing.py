from utils.torch_load import get_same_size_by_name
from utils.save_load import find_by_job_number, available_results
from utils.misc import make_list
import logging
import numpy as np


def testing_plan(model, wanted_epoch='last', min_samples=1000, epoch_tolerance=5,
                 available_by_compute=10000,
                 predict_methods='all', ood_sets='all', ood_methods='all', misclass_methods='all'):

    available = available_results(model,
                                  samples_available_by_compute=available_by_compute,
                                  predict_methods=predict_methods,
                                  misclass_methods=misclass_methods,
                                  ood_sets=ood_sets,
                                  ood_methods=ood_methods,
                                  wanted_epoch=wanted_epoch, epoch_tolerance=epoch_tolerance,
    )



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
    
    
