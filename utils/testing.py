from utils.torch_load import get_same_size_by_name
from utils.save_load import find_by_job_number, available_results
from utils.misc import make_list
import logging


def testing_plan(model, min_samples=1000, epoch_tolerance=10,
                 predict_methods='all', ood_sets='all', ood_methods='all', misclass_methods='all'):

    available = available_results(model, min_samples, epoch_tolerance,
                                  predict_methods=predict_methods,
                                  misclass_methods=misclass_methods,
                                  ood_sets=ood_sets,
                                  ood_methods=ood_methods)

    if isinstance(model, str):
        model = Model.load(model, load_state=False)
    elif isinstance(model, dict):
        model = model['net']

    predict_methods = make_list(predict_methods, model.predict_methods)
    ood_methods = make_list(ood_methods, model.ood_methods)

    testset = model.training_parameters['set']
    all_ood_sets = get_same_size_by_name(testset)
    ood_sets = make_list(ood_sets, all_ood_sets)

    sets = [testset] + ood_sets
    
    from_recorder = {s: {} for s in sets}
    from_compute = {s: {} for s in sets}

    methods = {testset: predict_methods}
    methods.update({s: ood_methods for s in ood_sets})

    # return available, None
    
    for s in sets:
        for m in available['recorders'][s]:
            # print('***', s, m)
            n, epoch = ({w: available[w][s][m][k] for w in ('json', 'recorders')} for k in ('n', 'epochs'))

            delta_epoch = epoch['recorders'] - epoch['json']
            # print('***', s, m, 'r', epoch['recorders'], 'j', epoch['json'])
            n = available['recorders'][s][m]['n']
            
            if delta_epoch > epoch_tolerance or (available['json'][s][m]['n'] < min_samples and n > min_samples):
                from_recorder[s][m] = {'delta_epoch': delta_epoch, 'n': n}

            n = 10000  # TBC 
            delta_epoch = model.trained - epoch['recorders']
            if delta_epoch > epoch_tolerance:
                from_compute[s][m] = {'delta_epoch': delta_epoch, 'n': n}
                
    return from_recorder, from_compute


def worth_computing(model, from_which='recorder', **kw):

    froms = make_list(from_which, ('recorder', 'compute')) 
    from_recorder, from_compute = testing_plan(model, **kw)

    resd = {}
    for w, d in zip(('recorder', 'compute'), (from_recorder, from_compute)):
        resd[w] = sum(1 if d[k] else 0 for k in d)
        if not list(d.values())[0] and resd[w]:
            resd[w] += 1
        
    if len(froms) > 1:
        return {f: resd[f] for f in froms}

    return resd[from_which]


if __name__ == '__main__':

    j = 137540
    j = 111076
    j = 169
    
    model = find_by_job_number(j, load_net=False)

    print('Loaded')
    from_r, from_c = testing_plan(model)
    
    
