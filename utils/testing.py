from cvae import ClassificationVariationalNetwork as Model
from utils.torch_load import get_same_size_by_name
from utils.save_load import LossRecorder, clean_results, last_samples, find_by_job_number
from utils.misc import make_list
import os

def testing_plan(model, min_samples=1000, epoch_tolerance=10,
                 predict_methods='all', ood_sets='all', ood_methods='all'):

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
    
    methods = {testset: predict_methods}
    methods.update({s: ood_methods for s in ood_sets})
    
    available = {'json': {}, 'recorders': {}}

    test_results = clean_results(model.testing, predict_methods)
    results = {s: clean_results(model.ood_results.get(s, {}), ood_methods) for s in ood_sets}

    results[testset] = test_results

    available['json'] = {s: {m: {k: results[s][m][k]
                                 for k in ('n', 'epochs')}
                             for m in results[s]}
                         for s in results}

    rec_dir = os.path.join(model.saved_dir, 'samples', 'last')

    available['recorders'] = {s: clean_results({}, methods[s]) for s in sets}

    if os.path.isdir(rec_dir):
        recorders = LossRecorder.loadall(rec_dir)
        epoch = last_samples(model)
        for s, r in recorders.items():
            n = len(r) * r.batch_size
            for m in methods[s]:
                if s == testset:
                    needed_components = model.predict_after_evaluate(None, None, method=m)
                else:
                    needed_components = model.batch_dist_measures(None, None, [m])[m]

                # print('***', s, m, 'c', *needed_components, 'r', *r.keys())
                all_components = all(c in r.keys() for c in needed_components)

                if all_components:
                    available['recorders'][s][m] = dict(n=n, epochs=epoch)

    from_recorder = {s: {} for s in sets}
    from_compute = {s: {} for s in sets}

    # return available, None
    
    for s in sets:
        for m in methods[s]:
            n, epoch = ({w: available[w][s][m][k] for w in ('json', 'recorders')} for k in ('n', 'epochs'))

            delta_epoch = epoch['recorders'] - epoch['json']
            # print('***', s, m, 'r', epoch['recorders'], 'j', epoch['json'])
            n = available['recorders'][s][m]['n']
            
            if delta_epoch > epoch_tolerance or (available['json'][s][m]['n'] < min_samples and n > min_samples):
                from_recorder[s][m] = {'delta_epoch': delta_epoch, 'n': n}

            n = len(r) * r.batch_size
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
    
    
