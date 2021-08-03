from cvae import ClassificationVariationalNetwork as Model
from utils.torch_load import get_same_size_by_name
from utils.save_load import LossRecorder, clean_results, last_samples
from utils.misc import make_list
import os

def testing_plan(model, min_samples=1000, predict_methods='all', ood_sets='all', ood_methods='all'):

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
                    needed_components = model.batch_dist_measures(None, None, m)[m]

                all_components = all(c in r.keys() for c in needed_components)

                if all_components:
                    available['recorders'][s][m] = dict(n=n, epochs=epoch)
                    
    return available



    
