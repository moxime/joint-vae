import os
import sys
from utils.save_load import LossRecorder, load_json, available_results, make_dict_from_model
from utils.torch_load import get_heldout_classes_by_name, set_dict
import numpy as np
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters, ParamFilter
import matplotlib.pyplot as plt
import logging

logging.getLogger().setLevel(logging.WARNING)

tset = 'cifar10-?'

rmodels = load_json('jobs', 'models-home.json')


if __name__ == '__main__':

    filters = DictOfListsOfParamFilters()

    filters.add('set', ParamFilter.from_string(tset, type=str))
    filters.add('done', ParamFilter.from_string('400..', type=int))
    filters.add('sigma_train', ParamFilter.from_string('coded', type=str))
    filters.add('type', ParamFilter.from_string('cvae', type=str))
    filters.add('features', ParamFilter.from_string('vgg19', type=str))

    mdirs = [_ for _ in rmodels if filters.filter(rmodels[_])]

    loaded_files = []
    with open('/tmp/files', 'w') as f:
        for mdir in mdirs:
            model = rmodels[mdir]
            ho = model['h/o']
            model['heldout'] = int(ho)
            ood_rec_file = 'record-' + tset.replace('-?', '+' + ho + '.pth')
            ind_rec_file = 'record-' + tset.replace('-?', '-' + ho + '.pth')
            rec_files = [os.path.join(mdir, 'samples', 'last', _) for _ in (ood_rec_file, ind_rec_file)]
            append = True
            for rec_file in rec_files:
                if not os.path.exists(rec_file):
                    logging.info('File does not exist: %s', rec_file)
                    f.write(rec_file + '\n')
                    append = False
            if append:
                loaded_files.append(mdir)

    print(len(loaded_files))
    if not loaded_files:
        logging.warning('Exiting, load files')
        sys.exit()

    fprs = []
    confs = []

    for mdir in loaded_files:

        model = M.load(mdir, load_net=False)
        fpr_ = make_dict_from_model(model, mdir, wanted_epoch='min-loss')['ood_fpr'][tset.replace('-', '+')]
        if fpr_ is not None and 0.95 in fpr_:
            fpr = fpr_[0.95]
        else:
            logging.info('Pb with %s', mdir)
            continue

        record_dir = os.path.join(mdir, 'samples', 'last')
        parent_set = rmodels[mdir]['set'].strip('-?')

        ho = rmodels[mdir]['heldout']
        parent_classes = set_dict[parent_set]['classes']
        ho_class = parent_classes[ho]

        hi_classes = [i for i, _ in enumerate(parent_classes) if mdir != ho_class]

        hi_classes = {i: parent_classes[_] for i, _ in enumerate(hi_classes)}

        recorders = LossRecorder.loadall(record_dir)
        
        ood_rec = [_ for _ in recorders if '+' in _][0]

        y_ood = recorders[ood_rec]._tensors['cross_y'].argmin(axis=0)

        
        y_pred_dist = {_: 0 for _ in hi_classes.values()}

        for y in y_ood:
            y_pred_dist[hi_classes[y.item()]] += 1 / len(y_ood)

        confusion = 0

        def H(p):
            n = len(p)
            entropy = - sum(_ * np.log(_ + 1e-100) / np.log(n) for _ in p)
            return entropy

        conf = H(y_pred_dist.values())
        conf = max(y_pred_dist.values())
        max_confusion = 0.5
        fprs.append(fpr)
        confs.append(conf)

        print('__', model.job_number)  #, model.training_parameters.get('early-min-loss'))
        print('heldout: {} ({})'.format(ho_class, ho))
        print('conf: {:.1%} fpr:{:.1%}'.format(conf, fpr))
        for c in sorted(y_pred_dist, key=y_pred_dist.get, reverse=True):
            if confusion <= max_confusion:
                print('{:10} {:5.1%}'.format(c, y_pred_dist[c]))
            confusion += y_pred_dist[c]

    logging.info('Plotting')
    fig, ax = plt.subplots()
    ax.scatter(fprs, confs)
    ax.set_xlabel('FPR')
    ax.set_ylabel('Classification Confidence')
    a = fig.show()

    input()
