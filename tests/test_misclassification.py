import os
import sys
import argparse
import torch
from utils.save_load import LossRecorder, load_json, needed_remote_files
from utils.torch_load import get_same_size_by_name, get_classes_by_name
from utils.parameters import parse_filters
import numpy as np
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters
import matplotlib.pyplot as plt
import logging
from utils.parameters import gethostname

parser = argparse.ArgumentParser()

parser.add_argument('--last', default=0, type=int)
parser.add_argument('--metrics', default='zdist')
parser.add_argument('--soft', action='store_true')
parser.add_argument('--by-classes', action='store_true')
parser.add_argument('-v', action='count', default=0)

rmodels = load_json('jobs', 'models-{}.json'.format(gethostname()))

col_width = 10
str_col_width = '13.13'
flt_col_width = '5.1f'

wanted_ood_methods = ['iws-a-4-1', 'iws-2s', 'iws']

# direction : where normal are supposed to be ('low', 'center', 'high')
direction_of_ind = {'zdist': 'low',
                    'kl': 'low',
                    'mahala': 'low',
                    'fisher_rao': 'low',
                    'iws': 'high'}

tpr = 0.95

prec_rec = 'P (R)'

if __name__ == '__main__':

    wanted = 'last'
    args_from_file = ('--dataset cifar10 '
                      '--type cvae '
                      '--gamma 500 '
                      '--sigma-train coded '
                      '--coder-dict learned '
                      '--last 1'
                      # '--job-num 149127'
                      ).split()

    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)

    logging.getLogger().setLevel(40 - 10 * args.v)
    
    metrics_for_mis = args.metrics

    filter_parser = parse_filters()
    filter_args = filter_parser.parse_args(ra)

    filters = DictOfListsOfParamFilters()

    for _ in filter_args.__dict__:
        filters.add(_, filter_args.__dict__[_])

    mdirs = [_ for _ in rmodels if filters.filter(rmodels[_])][-args.last:]

    total_models = len(mdirs)
    with open('/tmp/files', 'w') as f:

        for mdir, sdir in needed_remote_files(*mdirs, epoch=wanted, which_rec='ind', state=True):
            if mdir in mdirs:
                mdirs.remove(mdir)
            f.write(sdir + '\n')

    print(len(mdirs), 'complete model' + ('s' if len(mdirs) > 1 else ''), 'over', total_models)
    
    if not mdirs:
        logging.warning('Exiting, load files')
        logging.warning('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')

    fprs = []
    confs = []

    for mdir in mdirs:

        model = M.load(mdir, load_net=False)
        testset = model.training_parameters['set']

        oodsets = []

        if wanted == 'min-loss':
            epoch = model.training_parameters.get('early-min-loss', 'last')
        else:
            epoch = 'last'
        
        epoch_str = '{:0>4}'.format(epoch)

        print('__', model.job_number, testset,
              '@',  epoch)

        sample_dir =  os.path.join(mdir, 'samples')
        record_dir = os.path.join(sample_dir, epoch_str)
        recorders = LossRecorder.loadall(record_dir)

        if args.metrics not in recorders[testset].keys():
            model = M.load(mdir, load_state=True)
            model.to('cuda')
            logging.warning('We will train the model')
            i = 0
            while os.path.exists(os.path.join(sample_dir, f'{epoch_str}.{i}')):
                i += 1
            backup_dir = os.path.join(sample_dir, f'{epoch_str}.{i}')
            os.rename(record_dir, backup_dir)
            model.accuracy(batch_size=64, print_result='REC',
                           epoch=epoch, from_where='compute',
                           sample_dirs=[sample_dir])
            
        is_testset = True

        classes_ = get_classes_by_name(testset)  # + ['OOD']

        classes_.append('correct')
        classes_.append('missed')
        classes_.append(prec_rec)
        confusion_matrix = {}

        if args.by_classes:
            print(' ', f'{" ":{str_col_width}}', ' '.join(f'{_:{str_col_width}}' for _ in classes_))
        else:
            _c = ('correct', 'missed', prec_rec)
            print(' ', f'{" ":{str_col_width}}', ' '.join(f'{_:{str_col_width}}' for _ in _c))
            
        thresholds = (-np.inf, np.inf)
        for dset in [testset] + oodsets:

            try:
                rec = recorders[dset]
            except KeyError:
                print(record_dir)
                print(dset, *recorders)
                raise KeyError
            iws = rec._tensors['iws'].max(axis=0)[0].cpu()

            y_true = rec._tensors['y_true'].cpu()
            y_pred = rec._tensors['cross_y'].argmin(axis=0).cpu()

            if direction_of_ind[metrics_for_mis] == 'low':
                sign = -1
                which_threshold = 1
            else:
                sign = 1
                which_threshold = 0

            metrics_tensor = rec._tensors[metrics_for_mis]

            if args.soft:
                metrics_tensor = torch.nn.functional.softmax(-sign * metrics_tensor, dim=0)
                sign = 1
                which_threshold = 0

            classification_metrics = sign * (sign * metrics_tensor).max(axis=0)[0].cpu()
                
            if is_testset:

                y_correct = y_true == y_pred
                correct_metrics = classification_metrics[y_correct].sort()[0]

                n_correct = len(correct_metrics)
                
                if which_threshold == 1:
                    thresholds = -np.inf, correct_metrics[int(n_correct * tpr)]
                else:
                    thresholds = correct_metrics[-int(n_correct * tpr)], np.inf
                
                sorted_iws = iws.sort()[0]
                n = len(sorted_iws)
                ood_thresholds = (sorted_iws[n * 4 // 100], sorted_iws[-n * 1 // 100])

            classes = get_classes_by_name(dset)
            classes.append('set')
            confusion_matrix[dset] = {c: {c_: {True: 0., False: 0.} for c_ in classes_} for c in classes}

            as_ood = (iws < ood_thresholds[0]) | (iws > ood_thresholds[1])
            as_misclass = (classification_metrics < thresholds[0]) | (classification_metrics > thresholds[1])

            if False:
                print('{:.3g} ({:.3g} +-{:.3g}) {:.3g}'.format(thresholds[0], classification_metrics.mean(),
                                                               classification_metrics.std(),
                                                               thresholds[1]
                                                               ))
            # if not is_testset: print('FPR {:.2f}'.format(100 - 100 * as_ood.sum().item() / len(as_ood)))
            # y_pred[as_ood] = -1

            correct = y_pred == y_true
            missed = y_pred != y_true
            is_ood = not is_testset
            
            for y, y_, o, m in zip(y_true, y_pred, as_ood, as_misclass):
                c = classes[y]
                i_y = y_true == y
                c_ = classes_[y_]

                for c, i_y in zip((classes[y], 'set'), (y_true == y, torch.ones_like(y_true, dtype=bool))):

                    # Classfication, misclassification rates 
                    confusion_matrix[dset][c][c_][True] += 1. / i_y.sum()
                    if m:
                            # Detected as misclassified
                        confusion_matrix[dset][c][c_][False] += 1. / (i_y & (y_pred == y_)).sum()
                    if y_ == y:
                        # Accuracy
                        confusion_matrix[dset][c]['correct'][True] += 1. / (i_y).sum()
                        if m:
                            # 1 - TPR
                            confusion_matrix[dset][c]['correct'][False] += 1. / (i_y & correct).sum()
                        else:
                            # Precision = tp / (tp + fp)
                            confusion_matrix[dset][c][prec_rec][True] += 1. / (~as_misclass & i_y).sum()
                            # Recall = tp / (tp + fn)
                            confusion_matrix[dset][c][prec_rec][False] += 1. / (correct & i_y).sum()
                    else:
                        confusion_matrix[dset][c]['missed'][True] += 1. / (i_y).sum()
                        if m:
                            # TNR = 1 - FPR
                            confusion_matrix[dset][c]['missed'][False] += 1. / (i_y & missed).sum()

            if not args.by_classes:
                classes = ['set']
                classes_ = ['correct', 'missed', prec_rec]
            for c in classes:
                prefix = ' ' if is_testset else '*'
                line = confusion_matrix[dset][c]
                print(f'{prefix} {c:{str_col_width}}', end=' ')
                print(' '.join(f'{100 * line[_][True]:{flt_col_width}} ({100 * line[_][False]:{flt_col_width}})'
                               for _ in classes_))

            is_testset = False
