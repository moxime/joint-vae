import os
import sys
import argparse
from utils.save_load import LossRecorder, load_json, needed_remote_files
from utils.torch_load import get_same_size_by_name, get_classes_by_name
from utils.parameters import create_filter_parser
import numpy as np
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters
import matplotlib.pyplot as plt
import logging



parser = argparse.ArgumentParser()

parser.add_argument('--last', default=0, type=int)


logging.getLogger().setLevel(logging.WARNING)

tset = 'cifar10-?'

rmodels = load_json('jobs', 'models-home.json')

col_width = 10
str_col_width = '13.13'
flt_col_width = '5.1f'

wanted_ood_methods = ['iws-a-4-1', 'iws-2s', 'iws']

if __name__ == '__main__':

    args_from_file = ('--dataset cifar10 '
                      '--type cvae '
                      '--gamma 500 '
                      '--sigma-train coded '
                      '--coder-dict learned '
                      '--job-num 149127'
                      ).split()

    args, ra = parser.parse_known_args(None if sys.argv[0] else args_from_file)
    
    filter_parser = create_filter_parser()
    filter_args = filter_parser.parse_args(ra)
    
    # sys.exit()

    filters = DictOfListsOfParamFilters()

    for _ in filter_args.__dict__:
        filters.add(_, filter_args.__dict__[_])

    # filters.add('set', ParamFilter.from_string(tset, type=str))
    # filters.add('done', ParamFilter.from_string('400..', type=int))
    # filters.add('sigma_train', ParamFilter.from_string('coded', type=str))
    # filters.add('type', ParamFilter.from_string('cvae', type=str))
    # filters.add('features', ParamFilter.from_string('vgg19', type=str))

    mdirs = [_ for _ in rmodels if filters.filter(rmodels[_])][-args.last:]

    loaded_files = []
    with open('/tmp/files', 'w') as f:
        for mdir in mdirs:
            model = rmodels[mdir]
            ho = model['h/o']
            tset = model['set']
            if ho:
                model['ind'] = tset.replace('-?', '-' + ho)
                model['oods'] = [tset.replace('-?', '+' + ho)]
            else:
                model['ind'] = tset
                model['oods'] = get_same_size_by_name(tset)
            ood_rec_files = ['record-' + _ + '.pth' for _ in model['oods']]
            ind_rec_file = 'record-' + model['ind'] + '.pth'
               
            rec_files = [os.path.join(mdir, 'samples', 'last', _) for _ in ood_rec_files + [ind_rec_file]]
            append = True
            for rec_file in rec_files:
                if not os.path.exists(rec_file):
                    logging.info('File does not exist: %s', rec_file)
                    f.write(rec_file + '\n')
                    append = False
            if append:
                loaded_files.append(mdir)

    print(len(loaded_files), 'complete models')
    if not loaded_files:
        logging.warning('Exiting, load files')
        logging.warning('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')

        sys.exit()

    fprs = []
    confs = []

    for mdir in loaded_files:

        model = M.load(mdir, load_net=False)

        print('__', model.job_number, rmodels[mdir]['ind'],
              ':', *rmodels[mdir]['oods'])  # , model.training_parameters.get('early-min-loss'))

        record_dir = os.path.join(mdir, 'samples', 'last')
        recorders = LossRecorder.loadall(record_dir)

        is_testset = True

        classes_ = get_classes_by_name(rmodels[mdir]['ind'])  # + ['OOD']
        confusion_matrix = {}

        print(' ', f'{" ":{str_col_width}}', ' '.join(f'{_:{str_col_width}}' for _ in classes_))

        thresholds = (-np.inf, np.inf)
        for dset in [rmodels[mdir]['ind']] + rmodels[mdir]['oods']:

            rec = recorders[dset]
            iws = rec._tensors['iws'].max(axis=0)[0].cpu()
            if is_testset:
                sorted_iws = iws.sort()[0]
                n = len(sorted_iws)
                thresholds = (sorted_iws[n * 4 // 100], sorted_iws[-n * 1 // 100])

            classes = get_classes_by_name(dset)
            confusion_matrix[dset] = {c: {c_: {True: 0., False: 0.} for c_ in classes_} for c in classes}

            as_ood = (iws < thresholds[0]) | (iws > thresholds[1])

            # if not is_testset: print('FPR {:.2f}'.format(100 - 100 * as_ood.sum().item() / len(as_ood)))
            y_true = rec._tensors['y_true'].cpu()
            y_pred = rec._tensors['cross_y'].argmin(axis=0).cpu()
            # y_pred[as_ood] = -1

            for y, y_, o in zip(y_true, y_pred, as_ood):
                c = classes[y]
                c_ = classes_[y_]
                confusion_matrix[dset][c][c_][True] += 1. / (y_true == y).sum()
                if o:
                    confusion_matrix[dset][c][c_][False] += 1. / ((y_true == y) & (y_pred == y_)).sum()
            for c in classes:
                prefix = ' ' if is_testset else '*'
                line = confusion_matrix[dset][c]
                print(f'{prefix} {c:{str_col_width}}', end=' ')
                print(' '.join(f'{100 * line[_][True]:{flt_col_width}} ({100 * line[_][False]:{flt_col_width}})'
                               for _ in classes_))

            is_testset = False
    pass

