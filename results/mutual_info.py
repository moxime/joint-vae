import sys
import os
import logging
import argparse
import torch
from utils.save_load import LossRecorder, find_by_job_number, needed_remote_files, iterable_over_subdirs


@iterable_over_subdirs(0, iterate_over_subdirs=list)
def process_directory(folder, compare_with, prediction_method):

    global recorders
    recorders = LossRecorder.loadall(folder, map_location='cpu')

    if not recorders:
        return None

    print('*** Processing {} ***'.format(folder))

    folder_name = folder.split(os.sep)[-1]

    if '-' in folder_name:
        agg_type = 'cascad'
        sep = '-'
    elif '|' in folder_name:
        agg_type = 'parallel'
        sep = '|'

    jobs = [int(_) for _ in folder_name.split(sep)]
    models = find_by_job_number(*jobs)
    mdirs = {_: models[_]['dir'] for _ in models}

    dataset = args.dataset

    if not dataset:
        if len(recorders) > 1:
            dataset = [_ for _ in recorders if _.endswith('90')][0][: -2]
            has_ood = True
        else:
            dataset = next(iter(recorders))
            has_ood = False

    else:
        has_ood = len(recorders) > 1

    has_ood = has_ood and args.ood

    if not has_ood:
        recorders = {dataset: recorders[dataset]}

    if compare_with or prediction_method:
        has_all_recorders = True
        with open('/tmp/files-{}'.format(folder_name), 'w') as f:
            which_rec = 'all' if has_ood else 'ind'
            needed = needed_remote_files(*mdirs.values(), epoch='last', which_rec=which_rec, state=False)
            for mdir, sdir in needed:
                logging.debug('{} for {}'.format(sdir[-30:], 'last'))
                has_all_recorders = False
                f.write(sdir + '\n')
        if not has_all_recorders:
            logging.warning('Will not compare, missing recorders')
            compare_with = []
            prediction_method = None
        else:
            epochs = {_: '{:04d}'.format(max(models[_]['net'].testing)) for _ in jobs}
            recorders_by_job = {_: LossRecorder.loadall(os.path.join(mdirs[_], 'samples', epochs[_]))
                                for _ in mdirs}

    print('*** {} *** {} ***'.format(folder_name, dataset))

    oodsets = list(recorders)
    oodsets.remove(dataset)

    if agg_type == 'parallel':
        measures = {'Im': {}}
    else:
        measures = {}
        for i in range(len(jobs)):
            for j in range(i):
                measures['Im-{}-{}'.format(i, j)] = {}

    for s in recorders:

        temps = {float(k.split('-')[-1]): k for k in recorders[s].keys() if k.startswith('Im')}

        if agg_type == 'parallel':
            measures['Im'][s] = {T: recorders[s]._tensors[temps[T]] for T in temps}
        else:
            i_ = 0
            for i in range(len(jobs)):
                for j in range(i):
                    measures['Im-{}-{}'.format(i, j)][s] = {T: recorders[s]._tensors[temps[T]][i_]
                                                            for T in temps}
                    i_ += 1

    y = {_: recorders[dataset]._tensors['y_true'] for _ in measures}

    for meas in compare_with:
        which, what = meas.split('-')
        assert which in ['kl', 'zdist']
        assert what in ['mean', 'sumprod']

        y_true = {j: recorders_by_job[j][dataset]._tensors['y_true'] for j in jobs}

        assert all(y_true[jobs[0]] == y_true[jobs[1]])

        y[meas] = y_true[jobs[0]]

        signs = {'kl': -1, 'zdist': -1}

        measures[meas] = {}
        for s in recorders:
            first_meas_for_temps = 'Im' if agg_type == 'parallel' else 'Im-1-0'
            meas_ = {j: recorders_by_job[j][s]._tensors[which] for j in jobs}
            for j in meas_:
                # print('***', j, s, *meas_[j].shape)
                pass
            meas_ = {T: torch.stack([(signs[which] * meas_[j] / T).softmax(0) for j in jobs])
                     for T in measures[first_meas_for_temps][s]}
            if what == 'sumprod':
                measures[meas][s] = {T: meas_[T].prod(0).sum(0) for T in meas_}
            elif what == 'mean':
                measures[meas][s] = {T: meas_[T].mean(0).max(0)[0] for T in meas_}

    y_ = {}
    i_true = {}
    acc = {}

    for meas in measures:

        if prediction_method and agg_type == 'parallel':

            kls = torch.stack([recorders_by_job[j][dataset]._tensors['kl'] for j in jobs])
            p_y_x = {T: (-kls / T).softmax(1) for T in measures[meas][dataset]}

            y_[meas] = {T: p_y_x[T].mean(0).argmax(0) for T in p_y_x}

        elif agg_type == 'cascad':
            y_[meas] = {T: recorders[dataset]._tensors['kl'][0].argmin(0) for T in measures[meas][dataset]}
            y[meas] = recorders[dataset]._tensors['y_true']

        else:  # parallel and no prediction_method

            y_[meas] = {T: recorders[dataset]._tensors['y_'] for T in measures[meas][dataset]}
            y[meas] = recorders[dataset]._tensors['y_true']

        first_temp = next(iter(measures[meas][dataset]))
        n = min(len(y[meas]), len(measures[meas][dataset][first_temp]), len(y_[meas][first_temp]))
        print('***', meas, n)
        y[meas] = y[meas][:n]
        y_[meas] = {T: y_[meas][T][:n] for T in y_[meas]}
        measures[meas][dataset] = {T: measures[meas][dataset][T][:n] for T in y_[meas]}

        i_true[meas] = {T: y[meas].cpu() == y_[meas][T].cpu() for T in y_[meas]}
        acc[meas] = {T: i_true[meas][T].float().mean() for T in y_[meas]}

        measures[meas]['correct'] = {T: measures[meas][dataset][T][i_true[meas][T]]
                                     for T in measures[meas][dataset]}
        measures[meas]['incorrect'] = {T: measures[meas][dataset][T][~i_true[meas][T]]
                                       for T in measures[meas][dataset]}

    tpr = 0.95

    thr = {m: {T: [measures[m][_][T].sort()[0][int(len(measures[m][_][T]) * (1 - tpr))]
                   for _ in ('correct', dataset)]
               for T in measures[m][dataset]}
           for m in measures}

    max_precision = {}
    for meas in measures:
        print('***', meas, '***')
        for T in temps:
            print(T)
            quantiles = {}
            pr = {}

            for _ in (dataset, 'correct', 'incorrect', *oodsets):

                sortedI = measures[meas][_][T].sort()[0]
                n = len(measures[meas][_][T])

                iles = [0.01, 0.5, 0.99]
                iles = []
                i = [int(n * q) for q in iles]
                quantiles[_] = [sortedI[_] for _ in i]
                pr[_] = [(sortedI >= t).sum() / n for t in thr[meas][T]]

            prec_rec = {'recall': pr['correct']}
            _acc = acc[meas][T]
            prec_rec[dataset] = pr[dataset]
            prec_rec['precision'] = [_acc / (_acc + (1 - _acc) * _fpr / _tpr)
                                     for (_fpr, _tpr) in zip(pr['incorrect'], pr['correct'])]

            rstr = '{:10}: {:.2%} -- {:.2%} (acc={:.2%})'

            for _ in (dataset, 'precision', 'recall'):

                print(rstr.format(_, *prec_rec[_], acc[meas][T]))

                if _ == 'precision':
                    if not max_precision.get(meas) or prec_rec['precision'][0] > max_precision[meas][1]:
                        max_precision[meas] = (T, prec_rec['precision'][0])

    for meas in measures:
        print('{:10} {:.2%} at T={}'.format(meas, max_precision[meas][1], max_precision[meas][0]))


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.WARNING)

    job_dir = 'parallel-jobs'
    job_dir = '/tmp/jobs'
    job_dir = 'cascad-jobs/fashion'
    job_dir = 'parallel-jobs/fashion'
    job_dir = 'parallel-jobs/svhn'

    default_compare_with = ['kl-sumprod', 'kl-mean', 'zdist-mean']

    logging.getLogger().setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()

    parser.add_argument('job_dir')

    parser.add_argument('--dataset')

    parser.add_argument('--ood', action='store_true')

    parser.add_argument('--compare', nargs='*')
    parser.add_argument('--prediction', choices=['mean', 'max'])

    args_from_file = '--compare --prediction mean {}'.format(job_dir).split()

    args = parser.parse_args(None if len(sys.argv) > 1 else args_from_file)

    compare_with = args.compare
    prediction_method = args.prediction

    if compare_with is None:
        compare_with = []

    elif not compare_with:
        compare_with = default_compare_with

    process_directory(args.job_dir, compare_with, prediction_method)
