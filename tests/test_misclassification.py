from cvae import ClassificationVariationalNetwork as Model
from utils.save_load import find_by_job_number
import argparse
from utils.save_load import LossRecorder
import os
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve as prc, auc
from matplotlib import pyplot as plt

j = 133953
j = 134538
j = 133958
j = 172
j = 144418
tpr = 95


args = argparse.ArgumentParser()

args.add_argument('-j', default=j, type=int)
args.add_argument('--tpr', default=tpr, type=float)
args.add_argument('direct_load', nargs='?')
args.add_argument('--oodsets', nargs='*', default=[])
args.add_argument('--plot', action='store_true')
args.add_argument('--hide-correct', action='store_true')
args.add_argument('-N', type=int, default=1000)
args.add_argument('-T', type=int, nargs='+', default=[1])
args.add_argument('--soft', choices=['kl', 'iws'], default='iws')
args.add_argument('--hard', choices=['kl', 'iws'])
args.add_argument('--elbo', action='store_true')
args.add_argument('--2s', action='store_true', dest='two_sided')


a = args.parse_args()
j = a.j
tpr = a.tpr / 100

reload = False
if a.direct_load:
    net = Model.load(a.direct_load, load_state=False)
    print(net.job_number, 'loaded')
try:
    reload = net.job_number != j and not a.direct_load
except NameError:
    reload = True
    
if reload:
    net = find_by_job_number(j, load_state=False)['net']

dir_path = os.path.join(net.saved_dir, 'samples', 'last')
testset = net.training_parameters['set']

recorders = LossRecorder.loadall(dir_path, testset, *a.oodsets, device='cpu')

oodsets = [s for s in a.oodsets if s in recorders]

losses = recorders[testset]._tensors


for s in [testset] + oodsets:
    losses = recorders[s]._tensors
    y = losses.pop('y_true')

    logits = losses.pop('logits').T

    if a.elbo:
        logp_x_y_max = -losses['total'].min(axis=0)[0]
    else:
        logp_x_y_max = losses['iws'].max(axis=0)[0]
    y_ = net.predict_after_evaluate(logits, losses, method='iws')

    if s == testset:
        correct = (y == y_)
        missed = (y != y_)

    else:
        correct = (y != y)
        missed = (y == y)

    n_correct = correct.sum().item()
    accuracy = n_correct / len(y)

    if s == testset:        

        if a.two_sided:
            around = logp_x_y_max.mean()
            t_ = abs(around - logp_x_y_max).sort()[0]
            i = 0
            while ((around - t_[i] <= logp_x_y_max) * (logp_x_y_max <= around + t_[i])).sum() <= tpr * len(y):
                i += 1
            ood_thresholds = (around - t_[i], around + t_[i])

        else:
            ood_thresholds = (logp_x_y_max.sort()[0][int(np.floor(len(y) * (1 - tpr)))], np.inf)
        
    if a.elbo:
        logp_x_y_max = -losses['total'].min(axis=0)[0]
    else:
        logp_x_y_max = losses['iws'].max(axis=0)[0]

    pr = ((ood_thresholds[0] <= logp_x_y_max) * (logp_x_y_max <= ood_thresholds[1])).sum() / len(y)
    print('{} OOD {}PR {:.2f}'.format(s, 'T' if s == testset else 'F', pr.item() * 100))
    print('Accuracy: {:.2f}'.format(100 * accuracy))
    
    for T in a.T:

        metrics = a.hard or a.soft
        print('' if a.hard else 'soft', metrics)
        if a.hard:
            p_y_x = losses[metrics].max(axis=0)[0]
        else:
            p_y_x = torch.nn.functional.softmax(losses[metrics] / T, dim=0).max(axis=0)[0]

        if s == testset:
            _p = 5.1
            print(' ' * 8, end='|')
            print('|'.join(['{:_^{p}}'.format(_, p=int(_p) * 2 + 3) for _ in ('OOD', 'IND')]), end='|\n')

            print('{:_^79}'.format(f'T={T}'))

            pos_d = (logp_x_y_max >= ood_thresholds[0]) * (logp_x_y_max <= ood_thresholds[1])

            misclass_thresholds = p_y_x.sort()[0]

            num_of = {}

            idx_d = {'tp': pos_d, 'fn': ~pos_d}

            #                       Correctly detected as Ind Correctly
            cols = {'tptp': 'TP', # y                     y   y
                    'tpfn': 'FN', # y                     y   n
                    'tpfp': 'FP', # n                     y   y
                    'tptn': 'TN', # n                     y   n
                    'fntp': 'FN', # y                     n   y => n
                    'fnfn': 'FN', # y                     n   n
                    'fnfp': 'TN', # n                     n   y => n
                    'fntn': 'TN'} # n                     n   n

            totals = {_: None for _ in set(cols.values())}

            _n = len(misclass_thresholds)
            _i = [_ * _n // 100 for _ in range(10)]


            print('|{:_^8}'.format('t'), end='|')
            print('|'.join([f'{_:_^6}' for _ in cols]), end='|')
            print('|'.join([f'{_:_^7}' for _ in ['TPR', 'P', 'FPR']]), end='|\n')

            for t in misclass_thresholds[_i]:

                pos_c = (p_y_x >= t)
                idx_m = {'tp': correct * pos_c,
                         'fn': correct * ~pos_c,
                         'fp': ~correct * pos_c,
                         'tn': ~correct * ~pos_c}

                for ood_test in ('tp', 'fn'):
                    for misclass_test in ('tp', 'fn', 'fp', 'tn'):

                        idx = idx_d[ood_test] * idx_m[misclass_test]
                        num_of[ood_test + misclass_test] = idx.sum().item()  

                print('|{:8.1e}'.format(t), end='|')
                print('|'.join('{:6d}'.format(num_of[_]) for _ in num_of), end='|')

                for n in totals:
                    totals[n] = sum(num_of[_] for _ in cols if cols[_] == n)

                TPR = totals['TP'] / (totals['TP'] + totals['FN'])
                P = totals['TP'] / (totals['TP'] + totals['FP'])
                FPR = totals['FP'] / (totals['FP'] + totals['TN'])

                print('|'.join('{:6.1f}%'.format(100 * _) for _ in (TPR, P, FPR)), end='|\n')

        plt.figure()

        i_ = np.random.permutation(len(correct))[:a.N]
        logp_y_x = p_y_x.log()

        if s == testset:
            hthresholds = [p_y_x.sort()[0][-int(np.floor(tpr * len(y)))],
                           ood_thresholds[0],
                           max(logp_x_y_max) if not a.two_sided else ood_thresholds[1]]

        if not a.hide_correct:
            plt.semilogy(logp_x_y_max[i_][correct[i_]].numpy(), p_y_x[i_][correct[i_]].numpy(), 'b.')
        plt.semilogy(logp_x_y_max[i_][missed[i_]].numpy(), p_y_x[i_][missed[i_]].numpy(), 'r.')
        plt.title('{}: T={}'.format(s, T))

        if a.two_sided:
            plt.axvline(x=ood_thresholds[1], linestyle='--')
        plt.axvline(x=ood_thresholds[0], linestyle='--')

        plt.hlines(*hthresholds, linestyle='--')
        plt.show(block=False)

input()

    # tp = (correct * pos_d).sum().item()
    # fp = ((~correct) * pos_d).sum().item()
    # fn = (correct * (~pos_d)).sum().item()
    # tn = ((~correct) * (~pos_d)).sum().item()

    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # fpr = fp / (tn + fp)

    # print('Acc: {:.2f}% -> {:.2f}%'.format(accuracy * 100, accuracy_id * 100),
    #       'FPR: {:.2f}'.format(fpr * 100), 
    #       'Prec: {:.2f}'.format(precision * 100),
    #       'Rec:{:.2f}'.format(recall * 100))

    # meas = {'py': p_y_x.cpu(), 'log': logp_x_y_max.cpu(), 'kl': -kl.min(0)[0].cpu()}

    # p = {k: {} for k in meas}
    # r = {k: {} for k in meas}
    # aucpr = {k: {} for k in meas}

    # for k in meas:

    #     p[k]['correct'], r[k]['correct'], _ = prc(correct, meas[k])
    #     p[k]['error'], r[k]['error'], _ = prc(~correct, -meas[k])
    #     for w in ('correct', 'error'):
    #         aucpr[k][w] = auc(r[k][w], p[k][w])

    #     print(f'AUCPR {k:3}:', ' '.join(['{}: {:.1f}'.format(w, 100 * aucpr[k][w]) for w in ('correct', 'error')]))


