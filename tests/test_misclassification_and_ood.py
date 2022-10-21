from cvae import ClassificationVariationalNetwork as Model
from utils.save_load import find_by_job_number
import argparse
from utils.save_load import LossRecorder
import os
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve as prc, auc
from matplotlib import pyplot as plt
import sys

j = 133953
j = 134538
j = 133958
j = 172
j = 144418
tpr = 95


args = argparse.ArgumentParser()

args.add_argument('-j', default=j, type=int)
args.add_argument('--ood-tpr', default=100, type=float)
args.add_argument('--mis-tpr', default=tpr, type=float)
args.add_argument('direct_load', nargs='?')
args.add_argument('--oodsets', nargs='*', default=[])
args.add_argument('--plot', action='store_true')
args.add_argument('--hide-correct', action='store_true')
args.add_argument('-N', type=int, default=1000)
args.add_argument('-T', type=int, nargs='+', default=[1])
args.add_argument('--soft', choices=['kl', 'iws'], default='default')
args.add_argument('--hard', choices=['kl', 'iws'])
args.add_argument('--entropy', '-H', action='store_true')
args.add_argument('--elbo', action='store_true')
args.add_argument('--baseline', action='store_true')
args.add_argument('--2s', action='store_true', dest='two_sided')
args.add_argument('--print', action='store_true')

a = args.parse_args()
j = a.j
ood_tpr = a.ood_tpr / 100
mis_tpr = a.mis_tpr / 100

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

if net.type == 'vib':
    pass  # a.plot = False

if a.soft == 'default':
    if net.type == 'cvae':
        a.soft = 'iws'
    elif net.type == 'vib':
        a.hard = 'odin*'
    else:
        logging.error('Type %s of model not supported', net.type)
        sys.exit(1)

if a.elbo:
    a.soft = 'total'

if a.baseline:
    a.soft = 'logits'
    a.hard = None

recorders = LossRecorder.loadall(dir_path, testset, *a.oodsets, device='cpu')

oodsets = [s for s in a.oodsets if s in recorders]

losses = recorders[testset]._tensors

# for k in losses:
#     print(k, *losses[k].shape)

for s in [testset] + oodsets:

    losses = recorders[s]._tensors

    sign_for_ood = 1
    sign_for_mis = 1

    if a.hard == 'odin*':
        metrics_for_ood = [k for k in losses if k.startswith('odin')]
        metrics_for_mis = [k for k in losses if k.startswith('odin')]
        ndim_of_losses = dict(ood=1, mis=1)
        sign_for_mis = 1

    else:
        metrics_for_mis = [a.hard or a.soft]
        default_metrics = 'iws' if net.type == 'cvae' else metrics_for_mis[0]
        metrics_for_ood = ['total' if a.elbo else default_metrics]
        sign_for_ood = -1 if a.elbo else 1
        if 'kl' in metrics_for_mis or a.elbo:
            sign_for_mis = -1
        ndim_of_losses = dict(ood=2, mis=2)

    _s = '*** metrics {}{} of dim {} (ood) {}{}{} of dim {} (miss)'
    print(_s.format('' if sign_for_ood == 1 else '-',
                    metrics_for_ood[0],
                    ndim_of_losses['ood'],
                    '' if sign_for_mis == 1 else '-',
                    '' if a.hard else 'soft-',
                    metrics_for_mis[0],
                    ndim_of_losses['mis']
                    ))

    y = losses.pop('y_true')
    logits = losses['logits'].T

    y_ = net.predict_after_evaluate(logits, losses)

    if s == testset:
        correct = (y == y_)
        missed = (y != y_)

    else:
        correct = (y != y)
        missed = (y == y)

    mis_fpr_at_tpr = []
    for m_for_ood, m_for_mis in zip(metrics_for_ood, metrics_for_mis):

        print('*** OOD:', m_for_ood, '*** MIS:', m_for_mis)
        if ndim_of_losses['ood'] == 2:
            logp_x_y_max = (sign_for_ood * losses[m_for_ood]).max(axis=0)[0]
        else:
            logp_x_y_max = (sign_for_ood * losses[m_for_ood])

        n_correct = correct.sum().item()
        accuracy = n_correct / len(y)

        if s == testset:

            if a.two_sided:
                around = logp_x_y_max.mean()
                t_ = abs(around - logp_x_y_max).sort()[0]
                i = 0
                tp = ood_tpr = len(y)
                while ((around - t_[i] <= logp_x_y_max) * (logp_x_y_max <= around + t_[i])).sum() <= tp:
                    i += 1
                ood_thresholds = (around - t_[i], around + t_[i])

            else:
                ood_thresholds = (logp_x_y_max.sort()[0][int(np.floor(len(y) * (1 - ood_tpr)))], np.inf)

        pr = ((ood_thresholds[0] <= logp_x_y_max) * (logp_x_y_max <= ood_thresholds[1])).sum() / float(len(y))
        print('{} OOD {}PR {:.2f}'.format(s, 'T' if s == testset else 'F', pr.item() * 100))
        print('Accuracy: {:.2f}'.format(100 * accuracy))

        for T in a.T:

            print(m_for_mis)
            metrics = sign_for_mis * losses[m_for_mis]
            if a.hard:
                if ndim_of_losses['mis'] == 2:
                    p_y_x = metrics.max(axis=0)[0]
                else:
                    p_y_x = metrics
            else:
                assert ndim_of_losses['mis'] == 2
                p_y_x = torch.nn.functional.softmax(metrics / T, dim=0)
                if a.entropy:
                    # print(p_y_x.min())
                    p_y_x = ((p_y_x + 1e-10) * (p_y_x + 1e-10).log2()).sum(axis=0)
                else:
                    p_y_x = p_y_x.max(axis=0)[0]

            _d = {'s': s,
                  'c': p_y_x[correct].mean(),
                  'c_': p_y_x[correct].std(),
                  'm': p_y_x[missed].mean(),
                  'm_': p_y_x[missed].std()}
            _s = 'Metrics mean for {s} correct: {c:.4g} +- {c_:.4g} missed: {m:4g} +- {m_:.4g}'

            print(_s.format(**_d))

            if s == testset:
                _p = 5.1
                # print(' ' * 8, end='|')
                # print('|'.join(['{:_^{p}}'.format(_, p=int(_p) * 2 + 3) for _ in ('OOD', 'IND')]), end='|\n')

                print('{:_^79}'.format(f'T={T}'))

                pos_d = (logp_x_y_max >= ood_thresholds[0]) * (logp_x_y_max <= ood_thresholds[1])

                misclass_thresholds = p_y_x.sort()[0]

                num_of = {}

                idx_d = {'tp': pos_d, 'fn': ~pos_d}

                #                       Correctly detected as Ind Correctly
                cols = {'tptp': 'TP',  # y                     y   y
                        'tpfn': 'FN',  # y                     y   n
                        'tpfp': 'FP',  # n                     y   y
                        'tptn': 'TN',  # n                     y   n
                        'fntp': 'FN',  # y                     n   y => n
                        'fnfn': 'FN',  # y                     n   n
                        'fnfp': 'TN',  # n                     n   y => n
                        'fntn': 'TN'}  # n                     n   n

                totals = {_: None for _ in set(cols.values())}

                _n = len(misclass_thresholds)
                _i = [_ * _n // 100 for _ in range(10)]

                print('|{:_^8}'.format('t'), end='|')
                print('|'.join([f'{_:_^6}' for _ in cols]), end='|')
                print('|'.join([f'{_:_^7}' for _ in ['TPR', 'P', 'FPR']]), end='|\n')

                if m_for_mis == m_for_ood:
                    pass  # _i = [0]

                found_fpr_at_tpr = False
                for i, t in enumerate(misclass_thresholds):

                    found_fpr_at_tpr_now = False

                    pos_c = (p_y_x >= t)
                    idx_m = {'tp': correct * pos_c,
                             'fn': correct * ~pos_c,
                             'fp': ~correct * pos_c,
                             'tn': ~correct * ~pos_c}

                    for ood_test in ('tp', 'fn'):
                        for misclass_test in ('tp', 'fn', 'fp', 'tn'):

                            idx = idx_d[ood_test] * idx_m[misclass_test]
                            num_of[ood_test + misclass_test] = idx.sum().item()

                    for n in totals:
                        totals[n] = sum(num_of[_] for _ in cols if cols[_] == n)

                    TPR = totals['TP'] / (totals['TP'] + totals['FN'])
                    P = totals['TP'] / (totals['TP'] + totals['FP'])
                    FPR = totals['FP'] / (totals['FP'] + totals['TN'])

                    if TPR <= mis_tpr and not found_fpr_at_tpr:
                        mis_fpr_at_tpr.append((FPR, P))
                        found_fpr_at_tpr = True
                        found_fpr_at_tpr_now = True

                    if i in _i and a.print or found_fpr_at_tpr_now:
                        print('|{:8.1e}'.format(t), end='|')
                        print('|'.join('{:6d}'.format(num_of[_]) for _ in num_of), end='|')
                        print('|'.join('{:6.1f}%'.format(100 * _) for _ in (TPR, P, FPR)), end='|\n')

                    if found_fpr_at_tpr:
                        break

                if not found_fpr_at_tpr:
                    mis_fpr_at_tpr.append((FPR, P))

            if a.plot:
                plt.figure()

                i_ = np.random.permutation(len(correct))[:a.N]
                logp_y_x = p_y_x.log()

                if s == testset:
                    hthresholds = [p_y_x.sort()[0][-int(np.floor(mis_tpr * len(y)))],
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

    best_fpr = 1

    for m in metrics_for_mis:
        for T in a.T:
            FPR, P = mis_fpr_at_tpr.pop(0)
            if FPR < best_fpr:
                best_fpr = FPR
                best_P = P
                best_m = m
                best_T = T

    dP = best_P - accuracy
    print('{:16} {:4} at TPR {:.1f} : FPR={:.1f} P={:.1f} ({:.1f}{:+.1f})'.format(best_m,
                                                                                  best_T,
                                                                                  100 * mis_tpr,
                                                                                  100 * best_fpr,
                                                                                  100 * best_P,
                                                                                  100 * accuracy,
                                                                                  100 * dP
                                                                                  ))

if a.plot:
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
