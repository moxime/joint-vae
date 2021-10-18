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
args.add_argument('direct_load')
args.add_argument('--plot', action='store_true')
args.add_argument('-N', type=int, default=1000)
args.add_argument('-T', type=int, nargs='+', default=[1])

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

rf = os.path.join(net.saved_dir, 'samples', 'last', 'record-{}.pth'.format(net.training_parameters['set']))
recorder = LossRecorder.load(rf, device='cpu')

losses = recorder._tensors
y = losses.pop('y_true')

logits = losses.pop('logits').T

y_ = net.predict_after_evaluate(logits, losses, method='iws')

correct = (y == y_)
n_correct = correct.sum().item()
missed = (y != y_)
accuracy = n_correct / len(y)

# for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
logp_x_y_max = losses['iws'].max(axis=0)[0]
ood_threshold = logp_x_y_max.sort()[0][int(np.floor(len(y) * (1 - tpr)))]

for T in a.T:
    
    print('{:_^79}'.format(f'T={T}'))
    p_y_x = torch.nn.functional.softmax(losses['iws'] / T, dim=0).max(axis=0)[0]

    pos_d = (logp_x_y_max >= ood_threshold)

    misclass_thresholds = p_y_x.sort()[0]

    num_of = {}

    idx_d = {'tp': pos_d, 'fn': ~pos_d}

    cols = {'tptp': 'TP',
            'tpfn': 'FN',
            'tpfp': 'FP',
            'tptn': 'TN',
            'fntp': 'FN',
            'fnfn': 'FN',
            'fnfp': 'FN',
            'fntn': 'TN'}

    totals = {_: None for _ in set(cols.values())}
    
    _n = len(misclass_thresholds)
    _i = [_ * _n // 100 for _ in range(30)]


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
    plt.plot(logp_x_y_max[i_][correct[i_]].numpy(), p_y_x[i_][correct[i_]].numpy(), 'b.')
    plt.plot(logp_x_y_max[i_][missed[i_]].numpy(), p_y_x[i_][missed[i_]].numpy(), 'r.')
    plt.title(T)

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


