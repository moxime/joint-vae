import numpy as np
from utils.save_load import LossRecorder
from utils.roc_curves import roc_curve
import matplotlib.pyplot as plt

plt.close('all')

key = 'total'
sign = -1

tpr = 0.95
two_sided = (1, 1)

pth = {'in': '/tmp/r-in.pth', 'out': '/tmp/r.pth'}

bins = 200

r = {_: LossRecorder.load(pth[_], device='cpu') for _ in pth}

values = {_: sign * r[_][key] for _ in r}

v_min, v_max = values['out'].min().item(), values['out'].max().item()

h = {_: np.histogram(values[_], bins=bins, density=False, range=(v_min, v_max)) for _ in values}

for _ in values:
    v_ = (h[_][1][:-1] + h[_][1][1:])/2
    h_ = h[_][0] / h[_][0].max()
    plt.plot(v_, h_/h_.max(), label=_)

plt.legend()
plt.show(block=False)

n = len(values['in'])

v_sorted = values['in'].sort()[0]

t = v_sorted[n - int(n * tpr)]
fpr = (values['out'] > t).float().mean()

print('FPR@{:.0f}={:.2%}'.format(100*tpr, fpr))

auc, fpr, tpr_, _ = roc_curve(values['in'], values['out'], tpr, two_sided=False, validation=0)

print('FPR@{:.0f}={:.2%} AUC={:.1%}'.format(100*tpr, fpr[0], auc))

print('***', *two_sided, '***')

two_sided_ = tuple(two_sided[_] / sum(two_sided) for _ in (0, 1))
t_low = v_sorted[int(n * (1 - tpr) * two_sided_[0])]
t_high = v_sorted[int(n - n * (1 - tpr) * two_sided_[1])]

fpr = ((values['out'] > t_low) & (values['out'] <= t_high)).float().mean()

print('FPR@{:.0f}={:.2%}'.format(100*tpr, fpr))


auc, fpr, tpr_, _ = roc_curve(values['in'], values['out'], tpr, two_sided=two_sided, validation=0)

print('FPR@{:.0f}={:.2%} AUC={:.1%}'.format(100*tpr, fpr[0], auc))

for val in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8):

    auc, fpr, tpr_, _ = roc_curve(values['in'], values['out'], tpr, two_sided=two_sided, validation=val)
    print('val={} auc={:.1%} fpr={:.1%}'.format(val, auc, fpr[0]))

print('*** mean ***')

mean = values['in'].mean()

values_ = {_: (values[_] - mean).abs() for _ in values}
v_sorted = values_['in'].sort()[0]

t = v_sorted[n - int(n * tpr)]
fpr = (values_['out'] > t).float().mean()

print('FPR@{:.0f}={:.2%}'.format(100*tpr, fpr))

for val in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8):

    auc, fpr, tpr_, _ = roc_curve(values['in'], values['out'], tpr, two_sided='around-mean', validation=val)
    print('val={} auc={:.1%} fpr={:.1%}'.format(val, auc, fpr[0]))


auc, fpr, tpr_, _ = roc_curve(values['in'], values['out'], tpr, two_sided='around-mean', debug=True)
