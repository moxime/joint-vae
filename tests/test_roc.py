import time
import numpy as np
from utils.roc_curves import roc_curve
import argparse
from utils.save_load import LossRecorder

n = 1000

parser = argparse.ArgumentParser()
parser.add_argument('n', nargs='?', default=n, type=int)
parser.add_argument('-m', default=1., type=float)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--two-sided', nargs='*', type=int)
parser.add_argument('--nr', const=0, type=int, nargs='?')
parser.add_argument('--key', default='total')
parser.add_argument('--validation', default=0., type=float)
parser.add_argument('--in-rec')
parser.add_argument('--out-rec')

arg = parser.parse_args()

sign = {'total': -1, 'iws': +1}

if arg.two_sided is None:
    two_side = False

elif not arg.two_sided:
    two_side = 'around-mean'
else:
    two_side = tuple(arg.two_sided)

print('2S', two_side)

n_in = arg.n
n_out = arg.n

mean_in, std_in = arg.m, 1

mean_out, std_out = 0, 1

kept_tpr = [_/100 for _ in range(90, 100)]

if arg.nr is not None:
    np.random.seed(arg.nr)

if arg.in_rec:
    print('loading', arg.in_rec)
    ins = sign[arg.key] * LossRecorder.load(arg.in_rec, device='cpu')._tensors[arg.key]
    print(*ins.shape)
else:
    ins = mean_in + std_in * np.random.randn(n_in)

if arg.out_rec:
    print('loading', arg.out_rec)
    outs = sign[arg.key] * LossRecorder.load(arg.out_rec, device='cpu')._tensors[arg.key]
else:
    outs = mean_out + std_out * np.random.randn(n_out)

t0 = time.time()
auc, kept_fpr, kept_tpr, _ = roc_curve(ins, outs, *kept_tpr,
                                       debug=arg.debug,
                                       two_sided=two_side,
                                       validation=arg.validation)

print('AUC = {:.2%}'.format(auc))
for f, t in zip(kept_fpr, kept_tpr):
    print('{:6.2%} : {:6.2%}'.format(t, f))

print('T = {:.1f} us/i'.format((time.time() - t0) / (n_in + n_out) * 1e6))
