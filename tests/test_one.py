from cvae import ClassificationVariationalNetwork
import data.torch_load as dl

load_dir = './jobs/svhn/the'
load_dir = './jobs/fashion32/the'

net = ClassificationVariationalNetwork.load(load_dir)
net.to('cuda')

trainset_name = net.training['set']
trainset, testset = dl.get_dataset(trainset_name)
oodsets = [dl.get_dataset(n)[1] for n in testset.same_size]

oodset = oodsets[0]


batch_size = 20
x, y = {}, {}



x['ood'], y['ood'] = dl.get_batch(oodset, batch_size=batch_size)
x['test'], y['test'] = dl.get_batch(testset, batch_size=batch_size)


x_ = {}
y_ = {}
losses = {}
measures = {}
logpx_std = {}
logpx_mean = {}
logpx_min = {}
logpx_max = {}
logpx_delta = {}
logpx_nstd = {}

sets = ('test', 'ood')

types = ('mean', 'max', 'std', 'delta', 'nstd')
logpx_ = (logpx_mean, logpx_max, logpx_std, logpx_delta, logpx_nstd)


print('*\n' * 10)
for s in sets:
    print(f'Evaluating {s}_batch')
    x_[s], y_[s], losses[s], measures[s] = net.evaluate(x[s])

    logpx_s = -losses[s]['total']
    logpx_max[s], _ = logpx_s.max(axis=0)
    logpx_min[s], _ = logpx_s.min(axis=0)
    logpx_ref = logpx_max[s]
    logpx_delta[s] = (logpx_s - logpx_ref).exp()
    logpx_mean[s] = logpx_delta[s].mean(axis=0).log() + logpx_ref
    logpx_std[s] = logpx_delta[s].std(axis=0).log() + logpx_max[s]
    logpx_delta[s] = logpx_max[s] - logpx_mean[s]
    logpx_nstd[s] = logpx_mean[s] - logpx_std[s]

# print(f'*** {s} ***')


for t, logpx in zip(types, logpx_):
    print(f'\n*** {t} ***')
    for s in sets:
        print(s)
        print(' | '.join(f'{l:-7.2f}' for l in logpx[s]))
