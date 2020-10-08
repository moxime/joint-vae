from cvae import ClassificationVariationalNetwork
import data.torch_load as dl

load_dir = './jobs/svhn/the'
load_dir = './jobs/fashion32/the'

# load_dir = './jobs/fashion32/the'
# load_dir = './jobs/mnist/the'



print('Load net', end='') 
net = ClassificationVariationalNetwork.load(load_dir)
print(' to gpu')
net.to('cuda')


print('Getting sets')
trainset_name = net.training['set']
trainset, testset = dl.get_dataset(trainset_name)
oodsets = [dl.get_dataset(n)[1] for n in testset.same_size]

oodset = oodsets[0]


batch_size = 20
x, y = {}, {}


print('Getting batches')
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

methods = ('max', 'std', 'mag', 'mean', 'nstd', 'IYx')


sets = ('test', 'ood')

types = ('mean', 'max', 'std', 'delta', 'nstd')
logpx_ = (logpx_mean, logpx_max, logpx_std, logpx_delta, logpx_nstd)


print('*\n' * 10)


for t, logpx in zip(types, logpx_):
    print(f'\n*** {t} ***')
    for s in sets:
        print(s)
        print(' | '.join(f'{l:-7.2f}' for l in logpx[s]))

    x_[s], y_[s], losses[s], _ = net.evaluate(x[s])

    measures[s] = net.batch_dist_measures(None, losses[s], methods)
                  

for m in methods:
    print('*' * 80 + '\n', m)
    for s in sets:
        print(s)
        print('|'.join(f'{l:-7.2f}' for l in measures[s][m]) )
        
net.ood_detection_rates(batch_size=20, num_batch=20)

