from cvae import ClassificationVariationalNetwork
import data.torch_load as dl

# load_dir = './jobs/svhn/the'
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
loss_std = {}
loss_mean = {}
loss_max = {}

sets = ('test', 'ood')

for s in sets:
    print(f'Evaluating {s}_batch')
    x_[s], y_[s], losses[s], measures[s] = net.evaluate(x[s])

    loss_std[s] = losses[s]['total'].std(axis=0)
    loss_mean[s] = losses[s]['total'].mean(axis=0)
    loss_max[s], _ = losses[s]['total'].max(axis=0)

print('\n' * 10)
for s in sets:

    print(f'*** {s} ***')
    print('mean')
    print(' - '.join(f'{l:6.2f}' for l in loss_mean[s]))
    print('max')
    print(' - '.join(f'{l:6.2f}' for l in loss_max[s]))
    print('std')
    print(' - '.join(f'{l:6.2f}' for l in loss_std[s]))

