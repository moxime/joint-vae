from cvae import ClassificationVariationalNetwork
import data.torch_load as dl
import logging
from utils.save_load import collect_networks

logging.getLogger().setLevel(logging.DEBUG)

load_dir = './jobs/svhn/the'
load_dir = './jobs/fashion32/the'

# load_dir = './jobs/fashion32/the'
# load_dir = './jobs/mnist/the'
load_dir = 'jobs/cifar10/the-best'


print('Load net', end='') 
net = ClassificationVariationalNetwork.load(load_dir, load_state=False)
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

out = net.evaluate(x['test'], y['test'])

x_reco, y_est, mu, log_var, z = net.forward(x['test'], y['test'])
        

"""
x_ = {}
y_ = {}
losses = {}
measures = {}

methods = ('max', 'std', 'mag', 'mean', 'nstd', 'IYx')
sets = ('test', 'ood')

types = ('mean', 'max', 'std', 'delta', 'nstd')

print('*\n' * 10)


for s in sets:
    print(s)

    x_[s], y_[s], losses[s], _ = net.evaluate(x[s])
    measures[s] = net.batch_dist_measures(None, losses[s], methods)
                  

for m in methods:
    print('*' * 80 + '\n', m)
    for s in sets:
        print(s)
        print('|'.join(f'{l:-7.2f}' for l in measures[s][m]) )
        
# net.ood_detection_rates(batch_size=20, num_batch=20)

"""
