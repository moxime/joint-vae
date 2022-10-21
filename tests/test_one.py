from cvae import ClassificationVariationalNetwork
import data.torch_load as dl
import logging
from utils.save_load import collect_models, find_by_job_number
import torch
from scipy import stats
import numpy as np

from matplotlib import pyplot as plt

logging.getLogger().setLevel(logging.WARNING)


compute = False
compute = True

if compute:
    j = 107495
    j = 108183
    j = 37

    load_dir = find_by_job_number('./jobs', j, load_net=False)[j]['dir']

    print('Load net', end='')
    net = ClassificationVariationalNetwork.load(load_dir, load_state=True)
    print(' to gpu')
    net.to('cuda')
    net.latent_sampling = 16

    print('Getting sets')
    trainset_name = net.training['set']
    trainset, testset = dl.get_dataset(trainset_name, transformer=net.training['transformer'])
    oodsets = [dl.get_dataset(n)[1] for n in testset.same_size]
    oodset = oodsets[0]

    batch_size = 200
    n_batch = 1000 // batch_size

    loader = torch.utils.data.DataLoader(testset,
                                         shuffle=True,
                                         batch_size=batch_size)

    dictionary = net.encoder.latent_dictionary

    z_ = None
    K = net.latent_dim

    for i, data in enumerate(loader):
        if i > n_batch:
            break
        print(f'{i:2}/{n_batch}')
        x = data[0].to('cuda')
        y = data[1].to('cuda')

        with torch.no_grad():
            x_, _, _, _, z = net(x)
            if net.is_cvae:
                mu_y = dictionary.index_select(0, y).expand_as(z)
            else:
                mu_y = 0

        if z_ is None:
            z_ = (z - mu_y).reshape(-1, K)
        else:
            z_ = torch.cat((z_, (z - mu_y).reshape(-1, K)), dim=0)


f, a = plt.subplots(8, 16)
a_ = a.reshape(-1)


p = np.ndarray(z_.shape[1])
var = np.ndarray(z_.shape[1])

for k in range(z_.shape[1]):
    k2, p[k] = stats.normaltest(z_[:, k].cpu())
    var[k] = z_[:, k].var()
    print(f'{k:4}: {var[k]:.3g}')

for i, ax in enumerate(a_):

    ax.hist(z_[:, i].cpu(), bins=100)
    ax.set_title(f'{var[k]:g}')

plt.show(block=False)
