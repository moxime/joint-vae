from cvae import ClassificationVariationalNetwork
import data.torch_load as dl
from utils.save_load import find_by_job_number
import torch
from sklearn.metrics import auc, roc_curve
import numpy as np
from matplotlib import pyplot as plt
import logging
import os

from torchvision.transforms import ToPILImage

root_logger = logging.getLogger('')
root_logger.setLevel(logging.DEBUG)
file_logger = logging.FileHandler('/tmp/log')
file_logger.setLevel(logging.DEBUG)

sys_logger = logging.StreamHandler()
sys_logger.setLevel(logging.WARNING)
sys_logger.setLevel(logging.ERROR)

root_logger.addHandler(sys_logger)
root_logger.addHandler(file_logger)


search_dir = 'jobs'

# job_numbers = [37, 106366, 106687, 107009, 105605, 105541]
# job_numbers = [107066, 63, 107050]
# job_numbers = [107034]
job_numbers = [_ for _ in range(107000, 107100)]

def showable(x):

    if x.shape[0] == 1:
        x_ = x.expand((3,) + x.shape[1:]).detach().cpu()
    else:
        x_ = x.detach().cpu()

    return x_.permute(1, 2, 0)

def show_grid(net, x_in, x_out, y_in, y_out, order, axes):

    for i, axis in enumerate(axes):

        im = order[i // 2]
        which = i % 2

        if net.is_vib:
            x0_ = x_out[im]
        else:
            x0_ = x_out[0, im]

        x0 = x_in[im]

        if which:
            image = x0_
            _y = y_out[im].item()
        else:
            image = x0
            _y = y_in[im].item()

        axis.imshow(showable(image))
        snr = (x0_ - x0).pow(2).mean() / x0.pow(2).mean()
        snr_db = - 10 * np.log10(snr.item())
        _db = f' {snr_db:.1f}dB' if which else '' # f' (i={im})'
        axis.set_title(testset.classes[_y] + _db) #, y=0, pad=25, verticalalignment="top")
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

reload = True
reload = False

try:
    if reload:
        del jobs
    for j in job_numbers:
        jobs[j]

except (NameError, KeyError):
    print('Loading jobs')
    jobs = find_by_job_number(search_dir, *job_numbers, load_state=False, json_file='networks-lss.json')

    for j in jobs:
            jobs[j]['net'] = ClassificationVariationalNetwork.load(jobs[j]['dir'])

for job_number in jobs:
    net_dict = jobs[job_number]
    net = net_dict['net']

    device = 'cuda'
    net.to(device)

    trainset_name = net.training['set']
    transformer = net.training['transformer']
    trainset, testset = dl.get_dataset(trainset_name, transformer=transformer)
    oodsets = [dl.get_dataset(n, transformer=transformer)[1] for n in testset.same_size]

    oodset = oodsets[0]

    batch_size = 50

    batch_size_ = (4,)
    shape = net.input_shape
    C = len(trainset.classes)

    x, y = {}, {}

    x['ood'], y['ood'] = dl.get_batch(oodset, batch_size=batch_size)
    x['test'], y['test'] = dl.get_batch(testset, batch_size=batch_size)
    x['fake'], y['fake'] = torch.randn((batch_size,) + shape), torch.randint(C, (batch_size,) + shape)

    x_ = {}
    y_ = {}
    mu = {}
    log_var = {}
    z = {}
    losses = {}
    measures = {}
    loss_std = {}
    loss_mean = {}
    loss_min = {}

    sets = ('fake',)
    sets = ('test', 'ood', 'fake')
    for set_ in sets:
        print(f'Evaluating {set_}_batch')
        out = net.evaluate(x[set_].to(device), z_output=True)
        x_[set_], y_[set_], losses[set_], measures[set_], mu[set_], log_var[set_], z[set_] = out

        loss_std[set_] = losses[set_]['total'].std(axis=0)
        loss_mean[set_] = losses[set_]['total'].mean(axis=0)
        loss_min[set_], _ = losses[set_]['total'].min(axis=0)

    """
    ood_results = net.ood_detection_rates([oodset],
                                          num_batch=10, batch_size=100,
                                          update_self_ood=False, print_result='*')


    test_scores = net.batch_dist_measures(_, losses['test'], ['max'])['max']
    ood_scores = net.batch_dist_measures(_, losses['fake'], ['max'])['max']


    scores = np.concatenate([t.detach().cpu().numpy() for t in (test_scores, ood_scores)])

    truth = np.zeros(2*batch_size)
    truth[:batch_size] = 1



    fpr, tpr, thresholds = roc_curve(truth, scores)

    print(' tpr  fpr')
    for f, t in zip(fpr, tpr):

        print('{:>5.1f} {:>5.1f}'.format(100*t, 100*f))

    tpr_ = net.ood_results['mnist']['max']['tpr']
    fpr_ = net.ood_results['mnist']['max']['fpr']

    print(' tpr  fpr')
    for f, t in zip(fpr_, tpr_):

        print('{:>5.1f} {:>5.1f}'.format(100*t, 100*f))
    """
    f, a = plt.subplots(4, 6)
    a_ = a.reshape(-1)
    n = len(a_) // 2

    if net.type != 'vae':
        y_ = net.predict(x['test'], method='esty' if net.is_vib else 'loss')
    else:
        y_ = y['test']

    _true = (y_ == y['test']).cpu()

    i_ = np.arange(batch_size)

    i_true = np.random.permutation(i_[_true])
    i_false = np.random.permutation(i_[~_true])

    n_false = min(len(i_false), n // 2)
    n_false = max(n_false, n - len(i_true))

    if n_false:
        order = np.concatenate([i_true[:n - n_false], i_false[:n_false]])
    else:
        order = i_true[:n]

    show_grid(net, x['test'], x_['test'], y['test'], y_, order, a_)
    f.suptitle(f'{job_number} is a {net.type}')
    f.show()
    result_dir = os.path.join('results', f'{job_number:06d}')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    grid_png = os.path.join('results', f'{job_number:06d}', 'grid.png')
    f.savefig(grid_png)
    
    with torch.no_grad():
        if net.is_cvae or net.is_xvae:
            mu_y = net.encoder.latent_dictionary.index_select(0, y['test'][_true])
        else:
            mu_y = 0.

        mu_z = mu['test'][_true]
        var_z = log_var['test'][_true].exp()

        mu_ = (mu_z - mu_y).mean(0).cpu().numpy()
        _mu = (mu_z - mu_y).std(0).cpu().numpy()

        var_ = var_z.mean(0).cpu().numpy()
        _var = var_z.std(0).cpu().numpy()
        _var_ = np.quantile(var_z.cpu().numpy(), [0.25, 0.75], axis=0)

    np.set_printoptions(precision=1, suppress=True)
    # _f = {'float_kind': lambda x: f'{x:5.1f}'}

    p = list(net.encoder.parameters())[0]

    for t in zip(_mu, var_, p.norm(dim=0)): 

        # pre = '*' if t[0] > 0.1 else ' '
        # post = '*' if t[1] < 0.2 else ' '
        pre, post = '', ''
        print(pre + ' '.join(f'{_:6.2f}' for _ in t) + ' ' +  post)

    """
    for a, n in zip((mu_, var_, _mu,  _var),
                    ('<µ>', '<var>','_µ_', '_var_')):
        print(f'{n:^40}')
        print(np.array2string(a.reshape(16, 8) * 100,
              formatter=f))
        print('\n')
    """

    f, a = plt.subplots(1)

    a.errorbar(_mu, var_, _var_, fmt='o')
    acc = 0
    if net.predict_methods:
        acc = net.testing[net.predict_methods[0]]['accuracy']
    fpr = 0
    a.set_title(f'{job_number}: {net.type} ' +
                f'/ {trainset_name} ' +
                (f'/ acc={100 * acc:.2f}% ' if acc else '') +
                (f'/ fpr@tpr95={100 * fpr}% ' if fpr else ''))

    a.set_xlabel('Ecart-type des moyennes par dimension de Z')
    a.set_ylabel('Moyenne des variances +/ quartiles pazr dimension de Z')
    f.show()
    mu_z_var_z_png = os.path.join('results', f'{job_number:06d}', 'z_mu_var.png')
    f.savefig(mu_z_var_z_png)
    
input()
