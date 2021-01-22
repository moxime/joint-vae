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
job_numbers = [_ for _ in range(107120, 107200)]
job_numbers = [_ for _ in range(107050, 107400)]
job_numbers = [_ for _ in range(107360, 107400)]
job_numbers = [106754, 107365, 37, 107364, 107009]
job_numbers = [_ for _ in range(107384, 107400)]
job_numbers = [107384]
job_numbers = [107384, 107600, 107638, 107496, 107495, 107494]

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
    for j in job_numbers:
        jobs[j]

except (NameError, KeyError):
    print('Loading jobs')
    reload = True

if reload:
    jobs = find_by_job_number(search_dir, *job_numbers, load_state=False, json_file='networks-lss.json')

    to_be_removed = []
    for j in jobs:
        try:
            jobs[j]['net'] = ClassificationVariationalNetwork.load(jobs[j]['dir'])
        except RuntimeError:
            print(f'Error loading {j}')
            to_be_removed.append(j)

    for j in to_be_removed: jobs.pop(j)

fgrid = {}
fmuvar = {}
fhist = {}

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

    f, a = plt.subplots(4, 6)
    fgrid[job_number] = f
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

    _n = jobs[job_number]
    net_type = _n['type']
    K = _n['K']
    _s = _n['sigma']

    f.suptitle(f'{job_number} is a {net_type} K={K} sigma={_s}')
    # f.show()
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

    f, a = plt.subplots(1)
    fmuvar[job_number] = f
    
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
    # f.show()
    mu_z_var_z_png = os.path.join('results', f'{job_number:06d}', 'z_mu_var.png')
    f.suptitle(f'{job_number} is a {net.type} K={net.latent_dim} sigma={net.sigma:.2f}')

    f.savefig(mu_z_var_z_png)

    f, a = plt.subplots(2, 2)
    a_ = a.reshape(-1)
    fhist[job_number] = f
    f.suptitle(f'{job_number} is a {net_type} K={K} sigma={_s}')
    
    a_[0].plot(_mu, var_, '.')
    a_[0].set_xlabel('Variance sur un batch des moyennes par dimension')
    a_[0].set_ylabel('Moyenne sur un batch des variances par dimension')
    a_[0].set_xlim(0, 3.5)
    a_[0].set_ylim(0, 1.2)
    
    with torch.no_grad():
        a_[1].plot(mu_z[0].view(-1).cpu(), var_z[0].view(-1).cpu(), '.')
        a_[1].set_xlabel('moyennes sur une image')
        a_[1].set_ylabel('variances sur une image')
        a_[1].set_ylim(0, 1.2)
        a_[1].set_xlim(-6, 6)

        a_[2].hist(var_z[0].view(-1).cpu(), bins=10)
        a_[2].set_title('Histogramme des variances sur une image')
        _t1 = 0.25
        _t2 = 0.75
        _n1 = (var_z[0].view(-1) < _t1).sum().item()
        _n2 = (var_z[0].view(-1) > _t2).sum().item()
        a_[2].set_xlabel(f'variances sur une image ({_n1} sont < à {_t1} et {_n2} > à {_t2})')
        a_[2].set_xlim(0, 1.2)

        a_[3].hist(var_z.view(-1).cpu(), bins=100)
        a_[3].set_title('Histogramme des variances sur le batch')
        a_[3].set_xlim(0, 1.2)        
        
fx_ = {}

for j in jobs:
    f, a = plt.subplots(4, 6)
    fx_[j] = f
    a_ = a.reshape(-1)
    vae = jobs[j]['net']
    K = vae.latent_dim
    z = torch.randn(len(a_), K, device=device)    
    x_gen = vae.imager(vae.decoder(z))
    for i, axis in enumerate(a_):

        image = x_gen[i] 
        axis.imshow(showable(image))
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    _s = jobs[j]['sigma']
    f.suptitle(f'{j} is a {vae.type} K={vae.latent_dim} sigma={_s}')

    
def do_show_fig(grid=True, muvar=True, gen=True, hist=True):
    for j in jobs:
        if grid:
            fgrid[j].show()
        if muvar:
            fmuvar[j].show()
        if gen:
            fx_[j].show()
        if hist:
            fhist[j].show()
        
show_fig = False
show_fig = True
if show_fig:
    do_show_fig(
        grid=False,
        muvar=False,
        gen=False,
        hist=True,
    )
    input('Press ANY button to close figs\n')
    print('Closing')
    plt.close('all')
