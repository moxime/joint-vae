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
# root_logger.setLevel(logging.DEBUG)
file_logger = logging.FileHandler('/tmp/log')
file_logger.setLevel(logging.DEBUG)

sys_logger = logging.StreamHandler()
sys_logger.setLevel(logging.WARNING)
# sys_logger.setLevel(logging.ERROR)

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
# job_numbers = [108160]
# job_numbers = [65]
job_numbers = [107384, 107600, 107638, 107496, 107495, 107494, 37]
job_numbers = [108657]
job_numbers = [108182, 37]
job_numbers = [75]
job_numbers = [37]

def showable(x):
    
    if x.shape[0] == 1:
        x_ = x.expand((3,) + x.shape[1:]).detach().cpu()
    else:
        x_ = x.detach().cpu()

    return x_.permute(1, 2, 0)

def show_grid(net, x_in, x_out, order, axes, y_in=None, y_out=None):

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
            _y = y_out[im].item() if y_out is not None else -1
        else:
            image = x0
            _y = y_in[im].item() if y_in is not None else -1

        axis.imshow(showable(image))
        snr = (x0_ - x0).pow(2).mean() / x0.pow(2).mean()
        snr_db = - 10 * np.log10(snr.item())
        _db = f' {snr_db:.1f}dB' if which else '' # f' (i={im})'
        class_= testset.classes[_y] if _y > -1 else ''
        axis.set_title(class_ + _db) #, y=0, pad=25, verticalalignment="top")
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)


def show_examples(net, x_in, x_out, order, ax_matrix, y_in=None, y_out=None):

    rows, cols = ax_matrix.shape

    for c in range(cols):

        axis = ax_matrix[0, c]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.set_title('var[^X]={:.3g}'.format(x_out[1:, c].var(0).mean()))
        
        axis.imshow(showable(x_in[c]))

        axis = ax_matrix[1, c]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        
        _x = x_out[1:, c].mean(0)
        axis.imshow(showable(_x))
        
        axis.set_title('rmse={:.3g}'.format((_x - x_in[c]).pow(2).mean()))
        
        for r in range(2, rows):

            axis = ax_matrix[r, c]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

            _x = x_out[r - 2, c]
            axis.imshow(showable(_x))
            axis.set_title('rmse={:.3g}'.format((_x - x_in[c]).pow(2).mean()))
        
        
reload = True
reload = False
recompute = True
recompute = False


try:
    for job_number in job_numbers:
        jobs[job_number]

except (NameError, KeyError):
    print('Loading jobs')
    reload = True
    recompute=True
    
if reload:
    jobs = find_by_job_number(search_dir, *job_numbers, load_state=False, json_file='networks-lss.json')

    to_be_removed = []
    for job_number in jobs:
        try:
            jobs[job_number]['net'] = ClassificationVariationalNetwork.load(jobs[job_number]['dir'])
        except RuntimeError:
            print(f'Error loading {job_number}')
            to_be_removed.append(job_number)

    for job_number in to_be_removed: jobs.pop(job_number)

fgrid = {}
fexamples = {}
food = {}
foodexamples = {}
fmuvar = {}
fhist = {}
fx_ = {}

if recompute:
    data_dict = {}

plt.clf()

for job_number in jobs:

    if recompute:
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
        x['fake'] = torch.randn((batch_size,) + shape)
        y['fake'] = torch.randint(C, (batch_size,) + shape)

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

        def clone_dict(d):
            return {s: d[s].clone().detach() for s in d}

        data_dict[job_number] = dict(
            x=clone_dict(x),
            x_=clone_dict(x_),
            y=clone_dict(y),
            y_=clone_dict(y_),
            mu=clone_dict(mu),
            log_var=clone_dict(log_var),
            net=net,
            classes=testset.classes,
        )

    x = data_dict[job_number]['x']
    x_ = data_dict[job_number]['x_']
    y = data_dict[job_number]['y']
    y_ = data_dict[job_number]['y_']
    mu = data_dict[job_number]['mu']
    log_var = data_dict[job_number]['log_var']
    net = data_dict[job_number]['net']

    print(f'Creating grid for {job_number}')
    f, a = plt.subplots(4, 6)
    fgrid[job_number] = f
    a_ = a.reshape(-1)
    n = len(a_) // 2

    print(f'Creating grid of outputs for {job_number}')
    f, aex = plt.subplots(6, 10)
    fexamples[job_number] = f
    
    f, aoex = plt.subplots(6, 10)
    foodexamples[job_number] = f

    print(f'Creating ood figures for {job_number}')
    fo, aood = plt.subplots(4, 6)
    food[job_number] = fo
    aood_ = aood.reshape(-1)
    
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

    show_grid(net, x['test'], x_['test'], order, a_, y_in=y['test'], y_out=y_)

    show_examples(net, x['test'], x_['test'], order, aex, y_in=y['test'], y_out=y_)
        
    order = np.random.permutation(batch_size)[:n]
    show_grid(net, x['ood'], x_['ood'], order, aood_, y_out=y_)
    show_examples(net, x['ood'], x_['ood'], order, aoex, y_in=y['test'], y_out=y_)

    
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
            mu_y = net.encoder.latent_dictionary.index_select(0, y['test'][_true]).cpu()
        else:
            mu_y = 0.

        mu_z = mu['test'][_true].cpu()
        var_z = log_var['test'][_true].exp().cpu()

        mu_ = (mu_z - mu_y).mean(0).numpy()
        _mu = (mu_z - mu_y).std(0).numpy()

        var_ = var_z.mean(0).numpy()
        _var = var_z.std(0).numpy()
        _var_ = np.quantile(var_z.numpy(), [0.25, 0.75], axis=0)

    np.set_printoptions(precision=1, suppress=True)
    # _f = {'float_kind': lambda x: f'{x:5.1f}'}

    print(f'Creating latent quartiles for {job_number}')
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
    f.suptitle(f'{job_number} is a {net.type} K={net.latent_dim} sigma={net.sigma}')

    f.savefig(mu_z_var_z_png)

    print(f'Creating latent histograms for {job_number}')
    f, a = plt.subplots(2, 3)

    fhist[job_number] = f
    f.suptitle(f'{job_number} is a {net_type} K={K} sigma={_s}')

    a_ = a[0, 0]
    a_.plot(_mu, var_, '.')
    a_.set_xlabel('Variance sur un batch des moyennes par dimension')
    a_.set_ylabel('Moyenne sur un batch des variances par dimension')
    a_.set_xlim(0, 3.5)
    a_.set_ylim(0, 1.2)
    
    with torch.no_grad():
        a_ = a[0, 1]
        a_.plot(mu_z[0].view(-1).cpu(), var_z[0].view(-1).cpu(), '.')
        a_.set_xlabel('moyennes sur une image')
        a_.set_ylabel('variances sur une image')
        a_.set_ylim(0, 1.2)
        a_.set_xlim(-6, 6)

        a_ = a[0, 2]
        a_.hist(var_z[0].view(-1).cpu(), bins=10)
        a_.set_title('Histogramme des variances sur une image')
        _t1 = 0.25
        _t2 = 0.75
        _n1 = (var_z[0].view(-1) < _t1).sum().item()
        _n2 = (var_z[0].view(-1) > _t2).sum().item()
        a_.set_xlabel(f'variances sur une image ({_n1} sont < à {_t1} et {_n2} > à {_t2})')
        a_.set_xlim(0, 1.2)

        a_ = a[1, 0]
        a_.hist(var_z.view(-1).cpu(), bins=100)
        a_.set_title('Histogramme des variances sur le batch')
        a_.set_xlim(0, 1.2)        

        a_ = a[1, 1]

        ratio = (mu_z.pow(2) / var_z).mean(0).cpu()
        _, sorting_index = mu_z.pow(2).mean(0).cpu().sort(descending=True)
        qratio = np.quantile((mu_z.pow(2) / var_z).cpu().numpy(), [0.25, 0.75], axis=0)
        a_.semilogy(ratio[sorting_index])
        # a_.semilogy(qratio[0][sorting_index], '--')
        # a_.semilogy(qratio[1][sorting_index], '--')
        
        a_.set_title('Moyenne sur le batch du rapport mu_z(x)^2 / var_z(x) par dimension')

        a_ = a[1, 2]
        _v = var_z.mean(0).cpu()[sorting_index]
        _m = mu_z.var(0).cpu()[sorting_index]
        a_.bar(np.arange(len(_v)), _v)
        a_.bar(np.arange(len(_v)), _m, bottom=_v)
        a_.set_title('Moyenne sur le batch de var_z(x)'
                     ' / variance sur le batch de mu_z(x), par dimension')

        
        print(f'Generating artificial images for {job_number}')
        classes = data_dict[job_number]['classes']
        C = len(classes)
        f, a = plt.subplots(C, 20)
        fx_[job_number] = f

        vae = jobs[job_number]['net']
        K = vae.latent_dim
        z = torch.randn(a.shape+ (K,), device=device)
        mu_y = torch.zeros_like(z)
        if net.is_cvae:
            for c in range(C):
                for i, _ in enumerate(a[c]):
                    mu_y[c][i] = net.encoder.latent_dictionary[c]

        x_gen = vae.imager(vae.decoder(z + mu_y))
        for c in range(C):
            for i, axis in enumerate(a[c]):
                image = x_gen[c][i] 
                axis.imshow(showable(image))
                axis.get_xaxis().set_visible(False)
                axis.get_yaxis().set_visible(False)
 
        _s = jobs[job_number]['sigma']
        f.suptitle(f'{job_number} is a {vae.type} K={vae.latent_dim} sigma={_s}')

        

    
def do_show_fig(grid=False, examples=False, ood=False, muvar=False, gen=False, hist=False):
    for j in jobs:
        if grid:
            fgrid[j].show()
        if examples:
            fexamples[j].show()
            foodexamples[j].show()
        if ood:
            food[j].show()
        if muvar:
            fmuvar[j].show()
        if gen:
            fx_[j].show()
        if hist:
            fhist[j].show()
        
show_fig = False
show_fig = True

show_grid = True
show_grid = False

show_examples = True

show_ood = True
show_ood = False

show_muvar = True
show_muvar = False

show_gen = True
show_gen = False

show_hist = True
show_hist= False

if show_fig:
    do_show_fig(
        grid=show_grid,
        examples=show_examples,
        ood=show_ood,
        muvar=show_muvar,
        gen=show_gen,
        hist=show_hist,
    )
    input('Press any button to close figs\n')
    print('Closing')
    plt.close('all')
