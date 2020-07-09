from sklearn.metrics import roc_curve
from data.torch_load import choose_device, get_fashion_mnist, get_mnist
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

def ood_roc(vae, testset, oodset, method='px', batch_size=100, num_batch='all',
            print_result=False,
            device=None):

    # print('*****', device)
    shuffle = False
    test_n_batch = len(testset) // batch_size
    ood_n_batch = len(oodset) // batch_size

    if type(num_batch) is int:
    
        shuffle = True
        test_n_batch = min(num_batch, test_n_batch)
        ood_n_batch = min(num_batch, ood_n_batch)

    device = choose_device(device)
    vae.to(device)
        
    testloader = torch.utils.data.DataLoader(testset,
                                             shuffle=shuffle,
                                             batch_size=batch_size)

    oodloader = torch.utils.data.DataLoader(oodset,
                                            shuffle=shuffle,
                                            batch_size=batch_size)

    if method == 'px':

        log_px = np.ndarray(batch_size * (test_n_batch + ood_n_batch))
        label_ood = np.ndarray(batch_size * (test_n_batch + ood_n_batch))
        i = 0

        for loader, num_batch, is_ood in zip((testloader, oodloader),
                                             (test_n_batch, ood_n_batch),
                                             (False, True)):
            iter_ = iter(loader)
            for batch in range(num_batch):
                if print_result:
                    print(f'{method}: batch {batch:3d}/{num_batch} of',
                          f'{"ood" if is_ood else "test"}set', end='\r')
                data = next(iter_)
                x = data[0].to(device)
                log_px[i: i + batch_size] = vae.log_px(x).cpu() #.detach().numpy()
                label_ood[i: i + batch_size] = is_ood
                i += batch_size

    fpr, tpr, thresholds =  roc_curve(label_ood, -log_px)
    # return label_ood, log_px

    n = batch_size * min(test_n_batch, ood_n_batch)
    
    return fpr, tpr, -thresholds
    

def miss_roc(vae, testset, batch_size=100, num_batch='all',
             method='loss',
             predict_method='mean',
             verbose=0, device=None):

    if verbose > 0:
        print('Computing miss roc')
    shuffle = True
    if num_batch == 'all':
        num_batch = len(testset) // batch_size
        shuffle = False

    num_batch = min(num_batch, len(testset) // batch_size)
    testloader = torch.utils.data.DataLoader(testset,
                                             shuffle=shuffle,
                                             batch_size=batch_size)

    device = choose_device(device)

    vae.to(device)    

    score = np.ndarray(num_batch * batch_size)
    miss = np.ndarray(num_batch * batch_size)

    test_iter = iter(testloader)

    for batch in range(num_batch):

        if verbose > 1:
            print(f'batch {batch:3d}/{num_batch}', end='\r')
        data = next(test_iter)
        _, y_est, batch_losses = vae.evaluate(data[0].to(device))

        y_pred_batch = vae.predict_after_evaluate(y_est, batch_losses,
                                                  method=predict_method)
            
        if method == 'loss':
            log_py_x = vae.log_py_x(None, batch_losses=batch_losses)
            py_x = log_py_x.exp()

        if method == 'output':
            py_x = y_est.mean(0)
            log_py_x = py_x.log()
            if (py_x == 0).any():
                log_py_x[torch.where(py_x == 0)] = 0
        
        HY_x = (- log_py_x * py_x).sum(0)

        i = batch * batch_size
        score[i: i + batch_size] = HY_x.cpu()
        miss[i: i + batch_size] = data[1] != y_pred_batch.cpu()

    fpr, tpr, thresholds =  roc_curve(miss, score)
    # return label_ood, log_px
    return fpr, tpr


def fpr_at_tpr(fpr, tpr, a, thresholds=None,
               return_threshold=False):

    """fpr and tpr have to be in ascending order

    """
    assert(not return_threshold or thresholds is not None) 

    as_tpr = np.asarray(tpr)
    as_fpr = np.asarray(fpr)
    i_ = np.where(as_tpr >= a)[0].min()

    fpr_ = as_fpr[i_]
    
    if not return_threshold:
        return fpr_
    
    thr_ = thresholds[i_]

    return fpr_, thr_



def tpr_at_fpr(fpr, tpr, a):

    as_tpr = np.asarray(tpr)
    as_fpr = np.asarray(fpr)
    i_fpr = np.where(as_fpr <= a)[0]
    return as_tpr[i_fpr].max()


def ood_dict(fpr, tpr, rate, n):
    return {'tpr': rate,
            'fpr': fpr_at_tpr(fpr, tpr, rate),
            'n': n}


def plot_roc(fpr, tpr, ax=None, rates=[95, 98]):

    return_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots()

    minor_ticks = np.linspace(0, 100, 101)
    major_ticks = np.linspace(0, 100, 21)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    ax.plot(fpr * 100, tpr * 100)

    fpr_ = {r: fpr_at_tpr(fpr, tpr, r / 100) for r in rates}
    
    title = ' -- '.join([f'{r}: {fpr_[r]*100:.2f}' for r in rates])
    ax.set_title('FPR AT TPR ' + title)

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major')

    if return_fig:
        return fig, ax

def load_roc(file_name):

    try:
        with open(file_name, 'r') as f:
            dict = json.load(f)
            return dict['n'], dict['fpr'], dict['tpr']

    except(FileNotFoundError):
        return 0, None, 0.


def save_roc(file_name, fpr, tpr, true_pos, n):

    n_, f_, t_ = load_roc(file_name)
    print('save_roc', n_, f_, t_, len(fpr), len(tpr), true_pos, n)
    if n_ < n:
        with open(file_name, 'w') as f:
            d = ood_dict(fpr, tpr, true_pos, n)
            json.dump(d, f)
            return d['fpr']
    return f_





if __name__ == '__main__':

    load_dir = './jobs/fashion/thejob'
    net = ClassificationVariationalNetwork.load(load_dir)

    trainset, testset = get_fashion_mnist()
    _, oodset = get_mnist()

    with torch.no_grad():
        print('Computing miss roc')
        fpr_miss, tpr_miss = miss_roc(net, testset, verbose=2)
        print('Computing ood roc')
        fpr_ood, tpr_ood = ood_roc(net, testset, oodset, verbose=2)
        
    plot_roc(fpr_ood, tpr_ood)[0].show()
    plot_roc(fpr_miss, tpr_miss)[0].show()
    
