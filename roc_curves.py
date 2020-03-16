from cvae import ClassificationVariationalNetwork
from sklearn.metrics import roc_curve
from data.torch_load import choose_device
import torch
import numpy as np
import json

def ood_roc(vae, testset, oodset, method='px', batch_size=100, num_batch='all', device=None):
    
    test_n_batch = len(testset) // batch_size
    ood_n_batch = len(oodset) // batch_size

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

        log_px = np.zeros(batch_size * num_batch * 2)
        label_ood = np.zeros(batch_size * num_batch * 2)
        i = 0
        for loader, num_batch, is_ood in zip((testloader, oodloader),
                                             (test_n_batch, ood_n_batch),
                                             (False, True)):
            iter_ = iter(loader)
            for batch in range(num_batch):
                data = next(iter_)
                x = data[0].to(device)
                log_px[i: i + batch_size] = vae.log_px(x).cpu() #.detach().numpy()
                label_ood[i: i + batch_size] = is_ood
                i += batch_size

    fpr, tpr, thresholds =  roc_curve(label_ood, -log_px)
    # return label_ood, log_px
    return fpr, tpr
    

def miss_roc(vae, testset, batch_size=100, num_batch='all',
             method='loss', predict_method='mean', device=None):

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


def fpr_at_tpr(fpr, tpr, a):

    as_tpr = np.asarray(tpr)
    as_fpr = np.asarray(fpr)
    i_tpr = np.where(as_tpr >= a)[0]
    return as_fpr[i_tpr].min()


def tpr_at_fpr(fpr, tpr, a):

    as_tpr = np.asarray(tpr)
    as_fpr = np.asarray(fpr)
    i_fpr = np.where(as_fpr <= a)[0]
    return as_tpr[i_fpr].max()


def ood_dict(fpr, tpr, rate, n):
    return {'tpr': rate,
            'fpr': fpr_at_tpr(fpr, tpr, rate),
            'n': n}


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
