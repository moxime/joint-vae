import numpy as np
import os
import matplotlib.pyplot as plt
from data import generate as dg
from utils.save_load import load_json, save_object, collect_networks, get_path_from_input
import argparse
import torch
from sklearn.metrics import roc_curve

def showable_tensor(x):

    return x.squeeze().cpu().detach().numpy()

def show_y_matrix(vae, x):

    x_, y_, loss = vae.evaluate(x)
    logits = (y_ / (1 - y_)).log()

    plt.imshow(logits.cpu().detach().numpy())
    plt.show()
    pass


def show_x(vae, x):

    x_, y_, oneloss = vae.evaluate(x)
    
    for i, x in enumerate(x_):

        plt.figure()
        plt.imshow(showable_tensor(x))
        # plt.show()
        plt.title(f'y={i} loss = {oneloss[i]}')

    plt.show()

    pass


def show_x_y(vae, x, title='', predict_method='mean'):
    """ x is ONE sample 
    """
    
    x_reco, y_est_, oneloss = vae.evaluate(x)
    y_pred_ = y_est_.mean(0).cpu().numpy()

    y_pred = vae.predict(x, method=predict_method)

    y_ = torch.cat([y_est_.cpu(),
                    y_est_.mean(0).cpu().unsqueeze(0)], 0)
    # print(y_)

    log_px = vae.log_px(x)
    print(f'{title}{" :" if title else ""} log_px = {log_px}\n')
    
    f, axes = plt.subplots(3, 4)

    axes = axes.reshape(12)
    
    axes[0].imshow(showable_tensor(x), cmap='gray')
    axes[0].set_title(f'original ({title}) log_px={log_px:.3e}')

    ax_i = 1
    for i, x in enumerate(x_reco):

        axes[ax_i].imshow(showable_tensor(x), cmap='gray')
        # plt.show()
        warn = '[]' if i == y_pred else ('', '')
        axes[ax_i].set_title(f'{warn[0]}y={i} loss = {oneloss[i]:.3e}{warn[-1]}')
        ax_i += 1
        
    logits = (y_ / (1 - y_)).log()

    axes[-1].imshow(showable_tensor(logits), cmap='gray')

    axes[-1].set(xlabel='p(y) output', ylabel='y input')
    axes[-1].set_title(f'y_pred={np.round(100 * y_pred_)}')

    return f


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

    if device is None:
        has_cuda = torch.cuda.is_available
        device = torch.device('cuda' if has_cuda else 'cpu')

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

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

                
def show_examples(vae, testset, oodset, num_of_examples=10,
                  batch_size=100, num_batch=5, export=False,
                  export_dir='/tmp'):

    y_pred = vae.blind_predict(x_test)
    y_pred_ = y_pred.argmax(axis=-1)
    y_test_ = y_test.argmax(axis=-1)

    i_pred_ = np.argwhere(y_test_ == y_pred_).squeeze()
    i_miss_ = np.argwhere(y_test_ != y_pred_).squeeze()

    acc = len(i_pred_) / len(x_test)

    print('*' * 80 + f'\naccurary {acc}\n' + '*' * 80 + '\n') 

    example = 0
    continued = True
    while example < num_of_examples and continued:
        
        i_test = np.random.randint(0, x_test.shape[0])
        i_pred = i_pred_[np.random.randint(0, len(i_pred_))]
        
        f0 = show_x_y(vae, x_test[i_pred], title=f'y_true={y_test[i_pred]}')
        f0.show()
    
        i_ood = np.random.randint(0, x_ood.shape[0])
        y_true = y_ood[i_ood]
        x_true = x_ood[i_ood]
        # x_true /= x_true.mean()
        f1 = show_x_y(vae, x_true, title=f'y ood')
        f1.show()

        i_miss = i_miss_[np.random.randint(0, len(i_miss_))]
        x_miss, y_miss = x_test[i_miss], y_test[i_miss]
        f2 = show_x_y(vae, x_miss, title=f'y_missed={y_miss}')
        f2.show()

        log_px_test, log_px_ood = ood_roc(vae, x_test, y_test,
                                          x_ood, y_ood,
                                          stats=stats,
                                          verbose=0)

        HY_x_pred, HY_x_miss = miss_roc(vae, x_test, y_test,
                                        stats=stats,
                                        verbose=0)

        [_.sort() for _ in [log_px_test, log_px_ood, HY_x_miss, HY_x_pred]]
        
        with np.printoptions(precision=0):
            print(f'test:\n{log_px_test}\n')
            print(f'ood:\n{log_px_ood}\n')

        with np.printoptions(precision=0):
            print(f'pred:\n{np.exp(HY_x_pred)}\n')
            print(f'miss:\n{np.exp(HY_x_miss)}\n')

        f, a = plt.subplots(2)

        bins = stats // 10
        a[0].hist((HY_x_pred), bins, histtype='step', label='pred')
        a[0].hist((HY_x_miss), bins, histtype='step', label='miss')
        a[1].hist(log_px_test, bins, histtype='step', label='test')
        a[1].hist(log_px_ood, bins, histtype='step', label='ood')
        a[0].legend()
        a[1].legend()
        f.show()

        f, (a1, a2) = plt.subplots(1, 2)
        p_fa, p_d = roc_curves(-log_px_test, -log_px_ood)
        a1.plot(100*p_fa, 100*p_d)
        a1.plot(100*p_fa, 100*p_fa, 'r--')
        a1.set_xlabel('false positive of ood')
        a1.set_ylabel('true positive of ood')

        
        p_fa, p_d = roc_curves(np.exp(HY_x_pred), np.exp(HY_x_miss))
        a2.plot(100*p_fa, 100*p_d)
        a2.plot(100*p_fa, 100*p_fa, 'r--')
        a2.set_xlabel('false positive of miss')
        a2.set_ylabel('true positive of miss')

        for a in [a1, a2]:
            a.set_xticks(np.linspace(0, 100, 101), minor=True)
            a.set_yticks(np.linspace(0, 100, 101), minor=True)
            a.set_xticks(np.linspace(0, 100, 21))
            a.set_yticks(np.linspace(0, 100, 21))
            
            a.grid(which='both')
            
        f.show()
        
        char = input()
        if char != '':
            plt.close('all')
        if char.lower() == 'q':
            continued = False
            
def plot_results(list_of_vae, ax_lin, ax_log):

    beta_ = []
    acc_ = []

    for vae_dict in list_of_vae:
        acc = vae_dict['acc']
        if acc is not None:
            beta_.append(vae_dict['net'].beta)
            acc_.append(acc)

    beta_sorted = np.sort(beta_)
    i = np.argsort(beta_)
    acc_sorted = [acc_[_] for _ in i]
      
    for (b, a) in zip(beta_sorted, acc_sorted):
        print(f'{b:.2e}: {100-100*a:4.1f} %\n')

    legend = vae_dict['net'].print_architecture()
    ax_log.semilogx(beta_sorted, [1 - _ for _ in acc_sorted], 'o', label=legend)
    
    ax_lin.plot(beta_sorted, [1 - _ for _ in acc_sorted], '.', label=legend)
    
    return beta_sorted, acc_sorted
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="show results of networks in directory")
    parser.add_argument('--dataset', default='fashion',
                        choices=['fashion', 'mnist'])
    parser.add_argument('directories',
                        help='where to find the networks',
                        nargs='*', default=None)
    parser.add_argument('-s', '--stats', type=int, default=200)
    
    parser.add_argument('-x', '--export', action='store_true')

    parser.add_argument('-d', '--export_dir', default=None,
                        help='directory to export results')
    
    args = parser.parse_args()

    export_dir = args.export_dir

    export = args.export or export_dir is not None

    stats = args.stats
    dataset = args.dataset
    directories = args.directories
    print(directories)
    if len(directories) == 0:
        jobs = os.path.join(os.getcwd(), 'jobs')
        if not os.path.exists(jobs):
            jobs = os.getcwd()
        directories = [get_path_from_input(jobs)]
    print(directories)

    # set = 'fashion'
    # set = 'mnist'

    x_test, y_test = None, None
    if dataset == 'fashion':
        (x_train, y_train, x_test, y_test, x_ood, y_ood) = \
            dg.get_fashion_mnist(ood='mnist')

    if dataset == 'mnist':
        (x_train, y_train, x_test, y_test, x_ood, y_ood) = \
             dg.get_mnist(ood='mean')

    # print(dir_)

    list_of_lists_of_vae = []
    for directory in directories:
        collect_networks(directory, list_of_lists_of_vae,
                         x_test=x_test, y_test=y_test)
    
    num_of_nets = sum([len(_) for _ in list_of_lists_of_vae])

    if num_of_nets == 0:

        print('NOTHING TO SEE HERE\n')

    if num_of_nets == 1:
        vae_dict = list_of_lists_of_vae[0][0]
        vae = vae_dict['net']
        directory = vae_dict['dir']
        vae.compile()
        show_examples(vae, x_test, y_test, x_ood, y_ood, stats=stats)

    if num_of_nets > 1:
        f_lin, ax_lin = plt.subplots()
        f_log, ax_log = plt.subplots()
        
        for list_of_nets in list_of_lists_of_vae:
            
            beta_, acc_ = plot_results(list_of_nets, ax_lin, ax_log)

            if export_dir is not None:
                file_path = os.path.join(export_dir,
                                         list_of_nets[0]['net'].print_architecture()
                                         + '.tab')
                with open(file_path, 'w+') as f:
                    for b, a in zip(beta_, acc_):
                        f.write(f'{b:.2e} {a:6.4f}\n')
        
        ax_log.legend()
        # f_lin.show()
        f_log.show()
        input()
