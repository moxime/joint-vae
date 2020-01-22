import numpy as np
import os
import matplotlib.pyplot as plt
from data import generate as dg
from utils.save_load import load_json, save_object, collect_networks, get_path_from_input
import argparse


def show_y_matrix(vae, x):

    x_, y_ = vae.naive_call(x)
    logits = np.log(y_ / (1 - y_))

    plt.imshow(logits)
    plt.show()
    pass


def show_x(vae, x):

    x_, y_ = vae.naive_call(x)

    l_ = vae.naive_evaluate(x)
    
    for i, x in enumerate(x_):

        plt.figure()
        plt.imshow(x.numpy().reshape(28, 28))
        # plt.show()
        plt.title(f'y={i} loss = {l_[i]}')

    plt.show()

    pass


def show_x_y(vae, x, title=''):

    x_, y_ = vae.naive_call(x)
    y = vae.blind_predict(np.atleast_2d(x))
    y_ = np.vstack([y_, y, y_.numpy().mean(axis=0)])
    # print(y_)
    l_ = vae.naive_evaluate(x, verbose=0)

    beta2pi = vae.beta * 2 * np.pi

    d = x.size

    log_px = vae.log_px(x)
    print(f'{title} : log_px = {log_px}\n')
    
    f, axes = plt.subplots(3, 4)

    axes = axes.reshape(12)
    
    axes[0].imshow(x.reshape(28, 28), cmap='gray')
    axes[0].set_title(f'original ({title}) log_px={log_px}')

    ax_i = 1
    for i, x in enumerate(x_):

        axes[ax_i].imshow(x.numpy().reshape(28, 28), cmap='gray')
        # plt.show()
        axes[ax_i].set_title(f'y={i} loss = {l_[i]}')
        ax_i += 1
        
    logits = np.log(y_ / (1 - y_))

    axes[-1].imshow(logits, cmap='gray')

    axes[-1].set(xlabel='p(y) output', ylabel='y input')
    axes[-1].set_title(f'y_pred={np.round(100*y)}')

    return f


def roc_curves(likelihood_h0, likelihood_h1):
    """
    returns p_fa, p_d (false detextion of H1, true detection of H1) 
    """
    n_h0 = likelihood_h0.size
    n_h1 = likelihood_h1.size

    lh0_sorted = np.sort(likelihood_h0)
    p_fa = np.ndarray(n_h0)
    p_d = np.ndarray(n_h0)
    for i, lh in enumerate(lh0_sorted):
        p_fa[i] = sum(likelihood_h0 > lh) / n_h0
        p_d[i] = sum(likelihood_h1 > lh) / n_h1

    return p_fa, p_d


def ood_roc(vae, x_test, y_test, x_ood, y_ood,
            method='px',
            i_test=None, losses=None,
            stats=100, **kw):

    stats = min(stats, x_test.shape[0], x_ood.shape[0])
    
    if losses is None or i_test is None:
        i_test = np.random.permutation(x_test.shape[0])[:stats]
        losses = None
        
    if losses is None:
        losses = vae.evaluate(x_test[i_test], **kw)
    
    i_ood = np.random.permutation(x_ood.shape[0])[:stats]

    if method == 'px':
        log_px_test = vae.log_px(x_test[i_test], losses=losses, **kw)
        log_px_ood = vae.log_px(x_ood[i_ood], **kw)

        return log_px_test, log_px_ood


def miss_roc(vae, x_test, y_test, y_pred=None, stats=100, method='output', **kw):

    if y_pred is None:
        y_pred = vae.blind_predict(x_test)

    y_pred_ = y_pred.argmax(axis=-1)
    y_test_ = y_test.argmax(axis=-1)

    i_pred_ = np.argwhere(y_test_ == y_pred_).squeeze()
    n_pred = len(i_pred_)
    i_pred = i_pred_[np.random.permutation(n_pred)[:min(stats, n_pred)]]
    i_miss_ = np.argwhere(y_test_ != y_pred_).squeeze()
    n_miss = len(i_miss_)
    i_miss = i_miss_[np.random.permutation(n_miss)[:min(stats, n_miss)]]
    
    if method == 'loss':
        log_py_x_pred = vae.log_py_x(x_test[i_pred], **kw)
        log_py_x_miss = vae.log_py_x(x_test[i_miss], **kw)

        HY_x_pred = -(np.exp(log_py_x_pred) * log_py_x_pred).sum(axis=-1)
        HY_x_miss = -(np.exp(log_py_x_miss) * log_py_x_miss).sum(axis=-1)

        return HY_x_pred, HY_x_miss

    if method == 'output':
        
        H_pred = (-y_pred[i_pred] * np.log(y_pred[i_pred])).sum(axis=-1)
        H_miss = (-y_pred[i_miss] * np.log(y_pred[i_miss])).sum(axis=-1)

        return H_pred, H_miss
    
    
        
def show_examples(vae, x_test, y_test, x_ood, y_ood,
                  num_of_examples=10, stats=100, export=False, export_dir='/tmp'):

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
