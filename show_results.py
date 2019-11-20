import numpy as np
import os
import matplotlib.pyplot as plt
from data import generate as dg
from utils.save_load import load_json, save_object, collect_networks
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
    print(y_)
    l_ = vae.naive_evaluate(x)

    beta2pi = vae.beta * 2 * np.pi

    d = x.size

    log_pxy = [- l  / (2 * vae.beta) - d / 2 * np.log(d * beta2pi) for l in l_]
    print(log_pxy)
    
    log_px = np.log(np.sum(np.exp(log_pxy)))
    
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



def compute_losses(vae, x_test, x_ood, num_labels, save_dir=None):

    C = num_labels

    N_test = x_test.shape[0]
    losses_test = np.ndarray((N_test, C))
    N = N_test
    
    ood = x_ood is not None
    if ood:
        N_ood = x_ood.shape[0]
        losses_ood = np.ndarray((N_ood, C))
        N = min(N_test, N_ood)
    
    for i in range(N):

        l_ = np.array(vae.naive_evaluate(x_test[i], verbose=0))
        losses_test[i, :] = l_
        
        if ood:
            l_ = np.array(vae.naive_evaluate(x_ood[i], verbose=0))
            losses_ood[i, :] = l_

        if i%100 == 10:
            losses_ = [losses_test]
            if ood:
                losses_.append(losses_ood)

            for losses in losses_:
                print(f'===*=*=* i = {i} *=*=*====\n')
                mu = losses[:i].mean()
                s = losses[:i].std()
                mini = losses[:i].min()
                maxi = losses[:i].max()
                K = int((maxi - mu) / s)
                q = [mini] + [mu  + k * s for k in range(-1, K)] + [maxi]

                n = [(losses[:i] < q[k]).sum() for k in range(len(q))]

                n = np.array(n) / losses[:i].size

                f = n[1:] - n[:-1]

                str_ = ''
                for k in range(len(q) - 1):
                    str_ = str_ + f'{q[k]:.1e} [{100*f[k]:.1f}] '
                str_ +=  f'{q[-1]:.1e}\n'
                print(str_)

    if save_dir is not None:
        save_object(losses_test, save_dir, 'losses_test')
        if ood:
            save_object(losses_ood, save_dir, 'losses_ood')

            
def show_examples(vae, x_test, y_test, x_ood, y_ood, num_of_examples=10):

    y_pred = vae.blind_predict(x_test)
    y_pred_ = y_pred.argmax(axis=-1)
    y_test_ = y_test.argmax(axis=-1)

    i_pred_ = np.argwhere(y_test_ == y_pred_)
    i_miss_ = np.argwhere(y_test_ != y_pred_)

    acc = len(i_pred_) / len(x_test)
       
    for example in range(num_of_examples):
        
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
        
        char = input()
        if char != '':
            plt.close('all')

            
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
    ax_log.semilogx(beta_sorted, [1 - _ for _ in acc_sorted], '.', label=legend)
    
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

    args = parser.parse_args()

    dataset = args.dataset
    directories = args.directories

    if directories is None:
        directories = ['./jobs/fashion-mnist']
    print(directories)

    # set = 'fashion'
    # set = 'mnist'

    if dataset == 'fashion':
        (x_train, y_train, x_test, y_test, x_ood, y_ood) = \
            dg.get_fashion_mnist(ood='mnist')

    if dataset == 'mnist':
        (x_train, y_train, x_test, y_test, x_ood, y_ood) = \
             dg.get_mnist(ood='mean')

    # print(dir_)

    list_of_lists_of_vae = []
    for directory in directories:
        collect_networks(directory, list_of_lists_of_vae)
    
    num_of_nets = sum([len(_) for _ in list_of_lists_of_vae])

    if num_of_nets == 0:

        print('NOTHING TO SEE HERE\n')

    if num_of_nets == 1:
        vae = list_of_lists_of_vae[0][0]['net']
        vae.compile()
        show_examples(vae, x_test, y_test, x_ood, y_ood)

    if num_of_nets > 1:
        f_lin, ax_lin = plt.subplots()
        f_log, ax_log = plt.subplots()
        
        for list_of_nets in list_of_lists_of_vae:
            plot_results(list_of_nets, ax_lin, ax_log)

        ax_log.legend()
        # f_lin.show()
        f_log.show()
        input()
