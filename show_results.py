import numpy as np
from cvae import ClassificationVariationalNetwork
import os
import matplotlib.pyplot as plt
from data import generate as dg
from utils.save_load import load_json, save_object
import time



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

    f, axes = plt.subplots(3, 4)

    axes = axes.reshape(12)
    
    axes[0].imshow(x.reshape(28, 28), cmap='gray')
    axes[0].set_title(f'original ({title})')

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


def load_vae(dir_, i):

    return ClassificationVariationalNetwork.load(dir_[i])


def find_beta(dir_, beta):

    param_ = [load_json(d, 'params.json') for d in dir_]
    beta_ = [p['beta'] for p in param_]
    i_ = np.array(beta_).argsort()

    i_b = i_[0]
    for i in i_:

        if beta_[i] <= beta:
            i_b = i

    return i_b


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



if __name__ == '__main__':

    set = 'fashion'
    # set = 'mnist'

    if set == 'fashion':
        (x_train, y_train, x_test, y_test) = dg.get_fashion_mnist()
        load_dir = './jobs/fashion-mnist/latent-dim=100-sampling=500-encoder-layers=3'
        (_, _, x_ood, y_ood) = dg.get_mnist()

    if set == 'mnist':
        (x_train, y_train, x_test, y_test) = dg.get_mnist()
        load_dir = './jobs/mnist/sampling=1000/betas/'
        x_ood_ = x_test[None] # expand dims
        y_ood_ = y_test[None]
        perms = [np.random.permutation(x_test.shape[0]) for i in range(4)]

        x_ood = np.vstack([x_ood_[:, p, :] for p in perms]).mean(axis=0)
        y_ood = np.vstack([y_ood_[:, p, :] for p in perms]).mean(axis=0)


    dir_ = [os.path.join(load_dir, o) for o in os.listdir(load_dir) if
            os.path.isdir(os.path.join(load_dir, o))]

    # print(dir_)

    param_ = [load_json(d, 'params.json') for d in dir_]

    beta_ = [p['beta'] for p in param_]
    i_ = np.array(beta_).argsort()

    beta = 2e-4
    i = find_beta(dir_, beta)    
    
    vae = ClassificationVariationalNetwork.load(dir_[i])
    vae.compile()

    print(f'beta = {vae.beta}\n')
    
    param = param_[i]

    y_pred = vae.blind_predict(x_test)
    y_pred_ = y_pred.argmax(axis=-1)
    y_test_ = y_test.argmax(axis=-1)

    i_pred_ = np.argwhere(y_test_ == y_pred_)
    i_miss_ = np.argwhere(y_test_ != y_pred_)

    acc = len(i_pred_) / len(x_test)

    for example in range(10):
        
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
