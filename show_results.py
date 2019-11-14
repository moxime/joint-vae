import numpy as np
from cvae import ClassificationVariationalNetwork
import os
import matplotlib.pyplot as plt
from data import generate as dg
from utils.save_load import load_json
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

    for example in range(10):
        i_test = np.random.randint(0, x_test.shape[0])
    
        f0 = show_x_y(vae, x_test[i_test], title=f'y_true={y_test[i_test]}')
        f0.show()
    
        i_ood = np.random.randint(0, x_ood.shape[0])
        y_true = y_ood[i_ood]
        x_true = x_ood[i_ood]
        # x_true /= x_true.mean()
        f1 = show_x_y(vae, x_true, title=f'y_true={y_true}')
        f1.show()

        acc, i_miss_ = vae.accuracy(x_test, y_test, return_mismatched=True)

        i_miss = i_miss_[np.random.randint(0, len(i_miss_))]
        x_miss, y_miss = x_test[i_miss], y_test[i_miss]
        f2 = show_x_y(vae, x_miss, title=f'y_missed={y_miss}')
        f2.show()
        
        char = input()
        if char != '':
            plt.close('all')
