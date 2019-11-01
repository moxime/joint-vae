import numpy as np
from cvae import ClassificationVariationalNetwork
import os
import matplotlib.pyplot as plt
from data import generate as dg
from utils.save_load import load_json




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
    l_ = vae.naive_evaluate(x)

    plt.figure()
    plt.imshow(x.reshape(28, 28))
    plt.title(f'original ({title})')
    
    for i, x in enumerate(x_):

        plt.figure()
        plt.imshow(x.numpy().reshape(28, 28))
        # plt.show()
        plt.title(f'y={i} loss = {l_[i]}')


    logits = np.log(y_ / (1 - y_))

    plt.imshow(logits)
    plt.xlabel('p(y) output')
    plt.ylabel('y input')
    
    plt.show()

    pass


    
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


(x_train, y_train, x_test, y_test) = dg.get_mnist()

load_dir = './jobs/mnist/job-betas/'

dir_ = [os.path.join(load_dir, o) for o in os.listdir(load_dir) if
        os.path.isdir(os.path.join(load_dir, o))]

param_ = [load_json(d, 'params.json') for d in dir_]

beta_ = [p['beta'] for p in param_]
i_ = np.array(beta_).argsort()


if __name__ == '__main__':

    beta = 5e-3
    i = find_beta(dir_, beta)

    vae = ClassificationVariationalNetwork.load(dir_[i])
    vae.compile()
    param = param_[i]

    i_test = np.random.randint(0, 10000)
    show_x_y(vae, x_test[i_test], title=f'y_true={y_test[i_test]}')

    i_test_ = np.random.randint(0, 10000, 10)
    y_true = y_test[i_test_].mean(axis=0)
    x_true = x_test[i_test_].mean(axis=0)
    x_true /= x_true.mean()
    show_x_y(vae, x_true, title=f'y_true={y_true}')
    
