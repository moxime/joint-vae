import numpy as np
from cvae import ClassificationVariationalNetwork
import os
import matplotlib.pyplot as plt

import argparse
import data.torch_load as torchdl
import torch


e_ = [1024, 1024]
d_ = e_.copy()
d_.reverse()

e_ = [1024, 1024, 512, 512, 256, 256] # add 256
latent_dim = 50
# d_ = e_.copy()
# d_.reverse()
d_ = [256, 512]
c_ = [10]
batch_size = 100

latent_sampling = 100 


e_ = [1024, 512, 512]
latent_sampling = 100
latent_dim = 100
d_ = e_.copy()
d_.reverse()
c_ = [20, 20]

e_ = [1024, 512, 512]
# e_ = []
d_ = e_.copy()
d_.reverse()
c_ = [20, 20]
                              
latent_dim = 100
latent_sampling = 100
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
output_activation = 'sigmoid'

job_dirs = './jobs/fashion'
useless_vae = ClassificationVariationalNetwork((1, 28, 28), 10, e_,
                                               latent_dim, d_, c_,
                                               latent_sampling=latent_sampling,
                                               output_activation='sigmoid')

save_dir = os.path.join(job_dirs, useless_vae.print_architecture())

beta_pseudo_log = np.array([1, 2, 5])

beta_lin = np.linspace(1e-4, 5e-4, 5)

beta_ = np.hstack([beta_pseudo_log * p for p in np.logspace(-8, -5, 4)] * 4)
beta_ = np.hstack([beta_pseudo_log * p for p in np.logspace(-4, -2, 3)] * 4)
 
# beta_ = np.hstack([beta_pseudo_log * 1e-7] * 2)
# beta_ = np.hstack([1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]*5)
# beta_ = beta_lin
# beta_ = np.hstack([beta_lin]*3)


def read_float_list(path):

    list = None
    if os.path.exists(path):
        with open(path, 'r') as f:
            list = [float(_) for _ in f.readlines()]

    return list


def write_float_list(list, path):
    with open(path, 'w+') as f:
        for _ in list:
            f.write(f'{_:.3e}' + '\n')

        
def training_loop(input_dim, num_labels, encoder_layers, latent_dim,
                  decoder_layers, classifier_layers, trainset, 
                  testset, betas, directory='./jobs', device=None,
                  epochs=30, latent_sampling=100, batch_size=100, output_activation=None):

    methods = ClassificationVariationalNetwork.predict_methods

    if not os.path.exists(directory):
        os.makedirs(directory)
    betas_file = os.path.join(directory, 'betas_to_be_run')

    betas_from_file = read_float_list(betas_file)

    if betas_from_file is not None:
        betas = betas_from_file

    while len(betas) > 0:

        write_float_list(betas, betas_file)
        beta = betas[0]
            
        print(f'\n\nbeta={beta:.5e}\n\n')
        vae = ClassificationVariationalNetwork(input_dim,
                                               num_labels,
                                               encoder_layers,
                                               latent_dim,
                                               decoder_layers,
                                               classifier_layers,
                                               latent_sampling=latent_sampling,
                                               output_activation='sigmoid',
                                               beta=beta) 

        print(vae.print_architecture(), '\n', vae._sizes_of_layers)
        print(f'beta={vae.beta}')

        vae.to(device)
        vae.train(trainset, epochs=epochs, testset=testset,
                  batch_size=batch_size)

        acc = vae.accuracy(testset, method='all')

        for m in methods:
            print(f'{beta:.2e}: {acc[m]*100:5.2 %} (by {m})')

        dir_beta_ = os.path.join(directory, f'beta={beta:.5e}')
        dir_beta = f'{dir_beta_}-0'
        i = 0
        print(dir_beta, os.path.exists(dir_beta))
        while os.path.exists(dir_beta):
            i += 1
            dir_beta = f'{dir_beta_}-{i}'
            print(dir_beta, os.path.exists(dir_beta))
        vae.save(dir_beta)

        results_path_file = os.path.join(dir_beta, 'test_accuracy_')
        for m in methods:
            with open(results_path_file + m, 'w+') as f:
                f.write(str(acc[m]) + '\n')
        
        betas = read_float_list(betas_file)
        if len(betas) > 0:
            betas.pop(0)

    os.remove(betas_file)    
   
                
def plot_results(save_dir, x_test, y_test):

    beta_ = []
    acc_ = []

    dir_ = [os.path.join(save_dir, o) for o in os.listdir(save_dir) if
            os.path.isdir(os.path.join(save_dir, o))]


    for d in dir_:
        vae = ClassificationVariationalNetwork.load(d)
        beta_.append(vae.beta)

        results_path_file = os.path.join(d, 'test_accuracy') 

        try:
            # get result from file
            with open(results_path_file, 'r') as f:
                acc = float(f.read())
        except:
            acc = vae.accuracy(x_test, y_test)
            with open(results_path_file, 'w+') as f:
                f.write(str(acc) + '\n')
                
        acc_.append(acc)

    beta_sorted = np.sort(beta_)
    i = np.argsort(beta_)
    acc_sorted = [acc_[_] for _ in i]
      
    for (b, a) in zip(beta_sorted, acc_sorted):
        print(f'{b:.2e}: {100-100*a:4.1f} %\n')

    plt.semilogx(beta_sorted, [1 - _ for _ in acc_sorted], '.')
    
    """ 
    plt.figure()
    plt.loglog(beta_sorted, [1 - _ for _ in acc_sorted], '.')
    """

    plt.figure()
    plt.plot(beta_sorted, [1 - _ for _ in acc_sorted], '.')

    plt.show(block=False)

    input()

    # plt.close(fig='all')
    
    return beta_sorted, acc_sorted
    
                        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="train a network with idfferent values of beta")
    parser.add_argument('--dataset', default='fashion',
                        choices=['fashion', 'mnist'])

    parser.add_argument('-b', '--batch_size', type=int, default=100)

    args = parser.parse_args()
    batch_size = args.batch_size
    trainset, testset = torchdl.get_fashion_mnist()        

    epochs = 50

    print('Used device:', device)
    training_loop((1, 28, 28), 10,
                  e_, latent_dim, d_, c_,
                  trainset, testset, beta_,
                  output_activation='sigmoid',
                  latent_sampling=latent_sampling,
                  epochs=epochs,
                  batch_size=batch_size,
                  directory=save_dir,
                  device=device) 

    
