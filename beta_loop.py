import numpy as np
from cvae import ClassificationVariationalNetwork
import os
import matplotlib.pyplot as plt
from data import generate as dg



(x_train, y_train, x_test, y_test) = dg.get_mnist()

    
save_dir = './jobs/mnist/sampling=1000/betas/'


e_ = [1024, 1024]
d_ = e_.copy()
d_.reverse()

beta_ = np.logspace(-4, -6, 5)

# beta_ = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
digits = [i for i in range (1,10)]
beta_ = ([i*1e-6 for i in digits] + [i*1e-3 for i in digits])

latent_dim = 100

epochs = 20


latent_sampling = 1000

def training_loop(input_dim, num_labels, encoder_layers, latent_dim,
                  decoder_layers, x_train, y_train, x_test, y_test, betas,
                  dir=dir, epochs=30, latent_sampling=100, batch_size=10):


    for beta in betas:

        print(f'\n\nbeta={beta:.5e}\n\n')
        vae = ClassificationVariationalNetwork(input_dim,
                                               num_labels,
                                               encoder_layers,
                                               latent_dim,
                                               decoder_layers,
                                               latent_sampling=latent_sampling,
                                               beta=beta) 

        vae.compile(optimizer='Adam')

        history = vae.fit(x=[x_train, y_train],
                          epochs=epochs,
                          batch_size=batch_size)

        acc = vae.accuracy(x_test, y_test)
        print('\n'+'='*80 + '\n'+f'{beta:.2e}: {acc}\n')

        dir_beta_ = os.path.join(dir, f'{beta:.5e}')
        dir_beta = dir_beta_
        i = -1
        print(dir_beta, os.path.exists(dir_beta))
        while os.path.exists(dir_beta):
            i += 1
            dir_beta = f'{dir_beta_}-{i}'
            print(dir_beta, os.path.exists(dir_beta))
        vae.save(dir_beta)

    pass


def gogo():
    
    training_loop(28**2, 10, e_, latent_dim, d_, x_train, y_train,
                  x_test, y_test, beta_, dir=save_dir,
                  latent_sampling=latent_sampling, epochs=epochs) 

                
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
        print(f'{b:.2e}: {1-a:.3e}\n')

    plt.semilogx(beta_sorted, [1 - _ for _ in acc_sorted], '.')
    
    plt.figure()
    plt.plot(beta_sorted, [1 - _ for _ in acc_sorted], '.')

    plt.show(block=False)

    input()
    
    return beta_sorted, acc_sorted
    
    
    
                    
if __name__ == '__main__':
    
    b_, a_ = plot_results(save_dir, x_test, y_test)

    
