import numpy as np
from cvae import ClassificationVariationalNetwork
import os
import matplotlib.pyplot as plt
from data import generate as dg



(x_train, y_train, x_test, y_test) = dg.get_mnist()

    
save_dir = './jobs/mnist/job-betas/'

e_ = [1024, 1024]
d_ = e_.copy()
d_.reverse()

beta_ = np.logspace(-6.2, -8, 10)

latent_dim = 256

epochs = 30


def training_loop(input_dim, num_labels, encoder_layers, latent_dim,
                  decoder_layers, x_train, y_train, x_test, y_test, betas,
                  dir=dir, epochs=30, batch_size=10):


    for beta in betas:

        print(f'\n\nbeta={beta:.5e}\n\n')
        vae = ClassificationVariationalNetwork(input_dim,
                                               num_labels,
                                               encoder_layers,
                                               latent_dim,
                                               decoder_layers,
                                               beta=beta) 

        vae.compile(optimizer='Adam')

        history = vae.fit(x=[x_train, y_train],
            epochs=epochs,
            batch_size=batch_size)

        acc = vae.accuracy(x_test, y_test)
        print('\n'+'='*80 + '\n'+f'{beta:.2e}: {acc}\n')

        vae.save(dir + f'{beta:.5e}')

                
def gogo():
    
     training_loop(28**2, 10, e_, latent_dim, d_, x_train, y_train,
                   x_test, y_test, beta_, dir=save_dir, epochs=epochs) 

                
def plot_results(save_dir, x_test, y_test):

    beta_ = []
    acc_ = []

    dir_ = [os.path.join(save_dir, o) for o in os.listdir(save_dir) if
            os.path.isdir(os.path.join(save_dir, o))]


    for d in dir_:
        vae = ClassificationVariationalNetwork.load(d)
        beta_.append(vae.beta)
        acc_.append(vae.accuracy(x_test, y_test))

    for (b, a) in zip(beta_, acc_):
        print(f'{b:.2e}: {a}\n')

    beta_sorted = np.sort(beta_)
    i = np.argsort(beta_)
    acc_sorted = [acc_[_] for _ in i]
      
    plt.semilogx(beta_sorted, acc_sorted)

    plt.show()
    
    return beta_sorted, acc_sorted
    
    
    
                    
if __name__ == '__main__':
    
    b_, a_ = plot_results(save_dir, x_test, y_test)

    
