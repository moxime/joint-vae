import numpy as np
from cvae import ClassificationVariationalNetwork
import os
import matplotlib.pyplot as plt
from data import generate as dg



(x_train, y_train, x_test, y_test) = dg.get_mnist()

    


e_ = [1024, 1024]
d_ = e_.copy()
d_.reverse()

e_ = [1024, 1024, 512, 256]
d_ = e_.copy()
d_.reverse()


latent_dim = 20
latent_sampling = int(500)

save_dir = (f'./jobs/fashion-mnist/latent-dim=' + 
            f'{latent_dim}-sampling={latent_sampling}' +
            f'-encoder-layers={len(e_)}')


# beta_ = np.logspace(-4, -5, 5)
# beta_ = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
beta_ = [1e-5, 5e-5, 1e-4]
# digits = [i for i in range (1,10)]
# beta_ = ([i*1e-6 for i in digits] + [i*1e-3 for i in digits])
# beta_ = [2e-3, 3e-3]*2
# beta_ = [1e-3 + i*1e-4 for i in digits] * 3

epochs = 20






def read_float_list(path):

    list = None
    if os.path.exists(path):
        with open(path, 'r') as f:
            list = [float(_) for _ in f.readlines()]

    return list


def write_list(list, path):
    with open(path, 'w+') as f:
        for _ in list:
            f.write(str(_) + '\n')

            
def training_loop(input_dim, num_labels, encoder_layers, latent_dim,
                  decoder_layers, x_train, y_train, x_test, y_test, betas,
                  directory='./jobs', epochs=30, latent_sampling=100, batch_size=10):


    if not os.path.exists(directory):
        os.makedirs(directory)
    betas_file = os.path.join(directory, 'betas_to_be_run')

    betas_from_file = read_float_list(betas_file)

    if betas_from_file is not None:
        betas = betas_from_file

    while len(betas) > 0:

        write_list(betas, betas_file)
        beta = betas[0]
            
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

        dir_beta_ = os.path.join(directory, f'beta={beta:.5e}')
        dir_beta = dir_beta_
        i = -1
        print(dir_beta, os.path.exists(dir_beta))
        while os.path.exists(dir_beta):
            i += 1
            dir_beta = f'{dir_beta_}-{i}'
            print(dir_beta, os.path.exists(dir_beta))
        vae.save(dir_beta)

        betas = read_float_list(betas_file)
        if len(betas) > 0:
            betas.pop(0)

        
    os.remove(betas_file)
    
    pass


def gogo():
    
    training_loop(28**2, 10, e_, latent_dim, d_, x_train, y_train,
                  x_test, y_test, beta_, directory=save_dir,
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

    plt.close(fig='all')
    
    return beta_sorted, acc_sorted
    
    
    
                    
if __name__ == '__main__':
    
    b_, a_ = plot_results(save_dir, x_test, y_test)

    
