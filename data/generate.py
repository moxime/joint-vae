import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer, Dense, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

mnist = datasets.mnist
fashion_mnist = datasets.fashion_mnist


def get_fashion_mnist():

    return get_image_dataset(fashion_mnist)

def get_mnist():

    return get_image_dataset(mnist)

def get_image_dataset(dataset=mnist):
    
    (train_images, train_labels), (test_images, test_labels) = \
        dataset.load_data()       
    
    num_train = len(train_labels)
    num_test = len(test_labels)
    
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    train_images = train_images.reshape((num_train, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((num_test, 28 * 28))
    test_images = test_images.astype('float32') / 255

    return train_images, one_hot_train_labels, test_images, one_hot_test_labels


def gaussian_ball(N, mean, covar=1):

    dim = mean.size
    if np.ndim(covar) == 0:
        covar = covar * np.eye(dim)
    return np.random.multivariate_normal(mean, covar, size=N)


def gaussian_shell(N, dim, mean, var):

    gb = gaussian_ball(N, np.zeros(dim), 1).T

    dists = np.random.normal(loc=mean, scale=np.sqrt(var), size=N)

    def normed(x):
        n = np.sqrt((x**2).sum(axis=0))
        return (x / n)
        
    return (normed(gb) * dists).T


def generate_labels(batch_sizes, batch_labels):
    """generates_labels([100, 50, 50], [0, 1, 0]) creates a (200,) ndarray
    containing 100 '0', 50 '1' and 50 '0'

    """
    
    num_cat = max(batch_labels) + 1
    Y = np.zeros((sum(batch_sizes), num_cat))
    n0 = 0
    for i, size in enumerate(batch_sizes):
        n1 = n0 + size
        Y[n0:n1, batch_labels[i]] = 1
        n0 = n1

    return Y


def generate_random_means_covar(dim, num_parameters,
                                sigma=1,
                                sigma_mu=10,
                                sigma_A=1):
    """
    Generates parameters for a mixture of gaussians of dim dim.
    
    Parameters: 
    * dim: dimension of data 
    * N: number of parameters
    * sigma: the average diagonal of the covariance matrices
    * sigma_mu: the variance of the mean vectors
    * sigma_A: the variance of the elements creating the
    covraince matrices.

    Returns
    * mu_: the list of len N of mean-vectors
    * cov_: the list of covariance matrices 
    (cov = A.T*A is a covariance matrix)
    
    """
    cov_ = []
    mu_ = []
    p = dim

    for i in range(num_parameters):
        A = sigma_A*np.random.randn(p * p).reshape(p, p)
        + sigma*np.eye(p)

        cov = np.dot(A.T, A)
        cov_.append(cov)
        #       print('A_[{}].shape={}'.format(i, A_[i].shape))
        mu_.append(sigma_mu*np.random.randn(p))

    return mu_, cov_


def generate_gaussian_mixture(dim, sizes, mean_vectors, covariance_matrices=None):
    """
    Generates a mixture of gaussians of dim dim.
    
    Parameters: 
    * dim: dimension of data 
    * sizes: list of integers, e.g. [10, 30, 5]
    * mean_vectors: a list of mean_vectors
    * covariance_matrices: a list of covariance
    matrices (default is None, all are identity)

    Returns
    * X: a N*dim matrix (N=sum(sizes)) X[ni:ni+sizes[i],:]
    is a stack of gaussian sizes[i] vectors of mean mu_[i], and covar
    matrix cov_mat[i]
    
    """
    if covariance_matrices is None:
        covariance_matrices = []
        for s in sizes:
            covariance_matrices.append(np.eye(dim))
    p = dim
    n = sum(sizes)
    X = np.ndarray((n, p))
    n0 = 0
    for i, size in enumerate(sizes):
        n1 = n0 + size
        mu = mean_vectors[i]
        cov = covariance_matrices[i]
        X[n0:n1, :] = np.random.multivariate_normal(mu, cov, size)
        n0 = n1

    return X




def data_generator(input_dim, output_dim, layer_dims, output_var=0., activation='relu'):
    """ creates a network that will take a gaussian random variable z=N(0, I)
    as an input and generate x = N(f(z), \sigma^2 I) as aan output
    """

    input = Input(shape=(input_dim,))
    x = input
    
    for i, d in enumerate(layer_dims):
        x = Dense(d, activation=activation, name=f'intermediate-{i}',
                  bias_initializer='random_uniform',
                  kernel_initializer='random_uniform')(x)

    x = Dense(output_dim, name='mean')(x)
    
    class SamplingLayer(Layer):

        def __init__(self, var=0., *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.var = var
            
        def call(self, input):
            z_mean = input
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            # print(z_log_var, batch, dim)
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + self.var * epsilon

        
    s = SamplingLayer(var=output_var)(x)

    return Model(inputs=input, outputs=s)




if __name__ == '__main__':

    def test_gen_net(net=None, layers=None):

        P = 10
        N, T, M = 1000000, 5, P*2
        if layers is None:
            layers = np.random.randint(4, 12, 2)
        if net is None:
            net = data_generator(T, M, layers, output_var=0.001)
        z = np.random.randn(N, T)

        x = net(z)

        for p in range(P):
            plt.scatter(x[:,2*p], x[:,2*p+1], marker='.', s=4, alpha=10000/N)

        m = x.numpy().mean(axis=0)
        std = x.numpy().std(axis=0)
        
        print(f'layers are {layers}\nx is of mean {m} and std {std}')
        
        plt.show()

        return x, z, net, layers

        pass

    x, z, net, layers = test_gen_net()
