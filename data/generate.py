from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import numpy as np
mnist = datasets.mnist
from keras import backend as K


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon




def get_mnist():
    
    (train_images, train_labels), (test_images, test_labels) = \
        mnist.load_data()
    
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



if __name__ == '__main__':

    dim = 2
    sizes = [3, 2]
    m1 = np.zeros(dim)
    m2 = 10*np.ones(dim)

    s1 = np.eye(dim)
    
    means = [m1, m2]
    covars = [s1, s1] 

    X = generate_gaussian_mixture(dim, sizes, means)
    y = generate_labels(sizes, [1, 0])    
    print('done')

