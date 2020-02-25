import torch
from torch.nn import functional as F
import time


def compare_dims(small_dim, large_dim):
    """compare dims of tensor sizes

    - small_dim is (N1, N2,..., Ng) or (1, 1,..., N1, ..., Ng)

    - large_dim is (L1, L2,..,Lf, N1, ... , Ng, D1, ... , Dt)

    - returns f, t, True or _, _, False

    """
    print('Is ', small_dim, ' in ', large_dim, '?')
    if len(small_dim) == 0:
        return 0, len(large_dim), True
    
    if small_dim[0] == 1:
        f, t, ok = compare_dims(small_dim[1:], large_dim[1:])
        if ok:
            return f + 1, l , ok

    if large_dim[0] == small_dim[0]:
        f, t, ok = compare_dims(small_dim[1:], large_dim[1:])
        if ok:
            return f + 1, t, ok

    f, t, ok = compare_dims(small_dim, large_dim[1:])
    return f + 1, t, ok



def mse_loss(x_target, x_output, sampling_dims=1, ndim=0, batch_mean=True):
    """
    x_target of size (N1, .. ,Ng, D1, D2,..., Dt) 
    x_output of size (L, N1, ..., Ng, D1, D2,..., Dt) where L is sampling size, 
    """
    sampling_dims_ = [_ for _ in range(sampling_dims)]

    # print('sampling_dims: ', sampling_dims)
    
    if sampling_dims > 0:
        mean_output_sampling = x_output.mean(sampling_dims_)
        var_output_sampling = x_output.var(sampling_dims_, unbiased=False)
    else:
        mean_output_sampling = x_output.mean(sampling_dims_)
        var_output_sampling = x_output.var(sampling_dims_, unbiased=False)

        
    if batch_mean:
        return F.mse_loss(mean_output_sampling,
                          x_target) + var_output_sampling.mean()

    batch_ndim = x_target.dim()
    mean_dims = [_ for _ in range(batch_ndim - ndim, batch_ndim)]

    # print('Shapes: ',
    #       mean_output_sampling.shape,
    #       x_output.shape,
    #       var_output_sampling.shape)
    
    mse = F.mse_loss(mean_output_sampling,
                     x_target, reduction='none').mean(mean_dims)

    return mse + var_output_sampling.mean(mean_dims)


def kl_loss(mu_z, log_var_z, batch_mean=True):

    def kl_loss(self, mu, log_var, batch_mean=True):

    loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)

    if batch_mean:
        return loss.mean()

    return loss

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

if __name__ == '__main__':

    force_cpu = False
    has_cuda = torch.cuda.is_available and not force_cpu
    device = torch.device('cuda' if has_cuda else 'cpu')

    print(device)
    
    L = (100, 10)
    N = (200, 1, 3)
    # N = (200,)
    D = (1, 28, 28)
    # D = (784,)

    tick = time.time()
    x_target = torch.randn(*N, *D, device=device, requires_grad=False)
    x_output = torch.randn(*L, *N, *D, device=device)

    print(f'tensors ready in {1e3 * (time.time() - tick):.0f} ms')
    
    tick = time.time()

    tests = 1
    for t in range(tests):
        my_mse = mse_loss(x_target, x_output, ndim=len(D))

    print(f'my: {1e6*(time.time() - tick)/tests:.0f} us')

    tick = time.time()

    for t in range(tests):
        torch_mse = F.mse_loss(x_target, x_output)

    print(f'torch: {1e6*(time.time() - tick)/tests:.0f} us')
    
    print(my_mse.item(), torch_mse.item())

    loss = mse_loss(x_target, x_output, sampling_dims=2, ndim=len(D), batch_mean=False)

    print(loss.shape)
    
    
