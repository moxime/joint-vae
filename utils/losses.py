import torch
from torch.nn import functional as F
import time
import numpy as np

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

    loss = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum(-1)

    if batch_mean:
        return loss.mean()

    return loss


def x_loss(y_target, y_output, sampling_dims=1, batch_mean=True):

    """ Cross entropy

    - y_target of dims N1 x....x Ng(x1)
    - y_output of dims L1,.., Lf x N1 x...x Ng x C

    """
    C = y_output.shape[-1]
    
    sampling_dims_ = [_ for _ in range(sampling_dims)]    
    output_log_probas = y_output.log().mean(sampling_dims_).reshape(-1, C)

    if batch_mean:

        return F.nll_loss(output_log_probas, y_target.reshape(-1))

    shape = y_target.shape
    return F.nll_loss(output_log_probas,
                      y_target.reshape(-1),
                      reduction='none').reshape(shape)

    
def x_loss_pushy(y_target, y_output, sampling_dims=1, batch_mean=True):
    """

    y_target of dims N1 x....x Ng(x1)
    y_output of dims L1,.., Lf x N1 x...x Ng x C

    """
    L_ = y_output[:sampling_dims]
    s_y = y_input.squeeze_().shape

    one_dim = ()
    L_1 = ()
    for _ in s_y : one_dim += (1,)
    for _ in L_: L_1 += (1,)
    y_in_repeated = y_input.reshape(*L_1, *s_y).repeat(*L_, *one_dim)

    dims = tuple(_ for _ in range(y_output.dim()))
    out_perm_dims = (0,) + (-1,) + dims[1:-1] # from LxN1...xNgxC to LxCxN1...xNgxC

    if batch_mean:
        return F.nll_loss(y_output.permute(out_perm_dims).log(),
                          y_in_repeated)

    loss = F.nll_loss(y_output.permute(out_perm_dims).log(),
                      y_in_repeated, reduction='none')

    return loss.mean(0)

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

    test_mse = False
    test_xent = True

    print(device)
    
    L = (3,)
    N = (4,)
    # N = (10, 200)
    # N = (200,)
    D = (1, 28, 28)
    # D = (784,)
    C = 10

    if test_mse:
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
    
    
    if test_xent:

        y_target = torch.randint(C, N) 
        y_output = (torch.rand(*L, *N, C) * 5).softmax(dim=-1)

        loss = x_loss(y_target, y_output)
        print(loss)

        loss_ = x_loss(y_target, y_output, batch_mean=False)
