import torch
from torch.nn import functional as F
import time, logging
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


def mse_loss(x_target, x_output, ndim=3, sampling_dims=1, batch_mean=True):
    """
    x_target of size (N1, .. ,Ng, D1, D2,..., Dt) 
    x_output of size (L, (C,), N1, ..., Ng, D1, D2,..., Dt) where L is sampling size, 
    """

    output_dims_ = tuple(_ for _ in range(x_output.dim()))
    sampling_dims_ = output_dims_[:sampling_dims]
    batch_dims_ = output_dims_[-x_target.dim():-ndim]
    input_dims_ = output_dims_[-ndim:]
    
    mean_dims = sampling_dims_ + input_dims_
    if batch_mean:
        mean_dims += batch_dims_

    # print('****', mean_dims)
    return (x_target - x_output).pow(2).mean(mean_dims)


def kl_loss(mu_z, log_var_z, y=None, latent_dictionary=None, batch_mean=True, out_zdist=False):

    assert y is None or latent_dictionary is not None

    loss = -0.5 * (1 + log_var_z - log_var_z.exp()).sum(-1)

    if y is None:
        distances = mu_z.pow(2).sum(-1)
        loss += 0.5 * distances
        
    else:

        s_m = mu_z.shape
        K = mu_z.shape[-1]
        y_ = y.reshape(-1)
        
        mu_ = mu_z.exp().reshape(-1, K)
        centroids = latent_dictionary.index_select(0, y_)
        # print('*** gfdret ***', 'mu_', *mu_.shape, 'centroids', *centroids.shape)
        distances = (mu_ - centroids).pow(2).reshape(s_m).sum(-1)

        loss += 0.5 * distances
        
    # if torch.isnan(loss).any():
    #     for l in log_var_z:
    #         logging.error(l.sum().item())
    #     for l in mu_z:
    #         logging.error(l.sum().item())
    #     for l in loss:
    #         logging.error(l.item())
    
    if batch_mean:
        if out_zdist:
            return loss.mean(), distances.mean()
        return loss.mean()
    # print('losses l.76 | kl_loss', loss.shape)
    if out_zdist:
        return loss, distances
    return loss


def x_loss(y_target, logits, batch_mean=True):

    """ Cross entropy

    - y_target of dims N1 x....x Ng(x1)
    - logits of dims L x N1 x...x Ng x C

    """

    logits_dims_ = tuple(_ for _ in range(logits.dim()))
    batch_dims_ = logits_dims_[1:-1]
    
    C = logits.shape[-1]
    L = logits.shape[0]
    # print('*** klsde ***',
    #       'y_target', *y_target.shape,
    #       'logits',
    #       *logits.shape)

    y_ = y_target.reshape(1, -1).repeat(L ,1).reshape(-1)
    logits_ = logits.reshape(-1, C)
        
    if batch_mean:
        return F.cross_entropy(logits_, y_)

    shape = (L,) + y_target.shape
    return F.cross_entropy(logits_,
                           y_,
                           reduction='none').reshape(shape).mean(0)

    
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
    test_grad = True
    
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

        x = torch.rand(*L, *N, C, requires_grad=True)
        y_output = (x * 5).softmax(dim=-1)

        loss = x_loss(y_target, y_output)
        print(loss)

        loss_ = x_loss(y_target, y_output, batch_mean=False)

        if test_grad:

            loss.backward()
        

    
