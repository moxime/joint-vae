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


def mse_loss(x_target, x_output, ndim=3, batch_mean=True):
    """
    x_target of size (N1, .. ,Ng, D1, D2,..., Dt) 
    x_output of size (L, (C,), N1, ..., Ng, D1, D2,..., Dt) where L is sampling size, 
    """

    output_dims_ = tuple(_ for _ in range(x_output.dim()))
    batch_dims_ = output_dims_[-x_target.dim():-ndim]
    input_dims_ = output_dims_[-ndim:]
    
    mean_dims = input_dims_

    # print('****', mean_dims)

    if batch_mean:
        return F.mse_loss(x_output,
                          x_target.expand_as(x_output)) # + 1e-8 # * torch.randn_like(x_output))
    return F.mse_loss(x_output, x_target.expand_as(x_output),
                      reduction='none').mean(mean_dims) # + 1e-8
    # return (x_target - x_output).pow(2).mean(mean_dims)


def kl_loss(mu_z, log_var_z,
            z=None,
            y=None,
            latent_dictionary=None,
            prior_variance=1.,
            var_weighting=1.,
            batch_mean=True,
            out=['kl']):

    # logging.debug('TBR l:64 Computing KL')
    assert y is None or latent_dictionary is not None

    if z is None:
        assert 'sdist' not in out
    
    var_loss = -(1 + log_var_z - np.log(prior_variance) - log_var_z.exp() / prior_variance).sum(-1)
    # print('*** losses:59', 'warm', var_weighting, 'var_loss', loss.mean().detach().cpu().item())
    loss = 0.5 * var_weighting * var_loss
    
    _ = y.shape if y is not None else ('*',) 
    # print('*** losses:59', 'mu', *mu_z.shape, 'lv', *log_var_z.shape, 'y', *_)
    # logging.debug('TBR l:72 Computing KL')
    if y is None:
        distances = mu_z.pow(2).sum(-1) / prior_variance
        loss += 0.5 * distances
        sdistances = z.pow(2).sum(-1) / prior_variance
        
    else:

        K = mu_z.shape[-1]
        centroids_shape = y.shape + (K,)

        centroids = latent_dictionary.index_select(0, y.view(-1)).view(centroids_shape)
        # print('*** losses:74', 'mu_', *mu_z.shape, 'centroids', *centroids.shape)
        distances = (mu_z - centroids).pow(2).sum(-1) / prior_variance
        # print('*** z:', *z.shape, 'centroid:', *centroids.shape) 
        sdistances = (z.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(-1) / prior_variance
        # print('*** losses:76', 'loss', *loss.shape, 'dist', *distances.shape)
        loss = loss + 0.5 * distances
        
    if torch.isnan(loss).any():
        logging.error('NAN found in KL')
        for l in log_var_z:
            logging.error('log_var %s', l.sum().item())
        for l in mu_z:
            logging.error('mu_z %s', l.sum().item())
        for l in loss:
            logging.error('loss %s', l.item())

    output_dict = {'kl': loss.mean() if batch_mean else loss,
                   'dist': distances.mean() if batch_mean else distances,
                   'var': var_loss.mean() if batch_mean else var_loss,
                   'sdist': sdistances.mean() if batch_mean else sdistances}

    return tuple(output_dict[_] for _ in out)


def x_loss(y_target, logits, batch_mean=True):

    """ Cross entropy

    - y_target of dims N1 x....x Ng(x1)
    - logits of dims L x N1 x...x Ng x C

    """

    # print('losses:118', type(y_target), 'logits:', *logits.shape)
    
    if y_target is None:

        log_p = (logits.softmax(dim=-1) + 1e-6).log()
        permutation = [-1] + [_ for _ in range(len(log_p.shape) - 2)]
        # print('*** losses:125 target is none', *logits.shape[1:], '->', *permutation)
        # print(*[p.item() for p in log_p.mean(0)[0, :]])
        if log_p.shape[0] > 1:
            return -log_p[1:].mean(0).permute(permutation) # .max(-1)[0]
        else:
            return -log_p[0].permute(permutation) # .max(-1)[0]
    
    C = logits.shape[-1]
    L = logits.shape[0]

    y_ = y_target.reshape(1, -1).repeat(L, 1).reshape(-1)
    logits_ = logits.reshape(-1, C)

    # print('losses:134 L=', L, 'C=', C, 'shapes', 'L', *logits_.shape, 'T', *y_.shape) 
    
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
        

    
