import numpy as np

def cross_entropy(p, q):

    if p[q == 0].any():
        return np.infty

    h_pq = np.nansum(-p * np.log2(q + np.spacing(1)))
    return  h_pq

def d_kl(p, q, axis=None):
    """ if axis is not None then the output
    is of length shape[axis]
    """

    if axis==None:

        if p[q == 0].any():
            return np.infty

        i_pos = (q > 0)

        p_pos = p[i_pos]
        q_pos = q[i_pos]
    
        r = p_pos / q_pos

        dkl = np.nansum(p_pos * np.log2(r))

        return(dkl)
    # return max(0, dkl)

    N = p.shape[axis]
    dkl = np.ndarray(N)

    if axis==0:
        for i in range(N):
            dkl[i] = d_kl(p[i], q[i], axis=None)
        return dkl

    p0 = np.moveaxis(p, axis, 0)
    q0 = np.moveaxis(q, axis, 0)

    return d_kl(p0, q0, axis=0)
    
            

def d_ib(p_x_y, p_t_y):
    """ 
    x is (N, p) or a (p,) numpy array where p is the dim of the input
    p_x_y is p(y|x) is (N,k) where k is the dim of the output
    p_t_y, a (N,k) array is the value(s) of t associated whith x
    """
    return d_kl(p_x_y, p_t_y, axis=0)

if __name__ == '__main__':

    N = 12
    shape = (64, 16)
    sumed_axis = tuple(range(1,len(shape)+1))

    p = np.random.rand(N, *shape)
    p = (p.T / np.sum(p, axis=sumed_axis)).T

    q = np.ones(p.shape)
    q[0][0] = 0
    q = (q.T / np.sum(q, axis=sumed_axis)).T

    t = np.random.rand(N, *shape)
    t[q==0] = 0
    t = (t.T / np.sum(t, axis=sumed_axis)).T

    p_0 = np.random.rand(*shape)
    p_0 /= p_0.sum()
    p_ = np.repeat(p_0[np.newaxis, :], N, axis=0)

    q_0 = np.random.rand(*shape)
    q_0 /= q_0.sum()
    q_ = np.repeat(q_0[np.newaxis, :], N, axis=0)
 
    print(d_kl(p, q, axis=1))
    print(d_kl(p, p, axis=0))
    print(d_kl(t, q, axis=0))
    print(d_kl(q, q, axis=1))
    print(d_kl(p_, q_, axis=None))
    




