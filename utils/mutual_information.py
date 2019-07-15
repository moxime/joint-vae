import numpy as np

def cross_entropy(p, q):

    if p[q == 0].any():
        return np.infty

    h_pq = np.nansum(-p * np.log2(q + np.spacing(1)))
    return  h_pq

def d_kl(p, q):

    if p[q == 0].any():
        return np.infty

    i_pos = (q > 0)

    p_pos = p[i_pos]
    q_pos = q[i_pos]
    
    r = p_pos / q_pos

    dkl = np.nansum(p_pos * np.log2(r))

    return(dkl)
    return max(0, dkl)
    

if __name__ == '__main__':



    shape = (64, 16)
    
    p = np.random.rand(*shape)
    p = p / np.sum(p)

    q = np.ones(shape)
    q[0][0] = 0
    q = q / np.sum(q)

    t = np.random.rand(*shape)
    t[q==0] = 0
    t = t / np.sum(t)
    
    print(d_kl(p, q))
    print(d_kl(p, p))
    print(d_kl(t, q))
    print(d_kl(q, q))
    
    




