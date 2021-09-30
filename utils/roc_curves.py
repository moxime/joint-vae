import numpy as np
from sklearn.metrics import auc


def fpr_at_tpr(fpr, tpr, a, thresholds=None,
               return_threshold=False):

    """fpr and tpr have to be in ascending order

    """
    assert(not return_threshold or thresholds is not None) 

    as_tpr = np.asarray(tpr)
    as_fpr = np.asarray(fpr)

    i_ = np.where(as_tpr >= a)[0].min()

    fpr_ = as_fpr[i_]
    
    if not return_threshold:
        return fpr_
    
    thr_ = thresholds[i_]

    return fpr_, thr_


def tpr_at_fpr(fpr, tpr, a):

    as_tpr = np.asarray(tpr)
    as_fpr = np.asarray(fpr)
    i_fpr = np.where(as_fpr <= a)[0]
    return as_tpr[i_fpr].max()


def roc_curve(ins, outs, *kept_tpr, two_sided=False, around='mean', validation=1000, debug=False):

    if validation < 1:
        validation = int(validation * len(ins)) 
        
    ins_n_valid = validation if two_sided else 0

    permute_ins = np.random.permutation(len(ins))
    val_ins_idx = permute_ins[:ins_n_valid]
    test_ins_idx = permute_ins[ins_n_valid:]
    
    sorted_ins = np.sort(np.asarray(ins)[test_ins_idx])
    ins_validation = np.asarray(ins)[val_ins_idx]

    if two_sided:
        if around == 'mean':
            center = ins_validation.mean()
            
    if two_sided:
        all_thresholds = np.concatenate([[-np.inf], np.sort(abs(sorted_ins - center)), [np.inf]])
        
    else:
        all_thresholds = np.concatenate([[-np.inf], sorted_ins, [np.inf]])[::-1]

    relevant_thresholds = []
    relevant_fpr = []
    relevant_tpr = []
    
    last_fpr = -.01

    kept_tpr = np.sort(kept_tpr)
    kept_fpr = np.zeros_like(kept_tpr)
    kept_thresholds = np.zeros_like(kept_tpr)
    kept_tpr_i = 0

    n_ins = len(ins)
    n_outs = len(outs)
    for t in all_thresholds:

        if two_sided:
            tpr = (abs(ins - center) <= t).sum() / n_ins
            fpr = (abs(outs - center) <= t).sum() / n_outs
        else:
            tpr = (ins >= t).sum() / n_ins
            fpr = (outs >= t).sum() / n_outs

        if debug == 'hard':
            print('{:5.2f} -- {:5.2f}'.format(100 * fpr, 100 * tpr), end=' ')
        if fpr >= last_fpr:
            relevant_thresholds.append(t)
            relevant_tpr.append(tpr)
            relevant_fpr.append(fpr)
            last_fpr = fpr
            if debug == 'hard':
                print('*')

        elif debug == 'hard':
            print()

        if kept_tpr_i < len(kept_tpr) and tpr >= kept_tpr[kept_tpr_i]:

            kept_fpr[kept_tpr_i] = fpr
            kept_tpr[kept_tpr_i] = tpr
            kept_thresholds[kept_tpr_i] = t
            kept_tpr_i += 1

    auroc = auc(relevant_fpr, relevant_tpr)

    if two_sided:
        kept_thresholds = np.concatenate([np.array([center]), kept_thresholds])

    return auroc, kept_fpr, kept_tpr, kept_thresholds
    

if __name__ == '__main__':

    from time import time
    import sklearn.metrics
    
    n_ = [25000, 25000]
    s_ = [1, 0.5]
    m_ = [0, 2]
    
    ins = np.concatenate([np.random.randn(n) * s + m for (n, s, m) in zip(n_, s_, m_)])
    ins = np.random.permutation(ins)
    
    outs = np.random.randn(10000) + 1

    inandouts = np.concatenate([ins, outs])
    labels = np.concatenate([np.ones_like(ins), np.zeros_like(outs)])
    
    kept_tpr = [_ / 100 for _ in range(90, 100)]

    offset = 4
    for two_sided, factor in zip((False,) + (True,) * 1, (1,) + (1,) + (1,) * 1):
        print('2S' if two_sided else '1S')
        t0 = time()
        auroc, fpr, tpr, thr = roc_curve(factor * ins + offset, outs, *kept_tpr,
                                         two_sided=two_sided, validation=1000)

        t_home_made = time() - t0
        
        print('AUC={:.1f}'.format(100 * auroc))
        for h, f, t in zip(thr, fpr, tpr):

            f_ = (outs >= h).sum() / len(outs)
            t_ = (ins >= h).sum() / len(ins)
            print(('{:7.3f}--{:7.3f} ').format(100 * f, 100 * t))


        t0 = time()
        sklearn.metrics.roc_curve(labels, inandouts)
        t_skl = time() - t0

        print('HM: {:.4f}s SK: {:.4f}s'.format(t_home_made, t_skl))
