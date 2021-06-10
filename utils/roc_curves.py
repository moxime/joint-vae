import numpy as np


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


if __name__ == '__main__':
    pass
