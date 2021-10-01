from time import time
import numpy as np
from sklearn.metrics import auc, roc_curve as fast_roc_curve
import logging


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

    if debug:
        logging.debug('Computing fprs with a {}-sided test with data of lengths {} / {}'.format(
            'two' if two_sided else 'one', len(ins), len(outs)))
                      
    if validation < 1:
        validation = int(validation * len(ins)) 
        
    ins_n_valid = validation if two_sided else 0

    permute_ins = np.random.permutation(len(ins))
    val_ins_idx = permute_ins[:ins_n_valid]
    test_ins_idx = permute_ins[ins_n_valid:]
    
    ins_validation = np.asarray(ins)[val_ins_idx]

    if two_sided:
        if around == 'mean':
            center = ins_validation.mean()
        all_thresholds = np.concatenate([[-np.inf], np.sort(abs(ins[test_ins_idx] - center))])
        sorted_outs = np.sort(abs(outs - center))
            
    else:
        all_thresholds = np.concatenate([[-np.inf], np.sort(-ins[test_ins_idx])])
        sorted_outs = np.sort(-outs)

    relevant_thresholds = []
    relevant_fpr = []
    relevant_tpr = []
    
    last_fpr = -.01

    kept_tpr = np.sort(kept_tpr)
    kept_fpr = np.zeros_like(kept_tpr)
    kept_thresholds = np.zeros_like(kept_tpr)
    kept_tpr_i = 0

    n_ins = len(all_thresholds) - 1
    n_outs = len(outs)
        
    if debug == 'time':
        t0 = time()
        print('*** Going through thresholds')

    outs_idx = 0
        
    for ins_idx, t in enumerate(all_thresholds):

        while outs_idx < n_outs and sorted_outs[outs_idx] < t:
            outs_idx += 1

        tpr = (ins_idx) / n_ins 
        fpr = (outs_idx) / n_outs

        if not two_sided: t = -t
        if debug == 'hard':
            if two_sided:
                tpr_ = (abs(ins[test_ins_idx] - center) <= t).sum() / n_ins
                fpr_ = (abs(outs - center) <= t).sum() / n_outs
            else:
                tpr_ = (ins[test_ins_idx] >= t).sum() / n_ins
                fpr_ = (outs >= t).sum() / n_outs

            print('   {:7.4f} -- {:7.4f}'.format(100 * fpr, 100 * tpr))
            print('___{:7.4f} -- {:7.4f}'.format(100 * fpr_, 100 * tpr_), end=' ')
            
        if fpr >= last_fpr:
            relevant_thresholds.append(t)
            relevant_tpr.append(tpr)
            relevant_fpr.append(fpr)
            last_fpr = fpr
            if debug == 'hard':
                print('*___')

        elif debug == 'hard':
            print('___')

        if kept_tpr_i < len(kept_tpr) and tpr >= kept_tpr[kept_tpr_i]:

            kept_fpr[kept_tpr_i] = fpr
            kept_tpr[kept_tpr_i] = tpr
            kept_thresholds[kept_tpr_i] = t
            kept_tpr_i += 1

            if debug in ('medium', 'hard'):

                for what, data in zip(('TPR', 'FPR'), (ins, outs)):

                    n = len(data)
                    if two_sided:

                        t1 = center - t
                        t2 = center + t
                        bt = (data <= center - t).sum() / n * 100
                        at = (data > center + t).sum() / n * 100
                        pr = 100 - at - bt
                        _s = f'{what}: {bt:5.2f} ({t1:+.2e}) {pr:5.2f} ({t2:+.2e}) {at:5.2f}'

                    else:
                        bt = (data < t).sum() / n * 100
                        pr = (data >= t).sum() / n * 100
                        _s = f'{what}: {bt:5.2f} ({t:+.2e}) {pr:5.2f}'
                    
                    logging.debug(_s)

                logging.debug('--')
    
    if debug == 'time':
        print(f'Thresholding ins done in {time() - t0:3f}s')
        t0 = time()
    auroc = auc(relevant_fpr, relevant_tpr)
    if debug == 'time':
        print(f'AUC computed in {time() - t0:3f}s')

    if two_sided:
        kept_thresholds = np.concatenate([np.array([center]), kept_thresholds])
        
    return auroc, kept_fpr, kept_tpr, kept_thresholds
    

if __name__ == '__main__':

    from time import time
    import sklearn.metrics
    logging.getLogger().setLevel(logging.DEBUG)
    
    n_ = [1, 100-1]
    s_ = [1, 0.5]
    m_ = [2, 2]
    
    ins = np.concatenate([np.random.randn(n) * s + m for (n, s, m) in zip(n_, s_, m_)])
    ins = np.random.permutation(ins)
    
    outs = np.random.randn(250) + 1

    inandouts = np.concatenate([ins, outs])
    labels = np.concatenate([np.ones_like(ins), np.zeros_like(outs)])
    
    kept_tpr = [_ / 100 for _ in range(90, 100)]

    offset = 2

    two_sided_ = (True, False)
    factor_ = (1, 1)

    # two_sided_ = (False,)
    # two_sided_ = (True,)
    # factor_ = (1,)
    
    for two_sided, factor in zip(two_sided_, factor_):
        
        print('2S' if two_sided else '1S')
        t0 = time()
        auroc, fpr, tpr, thr = roc_curve(factor * ins + offset, outs, *kept_tpr,
                                         debug=False,
                                         two_sided=two_sided, validation=1)

        t_home_made = time() - t0
        
        print('AUC={:.2f}'.format(100 * auroc))

        if two_sided:
            center = thr[0]
            thr = thr[1:]
            
        for h, f, t in zip(thr, fpr, tpr):

            print((' {:7.3f}--{:7.3f} ').format(100 * f, 100 * t))
            if not two_sided: 
                f_ = (outs >= h).sum() / len(outs)
                t_ = (factor * ins + offset >= h).sum() / len(ins)
            else:
                f_ = (abs(outs - center) <= h).sum() / len(outs)
                t_ = (abs(factor * ins + offset - center) <= h).sum() / len(ins)
                
            print(('_{:7.3f}--{:7.3f} ').format(100 * f_, 100 * t_))

        t0 = time()
        sklearn.metrics.roc_curve(labels, inandouts)
        t_skl = time() - t0

        print('HM: {:.4f}s SK: {:.4f}s'.format(t_home_made, t_skl))
