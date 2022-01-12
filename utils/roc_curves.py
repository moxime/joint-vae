from time import time
import numpy as np
from sklearn.metrics import auc, roc_curve as fast_roc_curve
import logging
from scipy.interpolate import UnivariateSpline

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


def roc_curve(ins, outs, *kept_tpr, two_sided=False, validation=0.1, debug=False):

    t0 = time()
    if debug:
        logging.debug('Computing fprs with a {}-sided test with data of lengths {} / {}'.format(
            'two' if two_sided else 'one', len(ins), len(outs)))
                      
    if validation < 1:
        validation = int(validation * len(ins)) 
        
    ins_n_valid = validation if two_sided else 0
    # print('***', len(ins), len(outs), ins_n_valid)

    _s = np.random.get_state()
    np.random.seed()
    permute_ins = np.random.permutation(len(ins))
    np.random.set_state(_s)
    val_ins_idx = np.sort(permute_ins[:ins_n_valid])
    test_ins_idx = permute_ins[ins_n_valid:]
    
    ins_validation = np.sort(np.asarray(ins)[val_ins_idx])
    sorted_outs = np.sort(outs)
    sorted_ins = np.sort(ins[test_ins_idx])

    all_thresholds = {}

    if two_sided == 'around-mean':
        center = ins_validation.mean()
        delta_thresholds = np.concatenate([[0], np.sort(abs(ins[test_ins_idx] - center)), [np.inf]])
        all_thresholds['low'] = - delta_thresholds[::-1] + center
        all_thresholds['up'] = delta_thresholds + center

        
    elif isinstance(two_sided, tuple):
        
        old_indices = np.arange(0, len(ins_validation))
        new_length = len(ins)
        new_indices = np.linspace(0, len(ins_validation) - 1, new_length)
        spl = UnivariateSpline(old_indices, ins_validation, k = 3, s = 0)
        interpolated_ins_validation = spl(new_indices)
        for f, k in zip(two_sided, ('low', 'up')):
            t = ins_validation[::f]
            all_thresholds[k] = np.concatenate([[-np.inf], interpolated_ins_validation[::f], [np.inf]])
            
    else:
        
        all_thresholds['low'] = np.concatenate([[-np.inf], np.sort(ins[test_ins_idx])])
        all_thresholds['up'] = np.ones_like(all_thresholds['low']) * np.inf

    relevant_thresholds = []
    relevant_fpr = []
    relevant_tpr = []
    relevant_prec = []
    
    last_fpr = -.01

    kept_tpr = np.sort(kept_tpr)
    original_kept_tpr = kept_tpr.copy()
    kept_fpr = np.zeros_like(kept_tpr)
    kept_thresholds = {'low': np.zeros_like(kept_tpr), 'up': np.zeros_like(kept_tpr)}
    kept_precisions = np.zeros_like(kept_tpr)
        
    if debug == 'time':
        t0 = time()
        if debug: print('*** Going through thresholds')

    n = {'in': len(sorted_ins),
         'out': len(sorted_outs)}

    scores = {'in': sorted_ins, 'out': sorted_outs}
    
    idx = {'in': {'low': 0, 'up': -1},
           'out': {'low': 0, 'up': -1},
           'thr': {'low': 0, 'up': -1}}

    t = {_: all_thresholds[_][idx['thr'][_]] for _ in ('low', 'up')}

    last_fpr = 1.1
    last_tpr = 1.1
    
    kept_tpr_i = -1

    num_print = 50
    every_print = max(len(all_thresholds['low']) // num_print, 1)

    it = 0
    nt = min(len(all_thresholds[_]) for _ in ('up', 'low'))

    if debug :
        if two_sided == 'around-mean':
            print('mean: {:.6g} ({}) real: {:.6g} ({:.5g})'.format(center,
                                                                   len(ins_validation),
                                                                   ins.mean(),
                                                                   ins.std()
                                                                   ))
    
    while t['low'] < t['up'] and it < nt - 1:

        for w in ('out', 'in'):
            while idx[w]['low'] < n[w] - 1 and scores[w][idx[w]['low']] < t['low']:
                idx[w]['low'] += 1
            while idx[w]['up'] > -n[w] and scores[w][idx[w]['up']] > t['up']:
                idx[w]['up'] -= 1

        # neg = {w: max(0, idx[w]['low'] - 1) - idx[w]['up'] - 1 for w in ('out', 'in')}
        neg = {w: idx[w]['low'] - (idx[w]['up'] + 1) for w in ('out', 'in')}
        
        tpr = 1 - neg['in'] / n['in']
        fpr = 1 - neg['out'] / n['out']

        _s = ' <= '.join(['{:-13.7g}'] * 4)

        if not it % every_print:

            if debug:
                for w in ('in', 'out'):
                    print('{:3}:'.format(w), end=' ')
                    print(_s.format(t['low'],
                                    scores[w][idx[w]['low']],
                                    scores[w][idx[w]['up']],
                                    t['up']),
                          end=' | ')
                    print(idx[w]['low'], idx[w]['up'])
                
                print('|_FPR={:6.2%} TPR={:6.2%}'.format(fpr, tpr))
        
        it += 1
        idx['thr']['low'] +=1
        idx['thr']['up'] -=1

        t = {_: all_thresholds[_][idx['thr'][_]] for _ in ('low', 'up')}
            
        if fpr >= 0 and tpr >= 0: 
            relevant_thresholds.append((t['low'], t['up']))
            relevant_tpr.append(tpr)
            relevant_fpr.append(fpr)
            last_fpr = fpr
            last_tpr = tpr

        if kept_tpr_i >= - len(kept_tpr):
            if tpr < original_kept_tpr[kept_tpr_i]:
                 kept_tpr_i -= 1
            else:
                # print('  {:6.2%} {:6.2%}>={:6.2%}'.format(fpr, tpr, original_kept_tpr[kept_tpr_i]))
                kept_fpr[kept_tpr_i] = fpr
                kept_tpr[kept_tpr_i] = tpr
                for _ in t:
                    kept_thresholds[_][kept_tpr_i] = t[_]
    # print('*** iterations:', it)
    if debug:
        for w in ('in', 'out'):
            print('{:3}:'.format(w), end=' ')
            print(_s.format(t['low'],
                            scores[w][idx[w]['low']],
                            scores[w][idx[w]['up']], t['up']), end=' | ')
            print(idx[w]['low'], idx[w]['up'])
                
            print('FPR={:6.2%} TPR={:6.2%}'.format(fpr, tpr))

            
    relevant_fpr.append(0.0)
    relevant_tpr.append(0.0)
    
    auroc = auc(relevant_fpr, relevant_tpr)


    # return relevant_fpr, relevant_tpr
    # print('*** rc', len(ins), len(outs), time() - t0)
        
    return auroc, kept_fpr, kept_tpr, kept_thresholds
    

if __name__ == '__main__':

    from time import time
    import sklearn.metrics
    logging.getLogger().setLevel(logging.DEBUG)
    
    n_ = [1, 100000-1]
    s_ = [1, 1]
    m_ = [0, 0]
    
    ins = np.concatenate([np.random.randn(n) * s + m for (n, s, m) in zip(n_, s_, m_)])
    ins = np.random.permutation(ins)
    
    outs = np.random.randn(10000) 

    inandouts = np.concatenate([ins, outs])
    labels = np.concatenate([np.ones_like(ins), np.zeros_like(outs)])
    
    kept_tpr = [_ / 100 for _ in range(90, 100)]

    kept_tpr.append(0.999)
    
    offset = -6

    two_sided_ = (False, True)
    factor_ = (1, 1)

    # two_sided_ = (False,)
    # two_sided_ = (True,)
    # factor_ = (1,)
    
    for two_sided, factor in zip(two_sided_, factor_):
        
        print('2S' if two_sided else '1S')
        t0 = time()
        auroc, fpr, tpr, thr = roc_curve(factor * ins + offset, outs, *kept_tpr,
                                         debug=False,
                                         two_sided=two_sided, validation=100)

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
