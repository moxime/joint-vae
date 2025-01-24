import sys
import os

import logging

import torch
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def scrisk(y_true, y_est, scores):
    """srisk: computation of sc(od) risk

    -- y_true : class for indist , -1 for ood

    -- y_est : estimated class

    -- scores: the HIGHer the more likely to REJECT

    """

    i_ = scores.sort()[1]

    y_ = y_true[i_]

    y_est_ = y_est[i_]

    classif_errors = ((y_ != y_est_) & (y_ >= 0)).cumsum(0).squeeze()
    false_positives = (y_ < 0).cumsum(0).squeeze()
    fpr = false_positives / (y_true < 0).sum()
    true_positives = (y_ >= 0).cumsum(0).squeeze()
    tpr = true_positives / (y_true >= 0).sum()

    # print(tpr)
    # print(scores[i_])

    selective_risk = classif_errors / true_positives

    thr = {_: scores[i_][(tpr > _ / 100) & (y_ >= 0)].min() for _ in (0, 1, 95, 99)}

    thr[0] = 0.

    for _ in thr:
        tpr_at_thr = ((scores <= thr[_]) & (y_true >= 0)).sum() / (y_true >= 0).sum()
        fpr_at_thr = ((scores <= thr[_]) & (y_true < 0)).sum() / (y_true < 0).sum()
        logging.info('thr@{:2} = {:.1e} ({:5.1%}/{:5.1%})'.format(_, thr[_],
                                                                  tpr_at_thr, fpr_at_thr))

    return tpr, selective_risk, fpr


def scoring(losses, r=None, g='elbo', weight=1):
    """
    reject if high
    """
    y_est = losses['y_est_already']

    score = scoring_r(losses, score=r, y_est=y_est)
    score += weight * scoring_g(losses, score=g, y_est=y_est)
    return score


def scoring_r(losses, score='msp', y_est=None):
    """ estimation of 1 - max P(y|x)
    """
    if score is None:
        return 0.

    if y_est is None:
        y_est = losses['y_est_already']

    if score == 'predist':
        return 0.5 * losses['pre-zdist'].gather(0, y_est.unsqueeze(0)).squeeze()

    if score == 'dist':
        return 0.5 * losses['zdist'].gather(0, y_est.unsqueeze(0)).squeeze()

    if score == 'logmsp':
        return scoring_r(losses, 'dist', y_est=y_est) - (-0.5 * losses['zdist']).logsumexp(0)

    return 0.


def scoring_g(losses, score='elbo', y_est=None):
    """ estimation of pOut/pIn
    """

    def scoring_alt(k):

        s = losses[k + '@']
        if s.ndim == 2:
            return s[0, :]
        return s

    def scoring_in(k):

        return losses[k].gather(0, y_est.unsqueeze(0)).squeeze()

    if y_est is None:
        y_est = losses['y_est_already']

    key = score
    sign = 1
    if score == 'elbo':
        key = 'zdist'
        sign = 0.5

    if score == 'iws':
        sign = -1

    # score has to be high for ood
    return sign * (scoring_in(key) - scoring_alt(key))


if __name__ == '__main__':
    import argparse
    from utils.save_load import find_by_job_number, model_subdir, SampleRecorder, LossRecorder

    parser = argparse.ArgumentParser()

    parser.add_argument('--array-dir', default='wim-arrays-inspection.bak')
    parser.add_argument('jobs', nargs='*', type=int)
    parser.add_argument('-g', default='elbo')
    parser.add_argument('-r', default='none')
    parser.add_argument('-w', '--weight', default=1., type=float)
    parser.add_argument('-v', action='count')

    jobs = [655755, 680490]
    jobs = [660655]
    jobs = [683512]
    jobs = [680490]

    argv = [str(_) for _ in jobs]

    argv = None if sys.argv[0] else argv
    args = parser.parse_args(argv)

    if argv:
        print('[{}]\n'.format(args))

    logging.getLogger().setLevel(logging.ERROR)

    models = find_by_job_number(*args.jobs, job_dir=args.array_dir, force_dict=True)

    if args.v:
        logging.getLogger().setLevel(logging.INFO)

    figures = {}

    for j in models:

        print('\n{:=^80}'.format(j))

        mdict = models[j]
        dset = mdict['set']
        oodsets = mdict['wim_sets'].split('-')
        allsets = [dset] + oodsets

        rdir = model_subdir(mdict, 'samples', '{:04}'.format(mdict['done']))

        sample_recorders, loss_recorders = {}, {}

        for subdir, w in zip(('.', 'init'), ('after', 'before')):
            d = os.path.join(rdir, subdir)
            sample_recorders[w] = SampleRecorder.loadall(d)
            loss_recorders[w] = LossRecorder.loadall(d)

        # for _ in allsets:
        #     print(_, loss_recorders['after'][_]['y_true'].shape)

        rec = loss_recorders['after']

        y_est = torch.hstack([rec[_]['y_est_already'] for _ in allsets])
        scores = torch.hstack([scoring(rec[_], r=args.r, g=args.g, weight=args.weight)
                               for _ in allsets])
        y_true = torch.hstack([rec[_]['y_true'] * int(_ == dset) - int(_ != dset)
                               for _ in allsets])

        tpr, sr, fpr = scrisk(y_true, y_est, scores)

        fpr95 = fpr[tpr >= 0.95].min()
        sr95 = sr[tpr >= 0.95].min()
        auroc = auc(fpr, tpr)
        auscodrt = auc(tpr, 0.5 * sr + 0.5 * fpr)
        ausrt = auc(tpr, sr)

        _s = f'FPR@95 = {fpr95:.1%} -- SR95 = {sr95:.1%} -- '
        _s += f'AuROC = {auroc: .1%} '
        _s += f'-- AuST = {ausrt: .1%}'
        _s += f'-- AuSCODT = {auscodrt: .1%}'

        print(_s)

        figures[j] = plt.figure('{}'.format(j))

        a = figures[j].gca()
        a.plot(tpr, sr)
        a.plot(tpr, fpr)
        a.set_title('{}'.format(j))

        figures[j].show()

    if sys.argv[0]:
        input()
