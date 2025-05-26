import sys
import os

import logging

import torch
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from utils.save_load import needed_remote_files

default_scores = {}

default_scores['wim'] = {'r': 'zdist', 'g': 'elbo'}
default_scores['vib'] = {'r': 'odin-1-0.0040', 'g': 'none'}


def scrisk(y_true, y_est, r_scores, g_scores, weight=1):
    """srisk: computation of sc(od) risk

    -- y_true : class for indist , -1 for ood

    -- y_est : estimated class

    -- scores: the HIGHer the more likely to REJECT

    """

    if g_scores is None:
        scores = r_scores
    else:
        scores = r_scores + weight * g_scores

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

    selective_risk[selective_risk.isnan()] = 0.

    thr = {_: scores[i_][(tpr > _ / 100) & (y_ >= 0)].min() for _ in (0, 1, 95, 99)}

    thr[0] = 0.

    for _ in thr:
        tpr_at_thr = ((scores <= thr[_]) & (y_true >= 0)).sum() / (y_true >= 0).sum()
        fpr_at_thr = ((scores <= thr[_]) & (y_true < 0)).sum() / (y_true < 0).sum()
        logging.info('thr@{:2} = {:+.1e} ({:5.1%}/{:5.1%})'.format(_, thr[_],
                                                                   tpr_at_thr, fpr_at_thr))

    return tpr, selective_risk, fpr


def scoring(losses, r=None, g=None, weight=1, **kw):
    """
    reject if high
    """

    assert r is not None or g is not None

    try:
        y_est = losses['y_est_already']

        score = scoring_r(losses, score=r, y_est=y_est, **kw)
        score += weight * scoring_g(losses, score=g, y_est=y_est, **kw)
        return score
    except KeyError as e:
        logging.error('Possibles keys')
        logging.error(' -- '.join(losses))
        raise e


def scoring_r(losses, score='msp', y_est=None, mtype='cvae'):
    """ estimation of 1 - max P(y|x)
    """
    if score == 'default':
        score = default_scores[mtype]['r']  #

    logging.debug('r score is {}'.format(score))

    if score and score.startswith('odin'):

        # print('***', losses[score].shape, losses[score].mean())
        return 1 - losses[score]

    if score is None:
        return 0.

    if y_est is None:
        y_est = losses['y_est_already']

    if score == 'predist':
        return 0.5 * losses['pre-zdist'].gather(0, y_est.unsqueeze(0)).squeeze()

    if score == 'zdist':
        return 0.5 * losses['zdist'].gather(0, y_est.unsqueeze(0)).squeeze()

    if score == 'logmsp':
        return scoring_r(losses, 'dist', y_est=y_est) - (-0.5 * losses['zdist']).logsumexp(0)

    if score == 'msp':
        return losses['logits'].softmax(dim=0).max(dim=0)[0]

    raise ValueError('{} is unknwon for r(x)'.format(score))


def scoring_g(losses, score='elbo', y_est=None, mtype='wim'):
    """ estimation of pOut/pIn
    """

    if score == 'default':
        score = default_scores[mtype]['g']

    logging.debug('g score is {}'.format(score))

    if score is None:
        return 0.

    def scoring_alt(k):

        s = losses[k + '@']
        if s.ndim == 2:
            return s[0, :]
        return s

    def scoring_in(k):

        return losses[k].gather(0, y_est.unsqueeze(0).long()).squeeze()

    if y_est is None:
        y_est = losses['y_est_already']

    key = score

    """ sign has to be >0 if the score is higher for ood
    """
    sign = 1
    if score == 'elbo':
        key = 'total'
        sign = 1

    if score == 'iws':
        sign = -1

    if score == 'g':
        sign = 1

    if 'wim' in mtype and key.endswith('~@'):
        # score has to be high for ood
        return sign * (scoring_in(key) - scoring_alt(key))

    if 'wim' in mtype:

        return sign * scoring_in(key)

    return sign * losses[key]

    for _ in losses:
        logging.error('{:20} : {}'.format(_, 'x'.join(map(str, losses[_].shape))))
    raise ValueError('{} is unknwon for g(x). Losses key avaiblable:\n{}'.format(score, ' '.join(losses)))


if __name__ == '__main__':

    import sys
    import os
    import argparse
    from utils.save_load import find_by_job_number, model_subdir, SampleRecorder, LossRecorder
    import configparser
    from itertools import product

    plt.set_loglevel(level='warning')

    """ CONfIG AND ARGS
    """
    def parse_config(config_file='ft/results/scod.ini'):
        config = configparser.ConfigParser()
        config.read(config_file)

        jobs_by_dir = {}
        for model in config['jobs']:
            job_ = config.get('jobs', model)
            jobs_by_dir[model] = dict(j=int(job_.split()[0]), dir=job_.split()[1])
        jobs = {_: {'mdict': find_by_job_number(d['j'], job_dir=d['dir'], build_module=True, load_state=False)}
                for _, d in jobs_by_dir.items()}

        scores = dict(r=config['r-scores'], g=config['g-scores'])

        for j in jobs:

            jobs[j].update({_: scores[_].get(j).split() or [None] for _ in scores})

        options = dict(config['options'])

        return jobs, options

    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--weight', default=1., type=float)
    parser.add_argument('-v', action='count', default=0)

    args = parser.parse_args()

    jobs, opt = parse_config()

    """ ^^ CONFIG ^^
    """

    for j in jobs:
        print(j, jobs[j]['r'], jobs[j]['g'])

    print(opt['oodsets'])

    """ FETCH JOBS
    """
    mdirs = [jobs[_]['mdict']['dir'] for _ in jobs]

    with open('/tmp/files', 'w') as f:
        for d, _ in needed_remote_files(*mdirs, missing_file_stream=f):
            logging.debug(_[-20:])

    # for _ in jobs:
    #     print(_, type(jobs[_]['mdict']['net']).__name__)

    logging.getLogger().setLevel(logging.ERROR - 10 * args.v)

    """ ^^ FETCH ^^
    """

    logging.info('Starting with {} models'.format(len(jobs)))
    logging.debug('...')

    figures = {}

    oodsets = opt.get('oodsets', '').split()

    logging.info('OOD: {}'.format(' '.join(oodsets)))

    for j in jobs:

        print('\n{:=^80}'.format(j))

        for r, g in product(jobs[j]['r'], jobs[j]['g']):

            if not (r or g):
                continue

            print('\n{:=^80}'.format('r:{} g:{}'.format(r, g)))

            logging.info('Scores r:{} g:{}'.format(r, g))

            mdict = jobs[j]['mdict']
            dset = mdict['set']

            model = mdict['net']

            mtype = ''
            if type(model).__name__.lower().endswith('array'):
                mtype = type(model).__name__.lower()[:-5] + '-'

            mtype += mdict['type']

            rdir = model_subdir(mdict, 'samples', '{:04}'.format(mdict['done']))

            files = os.listdir(rdir)

            logging.debug('{}: {}'.format(rdir[-20:], ' - '.join(files)))
            rec = LossRecorder.loadall(rdir)

            allsets_ = list(rec)

            allsets = [dset] + oodsets

            if not set(allsets) <= set(allsets_):
                logging.error('Sets missing: {}'.format(' '.join(set(allsets) - set(allsets_))))
                continue

            logging.info('Sets: {}'.format('/'.join(allsets)))

            for s in allsets_:
                if 'y_est_already' not in rec[s]:
                    rec[s]._tensors['y_est_already'] = rec[s]['cross_y'].argmin(0)

            y_est = torch.hstack([rec[_]['y_est_already'] for _ in allsets])
            r_scores = torch.hstack([scoring_r(rec[_], score=r, mtype=mtype)
                                     for _ in allsets])
            if g:
                g_scores = torch.hstack([scoring_g(rec[_], score=g, mtype=mtype)
                                         for _ in allsets])
            else:
                g_scores = None
            y_true = torch.hstack([rec[_]['y_true'] * int(_ == dset) - int(_ != dset)
                                   for _ in allsets])

            tpr, sr, fpr = scrisk(y_true, y_est, r_scores, g_scores)

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
