from utils.texify import TexTab
import sys
import os

import logging

import torch
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from utils.save_load import needed_remote_files
import numpy as np

default_scores = {}

default_scores['wim'] = {'r': 'zdist', 'g': 'elbo'}
default_scores['vib'] = {'r': 'odin-1-0.0040', 'g': 'none'}


PRINTSTAT = True
PRINTSTAT = False


def printstat(func):

    def wrapped(*a, **kw):

        scores = func(*a, **kw)

        score = kw.get('score', '--')
        T = kw.get('T', 1.)

        try:
            smin = scores.min()
            smax = scores.max()
            smean = scores.mean()
            sstd = scores.std()
            if PRINTSTAT:  # 'msp' in score and 'log' not in score:
                print('*** {}-{} [{:.2g}--{:.2g}] {:.2g}+/-{:.2g}'.format(score, T, smin, smax, smean, sstd))
        except AttributeError:
            pass

        return scores

    return wrapped


def grid_search_odin(in_rec, *out_rec, metrics='scod', tpr=0.95):

    assert metrics in ('fpr', 'sel', 'scod')

    params = set([_ for _ in in_rec if _.startswith('odin')])

    if metrics == 'fpr':
        for r in out_rec:
            params = params & set(r)

    logging.debug('ODIN Params: {}'.format(', '.join(params)))

    old_rate = 1.0

    y_est = in_rec['y_est_already']
    y_true = in_rec['y_true']

    for p in params:
        risk = 0.

        in_scores = in_rec[p]
        thr = in_scores.sort()[0][int((1 - tpr) * len(in_scores)) - 1]

        if metrics in ('fpr', 'scod'):
            out_scores = torch.hstack([rec[p] for rec in out_rec])
            risk += (out_scores >= thr).float().mean()

        if metrics in ('sel', 'scod'):

            out_scores = in_rec[p][y_est != y_true]
            risk += (out_scores >= thr).float().sum() / len(y_est) / tpr

        if risk < old_rate:
            best_p = p
            old_rate = risk

    logging.info('Best odin param: {} with {}@95={:.1%}'.format(best_p, metrics, risk))

    return best_p


def scrisk(y_true, y_est, r_scores, g_scores, weight=0.5, target_tpr=None):
    """srisk: computation of sc(od) risk

    -- y_true : class for indist , -1 for ood

    -- y_est : estimated class

    -- scores: the HIGHer the more likely to REJECT

    """

    assert not (r_scores is None and g_scores is None)

    if r_scores is None:
        r_scores = torch.zeros_like(g_scores)

    if g_scores is None:
        g_scores = torch.zeros_like(r_scores)

    scores = weight / (1 - weight) * g_scores + r_scores if weight < 1. else g_scores

    i_ = scores.argsort()

    i_in = y_true >= 0
    i_ok = y_true == y_est

    if target_tpr is not None:

        thr = scores[i_in].sort()[0][int(target_tpr * i_in.sum())]
        i_pos = scores <= thr

        fpr = ((~i_in) & i_pos).sum() / (~i_in).sum()
        tpr = (i_in & i_pos).sum() / i_in.sum()
        selective_risk = (~i_ok & i_in & i_pos).sum() / (i_in & i_pos).sum()

        # print('TPR = {:.1%} FPR = {:.1%} SR = {:.1%}'.format(tpr, fpr, selective_risk))

        if weight == 0.5:

            k_i_ = {'in': i_in, 'out': ~i_in, 'ok': i_in & i_ok, 'ko': ~i_ok & i_in}
            k_s_ = {'g': g_scores, 'r': r_scores}

            for k_s in k_s_:
                for k_i in k_i_:
                    _score = k_s_[k_s][k_i_[k_i]]
                    _q = _score.quantile(torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
                    __q = '--'.join(map('{: .2e}'.format, _q))
                    print('{}[{:3}]: [{}] {: .2e} +/-{:.1e}'.format(k_s, k_i, __q,  _score.mean(),
                                                                    _score.std()))

    tpr = i_in[i_].cumsum(0) / i_in.sum()
    fpr = (~i_in)[i_].cumsum(0) / (~i_in).sum()
    selective_risk = (i_in & (~i_ok))[i_].cumsum(0) / i_in[i_].cumsum(0)

    if target_tpr is not None:
        fpr = fpr[tpr >= target_tpr].min()
        selective_risk = selective_risk[tpr >= target_tpr].min()
        tpr = tpr[tpr >= target_tpr].min()

    return tpr, selective_risk, fpr


@printstat
def scoring_r(losses, score='msp', y_est=None, mtype='cvae', T=1.):
    """ estimation of 1 - max P(y|x)
    """

    # print('*** r', score, mtype)

    if score == 'default':
        score = default_scores[mtype]['r']  #

    logging.debug('r score is {}'.format(score))

    if score and score.startswith('odin'):

        return 1 - losses[score]

    if score is None:
        return None

    if y_est is None:
        y_est = losses['y_est_already']

    if score == 'predist':
        return 0.5 * losses['pre-zdist'].gather(0, y_est.unsqueeze(0)).squeeze() / T

    if score == 'dist':
        return 0.5 * losses['zdist'].gather(0, y_est.unsqueeze(0)).squeeze() / T

    if score == 'logmsp':
        return scoring_r(losses, score='dist', y_est=y_est, T=T) + (-0.5 * losses['zdist'] / T).logsumexp(0)

    if score == 'prelogmsp':
        return (scoring_r(losses, score='predist', y_est=y_est, T=T)
                + (-0.5 * losses['pre-zdist'] / T).logsumexp(0))

    if score == 'msp':
        return 1 - losses['logits'].softmax(dim=0).max(dim=0)[0]

    if 'msp-' in score:
        T = float(score.split('-')[1])
        score = score.split('-')[0].replace('msp', 'logmsp')
        # print('***', score, T)
        return 1 - (-scoring_r(losses, score=score, y_est=y_est, mtype=mtype, T=T)).exp()

    raise ValueError('{} is unknwon for r(x)'.format(score))


def scoring_g(losses, score='elbo', y_est=None, mtype='wim'):
    """ estimation of pOut/pIn
    """
    # print('*** g', score, mtype)

    if score == 'default':
        score = default_scores[mtype]['g']

    logging.debug('g score is {}'.format(score))

    if score is None:
        return None

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
    if score.startswith('elbo'):
        key = 'total'
        sign = 1

    if score == 'iws':
        sign = -1

    if score == 'g':
        sign = 1

    if 'wim' in mtype and score.endswith('~@'):
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
    from utils.texify import TexTab

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

        def texify(s):

            s_ = s.split('-')

            return '-'.join([config['texify'].get(_, _) for _ in s_])

        return jobs, options, texify

    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--weight', default=0.5, type=float)
    parser.add_argument('-v', action='count', default=0)
    parser.add_argument('-f', action='store_true')
    parser.add_argument('--tab', nargs='?', const='/dev/stdout', default='/dev/null')

    args = parser.parse_args()

    jobs, opt, texify = parse_config()

    """ ^^ CONFIG ^^
    """

    for j in jobs:

        if jobs[j]['mdict'] is None:
            logging.error('{} not found'.format(j))

    for j in jobs:
        print(j, jobs[j]['r'], jobs[j]['g'])

    print(opt['oodsets'])

    """ FETCH JOBS
    """
    mdirs = [jobs[_]['mdict']['dir'] for _ in jobs]

    with open('/tmp/files', 'w') as f:
        for d, s in needed_remote_files(*mdirs, missing_file_stream=f):
            logging.debug(s[-20:])

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

    col_headers = ['fpr', 'rs', 'auroc', 'aust', 'auscodt']
    cols = ['s3.1'] * len(col_headers)
    tex_tab = TexTab('l', *cols, float_format='{:.1f}')

    tab_row = 'header'

    tex_tab.append_cell('', row='header')
    for s in col_headers:
        tex_tab.append_cell(texify(s), multicol_format='c', row='header')

    for j in jobs:

        mdict = jobs[j]['mdict']
        dset = mdict['set']

        model = mdict['net']

        mtype = ''
        if type(model).__name__.lower().endswith('array'):
            mtype = type(model).__name__.lower()[:-5] + '-'

        rdir = model_subdir(mdict, 'samples', '{:04}'.format(mdict['done']))

        files = os.listdir(rdir)

        logging.debug('{}: {}'.format(rdir[-20:], ' - '.join(files)))
        rec = LossRecorder.loadall(rdir, device='cpu')

        allsets_ = list(rec)

        allsets = [dset] + oodsets

        mtype += mdict['type']

        if not set(allsets) <= set(allsets_):
            logging.error('Sets missing: {}'.format(' '.join(set(allsets) - set(allsets_))))
            continue

        print('\n{:=^70}{:>30}'.format(j, ' '.join(allsets)))

        for r, g in product(jobs[j]['r'], jobs[j]['g']):

            if r and r.lower() == 'none':
                r = None
            if g and g.lower() == 'none':
                g = None

            if not (r or g):
                continue

            tab_row = '{}-{}-{}'.format(j, r, g)
            tex_tab.append_cell(texify(tab_row), row=tab_row)

            print('\n{:_^100}'.format('r:{} g:{}'.format(r, g)))

            logging.info('Scores r:{} g:{}'.format(r, g))

            logging.info('Sets: {}'.format('/'.join(allsets)))

            for s in allsets_:
                if 'cvae' in mtype:
                    if 'pre-zdist' in rec[s]:
                        rec[s]._tensors['y_est_already'] = rec[s]['pre-zdist'].argmin(0)
                    else:
                        rec[s]._tensors['y_est_already'] = rec[s]['pre-zdist'].argmin(0)
                assert 'y_est_already' in rec[s]

            y_est = torch.hstack([rec[_]['y_est_already'] for _ in allsets])

            if r == 'odin':
                r = grid_search_odin(rec[dset], *[rec[_] for _ in oodsets], metrics='sel')

            logging.warning('r={}'.format(r))

            r_scores = None
            g_scores = None

            if r:
                r_scores = torch.hstack([scoring_r(rec[_], score=r, mtype=mtype)
                                         for _ in allsets])
            if g:
                g_scores = torch.hstack([scoring_g(rec[_], score=g, mtype=mtype)
                                         for _ in allsets])

            y_true = torch.hstack([rec[_]['y_true'] * int(_ == dset) - int(_ != dset)
                                   for _ in allsets])

            if g and r:
                min_scod_risk = 1.0
                for weight in np.linspace(0, 1, 21):
                    tpr, sr, fpr = scrisk(y_true, y_est, r_scores, g_scores, weight=weight, target_tpr=0.95)
                    print('gamma:{:.2f} fpr: {:.1%} sr: {:.1%}'.format(weight, fpr, sr))
                    scod_risk = 0.5 * fpr + 0.5 * sr
                    if scod_risk < min_scod_risk:
                        weight_opt = weight

                tpr, sr, fpr = scrisk(y_true, y_est, r_scores, g_scores, weight=weight_opt)
                fpr95 = fpr[tpr >= 0.95].min()
                sr95 = sr[tpr >= 0.95].min()

                auroc = 1 - auc(tpr, fpr)
                auscodrt = auc(tpr, 0.5 * sr + 0.5 * fpr)
                ausrt = auc(tpr, sr)

            _s = f'gamma={weight_opt:.2f}: '
            _s += f'FPR@95 = {fpr95:.1%} -- SR95 = {sr95:.1%} -- '
            _s += f'AuROC = {auroc: .1%} '
            _s += f'-- AuST = {ausrt: .1%}'
            _s += f'-- AuSCODT = {auscodrt: .1%}'

            print(_s)

            #  col_headers = ['FPR', 'Rs', 'AUROC', 'AuST', 'AuSCODT']

            for val in (fpr95, sr95, auroc, ausrt, auscodrt):
                tex_tab.append_cell(100 * val, row=tab_row)

            fig_name = '{} - {} - {}'.format(j, r, g)
            if args.f:
                figures[fig_name] = plt.figure(fig_name)

                a = figures[j].gca()
                a.set_xlabel('TPR')
                a.plot(tpr, sr, label='Rs')
                a.plot(tpr, fpr, label='FPR')
                a.set_title(fig_name)
                a.legend()

                figures[j].show()
    if args.tab:
        with open(args.tab, 'w') as f:
            tex_tab.render(f)

    if sys.argv[0] and args.f:
        input()
