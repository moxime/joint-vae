from utils.texify import TexTab
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


def scrisk(y_true, y_est, r_scores, g_scores, weight=0.5, n_tpr=1000):
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

    n_samples = len(y_true)

    checked_tpr = torch.linspace(1 / n_tpr, 1, n_tpr)

    if n_tpr == 1:
        checked_tpr = torch.tensor([0.95])

    # g_ shape: n_samples x n_tpr
    g_ = g_scores.unsqueeze(-1) * checked_tpr.unsqueeze(0)

    scores = weight * g_ + (1 - weight) * r_scores.unsqueeze(-1)

    in_samples = (y_true >= 0).sum()
    out_samples = len(y_true) - in_samples

    thresholds = torch.zeros(n_tpr)

    for i in range(n_tpr):

        i_thr = max(-int(-(checked_tpr[i] * in_samples - 1)), 0)
        thresholds[i] = scores[y_true >= 0, i].sort()[0][i_thr]

    # i_pos n_samples x n_tpr
    # pos : s(x) <= threshold
    i_pos = scores <= thresholds.unsqueeze(0)

    i_true_pos = (i_pos & (y_true >= 0).unsqueeze(-1))

    i_false_pos = (i_pos & (y_true < 0).unsqueeze(-1))

    classif_errors = i_pos & ((y_true >= 0) & (y_true != y_est)).unsqueeze(-1)

    tpr = i_true_pos.sum(0) / in_samples
    fpr = i_false_pos.sum(0) / out_samples
    selective_risk = classif_errors.sum(0) / tpr / in_samples

    return tpr, selective_risk, fpr


def scoring_r(losses, score='msp', y_est=None, mtype='cvae'):
    """ estimation of 1 - max P(y|x)
    """
    if score == 'default':
        score = default_scores[mtype]['r']  #

    logging.debug('r score is {}'.format(score))

    # for _ in losses:
    #     if _.startswith('odin'):
    #         print('{} {:.2f}--{:.2f}'.format(_, losses[_].min(), losses[_].max()), *losses[_].shape)

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
        return 1 - losses['logits'].softmax(dim=0).max(dim=0)[0]

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
    parser.add_argument('--tab', nargs='?', const='/dev/stdout')

    args = parser.parse_args()

    jobs, opt, texify = parse_config()

    """ ^^ CONFIG ^^
    """

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

        print('\n{:=^80}'.format(j))

        for r, g in product(jobs[j]['r'], jobs[j]['g']):

            if not (r or g):
                continue

            tab_row = '{}-{}-{}'.format(j, r, g)
            tex_tab.append_cell(texify(tab_row), row=tab_row)

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

            # for s in allsets:
            #     print('{:=^20}'.format(s))
            #     print('\n'.join([_ for _ in rec[s] if _.startswith('pre-')]))

            y_est = torch.hstack([rec[_]['y_est_already'] for _ in allsets])

            if r == 'odin':

                r = grid_search_odin(rec[dset], *[rec[_] for _ in oodsets], metrics='sel')

            logging.warning('r={}'.format(r))
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
            auroc = 1 - auc(tpr, fpr)
            auscodrt = auc(tpr, 0.5 * sr + 0.5 * fpr)
            ausrt = auc(tpr, sr)

            _s = f'FPR@95 = {fpr95:.1%} -- SR95 = {sr95:.1%} -- '
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

    with open(args.tab, 'w') as f:
        tex_tab.render(f)

    if sys.argv[0] and args.f:
        input()
