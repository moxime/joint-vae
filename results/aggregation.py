import os
import sys
import argparse
from utils.save_load import load_json, needed_remote_files, develop_starred_methods, LossRecorder
from utils.parameters import parse_filters
import numpy as np
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters
import logging
from utils.parameters import gethostname
import pandas as pd
from utils.roc_curves import fpr_at_tpr
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import torch
from torch.nn.functional import one_hot
from scipy.stats import mode

parser = argparse.ArgumentParser()

parser.add_argument('--last', default=0, type=int)
parser.add_argument('--method', default='iws-2s')
parser.add_argument('-v', action='count', default=0)
parser.add_argument('--result-dir', default='/tmp')
parser.add_argument('--when', default='last')
parser.add_argument('--plot', nargs='?', const='p')
parser.add_argument('--tex', nargs='?', default=None, const='/tmp/r.tex')

rmodels = load_json('jobs', 'models-{}.json'.format(gethostname()))

col_width = 10
str_col_width = '13.13'
flt_col_width = '5.1f'

tpr = 0.95


np.seterr(divide='ignore', invalid='ignore')


def log_mean_exp(*tensors, normalize=False):

    t = torch.cat([_.unsqueeze(0) for _ in tensors])

    tref = t.max(axis=0)[0]
    dt = t - tref

    return (dt.exp().mean(axis=0).log() + tref).squeeze(0)


def counting_values(a, axis=-1, classes=None):

    if not classes:
        classes = y_.max()

    return sum(one_hot(torch.index_select(y_, axis, torch.tensor([i])).squeeze()) for i in range(a.shape[axis]))


def likelihood_of_classification(counted_values, accuracy, axis=-1):

    N = max(counted_values.sum(axis=axis)).item()
    C = counted_values.shape[axis]

    return 1 / C * (accuracy / (1 - accuracy) * (C - 1)) ** counted_values * ((1 - accuracy) / (C - 1)) ** N

    
if __name__ == '__main__':

    args_from_file = ('--dataset cifar10 '
                      '--type cvae '
                      # '--gamma 500 '
                      '--sigma-train coded '
                      '--coder-dict learned '
                      # '--last 1 '
                      '-vv '
                      '--tex '
                      '--method iws-a-4-1 '
                      '--job-num 192000.. '
                      '--when min-loss '
                      ).split()

    # args_from_file = '--job-num 192000... --when min-loss'.split()
    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)
    wanted = args.when

    max_per_rep = 4
    
    logging.getLogger().setLevel(40 - 10 * args.v)

    filter_parser = parse_filters()
    filter_args = filter_parser.parse_args(ra)
    
    filters = DictOfListsOfParamFilters()

    for _ in filter_args.__dict__:
        filters.add(_, filter_args.__dict__[_])

    mdirs = [_ for _ in rmodels if filters.filter(rmodels[_])][-args.last:]

    total_models = len(mdirs)
    logging.info('{} models found'.format(total_models))
    removed = False
    with open('/tmp/files', 'w') as f:

        for mdir, sdir in needed_remote_files(*mdirs, epoch=wanted, which_rec='all', state=False):
            logging.debug('{} for {}'.format(sdir[-30:], wanted))
            if mdir in mdirs:
                mdirs.remove(mdir)
                removed = True
                logging.info('{} is removed (files not found)'.format(mdir.split('/')[-1]))
            f.write(sdir + '\n')

    # logging.info((len(mdirs), 'complete model' + ('s' if len(mdirs) > 1 else ''), 'over', total_models))
    
    if not mdirs or removed:
        logging.warning('Exiting, load files')
        logging.warning('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')

    key_loss = args.method.split('-')[0]

    p_y_x = {}

    zdist = {}
    iws = {}
    
    normalize = False

    kept_testset = None
    thresholds = {}

    n_by_rep = dict(hsv=0, rgb=0)

    oods = {}
    ood_rates = {}
    classif = {}
    classif_rates = {}
    agreement = {}
    likelihood = {}

    for mdir in mdirs:

        model = M.load(mdir, load_net=False)
        rep = model.architecture['representation']
        name = rep.upper() + str(n_by_rep[rep])

        n_by_rep[rep] += 1
        testset = model.training_parameters['set']
        if kept_testset and testset != kept_testset:
            continue
        else:
            kept_testset = testset

        if n_by_rep[rep] > max_per_rep:
            continue

        if args.when == 'min-loss':
            epoch = model.training_parameters.get('early-min-loss', 'last')

        if args.when == 'last' or epoch == 'last':
            epoch = max(model.testing)

        recorders = LossRecorder.loadall(os.path.join(mdir, 'samples', '{:04d}'.format(epoch)), device='cpu')

        y_true = recorders[kept_testset]._tensors['y_true']
        
        sets = [*recorders.keys()]

        # exclude rotated set
        oodsets = [_ for _ in sets if not _.startswith(kept_testset)]
        # sets = [kept_testset, 'lsunr']  # + sets
        sets = [kept_testset] + oodsets
        
        iws[name] = {}
        zdist[name] = {}
        
        oods[name] = {}

        for dset in sets:
            t = recorders[dset]._tensors['iws']
            iws[name][dset] = t
            zdist[name][dset] = recorders[dset]._tensors['zdist']
            
    combos = []
    l_combos = [1, 2, 3, 4, 5]
    l_combos = [1]
    l_combos = [1, 7]
    l_combos = [1, 3, 5, 7]

    df_idx = ['ag', 'T', 'l', 'name']
    df = pd.DataFrame(columns=df_idx + sets + ['mis'])
    df.set_index(df_idx, inplace=True)
    df.columns.name = 'set'
    
    for _ in l_combos:
        combos += [*itertools.combinations(sorted(oods), _)]

    logging.info('Will work on {} combos'.format(len(combos)))
    
    i_ax = 0

    results_computed = '+,'

    corrs = {}

    accuracies = {}

    nan_temp = -1

    temps = [1, 5]
    temps_ = [nan_temp] + temps
    
    for combo in combos:

        logging.info('Working on {}'.format('--'.join(combo)))

        if '&' in results_computed:
            combo_name = '&'.join(combo)

            df.loc[('&', len(combo), combo_name)] = {}
            
            if len(combo) == 1:
                classif[combo_name] = {s: torch.argmax(iws[combo_name][s], dim=0).unsqueeze(-1)
                                       for s in sets}
                acc = (classif[combo_name][kept_testset].squeeze() == y_true).float().mean().item()
                accuracies[combo_name] = acc
                
            else:

                for s in sets:
                    y_ = torch.cat([classif[_][s] for _ in combo], dim=-1)
                    n = y_.shape[0]

                    for d in combo_name, classif, agreement, likelihood:
                        if combo_name not in d:
                            d[combo_name] = {}

                    count = counting_values(y_)
                    c, v = count.max(axis=-1)

                    k = (count == c.unsqueeze(-1)).sum(-1)
                    
                    C = count.shape[-1]

                    acc = np.mean([accuracies[_] for _ in combo])
                    likelihood[combo_name][s] = likelihood_of_classification(count, acc) * C ** len(combo)
                    
                    agreement[combo_name][s] = c
                    classif[combo_name][s] = - torch.ones((n, len(combo) + 1), dtype=int)
                    for m in range(0, len(combo) + 1):
                        i = c == m
                        classif[combo_name][s][i, m] = v[i]
        # # ax.set_title(combo_name)

        # if combo_name not in oods:
        #     oods[combo_name] = {}

        # for s in sets:
            
        #     oods[combo_name][s] = sum(oods[_][s] for _ in combo)

        #     n_oods = {}
        #     for i, _ in enumerate(combo):
        #         n_oods[i] = (oods[combo_name][s] > i).to(float).mean()
        #     _s = ' | '.join('>= {}: {:6.2%}'.format(i + 1, p) for i, p in n_oods.items())

        #     if '&' in print_results:
        #         print('{:20} {:8} {}'.format(combo_name, s, _s))

        # df = pd.DataFrame(oods[combo_name])    
        # # sns.histplot(df, multiple='dodge', ax=ax)  #

        # if len(combo) == 2:
        #     combo_name = 'x'.join(combo)
        #     corrs[combo_name] = {}
        #     for s in sets:
        #         t = torch.cat([oods[_][s].float().unsqueeze(0) for _ in combo])
        #         rho = t.corrcoef()[0, 1]

        #         corrs[combo_name][s] = rho
        #         if 'x' in print_results:
        #             print('{:20} {:8} {:6.2%}'.format(combo_name, s, rho))

        w_ = [_ for _ in results_computed if _ not in '&x']

        for w in w_:

            combo_name = w.join(combo)

            for t in temps_:
                df.loc[(w, t, len(combo), combo_name)] = {}

            if combo_name not in iws and w == '+':
                iws[combo_name] = {}
            if combo_name not in iws and w == ',':
                zdist[combo_name] = {}
                
            for s in sets:
                if w == '+':
                    t = log_mean_exp(*(iws[_][s] for _ in combo), normalize=normalize)
                    iws[combo_name][s] = t
                elif w == ',':
                    t = torch.stack([zdist[_][s] for _ in combo]).mean(0)
                    zdist[combo_name][s] = t

            if w == ',':
                z = zdist[combo_name][kept_testset]
                p_y_x[combo_name] = {t: (-z / 2 / t).softmax(0) for t in temps}
                p_y_x[combo_name][nan_temp] = (-z / 2)

            elif w == '+':
                if len(combo) == 1:
                    p_y_x[combo_name] = {t: (iws[combo_name][kept_testset] / t).softmax(axis=0)
                                         for t in temps}
                else:
                    p_y_x[combo_name] = {t: torch.stack([p_y_x[_][t] for _ in combo]).mean(0) for t in temps}
                    
            classif[combo_name] = torch.argmax(p_y_x[combo_name][temps[0]], dim=0)

            acc = (classif[combo_name] == y_true).to(float).mean().item()
            classif_rates[combo_name] = acc
            df.loc[(w, nan_temp, len(combo), combo_name)][kept_testset] = acc

            oods[combo_name] = {}
            ood_rates[combo_name] = {}

            if w == '+':
                t_in = iws[combo_name][kept_testset].max(0)[0]
            else:
                t_in = (-zdist[combo_name][kept_testset]).max(0)[0]
            n = len(t_in)
            i1, i2 = int(n * 0.04), int(n * 0.99)
            thr = sorted(t_in)[i1], sorted(t_in)[i2]
            thresholds[combo_name] = thr

            for s in (oodsets if w == '+' else []):
                if w == '+':
                    t = iws[combo_name][s].max(0)[0]
                else:
                    t = (-zdist[combo_name][s]).max(0)[0]
                oods[combo_name][s] = (t < thr[0]) + (t >= thr[1])
                tnr = oods[combo_name][s].to(float).mean().item()
                ood_rates[combo_name][s] = tnr
                df.loc[(w, nan_temp, len(combo), combo_name)][s] = tnr

            i_true = classif[combo_name] == y_true
                                               
            p_y_max = {_: p_y_x[combo_name][_].max(0)[0] for _ in p_y_x[combo_name]}

            for t in p_y_max:
                                               
                thr = sorted(p_y_max[t][i_true])[-int(sum(i_true) * tpr)]
                tnr = (p_y_max[t][~i_true] < thr).float().mean().item()

                df.loc[(w, t, len(combo), combo_name)]['mis'] = tnr

    if '&' in results_computed:
        for s in sets:
            print('***', s, '***')
            for _ in classif:
                voting = classif[_][s].shape[-1]
                if s == kept_testset:
                    _s = ' '.join(['{:6.2%} {:5.2%}'] * (voting - 1))
                    _p = sum([[(classif[_][s][:, v] == y_true).to(float).mean(),
                               (classif[_][s][:, v] == -1).to(float).mean()] for v in range(1, voting)], [])
                else:
                    _s = ' '.join(['{:6.2%}'] * (voting - 1))
                    _p = [(classif[_][s][:, v] == -1).to(float).mean() for v in range(1, voting)]
                    
                print(('{:25} ' + _s).format(_, *_p))
            
    df_mean = df.groupby(['ag', 'T', 'l']).agg('mean').stack().unstack('l')
    df_mean.rename({nan_temp: ''}, inplace=True)
    
    def float_format(u):
        return '{:.1f}'.format(100 * u)

    print(df_mean.to_string(float_format=float_format, index_names=False))

    if args.tex:

        f, ext = os.path.splitext(args.tex)
        names = dict(cifar10=r'\cifar')
        df_mean.columns.names = ['$N$']
        df_mean.rename({_: names.get(_, '\\' + _) for _ in sets}, inplace=True)

        column_format = 'l' * df_mean.index.nlevels + '%\n'
        column_format += 'S[table-format=2.{}]%\n'.format(1) * len(df_mean.columns)

        df_mean.to_latex(buf=args.tex,
                         float_format=float_format,
                         index_names=True,
                         column_format=column_format,
                         escape=False)

        for w in w_:
            d = df_mean.loc[w]
            column_format = 'l' * d.index.nlevels + '%\n'
            column_format += 'S[table-format=2.{}]%\n'.format(1) * len(d.columns)
            d.to_latex(buf=f + '-' + w + ext,
                       float_format=float_format,
                       column_format=column_format,
                       index_names=True,
                       escape=False)
        
    col_w = 10
    col_sep = '&' if args.tex else ' '
    row_sep = r'\\' if args.tex else ''

    if args.tex and '&' in results_computed:
        combo = combos[-1]
        combo_name = '&'.join(combo)
        N = len(combo)
        print('$n$     ', col_sep, col_sep.join('{:{}} '.format(l, col_w) for l in range(1, N+1)), end=' ')
        print(row_sep)
        
        for s in sets:    
            print('{:{}} '.format(names.get(s, name_prefix + s), col_w), end=col_sep)
            n = [(classif[combo_name][s][:,_] >= 0).float().mean() for _ in range(1, N+1)]
            print(col_sep.join('{:{}.1f} '.format(100 * _, col_w) for _ in n), end='')
            print(row_sep)
        s = kept_testset
        v, c = classif[combo_name][s].max(-1)
        i_true = v == y_true
        i = [*range(len(y_true))]
        i_false = ~i_true
        for w, idx in zip(('Total', 'Y', 'N'), (i, i_true, i_false)):
            print('{:{}} '.format(w, col_w), end=col_sep)
            n = [(classif[combo_name][s][idx,_] >= 0).float().mean() for _ in range(1, N+1)]
            print(col_sep.join('{:{}.1f} '.format(100 * _, col_w) for _ in n), end='')
            print(row_sep)

        print('Accuracy: \\SI{{{:.1}}}\\%'.format(acc))
            
    # if args.plot:
    #     plt.close('all')
    #     logging.getLogger().setLevel(logging.WARNING)

    #     for _ in losses_by_set:
    #         p = sns.pairplot(pd.DataFrame(losses_by_set[_]))
    #         p.fig.suptitle(_, size=48)
    #         p.fig.show()

    #     for j in iws:

    #         df = pd.DataFrame(iws[j])
    #         df.drop(columns=kept_testset + '90', inplace=True)
    #         df.columns = df.columns.rename('set')
    #         df_ = df.stack().reset_index()

    #         f, a = plt.subplots(1)
    #         sns.histplot(df_, x=0, hue="set", element="step", ax=a)
    #         f.suptitle(j, size=48)
    #         f.show()

    #     print('input')
    #     input()
    #     print('douune')
