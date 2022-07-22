import os
import sys
import logging
import argparse
import numpy as np
from utils.save_load import load_json, needed_remote_files, LossRecorder
from utils.parameters import parse_filters
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters
from utils.texify import tex_command, tabular_rule, tabular_env
from utils.parameters import gethostname
import pandas as pd
from utils.roc_curves import fpr_at_tpr
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import torch
from torch.nn.functional import one_hot
from module.aggregation import log_mean_exp, joint_posterior, mean_posterior, voting_posterior

agg_types = {'and': '&', 'joint': ',', 'sum': '+'}

parser = argparse.ArgumentParser()


parser.add_argument('--last', default=0, type=int)
parser.add_argument('--method', default='iws-2s')
parser.add_argument('--tpr', type=float, default=0.95)
parser.add_argument('-v', action='count', default=0)
parser.add_argument('--result-dir', default='/tmp')
parser.add_argument('--agg-type', nargs='*', choices=list(agg_types))
parser.add_argument('--when', default='last')
parser.add_argument('--plot', nargs='?', const='p')
parser.add_argument('--tex', action='store_true')
parser.add_argument('--sets-to-exclude', nargs='*', default=[])
parser.add_argument('--combos', nargs='+', type=int)

rmodels = load_json('jobs', 'models-{}.json'.format(gethostname()))

col_width = 10
str_col_width = '13.13'
flt_col_width = '5.1f'



np.seterr(divide='ignore', invalid='ignore')


if __name__ == '__main__':

    args_from_file = ('--dataset cifar10 '
                      '--type cvae '
                      '--gamma 1000 '
                      '--features vgg19 ' 
                      '--representation rgb '
                      '--sigma-train coded '
                      '--coder-dict learned '
                      # '--last 1 '
                      '-vv '
                      '--tex '
                      '--method iws-a-4-1 '
                      '--job-num 190000.. '
                      # '--job-num 140000...144000 '
                      '--when min-loss '
                      '--sets-to-exclude cifar100 '
                      '--agg-type sum joint '
                      '--combos 3 5 '
                      ).split()

    # args_from_file = ('-vvvv '
    #                   '--job-num 193080 193082 '
    #                   # '--job-num 169381 '
    #                   '--when min-loss '
    #                   ).split()

    # args_from_file = '--job-num 192000... --when min-loss'.split()

    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)
    wanted = args.when

    args.agg_type.insert(0, 'and')
    tpr = args.tpr
    
    max_per_rep = 100
    
    logging.getLogger().setLevel(40 - 10 * args.v)

    filter_parser = parse_filters()
    filter_args = filter_parser.parse_args(ra)
    
    filters = DictOfListsOfParamFilters()

    for _ in filter_args.__dict__:
        filters.add(_, filter_args.__dict__[_])

    filter_str = '--'.join(f'{d}:{f}' for d, f in filters.items() if not f.always_true)

    mdirs = [_ for _ in rmodels if filters.filter(rmodels[_])][-args.last:]

    total_models = len(mdirs)
    logging.info('{} models found'.format(total_models))
    removed = False
    which_rec = '-'.join(['all'] + args.sets_to_exclude)
    with open('/tmp/files', 'w') as f:

        for mdir, sdir in needed_remote_files(*mdirs, epoch=wanted, which_rec=which_rec, state=False):
            logging.debug('{} for {}'.format(sdir[-30:], wanted))
            if mdir in mdirs:
                mdirs.remove(mdir)
                removed = True
                logging.info('{} is removed (files not found)'.format(mdir.split('/')[-1]))
            f.write(sdir + '\n')

    # logging.info((len(mdirs), 'complete model' + ('s' if len(mdirs) > 1 else ''), 'over', total_models))
    
    if not mdirs:
        logging.error('Exiting, load files')
        logging.error('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')
        logging.error(' Or: %s', '$ . /tmp/rsync-files remote:dir/joint-vae')
        with open('/tmp/rsync-files', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('rsync -avP --files-from=/tmp/files $1 .\n')
        raise ValueError

    key_loss = args.method.split('-')[0]

    p_y_x = {}

    t = {_: {} for _ in ('iws', 'zdist')}
    
    kept_testset = None
    ind_thr = {}

    n_by_rep = dict(hsv=0, rgb=0)
    
    as_ind = {}
    ind_pr = {}
    y_classif = {}
    classif_acc = {}
    as_correct = {}
    correct_pr = {}
    agreement = {}
    distribution = {}

    for mdir in mdirs:

        model = M.load(mdir, load_net=False)
        rep = model.architecture['representation']
        name = rep.upper() + str(n_by_rep[rep])
        name = str(model.job_number)
        
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
        oodsets = [_ for _ in sets if (not _.startswith(kept_testset) and _ not in args.sets_to_exclude)]
        # sets = [kept_testset, 'lsunr']  # + sets
        sets = [kept_testset] + oodsets
        
        t['iws'][name] = {}
        t['zdist'][name] = {}
        
        as_ind[name] = {}

        for _ in ('zdist', 'iws'):
            for dset in sets:
                t[_][name][dset] = recorders[dset]._tensors[_]
                t[_][name][dset] = recorders[dset]._tensors[_]
            
    l_combos = sorted(args.combos)
    if 1 not in l_combos:
        l_combos.insert(0, 1)

    all_sets = [kept_testset, 'correct' , 'incorrect', *oodsets]

    combos = []
    for _ in l_combos:
        combos += [*itertools.combinations(sorted(as_ind), _)]

    logging.info('Will work on {} combos'.format(len(combos)))
    
    i_ax = 0

    """
    + : work with p(x|y) = sum p_k(x|y)
    , : work with z = [z1,...,zn]
    . : work wiht p(x|y) = prod p_k(yx|y)
    """
    results_computed = args.agg_type

    result_dir = args.result_dir
    
    nan_temp = None

    temps_ = {_: [nan_temp, 1, 5] for _ in results_computed}
    temps_['and'] = [None]

    p_y_x = {}
    log_p_x_y = {}
    ind_thr = {}
    correct_thr = {}
    accuracies = {}
    
    for combo in combos:

        logging.info('Working on {}'.format('--'.join(combo)))

        for _ in ('iws', 'zdist'):
            t[_][combo] = {s: [t[_][m] for m in combo] for s in sets}

        if len(combo) == 1:
            combo_name = combo[0]
            y_classif[combo_name] = {s: torch.argmax(t['iws'][combo_name][kept_testset], dim=0)
                                     for s in sets}

        for w in (results_computed if len(combo) >= 1 else ['and']):

            temps = temps_[w]
            combo_name = agg_types[w].join(combo)
            logging.info(w)
            
            if w == 'joint':
                p_y_x[combo_name] = {s: joint_posterior(*[t['zdist'][_][s] for _ in combo], temps=temps)
                                     for s in sets}
            elif w == 'sum':
                p_y_x[combo_name] = {s: mean_posterior(*[t['iws'][_][s] for _ in combo], temps=temps)
                                     for s in sets}
            elif w == 'and':
                p_y_x[combo_name] = {s: voting_posterior(*[y_classif[_][s] for _ in combo], temps=temps)
                                     for s in sets}

            y_classif[combo_name] = {s: p_y_x[combo_name][s][temps[0]].argmax(0) for s in sets}
            
            acc = (y_classif[combo_name][kept_testset].squeeze() == y_true).float().mean().item()
            i_true = (y_classif[combo_name][kept_testset] == y_true).squeeze()
            accuracies[combo_name] = acc

            for s, i_ in zip(('correct', 'incorrect'), (i_true, ~i_true)):
                p_y_x[combo_name][s] = {t: p_y_x[combo_name][kept_testset][t][:, i_] for t in temps} 

            # df.loc[i_df] = acc

            if w != 'and' or True:
                i0 = int(i_true.sum() * tpr)

                max_py = {s: {t: p_y_x[combo_name][s][t].max(0)[0] for t in temps} for s in all_sets}
                correct_thr[combo_name] = {t: sorted(max_py['correct'][t])[-i0] for t in temps}

                as_correct[combo_name] = {s: {t: max_py[s][t] >= correct_thr[combo_name][t] for t in temps}
                                          for s in all_sets}

                correct_pr[combo_name] = {s: {t: as_correct[combo_name][s][t].float().mean().item() for t in temps}
                                      for s in all_sets}
                                
            if w == 'sum' or len(combo) == 1:

                tprs = {tpr: tpr}
                if len(combo) > 1:
                    and_combo = combo_name.replace(agg_types[w], agg_types['and'])
                    tprs['and'] = ind_pr[and_combo]['and'][kept_testset]

                log_p_x_y[combo_name] = {s: log_mean_exp(*[t['iws'][_][s] for _ in combo]).max(0)[0] for s in sets}
                t_in = log_p_x_y[combo_name][kept_testset]
                n = len(t_in)

                ind_thr[combo_name] = {}
                ind_pr[combo_name] = {}
                as_ind[combo_name] = {}
                for r in tprs:
                    tpr_l = 4/5 * (1 - tprs[r])
                    tpr_r = 1 - 1/5 * (1 - tprs[r])
                    i1, i2 = int(n * tpr_l), int(n * tpr_r)
                    # print('*** {:30} {} {} {} {:.3%}--{:.3%}'.format(combo_name, r, i1, i2, tpr_l, tpr_r))
                    thr = sorted(t_in)[i1], sorted(t_in)[i2]
                    ind_thr[combo_name][r] = thr
                    as_ind[combo_name][r] = {}
                    ind_pr[combo_name][r] = {}
                
                    for s in all_sets:
                        if s.endswith('correct'):
                            s_ = kept_testset
                            i_ = i_true ^ (s != 'correct')
                        else:
                            s_ = s
                            i_ = torch.ones(len(log_p_x_y[combo_name][s]), dtype=bool)

                        t_ = log_p_x_y[combo_name][s_][i_]
                        as_ind[combo_name][r][s] = (t_ > thr[0]) * (t_ <= thr[1])
                        ind_pr[combo_name][r][s] = as_ind[combo_name][r][s].float().mean().item()

            elif w == 'and':

                as_ind[combo_name] = {}
                ind_pr[combo_name] = {}
                count_as_oods = {s: sum([one_hot(as_ind[_][tpr][s].long(), num_classes=2) for _ in combo])
                                 for s in sets}
                for s, i_ in zip(('correct', 'incorrect'), (i_true, ~i_true)):
                    count_as_oods[s] = count_as_oods[kept_testset][i_]

                as_ind[combo_name]['and'] = {s: (count_as_oods[s].diff() >= 0).squeeze() for s in all_sets}
                ind_pr[combo_name]['and'] = {s: as_ind[combo_name]['and'][s].float().mean().item()
                                             for s in all_sets}

                agreement[combo_name] = {s: (p_y_x[combo_name][s][temps[0]] > 0).sum(0)
                                         for s in all_sets}

                distribution[combo_name] = {s: {} for s in all_sets}
                k_dist = [('=', _) for _ in range(1, len(combo) + 1)]
                for _ in range(int(np.ceil(len(combo) / 2)), len(combo) + 1):
                    k_dist.append(('>=', _))

                for s in all_sets:                        
                    c = (p_y_x[combo_name][s][temps[0]] * len(combo)).max(0)[0].long()

                    for k in k_dist:
                        freq = (c == k[1]).float().mean() if k[0] == '=' else (c >= k[1]).float().mean()
                        distribution[combo_name][s][k] = freq.item()
              
    df_idx = ['agg', 'T', 'l', 'name']
    df = {}
    
    for _ in ('acc', 'ood', 'misclass'):
        df[_] = pd.DataFrame(columns=df_idx + all_sets)

        df[_].set_index(df_idx, inplace=True)
        df[_].columns.name = 'set'

    for w in args.agg_type:
        for combo in combos:
            combo_name = agg_types[w].join(combo)
            i_df = (w, None, len(combo), combo_name)
            df['acc'].loc[i_df] = accuracies[combo_name]
                
    df['acc'].groupby('agg', 'l') # .agg('mean').unstack('l')
    df['acc'].unstack('l')
