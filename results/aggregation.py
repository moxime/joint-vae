import os
import sys
import logging
import argparse
import numpy as np
from utils.save_load import load_json, needed_remote_files, LossRecorder
from utils.parameters import parse_filters
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters
from utils.texify import tex_command, TexTab
from utils.parameters import gethostname
import pandas as pd
from utils.roc_curves import fpr_at_tpr
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import torch
from torch.nn.functional import one_hot
from module.aggregation import log_mean_exp, joint_posterior, mean_posterior, voting_posterior, posterior

agg_type_letter = {'vote': '&', 'joint': ',', 'mean': '+', 'mean~': '~'}

parser = argparse.ArgumentParser()


parser.add_argument('--last', default=0, type=int)
parser.add_argument('--method', default='iws-2s')
parser.add_argument('--tpr', type=float, default=0.95)
parser.add_argument('-v', action='count', default=0)
parser.add_argument('--result-dir', default='/tmp')
parser.add_argument('--agg-type', nargs='*', choices=list(agg_type_letter), default=[])
parser.add_argument('--when', default='last')
parser.add_argument('--plot', nargs='?', const='p')
parser.add_argument('--tex', action='store_true')
parser.add_argument('--sets-to-exclude', nargs='*', default=[])
parser.add_argument('--combos', nargs='+', type=int)
parser.add_argument('--device', default='cpu')

rmodels = load_json('jobs', 'models-{}.json'.format(gethostname()))

col_width = 10
str_col_width = '13.13'
flt_col_width = '5.1f'

_ = np.seterr(divide='ignore', invalid='ignore')


def diff(t):

    return t[...,1:] - t[...,:-1]

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
                      '--agg-type mean joint mean~ '
                      '--combos 5 '
                      ).split()

    # args_from_file = ('-vvvv '
    #                   '--job-num 193080 193082 '
    #                   # '--job-num 169381 '
    #                   '--when min-loss '
    #                   ).split()

    # args_from_file = '--job-num 192000... --when min-loss'.split()

    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)
    wanted = args.when

    args.agg_type.insert(0, 'vote')
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
        sys.exit(1)

    key_loss = args.method.split('-')[0]

    p_y_x = {}

    t = {_: {} for _ in ('iws', 'zdist')}
    
    testset = None
    ind_thr = {}

    n_by_rep = dict(hsv=0, rgb=0)

    as_in = {'ind': {}, 'correct': {}}
    pr = {'ind': {}, 'correct': {}}

    y_classif = {}
    classif_acc = {}
    agreement = {}
    distribution = {}

    y_true = None

    for mdir in mdirs:

        model = M.load(mdir, load_net=False)
        rep = model.architecture['representation']
        name = rep.upper() + str(n_by_rep[rep])
        name = str(model.job_number)
        
        n_by_rep[rep] += 1
        current_testset = model.training_parameters['set']
        if testset and current_testset != testset:
            continue
        else:
            testset = current_testset

        if n_by_rep[rep] > max_per_rep:
            continue

        if args.when == 'min-loss':
            epoch = model.training_parameters.get('early-min-loss', 'last')

        if args.when == 'last' or epoch == 'last':
            epoch = max(model.testing)

        recorders = LossRecorder.loadall(os.path.join(mdir, 'samples', '{:04d}'.format(epoch)),
                                         map_location=args.device)
        current_y_true = recorders[testset]._tensors['y_true']

        if y_true is not None and (y_true != current_y_true).any():
            logging.debug('{} has a diffrent shuffle, can not use!'.format(name))
            continue
        else:
            y_true = current_y_true
        
        sets = [*recorders.keys()]

        # exclude rotated set
        oodsets = [_ for _ in sets if (not _.startswith(testset) and _ not in args.sets_to_exclude)]
        # sets = [kept_testset, 'lsunr']  # + sets
        sets = [testset] + oodsets
        
        t['iws'][name] = {}
        t['zdist'][name] = {}
        
        as_in['ind'][name] = {}

        for _ in ('zdist', 'iws'):
            for dset in sets:
                t[_][name][dset] = recorders[dset]._tensors[_]
                t[_][name][dset] = recorders[dset]._tensors[_]
            
    combo_lengths = sorted(args.combos)
    if 1 not in combo_lengths:
        combo_lengths.insert(0, 1)

    all_sets = [testset, 'correct' , 'incorrect', *oodsets]

    combos = []
    for _ in combo_lengths:
        combos += [*itertools.combinations(sorted(as_in['ind']), _)]

    logging.info('Will work on {} combos'.format(len(combos)))
    
    i_ax = 0

    """
    + : work with p(x|y) = sum p_k(x|y)
    , : work with z = [z1,...,zn]
    . : work wiht p(x|y) = prod p_k(yx|y)
    """
    wanted_aggs = args.agg_type

    unwanted_aggs = {'acc': ['dist'], 'ood': ['dist', 'mean~', 'joint'], 'misclass': []}
    
    agg_types = {w: [_ for _ in wanted_aggs  if _ not in unwanted_aggs[w]] for w in unwanted_aggs}
                   
    result_dir = args.result_dir

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    _ = os.listdir(result_dir)
    
    nan_temp = -1

    temps_ = {_: [nan_temp, 1, 2, 5, 10, 20] for _ in wanted_aggs}
    temps_['dist'] = [nan_temp]

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
            y_classif[combo_name] = {s: torch.argmax(t['iws'][combo_name][testset], dim=0)
                                     for s in sets}

        for w in (wanted_aggs if len(combo) > 1 else ['vote']):

            t_pth = {}
            
            temps = temps_[w]
            combo_name = agg_type_letter[w].join(combo)
            logging.info(w)
            
            if w == 'joint':
                p_y_x[combo_name] = {s: joint_posterior(*[t['zdist'][_][s] for _ in combo], temps=temps)
                                     for s in sets}
            elif w == 'mean':
                p_y_x[combo_name] = {s: mean_posterior(*[t['iws'][_][s] for _ in combo], temps=temps)
                                     for s in sets}

            elif w == 'mean~':
                p_y_x_ = {s: [posterior(t['iws'][_][s], temps=temps) for _ in combo] for s in sets}
                p_y_x[combo_name] = {s: {temp: sum([_[temp] for _ in p_y_x_[s]]) for temp in temps}
                                     for s in sets}
                
            elif w == 'dist' or (w == 'vote'):  # and len(combo) == 1):
                p_y_x[combo_name] = {s: voting_posterior(*[y_classif[_][s] for _ in combo], temps=temps)
                                     for s in sets}
                                 
            y_classif[combo_name] = {s: p_y_x[combo_name][s][temps[0]].argmax(0) for s in sets}
                                 
            acc = (y_classif[combo_name][testset].squeeze() == y_true).float().mean().item()
            i_true = (y_classif[combo_name][testset] == y_true).squeeze()
            accuracies[combo_name] = acc
                                 
            for s, i_ in zip(('correct', 'incorrect'), (i_true, ~i_true)):
                p_y_x[combo_name][s] = {t: p_y_x[combo_name][testset][t][:, i_] for t in temps} 
                                 
            max_py = {s: {t: p_y_x[combo_name][s][t].max(0)[0] for t in temps} for s in all_sets}

            in_set = {'ind': 'ood', 'correct': 'misclass'}

            if w != 'vote' and (w in agg_types['ood'] or w in agg_types['misclass']) or len(combo) == 1:

                tprs = {_: {tpr: {t: tpr for t in temps}} for _ in ('ind', 'correct')}
                thr_balance = {'ind': (4/5, 1/5), 'correct': (1, 0)}
                if len(combo) > 1:
                    vote_combo = combo_name.replace(agg_type_letter[w], agg_type_letter['vote'])
                    for k in ('ind', 'correct'):
                        tprs[k]['vote'] = pr[k][vote_combo]['vote'][testset if k == 'ind' else 'correct']

                if w == 'mean' or len(combo) == 1:
                    log_p_x_y[combo_name] = {s: log_mean_exp(*[t['iws'][_][s] for _ in combo]).max(0)[0]
                                             for s in sets}

                    for s, i_ in zip(('correct', 'incorrect'), (i_true, ~i_true)):
                        log_p_x_y[combo_name][s] = log_p_x_y[combo_name][testset][i_]

                k_ = [_ for _ in ('ind', 'correct') if w in agg_types[in_set[_]]]

                for k in k_:
                    if k == 'ind':
                        t_in_out = {s: {nan_temp: log_p_x_y[combo_name][s]} for s in all_sets}
                        t_in_out['ind'] = t_in_out[testset]
                        _temps = [nan_temp]
                    else:
                        t_in_out = {s: max_py[s] for s in all_sets}
                        _temps = temps
                        
                    n = {t: len(t_in_out[k][t]) for t in _temps}

                    ind_thr[combo_name] = {}
                    pr[k][combo_name] = {}
                    as_in[k][combo_name] = {}
                    for r in tprs[k]:
                        tpr_l = {t: thr_balance[k][0] * (1 - tprs[k][r][t]) for t in _temps}
                        tpr_r = {t: 1 - thr_balance[k][1] * (1 - tprs[k][r][t]) for t in _temps}
                        i_l_r = {t: (int(n[t] * tpr_l[t]), int(n[t] * tpr_r[t]) - 1) for t in _temps}

                        thr = {t: (sorted(t_in_out[k][t])[i_l_r[t][0]], sorted(t_in_out[k][t])[i_l_r[t][1]])
                               for t in _temps}

                        ind_thr[combo_name][r] = thr
                        as_in[k][combo_name][r] = {}
                        pr[k][combo_name][r] = {}

                        for s in all_sets:

                            t_ = t_in_out[s]
                            as_in[k][combo_name][r][s] = {t: (t_[t] >= thr[t][0]) * (t_[t] <= thr[t][1])
                                                          for t in _temps}
                            pr[k][combo_name][r][s] = {t: as_in[k][combo_name][r][s][t].float().mean().item()
                                                       for t in _temps}
                            
            elif w == 'vote':
                for k in ('ind', 'correct'):

                    _temps = temps if k == 'correct' else [nan_temp]
                    as_in[k][combo_name] = {}
                    pr[k][combo_name] = {}
                    count_as_in = {s: {t: sum([one_hot(as_in[k][_][tpr][s][t].long(), num_classes=2)
                                           for _ in combo])
                                       for t in _temps}
                                   for s in sets}

                    as_in[k][combo_name]['vote'] = {s: {t: (diff(count_as_in[s][t]) >= 0).squeeze()
                                                        for t in _temps}
                                                    for s in sets}
                    
                    for s, i_ in zip(('correct', 'incorrect'), (i_true, ~i_true)):
                        as_in[k][combo_name]['vote'][s] = {t: as_in[k][combo_name]['vote'][testset][t][i_]
                                                           for t in _temps}
                        
                    pr[k][combo_name]['vote'] = {s: {t: as_in[k][combo_name]['vote'][s][t].float().mean().item()
                                                     for t in _temps}
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

    def make_dfs():
        df_idx = {_: ['agg', 'T', 'l', 'name', 'tpr'] for _ in ('acc', 'ood', 'misclass')}

        df_idx['acc'].remove('tpr')

        df_sets = {'acc': testset, 'ood': sets, 'misclass': [testset, 'correct', 'incorrect']}

        df = {}

        for _ in df_idx:
            df[_] = pd.DataFrame(columns=df_idx[_] + all_sets)

            df[_].set_index(df_idx[_], inplace=True)
            df[_].columns.name = 'set'

        for w in args.agg_type:
            for combo in combos:
                combo_name = agg_type_letter[w].join(combo)
                for T in temps_[w]:
                    i_df = (w, T, len(combo), combo_name)
                    if T is nan_temp:
                        df['acc'].loc[i_df] = accuracies[combo_name]
                    if w in agg_types['misclass']:
                        for r in (tpr, 'vote'):
                            i_df_m = tuple([*i_df, r])
                            prs = pr['correct'][combo_name].get(r)
                            if prs:
                                df['misclass'].loc[i_df_m] = {s: prs[s][T] for s in all_sets}

                    if w in agg_types['ood'] and T == nan_temp:
                        for r in (tpr, 'vote'):
                            i_df_ood = tuple([*i_df, r])
                            prs = pr['ind'][combo_name].get(r)
                            if prs:
                                df['ood'].loc[i_df_ood] = {s: prs[s][nan_temp] for s in all_sets}

        group_by = {_: ['agg', 'l', 'tpr'] for _ in df}
        group_by['acc'].remove('tpr')
        group_by['misclass'].append('T')
        for _ in df:
            df[_] = df[_][df_sets[_]].groupby(group_by[_]).agg('mean').unstack('l')
            if _ not in ['acc']:
                df[_] = df[_].reorder_levels(['l', 'set'], axis='columns').sort_values(by='l', axis=1)

        return df
    
    df = make_dfs()

    for _ in df:
        print('\n\n\n*** {} ***'.format(_))
        
        print(df[_].to_string(float_format='{:.1%}'.format, na_rep='--'))

    if args.tex:

        cols = ['s2.1'] * len(combo_lengths)
        tab = TexTab('l', *cols, float_format='{:.1f}')
        tab.append_cell('$M$', row='header')
        for l in combo_lengths:
            tab.append_cell(l, multicol_format='c', row='header', formatter='{}')

        for agg in agg_types['acc']:
            tab.append_cell(agg, row=agg)
            for l in combo_lengths:
                tab.append_cell(100 * df['acc'].loc[agg][l], row=agg)

        tab.add_midrule(agg_types['acc'][0])

        tex_file = 'acc-agg-{}--{}--{}.tex'.format(testset,
                                                   '-'.join(str(_) for _ in combo_lengths),
                                                   '-'.join(agg_types['acc']))
                                                 
        with open(os.path.join(args.result_dir, tex_file), 'w') as f:
            tab.render(f)
            
        agg_for_ood = agg_types['ood']
        combo_l_ = {'vote': [_ for _ in combo_lengths if _ != 1], tpr: combo_lengths}
        agg_ = {tpr: [_ for _ in agg_for_ood if _ != 'vote'], 'vote': agg_for_ood}
        sets_ = {'vote': sets, tpr: oodsets}

        tex_rename = {s: tex_command('makecommand', s) for s in sets}
        tex_rename.update({s: s.capitalize() for s in ('correct', 'incorrect')})
                           
        for r in (tpr, 'vote'):
            
            i_ = df['ood'].index
            i = i_.isin([r], level='tpr') & i_.isin(agg_[r], level='agg')
            df_ = df['ood'][i].stack('set').droplevel('tpr')[combo_l_[r]].unstack('agg')            
                           
            if r == 'vote':
                n_cols = (len(agg_[r]) + 1) * (len(combo_l_[r]))

            else:
                n_cols = len(agg_[r]) * (len(combo_l_[r]))
                
            df_tex = df_.rename(index=tex_rename)
            cols = ['s2.1'] * n_cols
            tab = TexTab('l', *cols, float_format='{:2.1f}')
            tab.append_cell('$M$', row='header')
            for l in combo_l_[r]:
                tab.append_cell(l, multicol_format='c', width=len(agg_[r]) + (r == 'vote'),
                                row='header', formatter='{}')

            if len(agg_[r]) > 1:
                tab.append_cell('', row='subheader')
                tab.append_cell('/'.join(agg_[r]), width=n_cols, row='subheader')
                tab.add_midrule('subheader')
                
            for s in sets_[r]:
                tab.append_cell(tex_command('makecommand', s), row=s)
                if s == testset:
                    for l in combo_l_[r]:
                        pr = df_.loc[s][(l, agg_[r][0])]
                        tab.append_cell(100 * pr, multicol_format='S[table-format=2.1]',
                                        width=len(agg_[r]), row=s, formatter='{:2.1f}')
                else:
                    for l in combo_l_[r]:
                        for agg in agg_[r]:
                            pr = df_.loc[s][(l, agg)]
                            tab.append_cell(100 * pr, row=s)
            tab.add_midrule(sets_[r][0])

            if len(agg_[r]) > 1:
                for i_l, _ in enumerate(combo_l_[r]):
                    for i_agg, _ in enumerate(agg_[r][:-1]):
                        tab.add_col_sep(1 + i_l * len(agg_[r]) + i_agg + 1, '/')
            
            tab.render()
            tex_file = 'ood-agg-{}-{}--{}--{}.tex'.format(testset,
                                                          r,
                                                          '-'.join(str(_) for _ in combo_lengths),
                                                          '-'.join(agg_for_ood))
            
            with open(os.path.join(args.result_dir, tex_file), 'w') as f:
                tab.render(f)
                
        df_ = df['misclass']
        incorrect_cols = df_.columns.isin(['incorrect', 'correct'], level='set')
        df_ = df_[df_.columns[incorrect_cols]] # .stack('set').droplevel('set')
        aggs = agg_types['misclass']
        combo_l_ = {'vote': [_ for _ in combo_lengths if _ != 1], tpr: combo_lengths}
        agg_ = {tpr: [_ for _ in aggs if _ != 'vote'], 'vote': aggs}

        tex_aggs_ = {'TPR': r'\acron{tpr}', 'mean~': r'mean\~'}
        
        for r in (tpr, 'vote'):

            aggs = ['TPR', *agg_[r]] if r == 'vote' else agg_[r]

            tex_aggs = [tex_aggs_.get(_, _) for _ in aggs]

            if r == 'vote':
                n_cols = len(aggs) * (len(combo_l_[r])) 

            else:
                n_cols = len(aggs) * (len(combo_l_[r])) 
                
            cols = ['s2.1'] * n_cols

            tab = TexTab('l', *cols, float_format='{:2.1f}')

            tab.append_cell('$M$', row='header')
            
            for l in combo_l_[r]:
                tab.append_cell(l, multicol_format='c', width=len(aggs), # if l > 1 else 1,
                                row='header', formatter='{}')

            tab.append_cell('T', row='subheader')
            tab.append_cell('/'.join(tex_aggs), multicol_format='c', width=n_cols, row='subheader')
            for T in temps:
                tab.append_cell(T if T != nan_temp else '--', row=T, formatter='{}')
                for l in combo_l_[r]:
                    if r == 'vote':
                        pr  = 100 * df_.loc[(agg, r, T)][l, 'correct']
                        tab.append_cell(pr, row=T)
                        
                    for agg in agg_[r]:
                        if T in temps_[agg]:
                            pr  = 100 * df_.loc[(agg, r, T)][l, 'incorrect']
                        else:
                            pr = None
                        tab.append_cell(pr, row=T)

            if len(aggs) > 1:
                for i_l, _ in enumerate(combo_l_[r]):
                    for i_agg, _ in enumerate(aggs[:-1]):
                        tab.add_col_sep(i_l * len(aggs) + i_agg + 2, '/')

            tab.add_midrule('subheader')
            tab.add_midrule(temps[0])

            tab.render()
            tex_file = 'misclass-ood-{}-{}--{}--{}.tex'.format(testset,
                                                          r,
                                                          '-'.join(str(_) for _ in combo_lengths),
                                                          '-'.join(agg_[r]))
            # tex_file = 'misclass-{}.tex'.format(r)  # 
            with open(os.path.join(args.result_dir, tex_file), 'w') as f:
                tab.render(f)
