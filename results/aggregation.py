import os
import sys
import logging
import argparse
import numpy as np
from utils.save_load import load_json, needed_remote_files, LossRecorder
from utils.parameters import create_filter_parser
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters
from utils.texify import tex_command, TexTab
from utils.parameters import gethostname
import pandas as pd
import itertools
import torch
from torch.nn.functional import one_hot
from module.aggregation import log_mean_exp, joint_posterior, mean_posterior, voting_posterior, posterior

agg_type_letter = {'vote': '&', 'joint': ',', 'mean': '+', 'mean~': '~', 'sumprod': '.'}

parser = argparse.ArgumentParser()


parser.add_argument('-v', action='count', default=0)
parser.add_argument('--tex', action='store_true')
parser.add_argument('--plot', nargs='?', const='p')
parser.add_argument('--result-dir', default='/tmp')
parser.add_argument('--device', default='cpu')
parser.add_argument('--last', default=0, type=int)
parser.add_argument('--method', default='iws')
parser.add_argument('--tpr', type=float, default=0.95)
parser.add_argument('--agg-type', nargs='*', choices=list(agg_type_letter), default=[])
parser.add_argument('--when', default='last')
parser.add_argument('--sets-to-exclude', nargs='*', default=[])
parser.add_argument('--combos', nargs='+', type=int)
parser.add_argument('--compute', action='store_true')
parser.add_argument('--min-models-to-keep-on', type=int, default=0)

rmodels = load_json('jobs', 'models-{}.json'.format(gethostname()))

col_width = 10
str_col_width = '13.13'
flt_col_width = '5.1f'

_ = np.seterr(divide='ignore', invalid='ignore')


def diff(t):

    # return t.diff()
    return t[..., 1:] - t[..., :-1]


def kept_names_and_sets(y):
    """
    y is a dict of dict of labels: y[name][set] = 5, 4...

    """

    allsets = set.union(*(set(y[_]) for _ in y))

    keys_set_name = {s: {n: (''.join('{}'.format(x) for x in y[n][s][:16]) if s in y[n] else None) for n in y}
                     for s in allsets}

    nums_set_key = {s: {k: sum(_ == k for _ in keys_set_name[s].values())
                        for k in set(keys_set_name[s].values()) if k is not None}
                    for s in allsets}

    keys_set = {s: max(nums_set_key[s], key=nums_set_key[s].get) for s in allsets}

    names_set = {s: [n for n in keys_set_name[s] if keys_set_name[s][n] == keys_set[s]] for s in allsets}

    lengths_set = {s: min(len(y[n][s]) for n in names_set[s]) for s in names_set}

    return lengths_set, names_set


if __name__ == '__main__':

    args_from_file = ('-vv '
                      '--tex '
                      '--dataset fashion '
                      '--type cvae '
                      '--gamma 1000 '
                      '--features vgg11 '
                      '--sigma-train learned '
                      '--learned-prior-means true '
                      '--latent-prior-variance scalar '
                      '--forced-var not '
                      '--data-augmentation not '
                      # '--last 3 '
                      '--job-num 229000... '
                      # '--method iws-a-4-1 '
                      # '--when min-loss '
                      '--sets-to-exclude fashion90 '
                      '--agg-type mean joint mean~ '
                      '--min-models-to-keep-on 0 '
                      '--combos 2 '
                      '--compute '
                      ).split()

    args_from_file = ('-vv '
                      '--tex '
                      '--job-num 226180 226397 '
                      # '--method iws-a-4-1 '
                      # '--when min-loss '
                      '--sets-to-exclude fashion90 '
                      '--agg-type mean joint mean~ '
                      '--min-models-to-keep-on 0 '
                      '--combos 2 '
                      '--compute '
                      ).split()

    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)
    wanted = args.when

    args.agg_type.insert(0, 'vote')
    tpr = args.tpr

    logging.getLogger().setLevel(40 - 10 * args.v)

    filter_parser = create_filter_parser()
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

    logging.info('{} complete model{} over {}'.format(len(mdirs), 's' if len(mdirs) > 1 else '', total_models))

    if len(mdirs) < max(args.min_models_to_keep_on, max(args.combos)):
        logging.error('Exiting, load files')
        logging.error('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')
        logging.error(' Or: %s', '$ . /tmp/rsync-files remote:dir/joint-vae')
        with open('/tmp/rsync-files', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('rsync -avP --files-from=/tmp/files $1 .\n')
        sys.exit(1)

    ood_method = args.method.split('-')
    key_loss = ood_method[0]

    if len(ood_method) > 1:
        left = int(ood_method[-2])
        right = int(ood_method[-1])
        ind_balance = (left / (left + right), (right / left + right))
    else:
        ind_balance = (1, 0)

    p_y_x = {}

    t = {_: {} for _ in ('iws', 'zdist', 'kl')}

    ind_thr = {}

    as_in = {'ind': {}, 'correct': {}}
    pr = {'ind': {}, 'correct': {}}

    y_classif = {}
    classif_acc = {}
    agreement = {}
    distribution = {}

    testset = None
    y_true = {}

    recorders = {}

    for mdir in mdirs:

        model = M.load(mdir, load_net=False)
        name = str(model.job_number)

        current_testset = model.training_parameters['set']
        if testset and current_testset != testset:
            continue
        else:
            testset = current_testset

        if args.when == 'min-loss':
            epoch = model.training_parameters.get('early-min-loss', 'last')

        if args.when == 'last' or epoch == 'last':
            epoch = max(model.testing)

        recorders[name] = LossRecorder.loadall(os.path.join(mdir, 'samples', '{:04d}'.format(epoch)),
                                               map_location=args.device)

        oodsets = [_ for _ in recorders[name] if (not _.startswith(testset) and _ not in args.sets_to_exclude)]

        y_true[name] = {_: recorders[name][_]._tensors['y_true'] for _ in recorders[name]}

    lengths_by_set, kept_names_by_set = kept_names_and_sets(y_true)

    y_true = {name: {s: y_true[name][s][:lengths_by_set[s]]
                     for s in kept_names_by_set if name in kept_names_by_set[s]}
              for name in y_true}

    kept_names = set.union(*[set(_) for _ in kept_names_by_set.values()])

    _s = ', '.join(['{}: {} ({})'.format(s, len(kept_names_by_set[s]), lengths_by_set[s])
                    for s in kept_names_by_set])

    logging.info('Kept models (images): {}'.format(_s))

    for s in lengths_by_set:
        for _ in t:
            t[_][s] = {}
        n = lengths_by_set[s]
        for name in kept_names_by_set[s]:
            for _ in t:
                t[_][s][name] = recorders[name][s]._tensors[_][..., :n]
                t[_][s][name] = recorders[name][s]._tensors[_][..., :n]

    del recorders

    combo_lengths = sorted(args.combos)
    if 1 not in combo_lengths:
        combo_lengths.insert(0, 1)

    oodsets = [s for s in lengths_by_set if (lengths_by_set[s] and s != testset)]
    sets = [testset, *oodsets]

    combos = []
    for _ in combo_lengths:
        combos += [*itertools.combinations(sorted(kept_names), _)]

    logging.info('Will work on {} combos'.format(len(combos)))

    i_ax = 0

    """
    + : work with p(x|y) = sum p_k(x|y)
    , : work with z = [z1,...,zn]
    . : work wiht p(x|y) = prod p_k(yx|y)
    """
    wanted_aggs = args.agg_type

    unwanted_aggs = {'acc': [], 'ood': ['mean~', 'joint'], 'misclass': []}

    agg_types = {w: [_ for _ in wanted_aggs if _ not in unwanted_aggs[w]] for w in unwanted_aggs}

    result_dir = args.result_dir

    saved_dir = os.path.join(result_dir, 'saved')

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    nan_temp = -1

    temps_ = {_: [nan_temp, 1, 2, 5, 10, 20, 50, 100, 200, 500] for _ in wanted_aggs}
    # temps_['vote'] = [nan_temp]

    p_y_x = {}
    log_p_x_y = {}
    ind_thr = {}
    correct_thr = {}
    accuracies = {}

    for combo in combos:

        sets = [s for s in kept_names_by_set if all(m in kept_names_by_set[s] for m in combo)]
        oodsets = [s for s in sets if s != testset]

        all_sets = [testset, 'correct', 'incorrect', *oodsets]

        logging.info('Working on {}'.format('--'.join(combo)))

        for _ in t:
            for s in sets:
                if all(m in t[_][s] for m in combo):
                    t[_][s][combo] = [t[_][s][m] for m in combo]

        if len(combo) == 1:
            combo_name = combo[0]
            y_classif[combo_name] = {s: torch.argmax(t['iws'][testset][combo_name], dim=0)
                                     for s in sets}

        for w in (wanted_aggs if len(combo) > 1 else ['mean']):

            combo_name = agg_type_letter[w].join(combo)
            saved_pth = os.path.join(saved_dir, '{}.pth'.format(combo_name))
            is_saved = os.path.exists(saved_pth)
            compute = args.compute or not is_saved

            if compute:
                t_pth = {}
            else:
                t_pth = torch.load(saved_pth)

            temps = temps_[w]
            logging.info(w)

            if compute:
                logging.info('Computing y|x for {}'.format(combo_name))
                if w == 'joint':
                    p_y_x[combo_name] = {s: joint_posterior(*[t['zdist'][s][_] for _ in combo], temps=temps)
                                         for s in sets}
                elif w == 'mean':
                    p_y_x[combo_name] = {s: mean_posterior(*[t['iws'][s][_] for _ in combo], temps=temps)
                                         for s in sets}

                elif w == 'mean~':
                    p_y_x_ = {s: [posterior(-t['kl'][s][_], temps=temps) for _ in combo] for s in sets}
                    p_y_x[combo_name] = {s: {temp: torch.stack([_[temp] for _ in p_y_x_[s]]).mean(0)
                                             for temp in temps}
                                         for s in sets}

                elif w == 'dist' or (w == 'vote'):  # and len(combo) == 1):
                    p_y_x[combo_name] = {s: voting_posterior(*[y_classif[_][s] for _ in combo], temps=temps)
                                         for s in sets}

                t_pth['p_y_x'] = p_y_x[combo_name]

            else:
                p_y_x[combo_name] = t_pth['p_y_x']

            y_classif[combo_name] = {s: p_y_x[combo_name][s][temps[0]].argmax(0) for s in sets}

            i_true = (y_classif[combo_name][testset] == y_true[combo[0]][testset]).squeeze()
            accuracies[combo_name] = i_true.float().mean().item()

            for s, i_ in zip(('correct', 'incorrect'), (i_true, ~i_true)):
                p_y_x[combo_name][s] = {t: p_y_x[combo_name][testset][t][:, i_]
                                        for t in p_y_x[combo_name][testset]}

            max_py = {s: {t: p_y_x[combo_name][s][t].max(0)[0] for t in
                          p_y_x[combo_name][s]}
                      for s in all_sets}

            in_set = {'ind': 'ood', 'correct': 'misclass'}

            if w != 'vote' and (w in agg_types['ood'] or w in agg_types['misclass']) or len(combo) == 1:

                tprs = {_: {tpr: {t: tpr for t in temps}} for _ in ('ind', 'correct')}
                thr_balance = {'ind': ind_balance, 'correct': (1, 0)}
                if len(combo) > 1:
                    vote_combo = combo_name.replace(agg_type_letter[w], agg_type_letter['vote'])
                    for k in ('ind', 'correct'):
                        tprs[k]['vote'] = pr[k][vote_combo]['vote'][testset if k == 'ind' else 'correct']

                if w == 'mean' or len(combo) == 1:
                    if compute:
                        logging.info('Computing x|y for {}'.format(combo_name))

                        log_p_x_y[combo_name] = {s: log_mean_exp(*[t['iws'][s][_] for _ in combo]).max(0)[0]
                                                 for s in sets}
                        t_pth['log_p_x_y'] = log_p_x_y[combo_name]

                    else:
                        log_p_x_y[combo_name] = t_pth['log_p_x_y']

                    for s, i_ in zip(('correct', 'incorrect'), (i_true, ~i_true)):
                        log_p_x_y[combo_name][s] = log_p_x_y[combo_name][testset][i_]

                k_ = [_ for _ in ('ind', 'correct') if w in agg_types[in_set[_]]]

                logging.debug('Computing prs')
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

                        # print('**** {} {} {} {} {} ({}) {:.1%} {:.1%}'.format(w, k, combo_name,
                        #                                                       *i_l_r[_temps[0]],
                        #                                                       n[_temps[0]],
                        #                                                       tpr_l[_temps[0]],
                        #                                                       tpr_r[_temps[0]]))
                        
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
                logging.debug('Done computing prs')

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
            if compute:
                torch.save(t_pth, saved_pth)

    def make_dfs():

        df_idx = {_: ['agg', 'T', 'l', 'name', 'tpr'] for _ in ('acc', 'ood', 'misclass')}
        df_idx['acc'].remove('tpr')

        oodsets = [s for s in lengths_by_set if (lengths_by_set[s] and s != testset)]
        sets = [testset, *oodsets]

        df_sets = {'acc': [testset], 'ood': sets, 'misclass': [testset, 'correct', 'incorrect']}

        df = {}

        for _ in df_idx:
            df[_] = pd.DataFrame(columns=df_idx[_] + df_sets[_])

            df[_].set_index(df_idx[_], inplace=True)
            df[_].columns.name = 'set'

        for w in args.agg_type:
            for combo in combos:
                available_sets = [s for s in kept_names_by_set if (all(m in kept_names_by_set[s] for m in combo)
                                                                   and s not in args.sets_to_exclude)]
                if testset in available_sets:
                    available_sets.remove(testset)
                    all_sets = [testset, 'correct', 'incorrect', *available_sets]

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

        oodsets = [s for s in lengths_by_set
                   if (lengths_by_set[s] and s != testset and s not in args.sets_to_exclude)]

        sets = [testset, *oodsets]

        cols = ['s2.1'] * len(combo_lengths)
        tab = TexTab('l', *cols, float_format='{:.1f}')
        tab.append_cell('$M$', row='header')
        for l in combo_lengths:
            tab.append_cell(l, multicol_format='c', row='header', formatter='{}')

        for agg in agg_types['acc']:
            tab.append_cell(agg, row=agg)
            for l in combo_lengths:
                tab.append_cell(100 * df['acc'].loc[agg][(dataset, l)], row=agg)

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
                            pr = 100 * df_.loc[s][(l, agg)] if s in df_.index else None
                            tab.append_cell(pr, row=s)
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
        df_ = df_[df_.columns[incorrect_cols]]  # .stack('set').droplevel('set')
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
                tab.append_cell(l, multicol_format='c', width=len(aggs),  # if l > 1 else 1,
                                row='header', formatter='{}')

            tab.append_cell('T', row='subheader')
            tab.append_cell('/'.join(tex_aggs), multicol_format='c', width=n_cols, row='subheader')
            for T in temps:
                tab.append_cell(T if T != nan_temp else '--', row=T, formatter='{}')
                for l in combo_l_[r]:
                    if r == 'vote':
                        pr = 100 * df_.loc[(agg, r, T)][l, 'correct']
                        tab.append_cell(pr, row=T)

                    for agg in agg_[r]:
                        if T in temps_[agg]:
                            pr = 100 * df_.loc[(agg, r, T)][l, 'incorrect']
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
            tex_file = 'misclass-agg-{}-{}--{}--{}.tex'.format(testset,
                                                               r,
                                                               '-'.join(str(_) for _ in combo_lengths),
                                                               '-'.join(agg_[r]))
            # tex_file = 'misclass-{}.tex'.format(r)  #
            with open(os.path.join(args.result_dir, tex_file), 'w') as f:
                tab.render(f)
