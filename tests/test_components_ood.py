import os
import sys
import argparse
from utils.save_load import gethostname, LossRecorder, load_json, needed_remote_files
from utils.torch_load import get_same_size_by_name, get_classes_by_name
from utils.parameters import create_filter_parser
import numpy as np
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters
import matplotlib.pyplot as plt
import pandas as pd
import logging


parser = argparse.ArgumentParser()

parser.add_argument('--last', default=0, type=int)


logging.getLogger().setLevel(logging.WARNING)

tset = 'cifar10-?'

rmodels = load_json('jobs', 'models-{}.json'.format(gethostname()))

col_width = 10
str_col_width = '13.13'
flt_col_width = '5.1f'

if __name__ == '__main__':

    # args_from_file = ('--dataset cifar10 '
    #                   '--type cvae '
    #                   '--gamma 500 '
    #                   '--sigma-train coded '
    #                   '--coder-dict learned '
    #                   '--job-num 149127'
    #                   ).split()

    args_from_file = '--job-num 255258'.split()
    args_from_file = '--job-num 259267'.split()
    args_from_file = '--job-num 255450'.split()
    args_from_file = '--job-num 255259 259267 259269 255258'.split()

    args, ra = parser.parse_known_args(None if sys.argv[0] else args_from_file)

    filter_parser = create_filter_parser()
    filter_args = filter_parser.parse_args(ra)

    filters = DictOfListsOfParamFilters()

    for _ in filter_args.__dict__:
        filters.add(_, filter_args.__dict__[_])

    mdirs = [_ for _ in rmodels if filters.filter(rmodels[_])][-args.last:]

    loaded_files = []
    with open('/tmp/files', 'w') as f:
        for mdir in mdirs:
            model = rmodels[mdir]
            ho = model['h/o']
            tset = model['set']
            if ho:
                model['ind'] = tset.replace('-?', '-' + ho)
                model['oods'] = [tset.replace('-?', '+' + ho)]
            else:
                model['ind'] = tset
                model['oods'] = get_same_size_by_name(tset)
            ood_rec_files = ['record-' + _ + '.pth' for _ in model['oods']]
            ind_rec_file = 'record-' + model['ind'] + '.pth'

            rec_files = [os.path.join(mdir, 'samples', 'last', _) for _ in ood_rec_files + [ind_rec_file]]
            found_files = 0
            for rec_file in rec_files:
                if not os.path.exists(rec_file):
                    logging.info('File does not exist: %s', rec_file)
                    f.write(rec_file + '\n')
                else:
                    found_files += 1

            if found_files > 1:
                loaded_files.append(mdir)

    print(len(loaded_files), '(in)complete models')
    if not loaded_files:
        logging.warning('Exiting, load files')
        logging.warning('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')

        sys.exit()

    fprs = []
    confs = []

    plt.close('all')

    for mdir in loaded_files:

        model = M.load(mdir, load_net=False)
        model_dict = rmodels[mdir]

        D = np.prod(model.architecture['input_shape'])

        sigma = model.sigma.value
        # hist = {_: {} for _ in [0., 1., 1 / (2 * sigma**2), np.inf, 'iws', 'elbo']}
        hist = {_: {} for _ in ['kl', 'elbo', 'mse']}

        record_dir = os.path.join(mdir, 'samples', 'last')
        recorders = LossRecorder.loadall(record_dir, map_location='cpu')

        all_sets = [model_dict['ind']]

        for _ in model_dict['oods']:
            if _ in recorders:
                all_sets.append(_)

        print('__', model.job_number, all_sets[0],
              ':', *all_sets[1:])

        is_testset = True

        thresholds = {_: [-np.inf, np.inf] for _ in hist}
        pr = {_: {} for _ in hist}

        all_sets.append('all_oods')
        recorders['all_oods'] = LossRecorder(recorders[all_sets[0]].batch_size)

        for dset in all_sets:

            rec = recorders[dset]

            mse = (rec._tensors['wmse'] * sigma ** 2) * D

            if model_dict['type'] in ('cvae',):
                kl = rec._tensors['kl'].min(axis=0)[0].cpu()
                elbo = rec._tensors['total'].min(axis=0)[0].cpu()
                iws = rec._tensors['iws'].max(axis=0)[0].cpu()
            else:
                kl = rec._tensors['kl'].cpu()
                iws = rec._tensors['iws'].cpu()
                elbo = rec._tensors['total'].cpu()

            if not is_testset and dset != 'all_oods':
                for i in range(len(rec)):
                    if rec.has_batch(i, only_full=True):
                        recorders['all_oods'].append_batch(**rec.get_batch(i))

            for weight in hist:

                if weight == 'iws':
                    score = (- iws).sort()[0]
                elif weight == 'elbo':
                    score = (kl + mse / (2 * sigma ** 2)).sort()[0]
                elif weight == 'mse':
                    score = mse.sort()[0]
                elif weight == 'kl':
                    score = kl.sort()[0]
                else:
                    kl_w, mse_w = (0., 1) if np.isinf(weight) else (1, weight)
                    score = (kl_w * kl + mse_w * mse).sort()[0]

                hist[weight][dset] = np.histogram(score, bins=100, density=True)

                n = len(score)
                if is_testset:
                    a_ = (0, 5)
                    a_ = (1, 4)
                    n_l, n_u = int(n * a_[0] / 100), int(n * a_[1] / 100)
                    if a_[0]:
                        thresholds[weight][0] = score[n_l]
                    if a_[1]:
                        thresholds[weight][1] = score[-n_u]

                as_ind = (score > thresholds[weight][0]) & (score <= thresholds[weight][1])

                pr[weight][dset] = as_ind.float().mean().item()

            is_testset = False

        df = pd.DataFrame.from_records(pr)

        print(df.to_string(float_format='{:.1%}'.format))

        relevant_sets = []
        for s in all_sets:
            if not s.startswith(all_sets[0]):
                relevant_sets.append(s)

        ncols = 2 * len(pr)
        nrows = len(relevant_sets) // 2 + len(relevant_sets) % 2
        fig = plt.figure(model.job_number, figsize=(15, 12))
        ax = fig.subplots(nrows, ncols).flatten()

        i_ax = 0
        for s in relevant_sets:
            for w in ('kl', 'elbo', 'mse'):
                a = ax[i_ax]
                val_in = hist[w][all_sets[0]][1]
                val_in = val_in[:-1] / 2 + val_in[1:] / 2
                val_out = hist[w][s][1]
                val_out = val_out[:-1] / 2 + val_out[1:] / 2
                # a.semilogx(val_in, hist[w][all_sets[0]][0], label='in')
                # a.semilogx(val_out, hist[w][s][0])
                a.plot(val_in, hist[w][all_sets[0]][0], label='in')
                a.plot(val_out, hist[w][s][0])
                a.set_title('{}: {} {:.1%}'.format(s, w, pr[w][s]))
                a.set_axis_off()
                if not i_ax:
                    a.legend()
                a.set_xticks([], minor=[])
                a.set_yticks([], minor=[])
                i_ax += 1
        fig.show()
        fig.savefig('/tmp/{}.png'.format(model.job_number))
