import os
import sys
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from utils.save_load import LossRecorder
from utils.torch_load import get_same_size_by_name, get_classes_by_name, get_shape_by_name
from module.iteration import IteratedModels

from utils.texify import TexTab, tex_command

classif_titles = {True: 'correct', False: 'incorrect'}

parser = argparse.ArgumentParser()

parser.add_argument('jobs', nargs='+')
parser.add_argument('-v', action='count', default=0)
parser.add_argument('--results-dir', default='/tmp')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--png', action='store_true')
parser.add_argument('--images', default=10, type=int)
parser.add_argument('--tex', action='store_true')

log = logging.getLogger(__name__)


def do_what_you_gotta_do(dir_name, result_dir, n_images=10, png=True, tex=['means'], prec=2, tpr=0.95):

    try:
        model = IteratedModels.load(dir_name, load_state=False)
    except FileNotFoundError:
        log.error('{} not a model'.format(dir_name))
        return

    testset = model.training_parameters['set']
    allsets = [testset]
    allsets.extend(get_same_size_by_name(testset))

    recorders = LossRecorder.loadall(dir_name, map_location='cpu')
    samples_files = LossRecorder.loadall(dir_name, file_name='sample-{w}.pth', output='path', map_location='cpu')
    samples = {_: torch.load(samples_files[_]) for _ in samples_files}

    dset = model.training_parameters['set']
    oodsets = list(recorders)
    oodsets.remove(dset)

    sets = [dset] + oodsets

    samples_idx = {}
    samples_i = {}
    y_pred_ = {}

    output = {}

    k_with_y = {_: _ for _ in ('kl', 'zdist', 'iws', 'loss')}
    k_with_y_moving = {_ + '_': _ for _ in k_with_y}

    k_without_y = {'mse': 'mse'}

    k_all = dict(**k_without_y, **k_with_y, **k_with_y_moving)

    signs = {_: 1 for _ in k_all}
    signs['iws'] = -1
    signs['iws_'] = -1

    def which_y(t, k, dim=0):

        s = signs[k]
        return s * (s * t).min(dim=dim)[0]

    pr = {}

    disagreement = {}

    for s in sets:

        log.debug('Working on %s', s)

        rec = recorders[s]
        t = rec._tensors
        kl = t['kl']

        i_mse = [0]
        for j in range(1, len(model)):
            i_mse.append(i_mse[-1] + j)

        beta = np.prod(get_shape_by_name(s)[0]) / 1e-3
        t['loss'] = t['kl'] + beta * t['mse'][i_mse].unsqueeze(-2)
        
        y_pred = kl.argmin(1)
        y_pred_[s] = y_pred

        disagreement[s] = torch.zeros_like(y_pred[0])

        for i in range(len(disagreement[s])):
            disagreement[s][i] = len(y_pred[:, i].unique())

        if s == dset:
            y_true = t['y_true']

            for i in range(y_pred.shape[0]):
                print('Acc of step {}: {:.2%}'.format(i, (y_true == y_pred[i]).float().mean()))

            i_true = y_true == y_pred[0]

            t_y = {}
            thr = {}
            pr['correct'] = {}
            pr['incorrect'] = {}

            for k in k_all:

                thr[k] = {}
                if k in k_with_y:
                    index_y = torch.ones_like(t[k_all[k]], dtype=int) * y_pred[0]
                    t_y[k] = t[k_all[k]].gather(1, index_y)[:, 0]

                elif k in k_with_y_moving:

                    t_y[k] = which_y(t[k_all[k]], k, dim=1)

                else:
                    t_y[k] = t[k]

                t_y[k] *= signs[k]

                i_tpr = int(len(y_true) * tpr)
                thr[k] = t_y[k].sort()[0][..., i_tpr]

                for w in ('correct', 'incorrect'):
                    pr[w][k] = torch.zeros(len(model))

                for i, w in zip((i_true, ~i_true), ['correct', 'incorrect']):
                    for m in range(len(model)):
                        mean = t_y[k][m][i].mean()
                        pr[w][k][m] = (t_y[k][m][i] <= thr[k][m]).sum() / i.sum()
                        print('*** {} {} {} {:.1%} {:.3e}'.format(w, k, m, pr[w][k][m], mean))

        else:
            i_true = torch.ones_like(y_pred[0], dtype=bool)
            pr[s] = {}

            for k in k_all:

                if k in k_with_y:
                    index_y = torch.ones_like(t[k_all[k]], dtype=int) * y_pred[0]
                    t_y[k] = t[k_all[k]].gather(1, index_y)[:, 0]

                elif k in k_with_y_moving:
                    t_y[k] = which_y(t[k_all[k]], k, dim=1)

                else:
                    t_y[k] = t[k]

                t_y[k] *= signs[k]

                pr[s][k] = torch.zeros(len(model))
                for m in range(len(model)):
                    pr[s][k][m] = (t_y[k][m] <= thr[k][m]).sum() / len(y_pred[0])

        w = (True, False) if s == dset else (True,)

        title = classif_titles if s == dset else {True: s}

        i_ = {'all': i_true + True,
              True: i_true,
              False: ~i_true}

        for _ in w:
            disagreement[title[_]] = disagreement[s][i_[_]]

        print(s)

        batch_size = recorders[s].batch_size
        num_batch = len(recorders[s])
        len_samples = len(samples[s]['y'])
        samples_per_batch = len_samples // num_batch

        samples_idx[s] = torch.tensor([_ % batch_size < samples_per_batch for _ in range(len(i_true))])

        samples_i[s] = {True: i_true[samples_idx[s]], False: ~i_true[samples_idx[s]]}

        y_pred = y_pred_[s]
        x = {_: samples[s]['x'][samples_i[s][_]][:n_images] for _ in (True, False)}
        x_ = {_: samples[s]['x_'][:, 0, samples_i[s][_]][:, :n_images] for _ in (True, False)}

        print('**** x_', *x_[True].shape)
        
        y_ = {_: y_pred[:, samples_idx[s]][:, samples_i[s][_]][:, :n_images] for _ in (True, False)}

        y = {_: samples[s]['y'][samples_i[s][_]][:n_images] for _ in (True, False)}

        if s != dset:
            pass
            # y = {_: -1 * torch.ones_like(y[_]) for _ in y}

        w = (True, False) if s == dset else (True,)

        for _ in w:
            x[_] = torch.cat([x[_].unsqueeze(0), x_[_]])
            y[_] = torch.cat([y[_].unsqueeze(0), y_[_]])

        classes = {_: get_classes_by_name(s) if not _ else get_classes_by_name(dset)
                   for _ in range(len(model) + 1)}

        for k in k_with_y_moving:
            t[k] = which_y(t[k_all[k]], k, dim=1)

        for k in k_with_y:
            index = torch.ones_like(t[k_all[k]], dtype=int) * y_pred[0]
            t[k] = t[k_all[k]].gather(1, index)[:, 0, ]

        averaged = {title[_]: {k: t[k][..., i_[_]].mean(-1) for k in k_all} for _ in title}

        for _ in title:
            output[title[_]] = averaged[title[_]]
            output[title[_]].update({'x': x[_], 'y': y[_], 'c': classes})
            output[title[_]]['disagree'] = torch.zeros(len(model))
            for m in range(len(model)):
                output[title[_]]['disagree'][m] = (disagreement[title[_]] == m + 1).float().mean()

        for _ in title:
            mse = {}
            n = 0
            for i in range(len(model) + 1):
                for j in range(i):
                    mse[(j, i)] = output[title[_]]['mse'][n]
                    n += 1

            output[title[_]]['mse'] = mse

        if png:
            with open(os.path.join(result_dir, 'arch.tex'), 'w') as f:
                f.write('\\def\\niter{{{}}}\n'.format(len(model)))
                f.write('\\def\\trainset{{{}}}\n'.format(dset))
                _sets = ','.join(oodsets)
                f.write('\\def\\oodsets{{{}}}\n'.format(_sets))
                _sets = ','.join([classif_titles[True], classif_titles[False], *oodsets])
                f.write('\\def\\allsets{{{}}}\n'.format(_sets))

            for _ in w:
                image_dir = os.path.join(result_dir, 'samples', title[_])
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                for i in range(min(n_images, x[_].shape[1])):
                    tex_file = os.path.join(image_dir, 'image_{}.tex'.format(i))
                    with open(tex_file, 'w') as f:

                        for k in range(len(model) + 1):

                            # print('***', _, 'k,i', k, i, 'x:', *x[_].shape)
                            image = x[_][k][i]
                            # print('*** image out {k} n° {i} in [{m:.2f} {M:.2f}]'.format(k=k, i=i,
                            #                                                              m=image.min(),
                            #                                                              M=image.max()))
                            image_name = 'x_{}_{}.png'.format(i, k)
                            save_image(image, os.path.join(image_dir, image_name))
                        f.write(r'\def\yin{{{}}}'.format(classes[0][y[_][0][i]]).replace('_', '-'))
                        f.write(r'\def\yout{')
                        out_classes = [classes[k][y[_][k][i]] for k in range(1, len(model) + 1)]
                        f.write('\"' +
                                '\",\"'.join(out_classes[k].replace('_', '-') for k in range(len(model))))
                        f.write('\"}\n')
                        f.write(r'\def\n{{{}}}'.format(len(model)))
                        f.write('\n')
                        f.write(r'\def\rotatedlabel{}'.format('{90}' if s.endswith('90') else '{0}'))
                        f.write('\n')

    if tex:
        """ MSE tex file """
        # tab_width = len(model) * 3
        first_row = True

        max_mse = max([max(output[_]['mse'].values()) for _ in output])
        min_mse = min([min(output[_]['mse'].values()) for _ in output])

        min_mse_exp = np.floor(np.log10(min_mse))
        mse_factor = int(np.ceil(-min_mse_exp / 3) * 3 - 3)
        max_mse_exp = int(np.floor(np.log10(max_mse)))

        swidth = mse_factor + max_mse_exp + 1
        stable_mse = 's{}.3'.format(swidth)
        stable_pr = 's2.1'

        col_format = [stable_mse, stable_pr] + [stable_mse, stable_pr, stable_mse] * (len(model) - 1)

        tab = TexTab('l', *col_format, float_format='{:.1f}')

        for m in range(len(model)):
            c = 3 * m + 1 + (m == 0)
            tab.add_col_sep(c, ' (')
            tab.add_col_sep(c + 1, '\\%) ')

        for _ in output:
            header = _.capitalize() if _.endswith('correct') else tex_command('makecommand', _)
            tab.append_cell(header, row='header' + _)
            subheader = tex_command('acron', 'mse') + ' avec'
            tab.append_cell(subheader, row='subheader' + _)
            tab.append_cell('', row=_)

            for j in range(1, len(model) + 1):
                tab.append_cell('Out {}'.format(j), width=2 + (j > 1), multicol_format='c', row='header' + _)
                tab.append_cell('In', width=2, multicol_format='c', row='subheader' + _)
                if j > 1:
                    tab.append_cell('Out {}'.format(j - 1), multicol_format='c', row='subheader' + _)
                tab.append_cell(output[_]['mse'][(0, j)] * 10 ** mse_factor, row=_, formatter='{:.3f}')
                tab.append_cell(100 * pr[_]['mse'][j - 1], row=_, formatter='{:.1f}')
                if j > 1:
                    tab.append_cell(output[_]['mse'][(j - 1, j)] * 10 ** mse_factor, row=_, formatter='{:.3f}')

            if first_row:
                first_row = False
            else:
                tab.add_midrule('header' + _)

            for j in range(1, len(model) + 1):
                start = 1 if j == 1 else 3 * j - 3
                tab.add_midrule(_, start=start, end=start + 1 + (j > 1))

        with open(os.path.join(result_dir, 'mse.tex'), 'w') as f:
            tab.render(f)
            f.write('\\def\\msefactor{{{}}}'.format('{:1.0f}'.format(-mse_factor)))

        """ ZDisT / KL tex file """
        for k in [*k_with_y, *k_with_y_moving]:

            first_row = True

            max_k = max([max(output[_][k].abs()) for _ in output])
            min_k = min([min(output[_][k].abs()) for _ in output])
            min_k_exp = np.floor(np.log10(min_k))

            # print('*** MIN / MAX {} = {:.2e} ({}) / {:.2e}'.format(k, min_k, min_k_exp, max_k))

            if min_k_exp <= -3:
                min_k_exp -= 3

            k_factor = int(np.ceil(-min_k_exp / 3) * 3)

            max_k_exp = int(np.floor(np.log10(max_k)))

            swidth = k_factor + max_k_exp + 1

            col_format = ['l'] + ['s{}.3'.format(swidth), 's2.1'] * len(model)
            tab = TexTab(*col_format)
            for m in range(len(model)):
                tab.add_col_sep(2 + 2 * m, ' (')
                tab.add_col_sep(3 + 2 * m, '\\%)' + ' ' * (m < len(model) - 1))

            tab.append_cell('', row='header')
            for j in range(len(model)):
                tab.append_cell('M{}'.format(j + 1), width=2, multicol_format='c', row='header')

            for _ in output:
                tab.append_cell(_.capitalize() if _.endswith('correct') else tex_command('makecommand', _),
                                row=_)

                for j in range(len(model)):

                    tab.append_cell(output[_][k][j] * 10 ** k_factor, row=_, formatter='{:.3f}')
                    tab.append_cell(100 * pr[_][k][j], row=_, formatter='{:.1f}')

                if first_row:
                    first_row = False
                    tab.add_midrule(_)

            with open(os.path.join(result_dir, '{}.tex'.format(k)), 'w') as f:
                tab.render(f)
                f.write('\\def\\{}factor{{{}}}\n'.format(k, '{:1.0f}'.format(-k_factor)))

        """ Agreement """
        col_format = ['l'] + ['s2.1'] * len(model)
        tab = TexTab(*col_format)
        tab.append_cell(r'$|\mathcal Y|$', row='header')
        for m in range(len(model)):
            tab.append_cell(m + 1, multicol_format='c', row='header')

        for _ in output:
            tab.append_cell(_, row=_)
            for m in range(len(model)):
                tab.append_cell(100 * output[_]['disagree'][m], row=_, formatter='{:.1f}')

        tab.add_midrule(next(iter(output)))

        with open(os.path.join(result_dir, 'disagree.tex'), 'w') as f:
            tab.render(f)

    return output


if __name__ == '__main__':

    args_from_file = ('-vvv '
                      '--png '
                      # '--plot '
                      '--tex '
                      'iterated-jobs/svhn/199384-203528-203529 '
                      # 'iterated-jobs/cifar10/173277-173278-173279 '
                      ).split()

    args = parser.parse_args(None if len(sys.argv) > 1 else args_from_file)
    n_images = args.images

    if args.tex == []:
        args.tex = ['mean']

    log.setLevel(40 - 10 * args.v)
    log.debug('Logging at level %d', log.level)

    plt.close('all')

    for model_dir in args.jobs:

        log.info('Processing %s', model_dir)
        model_name = os.path.split(model_dir)[-1]
        result_dir = os.path.join(args.results_dir, model_name)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        output = do_what_you_gotta_do(model_dir, result_dir, n_images=n_images, png=args.png)

        if args.plot:
            for s in output:
                x, y, classes = (output[s][_] for _ in 'xyc')

                image = torchvision.utils.make_grid(x.transpose(0, 1).flatten(end_dim=1), nrow=len(x))
                img = transforms.functional.to_pil_image(image)

                fig, ax = plt.subplots(1)
                ax.imshow(np.asarray(img))
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                fig.suptitle(s)

                fig.show()

                print(s)

                for row in y.T:

                    print(' -> '.join('{:2}'.format(_) for _ in row))

    if args.plot and sys.argv[0]:
        input('Press any key to close figures')
