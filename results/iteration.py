import os
import sys
import logging
from utils.print_log import turnoff_debug
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from utils.save_load import LossRecorder
from utils.torch_load import get_same_size_by_name, get_classes_by_name
from module.iteration import IteratedModels


classif_titles = {True: 'correct', False: 'incorrect'}

parser = argparse.ArgumentParser()

parser.add_argument('jobs', nargs='+')
parser.add_argument('-v', action='count', default=0)
parser.add_argument('--results-dir', default='/tmp')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--png', action='store_true')
parser.add_argument('--images', default=10, type=int)

log = logging.getLogger(__name__)


def do_what_you_gotta_do(dir_name, result_dir, n_images=10, png=True):

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
    
    for s in sets:

        log.debug('Working on %s', s)

        rec = recorders[s]
        t = rec._tensors
        kl = t['kl']

        y_pred = kl.argmin(1)
        y_pred_[s] = y_pred

        agreement = torch.zeros_like(y_pred[0])

        for k in range(len(agreement)):
            agreement[k] = len(y_pred[:, k].unique())

        if s == dset:
            y_true = t['y_true']

            for i in range(y_pred.shape[0]):
                print('Acc of step {}: {:.2%}'.format(i, (y_true == y_pred[i]).float().mean()))

            i_true = y_true == y_pred[0]
            
        else:
            i_true = (y_pred[0] >= 0)

        w = ('all', True, False) if s == dset else ('all',)

        disagrees_on = {}
        count = {}

        i_ = {'all': i_true + True,
              True: i_true,
              False: ~i_true}

        n_ = {'all': '*', True: 'v', False: 'x'}

        for _ in i_:
            disagrees_on[_], count[_] = agreement[i_[_]].unique(return_counts=True)

        print(s)
        for _ in w:
            for a, k in zip(disagrees_on[_], count[_]):
                print('Disagreement Lvl {} for {}: {:6.1%}'.format(a, n_[_], k / i_[_].sum()))
            print()

        batch_size = recorders[s].batch_size
        num_batch = len(recorders[s])
        len_samples = len(samples[s]['y'])
        samples_per_batch = len_samples // num_batch

        samples_idx[s] = torch.tensor([_ % batch_size < samples_per_batch for _ in range(len(i_true))])

        samples_i[s] = {True: i_true[samples_idx[s]], False: ~i_true[samples_idx[s]]}

        y_pred = y_pred_[s]
        x = {_: samples[s]['x'][samples_i[s][_]][:n_images] for _ in (True, False)}
        x_ = {_: samples[s]['x_'][:, 0, samples_i[s][_]][:, :n_images] for _ in (True, False)}
        
        y_ = {_: y_pred[:, samples_idx[s]][:, samples_i[s][_]][:, :n_images] for _ in (True, False)}

        y = {_: samples[s]['y'][samples_i[s][_]][:n_images] for _ in (True, False)}

        if s != dset:
            pass
            # y = {_: -1 * torch.ones_like(y[_]) for _ in y}
    
        w = (True, False) if s == dset else (True,)
    
        for _ in w:
            x[_] = torch.cat([x[_].unsqueeze(0), x_[_]])
            y[_] = torch.cat([y[_].unsqueeze(0), y_[_]])
            title = classif_titles if s == dset else {True: s}

        classes = {_: get_classes_by_name(s) if not _ else get_classes_by_name(dset)
                   for _ in range(len(model) + 1)}

        output.update({title[_]:
                       {'x': x[_],
                        'y': y[_],
                        'c': classes}
                       for _ in title})
                           
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
                for i in range(n_images):
                    tex_file = os.path.join(image_dir, 'image_{}.tex'.format(i))
                    with open(tex_file, 'w') as f:
                    
                        for k in range(len(model) + 1):

                            image = x[_][k][i]
                            image_name = 'x_{}_{}.png'.format(i, k)
                            save_image(image, os.path.join(image_dir, image_name))
                        f.write(r'\def\yin{{{}}}'.format(classes[0][y[_][0][i]]))
                        f.write(r'\def\yout{')
                        f.write(','.join(classes[k][y[_][k][i]] for k in range(1, len(model) + 1)))
                        f.write('}\n')
                        f.write(r'\def\n{{{}}}'.format(len(model)))
                        f.write('\n')
                        f.write(r'\def\rotatedlabel{}'.format('{90}' if s.endswith('90') else '{0}'))
                        f.write('\n')
    return output


if __name__ == '__main__':

    args_from_file = ('-vvv '
                      '--png '
                      '--plot '
                      'iterated-jobs/199384-203528-203529'
                      ).split()

    args = parser.parse_args(None if len(sys.argv) > 1 else args_from_file)
    n_images = args.images

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
                
                for _ in x:

                    image = torchvision.utils.make_grid(x[_].transpose(0, 1).flatten(end_dim=1), nrow=len(x[_]))
                    img = transforms.functional.to_pil_image(image)

                    fig, ax = plt.subplots(1)
                    ax.imshow(np.asarray(img))
                    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    fig.suptitle(_)

                    fig.show()

                    print(_)

                    for row in y[_].T:

                        print(' -> '.join('{:2}'.format(_) for _ in row))

