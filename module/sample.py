"""
Use a network ti generate new images, use the sampling of Z

"""
from cvae import ClassificationVariationalNetwork as Net
import utils.torch_load as tl
from utils.save_load import fetch_models, needed_remote_files
import torch

import numpy as np
from matplotlib import pyplot as plt
import logging
import os
from utils.save_load import job_to_str, LossRecorder
from utils.inspection import output_latent_distribution
from utils.parameters import DEFAULT_RESULTS_DIR, DEFAULT_JOBS_DIR, set_log, add_filters_to_parsed_args

from torchvision.utils import save_image

import argparse
import sys


class DefaultClasses(object):

    def __getitem__(self, k):
        return k

    
def sample(net, x=None, y=None, root=os.path.join(DEFAULT_RESULTS_DIR, '%j', 'samples'), directory='test',
           in_classes=DefaultClasses(), out_classes=DefaultClasses(),
           N=20, L=10):
    
    r"""Creates a grid of output images. If x is None the output images
    are the ones created when the decoder is fed with prior z"""

    if x is not None:
        N = min(N, len(x))
    elif net.is_cvae:
        N = net.num_labels

    wN = int(np.log10(N - 1)) + 1
    L = min(L, net.latent_sampling)
    wL = int(np.log10(L - 1)) + 1
    K = net.latent_dim

    dir_path = os.path.join(job_to_str(net.job_number, root), directory)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif not os.path.isdir(dir_path):
        raise FileExistsError

    """
    ftable = open(os.path.join(dir_path, 'generate.tab', 'w'))
    ftable.write('x_in y_in y_out var_out x_out_avge mse_out_avge x_out_mean mse_out_mean')
    ftable.write(' '.join(f'x_out_{l_:0{wL}} mse_out_{l_:0{wL}}' for l_ in range(l_, L)))
    """

    defy = r'\def\y{{{}}}'
    
    if x is not None:

        (D, H, W) = net.input_shape[-3:]

        x_grid = {'name': f'grid-{N}x{L}',
                  'tensor': torch.zeros((D, 0, L * W), device=x.device)}

        with torch.no_grad():
            x_, logits, batch_losses, measures, mu, log_var, z = net.evaluate(x[:N], None, z_output=True)
            y_ = net.predict_after_evaluate(logits, batch_losses)

        list_of_images = [x_grid]

        for row in range(N):
            
            x_row = torch.zeros((D, H, 0), device=x.device)
            list_of_images.append({'name': f'x_{row:0{wN}}_in',
                                   'tensor': x[row],
                                   'tex': defy.format(in_classes[y[row]])
                                   })

            x_row = torch.cat([x_row, x[row]], 2)

            list_of_images.append({'name': f'x_{row:0{wN}}_out_mean',
                                   'tensor': x_[0][row],
                                   'tex': defy.format(out_classes[y_[row]])
                                   })
            x_row = torch.cat([x_row, x_[0][row]], 2)

            list_of_images.append({'name': f'x_{row:0{wN}}_out_average',
                                   'tensor': x_[1:].mean(0)[row],
                                   'tex': defy.format(out_classes[y_[row]])
                                   })

            x_row = torch.cat([x_row, x_[1:].mean(0)[row]], 2)

            for l_ in range(1, L-2):
                list_of_images.append({'name':
                                       f'x_{row:0{wN}}_out_{l_:0{wL}}',
                                       'tensor': x_[l_, row],
                                       'tex': defy.format(out_classes[y_[row]])
                                       })
                x_row = torch.cat([x_row, x_[l_, row]], 2)

            if row < N:
                x_grid['tensor'] = torch.cat([x_grid['tensor'], x_row], 1)

    elif net.is_cvae or net.is_jvae or net.is_vae:

        (D, H, W) = net.input_shape[-3:]
        K = net.latent_dim

        z = torch.randn(L, N, K, device=net.device)

        if net.is_cvae:
            z = z + net.encoder.prior.mean.unsqueeze(0)

        x_grid = {'name': f'grid-{N}x{L}',
                  'tensor': torch.zeros((D, 0, L * W), device=net.device)}
        list_of_images = [x_grid]

        x_ = net.imager(net.decoder(z)).view(L, N, D, H, W)

        for row in range(N):

            x_row = torch.zeros((D, H, 0), device=net.device)

            for l_ in range(L):
                list_of_images.append(
                    {'name': f'x{row:0{wN}}_out_{l_:0{wL}}',
                     'tensor': x_[l_, row]})

                x_row = torch.cat([x_row, x_[l_, row]], 2)

            if row < N:
                x_grid['tensor'] = torch.cat([x_grid['tensor'], x_row], 1)

    elif net.is_vae:
        pass

    else:
        raise ValueError('You try to generate images with a net'
                         f'which is {net.type}')

    for image in list_of_images:

        path = os.path.join(dir_path, image['name'] + '.png')
        logging.debug('Saving image in %s', path)
        save_image(image['tensor'], path)
        if 'tex' in image:
            path = os.path.join(dir_path, image['name'] + '.tex')
            with open(path, 'w') as f:
                f.write(image['tex'])
                # f.write('\n')

    return list_of_images


def zsample(x, net, y=None, batch_size=128, root=os.path.join(DEFAULT_RESULTS_DIR, '%j', 'samples'), bins=10, directory='test'):
    r"""will sample variable latent and ouput scatter and histogram of
    variance and mean of z

    """
    N = len(x)
    N -= N % batch_size

    assert net.type != 'cvae' or y is not None

    mu_z = torch.zeros(N, net.latent_dim)
    var_z = torch.zeros(N, net.latent_dim)
    log_var_z = torch.zeros(N, net.latent_dim)

    for b in range(N // batch_size):

        start = b * batch_size
        end = start + batch_size

        with torch.no_grad():
            out = net.evaluate(x[start:end], z_output=True)

        x_,  _, _, _, mu_z[start:end], log_var_z[start:end], _ = out

    var_z = log_var_z.exp()

    if net.is_cvae:

        encoder_dictionary = net.encoder.latent_dictionary
        centroids = latent_dictionary.index_select(0, y[:N])
        print('*** centroids shape', *centroids.shape)

    dir_path = os.path.join(job_to_str(net.job_number, root), directory)

    f = os.path.join(dir_path, 'hist_var_z.dat')
    output_latent_distribution(mu_z, var_z, f, result_type='hist_of_var',
                               bins=bins, per_dim=True)

    f = os.path.join(dir_path, 'mu_z_var_z.dat')
    output_latent_distribution(mu_z, var_z, f, result_type='scatter', per_dim=True)

    if y is not None:

        for c in range(net.num_labels):
            i_ = y == c

            for rtype, fname in zip(('hist_of_var', 'scatter'),
                                    ('hist_var_z{}.dat', 'mu_z_var_z{}.dat')):
                f = os.path.join(dir_path, fname.format(c))
                output_latent_distribution(mu_z, var_z, f, result_type=rtype,
                                           bins=bins, per_dim=True)


def comparison(x, *nets, batch_size=128, root=os.path.join(DEFAULT_RESULTS_DIR, '%j', 'samples'), directory='ood'):
    r""" Comparison of different nets """

    root = root.replace('%j', '-'.join(str(n.job_number) for n in nets))

    for n in nets:

        n.to(x.device)
        m = n.compute_max_batch_size(batch_size, 'test')
        batch_size = min(m, batch_size)
    logging.info('Batch size for comparison: %s', batch_size)

    N = x.shape[0] // batch_size * batch_size
    jobs = [n.job_number for n in nets]

    x_ = {n.job_number: torch.zeros(N, *x.shape[1:]) for n in nets}
    y_pred = {n.job_number: torch.zeros(N, dtype=int) for n in nets}

    for n in nets:
        j = n.job_number
        for b in range(N // batch_size):

            start = b * batch_size
            end = start + batch_size
            with torch.no_grad():
                x__, logits, losses, _ = n.evaluate(x[start:end])
                y_pred[j][start:end] = n.predict_after_evaluate(logits, losses)
                x_[j][start:end] = x__[0]

    div = {j: {jj: np.zeros(len(x)) for jj in jobs if jj > j} for j in jobs}
    # div = np.zeros((len(nets), len(nets), len(x)))

    for j in jobs:
        for jj in [_ for _ in jobs if _ > j]:

            div[j][jj] = (x_[j] - x_[jj]).pow(2).mean([_ for _ in range(1, x.dim())])
            # div[jj][j] = div[j][jj]

    return div, y_pred


if __name__ == '__main__':

    root=os.path.join(DEFAULT_RESULTS_DIR, '%j', 'samples')
    
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--job-dir', default=root)
    parser.add_argument('--last', type=int, default=0)
    parser.add_argument('-m', '--batch-size', type=int, default=256)
    parser.add_argument('--num-batch-for-test', type=int, default=1)
    parser.add_argument('-W', '--grid-width', type=int, default=0)
    parser.add_argument('--total-width', type=int, default=30)
    parser.add_argument('-N', '--grid-height', type=int, default=0)
    parser.add_argument('-D', '--directory', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--seed', type=int, const=1, nargs='?', default=False)
    parser.add_argument('--z-sample', type=int, default=0)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--look-for-missed', type=int, default=0)
    parser.add_argument('--stop-if-missing', action='store_true')
    parser.add_argument('--list-jobs-and-quit', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-p', '--plot', nargs='?', const='all')

    jobs = [
        # 222622,
        224383,
        # 224385,
    ]

    args_from_file = ['-D', '/tmp/%j/samples',
                      '-N', '10',
                      # '--list',
                      # '--debug',
                      '-vv',
                      '--seed',
                      '-m', '32',
                      '--num-batch', '100',
                      '--job-num'] + [str(j) for j in jobs]

    args, ra = parser.parse_known_args(None if sys.argv[0] else args_from_file)

    filters = add_filters_to_parsed_args(parser, args, ra)

    set_log(args.verbose, args.debug, os.path.join(DEFAULT_JOBS_DIR, 'log'), name='results')

    models = fetch_models(args.job_dir, filter=filters)[-args.last:]

    L = args.grid_width
    N = args.grid_height
    m = args.batch_size
    root = args.directory
    z_sample = args.z_sample
    bins = args.bins
    # cross_sample = args.compare

    m = max(m, N, z_sample)
    
    device = tl.choose_device()
    logging.debug(device)

    """if args.sync:
        r = LossRecorder(1)
        computed_max_batch_size = False
    """

    logging.info('Will work in %s', root)

    if args.list_jobs_and_quit:
        models.sort(key=lambda n: n['job'] if isinstance(n['job'], int) else 0)
        for i, n in enumerate(models):
            #            print(f'{i:4d}:', n['job'])
            print(n['job'])
        logging.info('%d networks listed', i + 1)
        sys.exit(0)

    if len(models) > 2:
        args.plot = False

    mdirs = [_['dir'] for _ in models]
    removed = False

    nmodels = len(mdirs)

    with open('/tmp/files', 'w') as f:
        for mdir, sdir in needed_remote_files(*mdirs, which_rec='none', state=True, missing_file_stream=f):
            logging.debug('{} for {}'.format(sdir[-30:], 'last'))
            if mdir in mdirs:
                mdirs.remove(mdir)
                removed = True
                logging.info('{} is removed (files not found)'.format(mdir.split('/')[-1]))

    if removed:
        logging.error('Exiting, load files')
        logging.error('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')
        logging.error(' Or: %s', '$ . /tmp/rsync-files remote:dir/joint-vae')
        with open('/tmp/rsync-files', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('rsync -avP --files-from=/tmp/files $1 .\n')
        if not mdirs or args.stop_if_missing:
            sys.exit(1)
        else:
            logging.error('Will go anymay with {} model(s) remaining. Consider laodingmissing files')
            logging.error('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')
            logging.error(' Or: %s', '$ . /tmp/rsync-files remote:dir/joint-vae')

    logging.info('{} plot will be done'.format('One' if args.plot else 'No'))
    logging.info('{} models will be worked'.format(len(mdirs)))

    models = [_ for _ in models if _['dir'] in mdirs]

    dict_of_sets = dict()
    for n in models:
        model = n['net']
        trainset_ = tuple(model.training_parameters[k] for k in ('set', 'transformer'))
        if trainset_ not in dict_of_sets:
            dict_of_sets[trainset_] = [n]
        else:
            dict_of_sets[trainset_].append(n)

    for (testset, transformer), list_of_nets in dict_of_sets.items():
        logging.info('Going with %s (%s): %d nets to be done',
                     testset, transformer, len(list_of_nets))
        x, y = dict(), dict()

        out_classes = tl.get_classes_by_name(testset)
        _, test_dataset = tl.get_dataset(testset, transformer=transformer)

        num_batch = args.num_batch_for_test
        x[testset], y[testset] = tl.get_batch(test_dataset, device=device, shuffle=args.seed,
                                              batch_size=m * num_batch)

        oodsets = test_dataset.same_size

        in_classes = {}
        for o in oodsets:
            _, ood_dataset = tl.get_dataset(o, transformer=transformer, splits=['test'])
            x[o], y[o] = tl.get_batch(ood_dataset, device=device, batch_size=m)
            in_classes[o] = tl.get_classes_by_name(o)
            
        if not L:
            L = args.total_width // (1 + len(x))

        for n in list_of_nets:

            logging.info('loading state of %s', n['job'])

            model = Net.load(n['dir'])
            model.to(device)
            logging.info('done')
            logging.info('Compute max batch size')

            batch_size = min(m, model.compute_max_batch_size(batch_size=2*m, which='test'))
            logging.info(f'done ({batch_size})')

            completed = {_: 0 for _ in ('correct', 'incorrect')}

            for _ in completed:

                x_correct_or_not = []
                y_correct_or_not = []

                for b in range(num_batch):
                    if completed[_] >= N:
                        break
                    i0 = b * batch_size
                    with torch.no_grad():
                        x_batch = x[testset][i0: i0 + batch_size]
                        y_batch = y[testset][i0: i0 + batch_size]
                        y_ = model.predict(x_batch)
                        i_ = (y_ == y_batch) ^ (_ == 'incorrect')
                        completed[_] += i_.sum().item()
                        x_correct_or_not.append(x_batch[i_])
                        y_correct_or_not.append(y_batch[i_])
                _s = 'Collected {n} images for {w} amongst {b} images'
                logging.info(_s.format(n=completed[_], w=_, b=b*batch_size))

                x[_] = torch.cat(x_correct_or_not)
                y[_] = torch.cat(y_correct_or_not)

            for s in x:
                logging.info('sampling %s', s)

                if N:
                    # N_ = batch_size if s == '?correct' else N
                    list_of_images = sample(model,
                                            x[s][:N],
                                            y[s][:N],
                                            root=root,
                                            directory=s,
                                            in_classes=in_classes.get(s, out_classes),
                                            out_classes=out_classes,
                                            N=N, L=L)

                    if z_sample and not s.endswith('correct'):

                        zsample(x[s][:z_sample], model, y=y[s][:z_sample],
                                batch_size=m,
                                root=root, bins=args.bins, directory=s)

            if N:
                list_of_images = sample(model, root=root,
                                        directory='generate', N=N, L=L)

            # loss_comparisons(model, root=root, plot=args.plot, bins=args.bins)
