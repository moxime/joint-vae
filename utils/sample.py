"""
Use a network ti generate new images, use the sampling of Z

"""
from cvae import ClassificationVariationalNetwork as Net
import utils.torch_load as tl
from utils.save_load import find_by_job_number
import torch

import numpy as np
from matplotlib import pyplot as plt
import logging
import os
from utils.save_load import job_to_str, LossRecorder
from utils.inspection import output_latent_distribution

from torchvision.utils import save_image

import argparse, sys


def sample(net, x=None, y=None, root='results/%j/samples', directory='test',
           N=20, L=10, iteration=False):
    r"""Creates a grid of output images. If x is None tuhe output images
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
                  
    if x is not None:

        (D, H, W) = net.input_shape[-3:]

        x_grid = {'name': f'grid-{N}x{L}.png',
                  'tensor': torch.zeros((D, 0, L * W), device=x.device)}
                             
        with torch.no_grad(): 
            x_, y_, mu, log_var, z = net(x, y)

        list_of_images = [x_grid]
        
        for row in range(len(x)):
            x_row = torch.zeros((D, H, 0), device=x.device)
            list_of_images.append({'name': f'x{row:0{wN}}_in.png',
                                   'tensor': x[row]})
            x_row = torch.cat([x_row, x[row]], 2)
                              
            list_of_images.append({'name': f'x{row:0{wN}}_out_mean.png',
                                   'tensor': x_[0][row]})
            x_row = torch.cat([x_row, x_[0][row]], 2)            
                
            if not iteration:
                list_of_images.append({'name': f'x{row:0{wN}}_out_average.png',
                                       'tensor': x_[1:].mean(0)[row]})
            
                x_row = torch.cat([x_row, x_[1:].mean(0)[row]], 2)
                
                for l_ in range(1, L-2):
                    list_of_images.append({'name':
                                           f'x{row:0{wN}}_out_{l_:0{wL}}.png',
                                           'tensor': x_[l_, row]})
                    x_row = torch.cat([x_row, x_[l_, row]], 2)
                    
            else:

                for l_ in range(1, L-1):
                    x_, y_, mu, log_var, z = net(x_, y)
                    list_of_images.append({'name':
                                           f'x{row:0{wN}}_out_{l_:0{wL}}.png',
                                           'tensor': x_[0, row]})
                    x_row = torch.cat([x_row, x_[0, row]], 2)
                    
            if row < N:
                x_grid['tensor'] = torch.cat([x_grid['tensor'], x_row], 1)    
                                  
    elif net.is_cvae or net.is_jvae or net.is_vae:
    
        (D, H, W) = net.input_shape[-3:]
        K = net.latent_dim

        z = torch.randn(L, N, K, device=net.device)

        if net.is_cvae:
            z = z + net.encoder.latent_dictionary.unsqueeze(0)

        x_grid = {'name': f'grid-{N}x{L}.png',
                  'tensor': torch.zeros((D, 0, L * W), device=net.device)}
        list_of_images = [x_grid]

        x_ = net.imager(net.decoder(z))
        # print(*x_.shape)        

        for row in range(N):

            x_row = torch.zeros((D, H, 0), device=net.device)
                
            for l_ in range(L):
                list_of_images.append(
                    {'name': f'x{row:0{wN}}_out_{l_:0{wL}}.png',
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

        path = os.path.join(dir_path, image['name'])
        logging.debug('Saving image in %s', path)
        save_image(image['tensor'], path)
        
    return list_of_images


def zsample(x, net,y=None, batch_size=128, root='results/%j/samples', bins=10, directory='test'):
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

    print('*** net.type', net.type)
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
            i_ = y==c

            for rtype, fname in zip(('hist_of_var', 'scatter'),
                                    ('hist_var_z{}.dat','mu_z_var_z{}.dat')):
                f = os.path.join(dir_path, fname.format(c))
                output_latent_distribution(mu_z, var_z, f, result_type=rtype,
                                           bins=bins, per_dim=True)

            
def comparison(x, *nets, batch_size=128, root='results/%j/samples', directory='ood'):
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
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--job-dir', default='jobs')
    parser.add_argument('-m', '--batch-size', type=int, default=256)
    parser.add_argument('-W', '--grid-width', type=int, default=0)
    parser.add_argument('--total-width', type=int, default=30)
    parser.add_argument('-N', '--grid-height', type=int, default=0)
    parser.add_argument('-D', '--directory', default=root)
    parser.add_argument('--z-sample', type=int, default=0)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--look-for-missed', type=int, default=0)
    parser.add_argument('--list-jobs-and-quit', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-p', '--plot', nargs='?', const='all')
    
    jobs = [117281,
            129527,
            117250,
            112267
            ]
    
    args_from_file = ['-D', '/tmp/%j/samples',
                      '-N', '10',
                      # '--debug',
                      '-vv',
                      '-m', '64',
                      '--job-num'] + [str(j) for j in jobs]
    
    filter_args, remaining_args = get_filters_args(None if sys.argv[0] else args_from_file)
    
    args = parser.parse_args(args=remaining_args, namespace=filter_args)

    set_log(args.verbose, args.debug, 'jobs/log', name='results')
    
    L = args.grid_width
    N = args.grid_height
    m = args.batch_size
    root = args.directory
    z_sample = args.z_sample
    bins = args.bins
    # cross_sample = args.compare

    filters = args.filters

    device = tl.choose_device()
    logging.debug(device)
    
    """if args.sync:
        r = LossRecorder(1)
        computed_max_batch_size = False
    """
    
    logging.info('Will work in %s', root)

    list_of_networks = sum(collect_models(args.job_dir, load_net=False, load_state=False), [])

    networks_to_be_studied = []
    for n in list_of_networks:
        filter_results = sum([[f.filter(n[d]) for f in filters[d]] for d in filters], [])
        to_be_studied = all(filter_results)

        if to_be_studied:
            networks_to_be_studied.append(n)
            # for d in filters:
            #   print(d, n[d])
            #   for f in filters[d]:
            #       print(f, f.filter(n[d]))

    if args.list_jobs_and_quit:
        networks_to_be_studied.sort(key=lambda n: n['job'] if isinstance(n['job'], int) else 0)
        for i, n in enumerate(networks_to_be_studied):
            #            print(f'{i:4d}:', n['job'])
            print(n['job'])
        logging.info('%d networks listed', i + 1)
        sys.exit(0)

    if len(networks_to_be_studied) > 2:
        args.plot = False

    logging.info('{} plot will be done'.format('One' if args.plot else 'No'))
    logging.info('{} models will be worked'.format(len(networks_to_be_studied)))

    dict_of_sets = dict()
    for n in networks_to_be_studied:
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
        
        _, test_dataset = tl.get_dataset(testset, transformer=transformer)
        x[testset], y[testset] = tl.get_batch(test_dataset, device=device,
                                              batch_size=max(z_sample, N))

        oodsets = test_dataset.same_size

        for o in oodsets:
            _, ood_dataset = tl.get_dataset(o, transformer=transformer)
            x[o], y[o] = tl.get_batch(ood_dataset, device=device, batch_size=max(z_sample, N))

        if not L:
            L = args.total_width // (1 + len(x))

        for n in list_of_nets:

            logging.info('loading state of %s', n['job'])

            model = Net.load(n['dir'])
            model.to(device)
            logging.info('done')
            logging.info('Compute max batch size')

            batch_size = min(m, model.compute_max_batch_size(batch_size=m, which='test'))
            logging.info(f'done ({batch_size})')

            for s in x:
                
                logging.info('sampling %s', s)

                if N:
                    list_of_images = sample(model, x[s][:N], root=root,
                                            directory=s,
                                            N=N, L=L)

                if z_sample:

                    zsample(x[s][:z_sample], model, y=y[s][:z_sample],
                            batch_size=m,
                            root=root, bins=args.bins, directory=s)

            if N:
                list_of_images = sample(model, root=root,
                                        directory='generate', N=N, L=L)
                
            loss_comparisons(model, root=root, plot=args.plot, bins=args.bins)    

