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


def zsample(x, net, batch_size=128, root='results/%j/samples', directory='test'):
    r"""will sample varaible latent and ouput scatter and histogram of
    variance and mean of z

    """
    N = len(x)

    mu_z = torch.zeros(z_sample, net.latent_dim)
    var_z = torch.zeros(z_sample, net.latent_dim)
    log_var_z = torch.zeros(z_sample, net.latent_dim)
                       
    for b in range(z_sample // batch_size):

        start = b * batch_size
        end = start + batch_size

        with torch.no_grad():
            out = net.evaluate(x[start:end], z_output=True)

        x_,  _, _, _, mu_z[start:end], log_var_z[start:end], _ = out

    var_z = log_var_z.exp()
                       
    dir_path = os.path.join(job_to_str(net.job_number, root), directory)

    f = os.path.join(dir_path, 'hist_var_z.dat')
    output_latent_distribution(mu_z, var_z, f, result_type='hist_of_var',
                        bins=bins, per_dim=True)

    f = os.path.join(dir_path, 'mu_z_var_z.dat')
    output_latent_distribution(mu_z, var_z, f, result_type='scatter', per_dim=True)

    
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
    
    device = 'cuda'

    reload = True
    reload = False

    root = 'results/%j/samples'
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', type=int,
                        nargs='+',
                        default=[],)
                       
    parser.add_argument('-m', '--batch-size', type=int, default=256)
    parser.add_argument('-L', '--grid-width', type=int, default=0)
    parser.add_argument('-N', '--grid-height', type=int, default=0)
    parser.add_argument('-D', '--directory', default=root)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-v', action='count')
    parser.add_argument('--z-sample', type=int, default=0)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--compare', type=int, default=0)
    parser.add_argument('--look-for-missed', type=int, default=0)
    
    args_from_file = ['-D', '/tmp/%j/samples',
                      '--compare', '1024',
                      # '--debug',
                      '-vv',
                      '-m', '64',
                      '-j', '117281', '117250', '112267',
                      ]
    
    args = parser.parse_args(None if sys.argv[0] else args_from_file)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.v:
        logging.getLogger().setLevel(logging.WARNING if args.v==1 else logging.INFO)
        
    jobs = args.jobs
    L = args.grid_width
    N = args.grid_height
    m = args.batch_size
    root = args.directory
    z_sample = args.z_sample
    bins = args.bins
    cross_sample = args.compare
    
    if args.sync:
        r = LossRecorder(1)
        computed_max_batch_size = False


    logging.info('Will work in %s', root)
    
    shells = find_by_job_number(*jobs, load_net=False, force_dict=True)

    adapt_L = not L
    LL = 30

    if z_sample or N:
        for j in shells:

            if adapt_L:
                L = LL // (2 + len(shells[j]['net'].ood_results))

            x = dict()
            y = dict()
            logging.info('loading state of %s', j)

            net = Net.load(shells[j]['dir'])
            net.to(device)
            logging.info('done')
            logging.info('Compute max batch size')

            batch_size = min(m, net.compute_max_batch_size(batch_size=m, which='test'))
            logging.info(f'done ({batch_size})')

            if args.sync and computed_max_batch_size and  batch_size != computed_max_batch_size:
                logging.warning("Let's assume you know what you're doing")

            computed_max_batch_size = batch_size
            z_sample = z_sample // batch_size * batch_size

            testset, transformer = (shells[j]['net'].training_parameters[k] for k in ('set', 'transformer'))
            _, test_dataset = tl.get_dataset(testset, transformer=transformer)

            if args.sync:
                r.init_seed_for_dataloader()

            x[testset], y[testset] = tl.get_batch(test_dataset, device=device,
                                                  batch_size=max(z_sample, N))

            oodsets = test_dataset.same_size

            for o in oodsets:
                _, ood_dataset = tl.get_dataset(o, transformer=transformer)
                x[o], y[o] = tl.get_batch(ood_dataset, device=device, batch_size=max(z_sample, N))

            for s in x:
                logging.info('sampling %s', s)

                if N:
                    list_of_images = sample(net, x[s][:N], root=root,
                                            directory=s,
                                            N=N, L=L)

                if z_sample:

                    zsample(x[s][:z_sample], net, batch_size, root=root, directory=s)

            if N:
                list_of_images = sample(net, root=root,
                                        directory='generate', N=N, L=L)

    if cross_sample:

        first_net = next(iter(shells.values()))
        testset, transformer = (first_net['net'].training_parameters[k] for k in ('set', 'transformer'))
        _, test_dataset = tl.get_dataset(testset, transformer=transformer)

        x, y = {}, {}
        
        x[testset], y[testset] = tl.get_batch(test_dataset, device=device, batch_size=cross_sample)

        oodsets = test_dataset.same_size

        for o in oodsets:
            _, ood_dataset = tl.get_dataset(o, transformer=transformer)
            x[o], y[o] = tl.get_batch(ood_dataset, device=device, batch_size=cross_sample)

        nets = [Net.load(shells[j]['dir']) for j in shells]

        batch_size = m

        # logging.info('Computing min of max batch size. Starting with %s', m)
        # for n in nets:
        #     if torch.cuda.is_available():
        #         n.to('cuda')
                
        #     batch_size = min(batch_size, n.compute_max_batch_size(batch_size=m, which='test'))
        #     logging.info(f'Done for {n.job_number}: {batch_size}')

        div, y_pred = {}, {}
        
        for s in x:

            logging.info('Comparing nets %s for set %s', ' '.join(str(n.job_number) for n in nets), s)
            div[s], y_pred[s] = comparison(x[s], *nets, batch_size=batch_size, root=root, directory=s)

        print('Divergence')

        thr = {j :{} for j in div[testset]}
        for j in div[testset]:
            for jj in div[testset][j]:
                thr[j][jj] = np.quantile(div[testset][j][jj].cpu(), 0.9)
                
        for s in x:
            print('Set', s)
            for j in div[s]:
                for jj in div[s][j]:
                    d = div[s][j][jj].mean()
                    dv = div[s][j][jj].std()
                    fpr = float(sum(div[s][j][jj] <= thr[j][jj])) / len(div[s][j][jj])
                    q = np.quantile(div[s][j][jj].cpu(), [0.1, 0.5, 0.9])
                    # print(f'{j} -- {jj}: {d:7.2e} +- {dv:7.2e}')
                    print(f'{j} -- {jj}:',
                          ' - '.join(f'{_:7.2e}' for _ in q),
                          f'FPR: {100*fpr:.1f}%')

        print('Mismatch of detection')
        for s in x:
            print('Set', s)
            for j in div[s]:
                for jj in div[s][j]:
                    r = 1 - float(sum(y_pred[s][j] != y_pred[s][jj])) / len(y_pred[s][j])
                    print(f'{j} -- {jj}: {100*r:.1f}%')

        
