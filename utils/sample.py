"""
Use a network ti generate new images, use the sampling of Z

"""
from cvae import ClassificationVariationalNetwork as Net
import data.torch_load as tl
from utils.save_load import find_by_job_number
import torch

import numpy as np
from matplotlib import pyplot as plt
import logging
import os
from utils.save_load import job_to_str, LossRecorder
from utils.inspection import latent_distribution

from torchvision.utils import save_image

import argparse


def sample(net, x=None, y=None, axis=None, root='results/%j/samples', directory='test',
           N=20, L=10, iteration=False):

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

        x_grid = {'name': 'grid.png',
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
                                  
    elif net.is_cvae or net.is_jvae:
    
        (D, H, W) = net.input_shape[-3:]
        K = net.latent_dim

        z = torch.randn(L, N, K, device=net.device)

        if net.is_cvae:
            z = z + net.encoder.latent_dictionary.unsqueeze(0)

        x_grid = {'name': 'grid.png',
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
    
           
if __name__ == '__main__':

    j = 111028
    j = 109103
    j = 37
    j = 111620
    
    device = 'cuda'

    reload = True
    reload = False

    N = 10
    L = 10
    batch_size = 256
    z_sample = 512
    
    root = '/tmp/%j'
    root = 'results/%j/samples'
    

    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', type=int,
                        nargs='+',
                        default=[j],)
    parser.add_argument('-m', '--batch-size', type=int, default=batch_size)
    parser.add_argument('-L', '--grid-width', type=int, default=L)
    parser.add_argument('-N', '--grid-height', type=int, default=N)
    parser.add_argument('-D', '--directory', default=root)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--z-sample', type=int, default=z_sample)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--sync', action='store_true')
    
    a = parser.parse_args()

    if a.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    jobs = a.jobs
    L = a.grid_width
    N = a.grid_height
    m = a.batch_size
    root = a.directory
    z_sample = a.z_sample
    bins = a.bins

    if a.sync:
        r = LossRecorder(1)
        computed_max_batch_size = False
    
    shells = find_by_job_number('jobs', *jobs, load_net=False)

    for j in shells:

        x = dict()
        y = dict()
        print('loading state of', j, flush=True, end='...')

        net = Net.load(shells[j]['dir'])
        net.to(device)
        print('done')
        print('Compute max batch size', flush=True, end='...')

        batch_size = min(m, net.compute_max_batch_size(batch_size=m, which='test'))
        print(f'done ({batch_size})')

        if a.sync and computed_max_batch_size and  batch_size != computed_max_batch_size:
            print("Let's assume you know what you're doing")
            
        computed_max_batch_size = batch_size

        z_sample = z_sample // batch_size * batch_size

        testset, transformer = (shells[j]['net'].training[k] for k in ('set', 'transformer'))
        _, test_dataset = tl.get_dataset(testset, transformer=transformer)

        r.init_seed_for_dataloader()
        
        x[testset], y[testset] = tl.get_batch(test_dataset, device=device, batch_size=z_sample)

        oodsets = test_dataset.same_size

        for o in oodsets:
            _, ood_dataset = tl.get_dataset(o, transformer=transformer)
            x[o], y[o] = tl.get_batch(ood_dataset, device=device, batch_size=z_sample)

        if z_sample:
            mu_z = {s: torch.zeros(z_sample, net.latent_dim) for s in x}
            var_z = {s: torch.zeros(z_sample, net.latent_dim) for s in x}
            log_var_z = torch.zeros(z_sample, net.latent_dim)
            mse = torch.zeros(z_sample)
            
        for s in x:
            print('sampling', s)

            list_of_images = sample(net, x[s][:N], root=root,
                                    directory=s,
                                    N=N, L=L)

            _q = [0.05, 0.1, 0.5, 0.9, 0.95]

            print('qtl: ' + ' -- '.join(['{:8.2e}'] * len(_q)).format(*_q))
            for b in range(z_sample // batch_size):

                start = b * batch_size
                end = start + batch_size

                with torch.no_grad():
                    out = net.evaluate(x[s][start:end], z_output=True)

                x_,  _, _, _, mu_z[s][start:end], log_var_z[start:end], _ = out

                d_ = [_ for _ in range(1, x[s].dim())]
                
                mse[start:end] = (x_[0] - x[s][start:end]).pow(2).mean(d_)

            q = np.quantile(mse.cpu(), _q)
            print('mse: ' + ' -- '.join(['{:8.2e}'] * len(q)).format(*q))
            var_z[s] = log_var_z.exp()

            dir_path = os.path.join(job_to_str(net.job_number, root), s)

            f = os.path.join(dir_path, 'hist_var_z.dat')
            latent_distribution(mu_z[s], var_z[s], result_type='hist_of_var',
                                bins=bins, per_dim=True, output=f)

            f = os.path.join(dir_path, 'mu_z_var_z.dat')
            latent_distribution(mu_z[s], var_z[s], result_type='scatter', per_dim=True, output=f)

        list_of_images = sample(net, root=root,
                                directory='generate', N=N, L=L)
