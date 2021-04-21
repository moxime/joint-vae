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
from utils.save_load import job_to_str

from torchvision.utils import save_image

import logging
import argparse


def sample(net, x=None, y=None, axis=None, root='results/%j/samples', directory='test',
           N=20, L=10):

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
                  'tensor': torch.zeros((D, 0, (L + 2) * W), device=x.device)}
                             
        with torch.no_grad(): 
            x_, y_, mu, log_var, z = net(x, y)

        list_of_images = [x_grid]
        
        for row in range(len(x)):
            x_row = torch.zeros((D, H, 0), device=x.device)
            list_of_images.append({'name': f'x{row:0{wN}}_in.png',
                                   'tensor': x[row]})
            x_row = torch.cat([x_row, x[row]], 2)
                              
            list_of_images.append({'name': f'x{row:0{wN}}_out_average.png',
                                   'tensor': x_[1:].mean(0)[row]})
            
            x_row = torch.cat([x_row, x_[1:].mean(0)[row]], 2)

            list_of_images.append({'name': f'x{row:0{wN}}_out_mean.png',
                                   'tensor': x_[0][row]})
            x_row = torch.cat([x_row, x_[0][row]], 2)            
                
            for l_ in range(1, L):
                list_of_images.append({'name':
                                       f'x{row:0{wN}}_out_{l_:0{wL}}.png',
                                       'tensor': x_[l_, row]})
                x_row = torch.cat([x_row, x_[l_, row]], 2)
                
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
    

def losses_histogram(net, directory):

    sample_dir = os.path.join(directory, 'samples')

    if not os.path.exists(sample_dir):
        raise FileNotFoundError(sample_dir)
    
    files = os.listdir(sample_dir)
    last_sample = None
    last_epoch = 0
    for f in files:
        try:
            epoch = int(f.rsplit('.', 1)[0])
            if epoch > last_epoch:
                last_epoch = epoch
                last_sample = f
        except ValueError:
            pass

    sample_dict = torch.load(os.path.join(sample_dir, last_sample))
    return sample_dict

           
if __name__ == '__main__':

    j = 111028
    j = 109103
    j = 37
    j = 111620
    
    device = 'cuda'

    reload = True
    reload = False

    N = 20
    batch_size = 512
    root = '/tmp/%j'
    root = 'results/%j/samples'
    
    logging.getLogger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--jobs', type=int,
                        # nargs='+',
                        default=j,)
    parser.add_argument('-m', '--batch-size', type=int, default=batch_size)
    parser.add_argument('-L', '--grid-width', type=int, default=N)
    parser.add_argument('-N', '--grid-height', type=int, default=N)
    parser.add_argument('-D', '--directory', default=root)
    
    a = parser.parse_args()
    j = a.jobs
    L = a.grid_width
    N = a.grid_height
    m = a.batch_size
    root = a.directory
    
    m = max(m, N)
    
    try:
        if reload:
            del net
        net.trained
        if net.job_number != j:
            raise ValueError

    except Exception as e:
        x = dict()
        y = dict()
        print('loading state of', j, flush=True, end='...')
        shell = find_by_job_number('jobs', j, load_net=False)[j]
        net = Net.load(shell['dir'])
        print('done')
        dataset, transformer = (shell['net'].training[k] for k in ('set', 'transformer'))
        _, testset = tl.get_dataset(dataset=shell['set'], transformer=transformer)
        x['i'], y['i'] = tl.get_batch(testset,device=device, batch_size=m)
        _, oodset = tl.get_dataset(testset.same_size[0], transformer=transformer)
        x['o'], y['o'] = tl.get_batch(oodset, device=device, batch_size=m)
        net.to(device)

    list_of_images = sample(net, x['i'], root=root,
                            directory='test',
                            N=10, L=L)
    list_of_images = sample(net, x['o'], root=root,
                            directory='ood', N=10, L=L)
    list_of_images = sample(net, root=root,
                            directory='generate', N=N, L=L)
