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
    ftable.write(' '.join(f'x_out_{l:0{wL}} mse_out_{l:0{wL}}' for l in range(l, L)))
    """
                  
    if x is not None:

        (D, H, W) = net.input_shape[-3:]

        x_grid = {'name': 'grid.png', 'tensor': torch.zeros((D, 0, (L + 2) * W), device=x.device)}
                             
        with torch.no_grad(): 
            x_, y_, mu, log_var, z = net(x, y)

        list_of_images = [x_grid]
        
        for row in range(len(x)):
            x_row = torch.zeros((D, H, 0), device=x.device)
            list_of_images.append({'name': f'x{row:0{wN}}_in.png', 'tensor': x[row]})
            x_row = torch.cat([x_row, x[row]], 2)
                              
            list_of_images.append({'name': f'x{row:0{wN}}_out_average.png',
                                   'tensor': x_[1:].mean(0)[row]})
            
            x_row = torch.cat([x_row, x_[1:].mean(0)[row]], 2)

            list_of_images.append({'name': f'x{row:0{wN}}_out_mean.png', 'tensor': x_[0][row]})
            x_row = torch.cat([x_row, x_[0][row]], 2)            
                
            for l in range(1, L):
                list_of_images.append({'name': f'x{row:0{wN}}_out_{l:0{wL}}.png', 'tensor': x_[l, row]})
                x_row = torch.cat([x_row, x_[l, row]], 2)
                
            if row < N:
                x_grid['tensor'] = torch.cat([x_grid['tensor'], x_row], 1)    
                                  
    elif net.is_cvae or net.is_jvae:
    
        (D, H, W) = net.input_shape[-3:]
        K = net.latent_dim

        z = torch.randn(L, N, K, device=net.device)

        if net.is_cvae:
            z = z + net.encoder.latent_dictionary.unsqueeze(0)

        x_grid = {'name': 'grid.png', 'tensor': torch.zeros((D, 0, L * W), device=net.device)}
        list_of_images = [x_grid]

        x_ = net.imager(net.decoder(z))
        # print(*x_.shape)        

        for row in range(N):

            x_row = torch.zeros((D, H, 0), device=net.device)
                
            for l in range(L):
                list_of_images.append({'name': f'x{row:0{wN}}_out_{l:0{wL}}.png',
                                       'tensor': x_[l, row]})
                x_row = torch.cat([x_row, x_[l, row]], 2)
                
            if row < N:
                x_grid['tensor'] = torch.cat([x_grid['tensor'], x_row], 1)    
        
        
    elif net.is_vae:
        pass
        
    else:
        raise ValueError(f'You try to generate images with a net which is {net.type}')


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

    N = 20

    logging.getLogger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("j", type=int, default=j)
    j = parser.parse_args().j
    
    try:
        if reload: del net
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
        x['i'], y['i'] = tl.get_batch(testset,device=device, batch_size=N)
        _, oodset = tl.get_dataset(testset.same_size[0], transformer=transformer)
        x['o'], y['o'] = tl.get_batch(oodset,device=device, batch_size=N)
        net.to(device)

    L = 16
    list_of_images = sample(net, x['i'], # root='/tmp/%j',
                            directory='test',
                            N=10, L=16)
    list_of_images = sample(net, x['o'], # root='/tmp/%j',
                            directory='ood', N=20, L=16)
    list_of_images = sample(net, # root='/tmp/%j',
                            directory='generate', N=20, L=16)
