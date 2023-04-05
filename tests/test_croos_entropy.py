import torch
import numpy as np 
import sys
import torch.nn.functional as F
import time

sys.path.append('../')
from vae_layers import Sampling

device = torch.device('cuda')
device = torch.device('cpu')

D = (3, 2)
C = 5 # num of classes
N = (13, 11) # size of dataset
L = 7 # size of latent sampling

t = time.time()
y_in = (torch.LongTensor(*N, 1).random_() % C).to(device)
y_out = torch.randn(L, *N, C).softmax(dim=-1).to(device)
print(f'{(time.time() - t)*1e3:.0f} ms')


# C = 3
# N = (2,)
# y_in = torch.LongTensor([0, 1])
# y_out = torch.Tensor([[0.5, 0, 0.5], [0.5, 0.25, 0.25]]).reshape(1, *N, C)  # 


time_test = False
dataset_size = 60000
if time_test:
    t = time.time()
    for _ in range(dataset_size//np.prod(N)):
        y_in_repeated = y_in.repeat(L, 1)
        loss = F.nll_loss(y_out.log(), y_in_repeated, reduction='none')
    print(f'{(time.time() - t)*1e3:.0f} ms')

    loss.size()

    
one_dim = ()
for _ in y_in.squeeze_().shape: one_dim += (1,)
y_in_repeated = y_in.reshape((1,) + N).repeat(L, *one_dim)

print('y_in: ', y_in.shape)
print('y_in_repeated: ', y_in_repeated.shape)
print('y_out: ', y_out.shape)

dims = tuple(_ for _ in range(y_out.dim()))
out_perm_dim = (0,) + (-1,) + dims[1:-1] # from LxN1...xNgxC to LxCxN1...xNgxC


loss = F.nll_loss(y_out.permute(out_perm_dim).log(), y_in_repeated,
                  reduction='none')

# print(loss/np.log(2))
print(loss.shape)
