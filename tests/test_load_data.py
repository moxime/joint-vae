from __future__ import print_function

import torch
import data.torch_load as torchdl
import time

device = torch.device('cpu')
device = torch.device('cuda')
shuffle = False
shuffle = True
batch_size = 512
pin_memory = False
pin_memory = True
transformer = 'pad'
transformer = 'default'

trainset, testset = torchdl.get_dataset('letters', transformer=transformer)
trainset, testset = torchdl.get_dataset('lsunc', transformer=transformer)

testloader = torch.utils.data.DataLoader(testset,
                                         pin_memory=pin_memory,
                                         num_workers=10,
                                         batch_size=batch_size,
                                         shuffle=shuffle)

test_iterator = iter(testloader)

nbatch = len(testset) // batch_size

t0 = time.time()

for i in range(nbatch):

    data = next(test_iterator)
    x, y = (d.to(device) for d in data)

    t = time.time() - t0
    print(f'\r{t / (i + 1) / batch_size * 1e6:6.0f} mus / i', end='')

print()
