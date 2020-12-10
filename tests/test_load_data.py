from __future__ import print_function
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt

import data.torch_load as torchdl
import numpy as np

device = torch.device('cpu')
batch_size = 100

trainset, testset = torchdl.get_dataset('letters')

x, y = torchdl.get_batch(trainset, batch_size=batch_size)


plt.imshow(x[0, 0])

shape = (1, 2)

f, axis = plt.subplots(*shape)

for i, a in enumerate(axis.flat):

    im = x[i, 0, :, :]
    label = trainset.classes[y[i]-1]

    a.imshow(im, cmap='gray')
    a.set_title(label)

f.show()

