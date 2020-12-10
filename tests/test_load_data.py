from __future__ import print_function
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt

import data.torch_load as torchdl
import numpy as np

device = torch.device('cpu')
batch_size = 200

trainset, testset = torchdl.get_dataset('letters', transformer='pad')

x, y = torchdl.get_batch(trainset, batch_size=batch_size, device=device)


plt.imshow(x[0, 0])

shape = (6, 10)

f, axis = plt.subplots(*shape)

for i, a in enumerate(axis.flat):

    im = x[i, 0, :, :]
    label = trainset.classes[y[i]-1]

    a.imshow(im, cmap='gray')
    a.set_title(label)
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

    print(label, end=' ')
    
f.show()

