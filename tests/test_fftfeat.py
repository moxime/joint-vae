import argparse
from matplotlib import pyplot as plt
import PIL
import torch
from torchvision.transforms import ToPILImage
from module.fft_layers import FFTFeatures as F
from utils.torch_load import get_dataset, get_batch


dset = 'svhn'
dset = 'mnist'
device = 'cpu'
padding = 1

parser = argparse.ArgumentParser()

parser.add_argument('--set', default=dset)
parser.add_argument('--device', default=device)
parser.add_argument('-P', type=int, default=padding)

args = parser.parse_args()

dset = args.set
padding = args.P
device = args.device

dset, _ = get_dataset(dset)

x, y = get_batch(dset, device=device)

shape = x.shape[-3:]


# which = ['iphase', 'module', 'imodule', 'real', 'imag', 'phase']
which = ['phase', 'module']

m = F(shape, P=padding, which=which)

m.to(device)

f = m(x)

print('x:', *x.shape, 'f:', *f.shape)

shows = 4

fig, ax = plt.subplots(2, shows, squeeze=False)

for _ in range(shows):

    im = ToPILImage()(x[_].squeeze())
    ax[0, _].imshow(im)
    ax[1, _].imshow(f[_].to('cpu'))

fig.show()

input()
