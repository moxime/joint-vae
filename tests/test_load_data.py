from __future__ import print_function

import torch
import utils.torch_load as tl
import time

dataset = 'letters'
dataset = 'svhn'
dataset = 'lsunr'
dataset = 'cifar10+3'
dataset = 'mnist'
dataset = 'imagenet21k'

transformer = 'default'

da = ['flip', 'crop']
da = []

splits = ['test']
splits = ['train', 'test']
train, test = tl.get_dataset(dataset, data_augmentation=da, transformer=transformer, splits=splits)

print('*** TRAIN')
print(train)

print('*** TEST')
print(test)

x, y = tl.get_batch(test)
print(y.max().item())
tl.show_images(test, num=4)
