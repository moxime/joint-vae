from __future__ import print_function

import torch
import utils.torch_load as tl
import time

dataset = 'mnist+2+4'
dataset = 'letters'
dataset = 'cifar10+3'
dataset='svhn'
dataset = 'lsunr'

transformer = 'default'

da = ['flip', 'crop']
da = []

splits = ['train', 'test']
splits = ['test']
train, test = tl.get_dataset(dataset, data_augmentation=da, transformer=transformer, splits=splits)

print('*** TRAIN')
print(train)

print('*** TEST')
print(test)


tl.show_images(test, num=64)
