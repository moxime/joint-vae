from __future__ import print_function

import torch
import utils.torch_load as tl
import time

train, test = tl.get_dataset('mnist', data_augmentation=['flip', 'crop'], transformer='pad')

print('*** TRAIN')
print(train)

print('*** TEST')
print(test)
