import os
from utils import torch_load as dl
from torchvision.utils import save_image

output_dir = '/tmp/examples'

ind = 'cifar10'

ood = ['svhn', 'mnist32r', 'fashion32r', 'cifar100', 'lsunr']


ind_class = 1

batch_size = 100

sets = {_: dl.get_dataset(_, splits=['test'])[1] for _ in ood + [ind]}


batches = {_: dl.get_batch(sets[_], batch_size=batch_size) for _ in sets}

for _, s in sets.items():

    y_ = {c: 0 for c in range(len(s.classes))}

    for i in range(batch_size):

        x, y = batches[_][0][i], batches[_][1][i]
        y_[int(y)] += 1

        name = '{}_{}_{}'.format(_, y, y_[int(y)])

        save_image(x, os.path.join(output_dir, name + '.png'))
