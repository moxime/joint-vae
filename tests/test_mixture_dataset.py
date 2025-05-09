import logging
import torch
from torch.utils.data import Dataset
from utils.torch_load import MixtureDataset, EstimatedLabelsDataset, get_dataset, collate
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.INFO)


class ListofTensors(list):

    def to(self, device):

        return ListofTensors(_.to(device) for _ in self)


class FooDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(self):
        return 10000

    def __getitem__(self, i):

        return ListofTensors([torch.randn(4), 0, 1])


names = ('ind__', 'ood_a', 'ood_b')
sizes = (50000, 11100, 530)

wanted_length = 512

mix_ood = (1, 1)
mix_ind = 1

s = {_: [('{}_{:05}_{}'.format(_, i, i % 10), i % 10) for i in range(n)] for _, n in zip(names, sizes)}


ood = MixtureDataset(ood_a=s['ood_a'], ood_b=s['ood_b'], mix=mix_ood, shift_key=94)

mix = MixtureDataset(ind=s['ind__'], ood=ood, mix={'ind': mix_ind,
                                                   'ood': 1 - mix_ind}, length=wanted_length, shift_key=67)


print('=== mix')
count = {_: 0 for _ in s}
for i in range(len(mix)):

    x = mix[i][0]
    if i > len(mix) - 50:
        print(x)
    for _ in count:
        if _ in x:
            count[_] += 1

print('=== count')
count['total'] = sum(count.values())

th_count = {'ind__': mix_ind}
for _, m in zip(ood.classes, mix_ood):
    th_count[_] = (1 - th_count['ind__']) * m / sum(mix_ood)
    th_count['total'] = 1

print('\n'.join('{:5}:{:5} {:6.1%} ({:6.1%})'.format(k, v, v / count['total'], th_count[k])
                for k, v in count.items()))


ood_ = mix.extract_subdataset('ood')

print('=== ood')

for i in range(len(ood_)):

    if i > len(ood_) - 10:
        print(ood_[i][0])

ood_b = ood_.extract_subdataset('ood_b')

print('=== ood_b')

count = {_: 0 for _ in range(10)}
for i in range(len(ood_b)):
    x, y = ood_b[i]
    count[y] += 1
    if i > len(ood_b) - 10:
        print(ood_b[i][0])

print('\n'.join('{}:{:5} {:6.1%}'.format(k, v, v / (len(ood_b) + 1e-30))
                for k, v in count.items()))


# print('===REAL DATASETS')

# ood = MixtureDataset(lsunr=get_dataset('lsunr', splits=['test'])[1])

# print('***', len(ood))
# mix2 = MixtureDataset(ind=get_dataset('cifar10')[1], ood=ood, mix=(0.8, 0.2), length=2999)

print('=== ESTIMATED LABELS')

ood_b_y = EstimatedLabelsDataset(ood_b)

ood_b_y.append_estimated(torch.ones(len(ood_b), dtype=int))

ood_b_y.return_estimated = True

loader = torch.utils.data.DataLoader(FooDataset(), batch_size=16, collate_fn=collate)
loader = torch.utils.data.DataLoader(ood_b_y, batch_size=16, collate_fn=collate)

try:
    d = next(iter(loader))
except StopIteration:
    print('Empty dataset')
