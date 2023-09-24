from utils.torch_load import MixtureDataset, SubSampledDataset

names = ('ind__', 'ood_a', 'ood_b')
sizes = (50000, 11100, 2700)

wanted_length = 512

mix_ood = (1, 1)

s = {_: [('{}_{:05}_{}'.format(_, i, i % 10), i % 10) for i in range(n)] for _, n in zip(names, sizes)}


ood = MixtureDataset(ood_a=s['ood_a'], ood_b=s['ood_b'], mix=mix_ood)

mix_ind = (0.8, 0.2)
mix = MixtureDataset(ind=s['ind__'], ood=ood, mix=mix_ind, length=wanted_length)

# raise StopIteration

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

th_count = {'ind__': mix_ind[0] / (mix_ind[0] + mix_ind[1])}
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

print('\n'.join('{}:{:5} {:6.1%}'.format(k, v, v / len(ood_b))
                for k, v in count.items()))
