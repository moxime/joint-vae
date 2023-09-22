from utils.torch_load import MixtureDataset

names = ('ind__', 'ood_a', 'ood_b')
sizes = (50000, 11100, 2700)

wanted_length = 200

mix_ood = (1, 1)

s = {_: [('{}_{:03}'.format(_, i), i % 10) for i in range(n)] for _, n in zip(names, sizes)}


ood = MixtureDataset(ood_a=s['ood_a'], ood_b=s['ood_b'], mix=mix_ood)

mix_ind = (4, 1)
mix = MixtureDataset(ind=s['ind__'], ood=ood, mix=mix_ind, length=wanted_length)


print('=== mix')
count = {_: 0 for _ in s}
for i in range(len(mix)):

    x = mix[i][0]
    if i > len(mix) - 30:
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

ood_a = ood_._dataset.extract_subdataset('ood_a')

print('=== ood_a')

for i in range(len(ood_a)):

    if i > len(ood_a) - 10:
        print(ood_a[i][0])
