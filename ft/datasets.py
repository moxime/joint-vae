from itertools import accumulate

import numpy as np

from torch.utils.data import Dataset
import torch

import utils.torch_load as torchdl

import logging

logger = logging.getLogger('sets')


class ListofTensors(list):

    def to(self, device):

        return ListofTensors(_.to(device) for _ in self)


class EstimatedLabelsDataset(Dataset):

    def __init__(self, dataset):

        super().__init__()

        self._dataset = dataset

        self._return_estimated = False

        self._estimated_labels = []

    @property
    def name(self):
        return self._dataset.name

    @property
    def return_estimated(self):
        return self._return_estimated

    @return_estimated.setter
    def return_estimated(self, b):
        assert not b or len(self) == len(self._estimated_labels), 'You did not collect etimated labels'
        self._return_estimated = b

    def append_estimated(self, y_):
        if isinstance(y_, torch.Tensor):
            y_ = y_.cpu()
        tmp_s = 'Populating estimated laels with {} labels (dataset of length {})'
        logging.info(tmp_s.format(len(y_), len(self)))
        self._estimated_labels += list(np.asarray(y_))

    def __len__(self):

        return len(self._dataset)

    def __getitem__(self, i):

        x, y = self._dataset[i]

        if self.return_estimated:
            x = ListofTensors([x, self._estimated_labels[i]])

        return x, y


class SubSampledDataset(Dataset):

    def __init__(self, dataset, length=None, seed=0, task=0):
        """Args:

        -- dataset

        -- length: wanted length

        -- seed: seed for randomization

        -- task:

            -- if None (default): will statrify dataset in *length*
               bundle and will take one sample per bundle (randomly
               with seed seed).

            -- if int takes task-th batch of dataset that has been
               sliced in batches of size length after random
               reordering

        """
        super().__init__()

        d_str = 'Create subdataset with {} ({}) of len {} with seed {} and task {}'
        logger.debug(d_str.format(getattr(dataset, 'name', None), len(dataset), length, seed, task))
        self._dataset = dataset
        self._seed = seed
        self.maxlength = len(dataset)
        self._task = task

        self._length = self.maxlength

        self.name = 'sub-{}'.format(getattr(self._dataset, 'name', 'dataset'))

        self._bar = False

        self.shrink(length)

        try:
            dataset_name = dataset.name
        except AttributeError:
            dataset_name = 'unknown'

        i_str = 'Creating dataset from {} of length {} with seed {} (task {})'
        logging.info(i_str.format(dataset_name, len(self), seed, self._task))
        try:
            self.classes = dataset.classes
        except AttributeError:
            pass

    def bar(self, b=True):

        if b:
            if not self._bar:
                self._original_length = len(self)
                self._bar = True
                self.shrink()
        else:
            if self._bar:
                self._bar = False
                self.shrink(self._original_length)

    def _create_idx(self):

        rng = np.random.default_rng(self._seed)

        if self._task >= self._num_batches:
            w_str = 'Batch # {} >= {} sample_every'
            logger.debug(w_str.format(self._task, self._num_batches))

        self._idx = list(rng.permutation(self.maxlength))

        logger.debug('Created idx for {} of size {}. Firsts: {}'.format(self.name, len(self._idx),
                                                                        '-'.join(map(str, self._idx[: 10]))))

    def shrink(self, length=None):

        if length is None:
            if not self._bar:
                length = len(self._dataset)
            else:
                length = len(self._dataset) - self._original_length

        if not length:
            self._length = 0
            return

        if self._bar:
            length = min(length, self.maxlength - self._original_length)
            self._num_batches = len(self._dataset) // self._original_length

        else:
            length = min(length, self.maxlength)
            self._num_batches = len(self._dataset) // length

        old_length = self._length
        self._length = length
        logging.info('Shrunk dataset {} {} to {}'.format(getattr(self, 'name', 'unknowmn'),
                                                         old_length,
                                                         len(self)))
        self._create_idx()

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # if idx.stop >= self._length:
            #     raise IndexError
            raise NotImplementedError('slices for subdataset')
        if idx >= self._length:
            raise IndexError

        shift = (self._task + self._bar) * (self._original_length if self._bar else self._length)

        return self._dataset[self._idx[(idx + shift) % len(self._idx)]]


class MixtureDataset(Dataset):

    def __init__(self, *datasets, mix=None, length=None, seed=0, task=None, **dict_of_datasets):

        super().__init__()
        assert not datasets or not dict_of_datasets

        self._seed = seed
        self._task = task

        self._bar = False

        if not dict_of_datasets:
            dict_of_datasets = {getattr(d, 'name', str(i)): d for i, d in enumerate(datasets)}

        logger.debug('Create Mixture with {}'.format(','.join(dict_of_datasets)))
        self._build_classes_from_dict(**dict_of_datasets)

        self.num_datasets = len(self._datasets)

        if not mix:
            mix = [len(d) / sum(len(_) for _ in self._datasets) for d in self._datasets]

        if isinstance(mix, int):
            mix = tuple(1 / len(self._datasets) for _ in self._datasets)

        if isinstance(mix, dict):
            mix = [mix[_] for _ in self.classes]

        mix = [_ / sum(mix) for _ in mix]

        self._mix = mix

        self.maxlength = int(min(np.ceil(d.maxlength / m)
                                 for d, m in zip(self._datasets, np.array(self._mix)) if m > 0))

        self.shrink(length)

    def shrink(self, length=None):

        unit_length = int(min(np.floor(len(d) / m)
                              for d, m in zip(self._datasets, np.array(self._mix)) if m > 0))
        max_length = self.maxlength

        if length is None:
            length = unit_length
            # print('***', self.name, 'unit length:', unit_length)
        else:
            unit_length = min(unit_length, length)

        if length > max_length:
            logging.warning('Length {} non attainable, will stop at {}'.format(length, max_length))
            length = max_length
        # for i, (s, d, m) in enumerate(zip(self.classes, self._datasets, self._mix)):
        #     print('*** {} : {}/{}={}'.format(s, len(d), m, len(d) / m))

        if not length:
            self._length = 0
            self._lengths = [0 for _ in self._datasets]
            self._mix_ = self._mix
            for d in self._datasets:
                d.shrink(0)

            return

        lengths = [int(np.floor(unit_length * m)) for m in self._mix]
        target_lengths = [length * m for m in self._mix]

        # print('*** target', target_lengths)
        # print('*** lengths', lengths)
        logger.debug('Adjusting length of {}'.format(self.name))

        for d, l in zip(self._datasets, lengths):
            d.shrink(l)

        while sum(lengths) < length:
            i_d = np.argmax(np.array(np.array(target_lengths) - np.array(lengths)))
            # print('***', *lengths, i_d)
            lengths[i_d] += 1
            self._datasets[i_d].shrink(lengths[i_d])

        logger.debug('Adjusted length of {}'.format(self.name))

        self._lengths = [len(d) for d in self._datasets]
        self._length = sum(self._lengths)

        self._cum_lengths = [0] + list(accumulate(self._lengths))

        self._mix_ = [l / self._length for l in self._lengths]

        if length and 1 - self._length / length > 0.1:
            _s = 'In mixture dataset {}; wanted length: {}, maximum length: {}'
            logging.warning(_s.format('-'.join(self.classes), length, sum(self._lengths)))

    def _build_classes_from_dict(self, **datasets):

        self._classes = []
        self._datasets = []
        for _, d in datasets.items():
            self._classes.append(_)
            if isinstance(d, (MixtureDataset, SubSampledDataset)):
                self._datasets.append(d)
            else:
                self._datasets.append(SubSampledDataset(d, seed=self._seed, task=self._task))
        self.name = '-'.join(['{}:{}'.format(i, getattr(d, 'name', 'set'))
                              for i, d in enumerate(self._datasets)])

        self._classes = tuple(self._classes)

    @property
    def classes(self):
        return self._classes

    def rename(self, *a, **kw):

        assert not a or not kw, 'To rename, choose a list of name or a dict of old names: new names'
        assert not a or len(a) == len(self.classes)

        if a:
            self._classes = tuple(a)
        else:
            self._classes = tuple(kw.get(_, _) for _ in self._classes)

    def which_subsets(self, *y, which=None):

        for _ in y:
            if which:
                yield self.classes[_] == which
            else:
                yield self.classes[_]

    @property
    def subdatasets(self):
        return self._datasets

    @property
    def mix(self):
        return self._mix_

    def __len__(self):

        return self._length

    def __geti__(self, idx):

        which, l_ = max((w, l) for w, l in enumerate(self._cum_lengths) if l <= idx)

        sub_idx = idx - l_

        return which, sub_idx

    def __getitem__(self, idx):

        if not isinstance(idx, slice):
            if idx >= len(self):
                raise IndexError('Idx {} for length {}'.format(idx, len(self)))

        which, sub_idx = self.__geti__(idx)

        x, y = self._datasets[which][sub_idx]

        return x, which

    def __str__(self):

        return '\n\n'.join('Subdataset {}: {}\n{} \n({})'.format(i, n, d, len(d))
                           for i, (n, d) in enumerate(zip(self.classes, self._datasets)))

    def __repr__(self):

        return str(self)

    def extract_subdataset(self, name, new_name=None):

        i = self.classes.index(name)
        d = self._datasets[i]
        if new_name is None:
            new_name = self.classes[i]
        d.name = new_name
        return d

    def bar(self, b=True):
        for d in self._datasets:
            # self._bar ^ b is True if switch
            # (self._bar ^ b) ^ d._bar switches d._bar if switch
            d.bar((self._bar ^ b) ^ d._bar)

        self._bar = b
        self._lengths = [len(d) for d in self._datasets]
        self._length = sum(self._lengths)

        self._cum_lengths = [0] + list(accumulate(self._lengths))

        self._mix_ = [l / self._length for l in self._lengths]


def create_moving_set(ind, transformer, data_augmentation,
                      moving_size, ood_mix, oodsets,
                      padding_sets, padding=0., mix_padding=0.,
                      ood_mix_pad=0.5,
                      seed=0, task=None):

    trainset, testset = torchdl.get_dataset(ind, transformer=transformer,
                                            data_augmentation=data_augmentation)

    ood_sets = {_: torchdl.get_dataset(_, transformer=transformer, splits=['test'])[1] for _ in oodsets}

    ood_set = MixtureDataset(mix=1, seed=seed, task=task, **ood_sets, length=int(ood_mix * moving_size))
    ind_set = SubSampledDataset(testset, seed=seed, task=task, length=moving_size - len(ood_set))

    padding_sets = {_: torchdl.get_dataset(_, transformer=transformer, splits=['test'])[1]
                    for _ in padding_sets}

    for _ in padding_sets:
        if _ in oodsets:

            raise ValueError('{} is in ood sets and padding sets. Set padding_mix arg instead'.format(_))

    moving_sets = {'ood': ood_set, 'ind': ind_set}

    if padding:
        padding_set = MixtureDataset(seed=seed, task=task, **padding_sets,
                                     mix=1,
                                     length=int(padding * moving_size))
        moving_sets['pad'] = padding_set

    if mix_padding:
        padmix_sets = {}
        padmix_mix = {}
        ind_set_bar = SubSampledDataset(testset, seed=seed, task=task, length=len(ind_set))
        ind_set_bar.bar()

        ood_set_bar = MixtureDataset(mix=1, seed=seed, task=task, **ood_sets, length=len(ood_set))
        ood_set_bar.bar()

        padmix_sets['ood'] = ood_set_bar
        padmix_mix['ood'] = mix_padding * ood_mix_pad
        padmix_sets['ind'] = ind_set_bar
        padmix_mix['ind'] = mix_padding - padmix_mix['ood']

        moving_sets['padmix'] = MixtureDataset(seed=seed, task=task, **padmix_sets,
                                               mix=padmix_mix,
                                               length=int(mix_padding * moving_size))

        # for _ in ('ind', 'ood', 'padmix'):

        #     print('***', _, len(moving_sets[_]))
        # print('*** all', unique(*[moving_sets[_] for _ in ('ind', 'ood', 'padmix')]))

    moving_set = MixtureDataset(mix={_: len(moving_sets[_]) for _ in moving_sets},
                                seed=seed, task=task, **moving_sets)

    return moving_set


def inspect(dset, depth=0):

    prefix = '|  ' * depth

    print('{p} {b} {s}: {l}  ({ml})'.format(p=prefix, s=dset.name,
                                            b='*' if dset._bar else ' ',
                                            l=len(dset), ml=dset.maxlength))
    if isinstance(dset, MixtureDataset):
        for _ in dset._datasets:
            inspect(_, depth=depth + 1)


if __name__ == '__main__':

    import sys
    import argparse
    import time

    def hash_tensor(tensor):
        return hash(tuple((tensor * 2**16).int().reshape(-1).tolist()))

    def unique(*sets):

        alls = set()

        for s in sets:
            alls = alls.union([hash_tensor(_[0]) for _ in s])

        return len(alls)

    parser = argparse.ArgumentParser()

    parser.add_argument('ind')
    parser.add_argument('size', type=int)
    parser.add_argument('--oods', nargs='*')
    parser.add_argument('--mix', type=float, default=0.5)
    parser.add_argument('--pad', type=float, default=0.0)
    parser.add_argument('--pad-sets', nargs='*', default=['const32'])
    parser.add_argument('--pad-mix', default=0.0, type=float)
    parser.add_argument('--task', type=int)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    print('Creating moving set')
    t0 = time.time()
    moving_set = create_moving_set(args.ind, 'default', [], args.size, args.mix, args.oods,
                                   args.pad_sets, args.pad, mix_padding=args.pad_mix,
                                   seed=args.seed, task=args.task)

    inspect(moving_set)
    moving_set_bar = create_moving_set(args.ind, 'default', [], args.size, args.mix, args.oods,
                                       args.pad_sets, args.pad, mix_padding=args.pad_mix,
                                       seed=args.seed, task=args.task)

    print('Done (in {:.0f}s)'.format(time.time() - t0))
    t0 = time.time()

    if args.pad_mix > 0:
        check_ind = moving_set.extract_subdataset('ind')
        check_pad = moving_set.extract_subdataset('padmix')

        assert len(check_ind) + len(check_pad) == unique(check_ind, check_pad)

        print('Passed checkpoint in {:.3f}ms'.format(1000 * (time.time() - t0)))

    print('\n==BAR=')
    moving_set_bar.bar()
    inspect(moving_set_bar)

    print('total', len(moving_set))

    subsets = {_: moving_set.extract_subdataset(_) for _ in moving_set.classes}

    for _ in subsets:

        print('|_', _, len(subsets[_]))

    sets = {_: {} for _ in [args.ind] + args.oods + args.pad_sets}

    oodset = moving_set.extract_subdataset('ood')

    if args.pad:
        paddingset = moving_set.extract_subdataset('pad')

    if args.pad_mix:
        padmixset = moving_set.extract_subdataset('padmix')

    for _ in sets:
        if _ == args.ind:

            sets[_]['ind'] = moving_set.extract_subdataset('ind')

            if args.pad_mix:
                sets[_]['pad'] = padmixset.extract_subdataset('ind')

        if _ in args.oods:

            sets[_]['ood'] = oodset.extract_subdataset(_)
            if args.pad_mix:
                sets[_]['pad'] = padmixset.extract_subdataset('ood').extract_subdataset(_)

        if args.pad and _ in args.pad_sets:
            sets[_]['pad'] = paddingset.extract_subdataset(_)

    for _ in sets:
        for s in sets[_]:
            print(_, s, len(sets[_][s]))

    for s in sets:

        print(s)

        for _ in sets[s]:
            print(' -', _, len(sets[s][_]))

    for s, m in zip(moving_set.classes, moving_set.mix):
        print('{:6}'.format(s), '{:.1%}'.format(m), int(m * len(moving_set)))

    if args.pad_mix:
        for s, m in zip(padmixset.classes, padmixset.mix):
            print('|_{:6}'.format(s), '{:.1%}'.format(m), int(m * len(padmixset)))

    for s in sets:
        u = {_: unique(sets[s][_]) for _ in sets[s]}
        u['sum'] = sum(u.values())
        u['all'] = unique(*[sets[s][_] for _ in sets[s]])

        print(s, ' + '.join(map(str, list(u.values())[: -2])), '= {} ({})'.format(*list(u.values())[-2:]))

    print('=== IND')

    ind_in_moving_set = moving_set.extract_subdataset('ind')
    ind_in_moving_set_bar = moving_set_bar.extract_subdataset('ind')

    print('_*')
    print(len(ind_in_moving_set), len(ind_in_moving_set_bar), unique(ind_in_moving_set, ind_in_moving_set_bar))
    moving_set.bar()
    print('**')
    print(len(ind_in_moving_set), len(ind_in_moving_set_bar), unique(ind_in_moving_set, ind_in_moving_set_bar))
    moving_set_bar.bar(False)
    print('*_')
    print(len(ind_in_moving_set), len(ind_in_moving_set_bar), unique(ind_in_moving_set, ind_in_moving_set_bar))
    moving_set.bar(False)
    print('__')
    print(len(ind_in_moving_set), len(ind_in_moving_set_bar), unique(ind_in_moving_set, ind_in_moving_set_bar))

    print('=== OOD')

    moving_set_bar.bar(True)
    moving_set.bar(False)

    ood_in_moving_set = moving_set.extract_subdataset('ood')
    ood_in_moving_set_bar = moving_set_bar.extract_subdataset('ood')

    inspect(ood_in_moving_set)
    inspect(ood_in_moving_set_bar)

    print('_*')
    print(len(ood_in_moving_set), len(ood_in_moving_set_bar), unique(ood_in_moving_set, ood_in_moving_set_bar))
    moving_set.bar()
    print('**')
    print(len(ood_in_moving_set), len(ood_in_moving_set_bar), unique(ood_in_moving_set, ood_in_moving_set_bar))
    moving_set_bar.bar(False)
    print('*_')
    print(len(ood_in_moving_set), len(ood_in_moving_set_bar), unique(ood_in_moving_set, ood_in_moving_set_bar))
    moving_set.bar(False)
    print('__')
    print(len(ood_in_moving_set), len(ood_in_moving_set_bar), unique(ood_in_moving_set, ood_in_moving_set_bar))

    for s in args.oods:
        print('===', s)
        moving_set_bar.bar(True)
        moving_set.bar(False)
        sub_ood = ood_in_moving_set.extract_subdataset(s)
        sub_ood_bar = ood_in_moving_set_bar.extract_subdataset(s)

        print('_*')
        print(len(sub_ood), len(sub_ood_bar), unique(sub_ood, sub_ood_bar))
        moving_set.bar()
        print('**')
        print(len(sub_ood), len(sub_ood_bar), unique(sub_ood, sub_ood_bar))
        moving_set_bar.bar(False)
        print('*_')
        print(len(sub_ood), len(sub_ood_bar), unique(sub_ood, sub_ood_bar))
        moving_set.bar(False)
        print('__')
        print(len(sub_ood), len(sub_ood_bar), unique(sub_ood, sub_ood_bar))
