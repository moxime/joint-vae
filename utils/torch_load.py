from collections import namedtuple
from torchvision import datasets, transforms
import torchvision
import torch
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys
import os
import logging
import string
import re
import collections
import numpy as np
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
import configparser
from matplotlib import pyplot as plt
# from torch.utils.data._utils import collate

from utils.custom_sets import UniformDataset, ConstantDataset, FromNumpy
from utils.custom_sets import ImageFolderWithClassesInFile, ImageListDataset

CONF_FILE = 'data/sets.ini'

set_props = None

logger = logging.getLogger('sets')


def dataset_properties(conf_file='data/sets.ini', all_keys=True):

    global set_props

    if not set_props is None:
        return set_props

    parsed_props = configparser.ConfigParser()
    parsed_props.read(conf_file)

    properties = {}

    bool_keys = ('downloadable',)

    for s in parsed_props.sections():

        p_ = parsed_props[s]
        p = {}
        p['shape'] = tuple(int(_) for _ in p_['shape'].split())

        if 'classes_from_file' in p_:
            p['classes'] = []
            class_file = p_['classes_from_file']

            with open(class_file) as f:
                for i, line in enumerate(f):
                    if not line.startswith('#'):
                        splitted_line = line.split()
                        p['classes'].append(' '.join(splitted_line[1:]))

        elif 'classes' in p_:
            classes = p_.get('classes', '')
            if classes.startswith('$'):
                if classes == '$letters':
                    p['classes'] = list(string.ascii_lowercase)
                elif classes == '$numbers':
                    p['classes'] = [str(_) for _ in range(10)]
            elif classes:
                p['classes'] = classes.split()
        else:
            p['classes'] = None

        if p['classes']:
            p['classes'] = [_.replace('_', ' ') for _ in p['classes']]

        p['labels'] = 0 if not p['classes'] else len(p['classes'])

        if all_keys:
            keys = ('default_transform', 'pre_transform', 'target_transform', 'folder', 'kw_for_split',
                    'root', 'classes_from_file', 'downloadable', 'by_shape')
        else:
            keys = ()
        for k in keys:
            p[k] = p_.getboolean(k) if k in bool_keys else p_.get(k)

        properties[s] = p

    set_props = properties
    return properties


class LoggerAsfile(object):

    def write(self, s):
        logging.debug(s)


laf = LoggerAsfile()


@contextmanager
def suppress_stdout(log=False):
    with open(os.devnull, "w") as dev_null:
        orig = sys.stdout
        sys.stdout = laf if log else dev_null
        try:
            yield
        finally:
            sys.stdout = orig


@contextmanager
def stdout_as_debug():
    orig = sys.stdout
    sys.stdout = laf
    try:
        yield
    finally:
        sys.stdout = orig


def choose_device(device=None):
    """if device is None, returns cuda or cpu if not available.
    else returns device

    """

    if device is None:

        has_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if has_cuda else 'cpu')

    return device


target_transforms = {'y-1': lambda y: y - 1}


class DTDConcatTestVal(torch.utils.data.ConcatDataset):

    def __init__(self, *a, split='test', **kw):

        if split == 'train':
            splits = ['train']
        elif split == 'test':
            splits = ['test', 'val']
        else:
            raise ValueError('{} unknown split for DTDConcatTestVal'.format(split))

        super().__init__([datasets.DTD(*a, split=_, **kw) for _ in splits])

    @ property
    def transforms(self):
        return self.datasets[0].transforms

    @ property
    def transform(self):
        return self.datasets[0].transform

    @ property
    def target_transform(self):
        return self.datasets[0].target_transform


def create_folder_dataset(folder, classes_file):

    if classes_file:
        class Dataset(ImageFolderWithClassesInFile):

            def __init__(self, root, *a, **kw):
                super().__init__(root, classes_file, *a, **kw)

    else:
        class Dataset(ImageListDataset):

            def __init__(self, *a, **kw):
                super().__init__(folder, *a, **kw)

    return Dataset


def modify_const(cls, **added_kw):

    def const(*a, **kw):

        return cls(*a, **added_kw, **kw)

    return const


getters = {'const': ConstantDataset,
           'uniform': UniformDataset,
           'mnist': datasets.MNIST,
           'fashion': datasets.FashionMNIST,
           'letters': modify_const(datasets.EMNIST, split='letters'),
           'cifar10': datasets.CIFAR10,
           'cifar100': datasets.CIFAR100,
           'svhn': datasets.SVHN,
           'lsunc': datasets.LSUN,
           'lsunr': datasets.LSUN,
           'dtd': DTDConcatTestVal,
           'random300k': FromNumpy,
           'places365': modify_const(datasets.Places365, small=True)
           }


def get_dataset(dataset='mnist',
                transformer='default',
                data_augmentation=[],
                conf_file='data/sets.ini',
                splits=['train', 'test'],
                getters=getters,
                download=True,
                **kw):

    dataset = dataset.lower()

    rotated = dataset.endswith('90')

    if rotated:
        dataset = dataset[:-2]

    parent_set, heldout_classes = get_heldout_classes_by_name(dataset)

    set_props = dataset_properties(conf_file=conf_file, all_keys=True)[parent_set]

    first_target_transform = target_transforms.get(set_props.get('target_transform'), lambda y: y)

    if heldout_classes:
        dataset = parent_set
        C = get_shape_by_name(parent_set)[-1]
        heldin = [_ for _ in range(C) if _ not in heldout_classes]
        d = {c: i for (i, c) in enumerate(heldin)}
        d.update({_: -1 for _ in heldout_classes})
        def target_transform(y): return d.get(first_target_transform(y))

    else:
        target_transform = first_target_transform

    same_size = get_same_size_by_name(get_name_by_heldout_classes(dataset, *heldout_classes))

    pre_transforms = []
    train_transforms = []
    post_transforms = []

    if rotated:
        pre_transforms.append(transforms.Lambda(lambda img: transforms.functional.rotate(img, 90)))

    pre_transform = set_props.get('pre_transform') or ''
    post_to_tensor = True
    for t in pre_transform.split():

        if t.startswith('resize'):
            shape = t.split('-')[1:]
            if not shape:
                shape = tuple(set_props['shape'][1:])
            if len(shape) == 1:
                shape = int(shape[0])
            else:
                shape = tuple(int(_) for _ in shape)

            pre_transforms.append(transforms.Resize(shape, antialias=None))

        elif t.startswith('crop'):
            pre_transforms.append(transforms.RandomCrop(set_props['shape'][1:]))

        elif t.startswith('center-crop'):
            try:
                shape = int(t.split('-')[-1])
                shape = (shape, shape)
            except ValueError:
                shape = tuple(set_props['shape'][1:])

            pre_transforms.append(transforms.CenterCrop(shape))

        elif t.startswith('pad'):
            try:
                pad = int(t.split('-')[-1])
            except ValueError:
                pad = 2
            pre_transforms.append(transforms.Pad(2))

        elif t.startswith('rotate'):
            angle = int(t.split('-')[-1])
            pre_transforms.append(lambda img: transforms.functional.rotate(img, angle))

        elif t == 'hflip':
            pre_transforms.append(lambda img: transforms.functional.hflip(img))

        elif t == 'g2c':
            pre_transforms.append(lambda x: x.repeat(3, 1, 1))

        elif t == 'tensor':
            post_to_tensor = False
            pre_transforms.append(transforms.ToTensor())

        elif t == 'already_tensor':
            post_to_tensor = False

    for t in data_augmentation:
        if t == 'flip':
            t_ = transforms.RandomHorizontalFlip()

        if t == 'crop':
            size = set_props['shape'][1:]
            padding = 0 if 'imagenet' in dataset else size[0] // 8
            t_ = transforms.RandomCrop(size, padding=padding, padding_mode='edge')

        train_transforms.append(t_)

    if transformer == 'default':
        transformer = set_props['default_transform']

    if transformer == 'crop':
        post_transforms.append(transforms.CenterCrop(set_props['shape'][1:]))

    elif transformer == 'pad':
        post_transforms.append(transforms.Pad(2))

    if post_to_tensor:
        post_transforms.append(transforms.ToTensor())

    if set_props.get('folder'):
        getter = create_folder_dataset(set_props.get('folder'), set_props.get('classes_from_file'))
    else:
        i = len(dataset)
        while dataset[:i] not in getters:
            i -= 1
        if i:
            getter = getters[dataset[:i]]
        else:
            raise KeyError(dataset)
    root = set_props['root']
    directory = root  # os.path.join(root, parent_set)

    train_kw = {}
    test_kw = {}

    if set_props.get('folder'):
        root = set_props['folder']
        kw_ = (set_props.get('kw_for_split') or 'root {}/train {}/test').format(root, root).split()
        train_kw[kw_[0]] = kw_[1]
        test_kw[kw_[0]] = kw_[2]

    elif set_props.get('kw_for_split'):
        kw_ = set_props['kw_for_split'].split()
        train_kw = {}
        train_kw[kw_[0]] = kw_[1]
        test_kw = {}
        test_kw[kw_[0]] = kw_[2]
        for _ in train_kw, test_kw:
            _['root'] = directory
    elif set_props.get('by_shape'):
        train_kw = {'shape': set_props['shape']}
        test_kw = {'shape': set_props['shape']}
    else:
        train_kw = dict(train=True, root=directory)
        test_kw = dict(train=False, root=directory)

    if not set_props.get('folder') and set_props.get('downloadable'):
        train_kw['download'] = True and download
        test_kw['download'] = True and download

    with suppress_stdout(log=True):
        dsets = {'train': None, 'test': None}
        if 'train' in splits:
            dsets['train'] = getter(**train_kw,
                                    target_transform=target_transform,
                                    transform=transforms.Compose(pre_transforms + train_transforms + post_transforms))
        if 'test' in splits:
            dsets['test'] = getter(**test_kw,
                                   target_transform=target_transform,
                                   transform=transforms.Compose(pre_transforms + post_transforms))

    if set_props.get('classes_from_file'):
        for s in splits:
            dsets[s].classes_file = set_props['classes_from_file']

    for s in splits:
        dsets[s].name = dataset + ('90' if rotated else '')
        dsets[s].same_size = same_size
        dsets[s].transformer = transformer

        # if not hasattr(s, 'classes'):
        C = set_props['labels']
        if set_props.get('classes') or not hasattr(dsets[s], 'classes'):
            dsets[s].classes = set_props.get('classes', [str(i) for i in range(C)])

        dsets[s].heldout = []
        if heldout_classes:
            dsets[s].heldout = heldout_classes
            dsets[s].classes = [c for (i, c) in enumerate(dsets[s].classes) if i not in heldout_classes]
            if len(heldout_classes) < C / 2:
                dsets[s].name = dsets[s].name + '-' + '-'.join(str(_) for _ in heldout_classes)
            else:
                dsets[s].name = dsets[s].name + '+' + '+'.join(str(_) for _ in range(C) if _ not in heldout_classes)

        for label_attr in ('targets', 'labels'):
            if hasattr(dsets[s], label_attr):
                break
        else:
            label_attr = None

        for data_attr in ('data', 'imgs', 'samples'):
            if hasattr(dsets[s], data_attr):
                break
        else:
            data_attr = None

        # assert data_attr and label_attr, "You have to find out the attrs of {}".format(dataset)
        if not (data_attr and label_attr):
            logging.warning('Does not konw label/data attr of {}'.format(dataset))

        elif dsets[s].target_transform:
            labels = getattr(dsets[s], label_attr)
            y = torch.tensor([dsets[s].target_transform(int(_)) for _ in labels], dtype=int)
            if (y >= 0).sum() < len(y):
                setattr(dsets[s], data_attr, getattr(dsets[s], data_attr)[y >= 0])

            if isinstance(labels, (torch.Tensor, np.ndarray)):
                setattr(dsets[s], label_attr, labels[y >= 0])
            elif isinstance(labels, list):
                setattr(dsets[s], label_attr,
                        [_ for _ in labels if dsets[s].target_transform(_) >= 0])
            else:
                raise TypeError

    return dsets['train'], dsets['test']


def get_mnist(**kw):

    return get_dataset(dataset='MNIST', **kw)


def get_fashion_mnist(**kw):

    with suppress_stdout():
        print('get it')
        return get_dataset(dataset='fashion', **kw)


def get_svhn(**kw):

    return get_dataset(dataset='svhn', **kw)


def get_cifar10(**kw):

    return get_dataset(dataset='cifar10', **kw)


def get_batch(dataset, shuffle=True, batch_size=100, device=None):

    manual_seed = False
    if not isinstance(shuffle, bool):
        initial_seed = torch.seed()
        torch.manual_seed(shuffle)
        manual_seed = True
        shuffle = True
    loader = torch.utils.data.DataLoader(dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size)

    data = next(iter(loader))

    device = choose_device(device)

    if manual_seed:
        torch.manual_seed(initial_seed)

    x, y = data

    return data[0].to(device), data[1].to(device)


def get_shape(dataset):

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1)

    data = next(iter(loader))

    num_labels = len(dataset.classes)

    return tuple(data[0][0].shape), num_labels


def get_shape_by_name(set_name, transform='default', conf_file='data/sets.ini'):

    set_props = dataset_properties(conf_file)

    if set_name.endswith('90'):
        shape, labels = get_shape_by_name(set_name[:-2])
        shape = (shape[0], shape[2], shape[1])
        return shape, labels

    set_name, heldout = get_heldout_classes_by_name(set_name)

    if set_name not in set_props:
        return None, None

    shape = set_props[set_name]['shape']
    num_labels = set_props[set_name]['labels'] - len(heldout)
    if transform != 'pad':
        return set_props[set_name]['shape'], num_labels
    p = 2
    if len(shape) == 3:
        return (shape[0], shape[1] + 2 * p, shape[2] + 2 * p), num_labels


def get_same_size_by_name(set_name, rotated=False, conf_file='data/sets.ini'):

    set_props = dataset_properties(conf_file=conf_file)

    if set_name.endswith('-?'):
        return [set_name[:-2] + '+?']

    if set_name.endswith('90'):
        return get_same_size_by_name(set_name[:-2], rotated=True)

    parent_set, heldout = get_heldout_classes_by_name(set_name)
    if heldout:
        C = get_shape_by_name(parent_set)[-1]
        new_heldout = [_ for _ in range(C) if _ not in heldout]
        return [get_name_by_heldout_classes(parent_set, *new_heldout)]

    if set_name not in set_props:
        return []

    shape, _ = get_shape_by_name(set_name)
    same_size = [s for s in set_props if set_props[s]['shape'] == shape]
    if not rotated:
        same_size.remove(set_name)
        same_size.append(set_name + '90')

    return same_size


def get_classes_by_name(dataset, texify=False):

    if texify:
        def t(k): return str(k).replace('_', '-')
    else:
        def t(k): return k

    if dataset.endswith('90'):
        return get_classes_by_name(dataset[:-2])
        # return [_ + '-90' for _ in get_classes_by_name(dataset[:-2])]
    parent_set, ho = get_heldout_classes_by_name(dataset)

    dp = dataset_properties()[parent_set]

    parent_classes = dp.get('classes') or [parent_set]

    return [t(_) for i, _ in enumerate(parent_classes) if i not in ho]


def get_heldout_classes_by_name(dataset):

    if '-' in dataset:
        set_names = dataset.split('-')
        heldout_classes = [int(_) for _ in set_names[1:]]
        heldout_classes.sort()
        return set_names[0], heldout_classes

    if '+' in dataset:
        set_names = dataset.split('+')
        parent_set = set_names[0]
        C = get_shape_by_name(parent_set)[-1]
        heldout_classes = [_ for _ in range(C) if str(_) not in set_names]
        return parent_set, heldout_classes

    return dataset, []


def get_name_by_heldout_classes(dataset, *heldout):

    if not heldout:
        return dataset
    C = get_shape_by_name(dataset)[-1]
    heldout = sorted(heldout)

    if len(heldout) / C > 0.5:
        return dataset + '+' + '+'.join(str(_) for _ in range(C) if _ not in heldout)

    return dataset + '-' + '-'.join(str(_) for _ in heldout)


def get_dataset_from_dict(dict_of_sets, set_name, transformer):

    try:
        sets = dict_of_sets[set_name][transformer]
        logging.debug(f'{set_name} with {transformer} already loaded')
        # print('**** torch_load:226', len(sets))
    except KeyError:
        sets = get_dataset(set_name, transformer=transformer)
        logging.debug(f'Getting {set_name} with transform {transformer}')
        # print('**** torch_load:230', len(sets))
        if set_name not in dict_of_sets:
            dict_of_sets[set_name] = {}
        dict_of_sets[set_name][transformer] = sets
    return sets


def show_images(imageset, shuffle=True, num=4, ncols=4, **kw):

    if isinstance(imageset, str):
        imageset = get_dataset(imageset, splits=['test'])[1]

    batch_size = num
    loader = torch.utils.data.DataLoader(imageset,
                                         shuffle=shuffle,
                                         batch_size=batch_size)

    x, y = next(iter(loader))

    nrows = int(np.ceil(num / ncols))

    fix, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    for i, image in enumerate(x):
        r = i // ncols
        c = i - r * ncols
        img = transforms.functional.to_pil_image(image)
        axs[r, c].imshow(np.asarray(img), cmap='gray' if x.shape[-3] == 1 else None)

        try:
            label = imageset.classes[y[i]]
        except (AttributeError, TypeError):
            label = str(y[i].numpy())
        except IndexError:
            label = i
        axs[r, c].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title=label)
    fix.show()

    nrows = int(np.ceil(num / ncols))


def export_png(imageset, directory, by_class=False):

    loader = torch.utils.data.DataLoader(imageset)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    classes = imageset.classes

    if by_class:
        for c in classes:
            try:
                os.makedirs(os.path.join(directory, c))
            except FileExistsError:
                pass

    i = 0
    for x, y in loader:

        for image_tensor, c in zip(x, y):
            image_dir = os.path.join(directory, classes[c]) if by_class else directory
            if not os.path.isdir(image_dir):
                os.makedirs
            filename = os.path.join(image_dir, f'{i:05}.png')
            print(i, c, *image_tensor.shape)
            save_image(image_tensor, filename)
            i += 1


np_str_obj_array_pattern = re.compile(r'[SaUO]')

collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_err_msg_format.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return elem_type({key: collate([d[key] for d in batch]) for key in elem})
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return elem_type([collate(samples) for samples in transposed])

    raise TypeError(collate_err_msg_format.format(elem_type))


if __name__ == '__main__':

    import sys

    # plt.set_loglevel(level='warning')
    # import time

    # data = 'cifar10'
    # data = 'svhn'

    # for data in ('cifar10', 'svhn'):

    #     dset, _ = get_dataset(data, splits=['train'])

    #     batch = get_batch(dset, batch_size=int(1e6))[0]

    #     print('{}: {} samples'.format(data, batch.shape[0]))

    #     mean, std = batch.mean((0, 2, 3)), batch.std((0, 2, 3))

    #     for k, v in zip(('mean', 'std'), (mean, std)):
    #         print('{:4}: {}'.format(k, ', '.join('{:.4f}'.format(_) for _ in v)))
    trainset, testset = get_dataset('o_places365', splits=['test'])

    # dset = SubSampledDataset(trainset, length=200, seed=10, task=1)

    # mset = MixtureDataset(dset, dset)

    # x, y = get_batch(dset, shuffle=False, batch_size=32)

    # print(' '.join(map('{}'.format, y[:10])))

    show_images(testset, num=16, shuffle=True)

    if sys.argv:
        input()
