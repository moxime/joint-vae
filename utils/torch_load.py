from torchvision import datasets, transforms
import torchvision
import torch
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys
import os
import logging
import string
import numpy as np
from torchvision.utils import save_image
import configparser
from matplotlib import pyplot as plt


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


def letters_getter(**kw):

    s = datasets.EMNIST(split='letters',  **kw)
    # st0 = np.random.get_state()
    # np.random.seed(0)
    # i = np.random.permutation(len(s))
    # print(*i[:10])
    # s.targets = s.targets[i]
    # s.data = s.data[i]
    # np.random.set_state(st0)
    return s


target_transforms = {'y-1': lambda y: y-1}


class ImageFolderWithClassesInFile(datasets.ImageFolder):

    def __init__(self, root, classes_file, *a, **kw):

        logging.debug('Creating dataset in folder {} based on classes listed in {}'.format(root, classes_file))
        self.root = root
        self._classes_file = classes_file
        self._compile_dict()

        super().__init__(root, *a, **kw)

    def _compile_dict(self):
        self.node_to_idx = {}
        self.idx_to_class = {}
        self.idx_to_node = {}

        with open(self._classes_file) as f:
            i = 0
            for line in f:
                if not line.startswith('#'):
                    splitted_line = line.split()
                    node = splitted_line[0]
                    classi = ' '.join(splitted_line[1:])
                    self.node_to_idx[node] = i
                    self.idx_to_class[i] = classi
                    self.idx_to_node[i] = node
                    i += 1

            self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
            self.nodes = [self.idx_to_node[i] for i in range(len(self.idx_to_class))]

    def find_classes(self, directory):

        classes = self.nodes
        return classes, self.node_to_idx


def create_image_dataset(classes_file):

    class Dataset(ImageFolderWithClassesInFile):

        def __init__(self, root, *a, **kw):
            super().__init__(root, classes_file, *a, **kw)

    return Dataset


getters = {'mnist': datasets.MNIST,
           'fashion': datasets.FashionMNIST,
           'letters': letters_getter,
           'cifar10': datasets.CIFAR10,
           'cifar100': datasets.CIFAR100,
           'svhn': datasets.SVHN,
           'lsunc': datasets.LSUN,
           'lsunr': datasets.LSUN,
           }


def dataset_properties(conf_file='data/sets.ini', all_keys=True):

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
                    'root', 'classes_from_file', 'downloadable')
        else:
            keys = ()
        for k in keys:
            p[k] = p_.getboolean(k) if k in bool_keys else p_.get(k)

        properties[s] = p

    return properties


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
    for t in pre_transform.split():

        if t.startswith('resize'):
            shape = t.split('-')[1:]
            if not shape:
                shape = tuple(set_props['shape'][1:])
            if len(shape) == 1:
                shape = int(shape[0])
            else:
                shape = tuple(int(_) for _ in shape)

            pre_transforms.append(transforms.Resize(shape))

        elif t.startswith('crop'):
            pre_transforms.append(transforms.RandomCrop(set_props['shape'][1:]))

        elif t.startswith('center-crop'):
            try:
                shape = int(t.split('-')[-1])
                shape = (shape, shape)
            except ValueError:
                shape = tuple(set_props['shape'][1:])

            pre_transforms.append(transforms.CenterCrop(shape))

        elif t.startswith('rotate'):
            angle = int(t.split('-')[-1])
            pre_transforms.append(lambda img: transforms.functional.rotate(img, angle))

        elif t == 'hflip':
            pre_transforms.append(lambda img: transforms.functional.hflip(img))

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

    post_transforms.append(transforms.ToTensor())

    if set_props.get('folder'):
        getter = create_image_dataset(set_props['classes_from_file'])
    else:
        getter = getters[dataset]

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
    else:
        train_kw = dict(train=True, root=directory)
        test_kw = dict(train=False, root=directory)

    if not set_props.get('folder') and set_props.get('downloadable'):
        train_kw['download'] = True and download
        test_kw['download'] = True and download

    with suppress_stdout(log=False):
        if 'train' in splits:
            trainset = getter(**train_kw,
                              target_transform=target_transform,
                              transform=transforms.Compose(pre_transforms + train_transforms + post_transforms))
        else:
            trainset = None

        if 'test' in splits:
            testset = getter(**test_kw,
                             target_transform=target_transform,
                             transform=transforms.Compose(pre_transforms + post_transforms))
        else:
            testset = None
        returned_sets = (trainset, testset)

    if set_props.get('classes_from_file'):
        for s in returned_sets:
            if s is not None:
                s.classes_file = set_props['classes_from_file']

    for s in returned_sets:
        if s is not None:
            s.name = dataset + ('90' if rotated else '')
            s.same_size = same_size
            s.transformer = transformer

            # if not hasattr(s, 'classes'):
            C = set_props['labels']
            s.classes = set_props.get('classes', [str(i) for i in range(C)])

            s.heldout = []
            if heldout_classes:
                s.heldout = heldout_classes
                s.classes = [c for (i, c) in enumerate(s.classes) if i not in heldout_classes]
                if len(heldout_classes) < C / 2:
                    s.name = s.name + '-' + '-'.join(str(_) for _ in heldout_classes)
                else:
                    s.name = s.name + '+' + '+'.join(str(_) for _ in range(C) if _ not in heldout_classes)

            if s.target_transform:
                for attr in ('targets', 'labels'):
                    if hasattr(s, attr):
                        labels = getattr(s, attr)
                        y = torch.tensor([s.target_transform(int(_)) for _ in labels], dtype=int)
                        s.data = s.data[y >= 0]

                        if isinstance(labels, (torch.Tensor, np.ndarray)):
                            setattr(s, attr, labels[y >= 0])
                        elif isinstance(labels, list):
                            setattr(s, attr, [_ for _ in labels if s.target_transform(_) >= 0])
                        else:
                            raise TypeError

    return returned_sets


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
        imageset = get_dataset(imageset)[1]

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
        axs[r, c].imshow(np.asarray(img))

        try:
            label = imageset.classes[y[i]]
        except (AttributeError, TypeError):
            label = str(y[i].numpy())
        except IndexError:
            label = i
        axs[r, c].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title=label)
    fix.show()

    nrows = int(np.ceil(num / ncols))

    fix, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    for i, image in enumerate(x):
        r = i // ncols
        c = i - r * ncols
        img = transforms.functional.to_pil_image(image)
        axs[r, c].imshow(np.asarray(img))

        try:
            label = imageset.classes[y[i]]
        except (AttributeError, TypeError):
            label = str(y[i].numpy())
        axs[r, c].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title=label)
    fix.show()


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


if __name__ == '__main__':

    plt.set_loglevel(level='warning')
    import time

    dset = 'cifar100'
    dset = 'letters'
    dset = 'lsunr'
    splits = ['train', 'test']
    splits = ['test']

    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug('Going to build dataset')
    t0 = time.time()
    # dset = _imagenet_getter(transform=transforms.ToTensor())
    train, test = get_dataset(dset, splits=splits)
    logging.debug('Built in {:.1f}s'.format(time.time() - t0))

    if 'train' in splits:
        plt.figure()
        show_images(train, num=36, ncols=6)
    if 'test' in splits:
        plt.figure()
        show_images(test, num=36, ncols=6)
