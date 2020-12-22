from torchvision import datasets, transforms
import torchvision
import torch
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys
import os
import logging
import string


class LoggerAsfile(object):

    def write(self, s):
        logging.debug(s)

laf = LoggerAsfile()


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as dev_null:
        orig = sys.stdout
        sys.stdout = dev_null
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


def modify_getter(getter, pretransform=None, **added_kw):

    def modified_getter(*a, **kw):

        if 'transform' in kw and pretransform:
            kw['transform'] = transforms.Compose([pretransform, kw['transform']])
        return getter(*a, **added_kw, **kw)

    return modified_getter

        
def choose_device(device=None):
    """

    if device is None, returns cuda or cpu if not available.
    else returns device
    """    

    if device is None:

        has_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if has_cuda else 'cpu')

    return device

cifar_shape = (3, 32 , 32)
mnist_shape = (1, 28, 28)
set_dict = {'cifar10': {'shape': (3, 32, 32),
                        'labels':10,
                        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                    'dog', 'frog', 'horse', 'ship', 'truck'],
                        'default': 'simple',
                        'means': (0.4914, 0.4822, 0.4465), 
                        'stds': (0.2023, 0.1994, 0.2010),
                        'getter': datasets.CIFAR10},
            
            'mnist': {'shape': (1, 28, 28),
                      'labels':10,
                      'classes': [str(i) for i in range(10)],
                      'default': 'simple',
                      'getter': datasets.MNIST}
            }

set_dict['fashion'] = set_dict['mnist'].copy()
set_dict['fashion']['getter'] = datasets.FashionMNIST
set_dict['fashion']['classes'] = datasets.FashionMNIST.classes

set_dict['letters'] = set_dict['mnist'].copy()

pretransform = transforms.Compose([
                    lambda img: transforms.functional.rotate(img, -90),
                    lambda img: transforms.functional.hflip(img)])

set_dict['letters'].update({'classes': list(string.ascii_lowercase),
                            'labels': 26,
                            'getter': modify_getter(datasets.EMNIST,
                                                    pretransform=pretransform,
                                                    split='letters')
})


# set_dict['lsun-c'] = set_dict['cufar10'].copy()

# crop_transform = transforms.RandomCrop((32, 32))


"""
def _lsun_getter(train=True, **kw):
    set_ = datasets.LSUN(classes='train' if train else 'test', **kw)
    
set_dict['lsun'] = set_dict['cifar10'].copy()
set_dict['lsun'].pop('means')
set_dict['lsun'].pop('stds')
set_dict['lsun']['getter'] = datasets.LSUN
"""

def _svhn_getter(train=True, **kw):
    set_ = datasets.SVHN(split='train' if train else 'test', **kw)    
    set_.classes = [str(i) for i in range(10)]
    return set_

set_dict['svhn'] = set_dict['cifar10'].copy()
set_dict['svhn'].pop('means')
set_dict['svhn'].pop('stds')
set_dict['svhn']['classes'] = [str(i) for i in range(10)]
set_dict['svhn']['getter'] = _svhn_getter




transformers = {'simple': {n: transforms.ToTensor() for n in set_dict}}

transformers['normal'] = {n: transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(set_dict[n].get('means', 0),
                                                                     set_dict[n].get('stds', 1))])
                         for n in set_dict}

transformers['pad'] = {n: transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
                       for n in set_dict}


def get_dataset(dataset='MNIST', root='./data', ood=None,
                transformer='default', data_augmentation=[]):

    if transformer == 'default':
        transformer = set_dict[dataset]['default']
    transform = transformers[transformer][dataset]
    dataset = dataset.lower()

    shape = set_dict[dataset]['shape']
    same_size = [s for s in set_dict if set_dict[s]['shape'] == shape]
    same_size.remove(dataset)
    
    train_transforms = []

    for t in data_augmentation:
        if t == 'flip':
            t_ = transforms.RandomHorizontalFlip()

        if t== 'crop':
            size = set_dict[dataset]['shape'][1:]
            padding = size[0] // 8
            t_ = transforms.RandomCrop(size, padding=padding)

        train_transforms.append(t_)

    train_transforms.append(transform)
    train_transform = transforms.Compose(train_transforms)

    getter = set_dict[dataset]['getter']
    with suppress_stdout():
        trainset = getter(root=root, train=True,
                          download=True,
                          transform=train_transform)

        testset = getter(root=root, train=False,
                         download=True,
                         transform=transform)

    for s in (trainset, testset):
        s.name = dataset
        s.same_size = same_size
        s.transformer = transformer
        C = set_dict[dataset]['labels']
        s.classes = set_dict[dataset].get('classes', [str(i) for i in range(C)])
    
    return trainset, testset


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

    loader = torch.utils.data.DataLoader(dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size)

    data = next(iter(loader))

    device = choose_device(device)
    
    return data[0].to(device), data[1].to(device)

def get_shape(dataset):
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1)

    data = next(iter(loader))

    num_labels = len(dataset.classes)
    
    return tuple(data[0][0].shape), num_labels

def get_shape_by_name(set_name, transform='default'):

    shape = set_dict[set_name]['shape']
    num_labels = len(set_dict[set_name]['classes'])
    if transform != 'pad':
        return set_dict[set_name]['shape'], num_labels
    p = transformers['pad'][set_name].transforms[0].padding
    if len(shape)==3:
        return (shape[0], shape[1] + 2 * p, shape[2] + 2 *p), num_labels


def get_same_size_by_name(set_name):

    shape, _ = get_shape_by_name(set_name)
    same_size = [s for s in set_dict if set_dict[s]['shape'] == shape]
    same_size.remove(set_name)

    return same_size


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


def show_images(imageset, shuffle=True, num=4, **kw):

    loader = torch.utils.data.DataLoader(imageset,
                                         shuffle=shuffle,
                                         batch_size=num)

    data = next(iter(loader))
    npimages = torchvision.utils.make_grid(data[0]).numpy().transpose(1, 2, 0)
    labels = data[1]
    try:
        classes = [imageset.classes[y] for y in labels]
    except AttributeError:
        classes = [str(y.numpy()) for y in labels]

    legend = ' - '.join(classes)

    f, a = plt.subplots()
    a.axis('off')
    a.imshow(npimages, **kw)
    a.set_title(legend)
    f.show()

    
if __name__ == '__main__':

    t, T = get_dataset('cifar10', data_augmentation=['flip', 'crop'])

    show_images(t)
