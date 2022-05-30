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


def modify_getter(getter, pretransform=None, **added_kw):

    def modified_getter(*a, **kw):
        copied_kw = added_kw.copy()

        if 'transform' in kw and pretransform:
            kw['transform'] = transforms.Compose([pretransform, kw['transform']])

        k = 'target_transform'
        if added_kw.get(k):
            if kw.get(k):
                kw[k] = transforms.Compose([copied_kw.pop(k), kw[k]])
            else:
                kw[k] = copied_kw.pop(k)
        kw.update(copied_kw)
        return getter(*a, **kw)

    return modified_getter


def choose_device(device=None):
    """if device is None, returns cuda or cpu if not available.
    else returns device

    """    

    if device is None:

        has_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if has_cuda else 'cpu')

    return device


getters = {'mnist': datasets.MNIST,
           'fashion': datasets.FashionMNIST,
           'letters': datasets.EMNIST,
           'cifar10': datasets.CIFAR10,
           'svhn': datasets.SVHN,
           'lsunc': datasets.LSUN,
           'lsunr': datasets.LSUN}

letters_rotate = transforms.Compose([lambda img: transforms.functional.rotate(img, -90),
                                     lambda img: transforms.functional.hflip(img)])


getter_adapters = {'letters': dict(pretransform=letters_rotate, target_transform=lambda x: x - 1)}


def dataset_properties(conf_file='data/sets.ini'):

    parsed_props = configparser.ConfigParser()
    parsed_props.read(conf_file)

    properties = {}
    
    for s in parsed_props.sections():

        p_ = parsed_props[s]
        print('***', *list(p_.keys()))
        p = {}
        p['shape'] = tuple(int(_) for _ in p_['shape'].split())

        if 'classes_from_files' in p_:
            pass
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
            
        properties[s] = p

    return properties


def _imagenet_getter(train=True, download=False, root='./data', **kw):

    dset = datasets.ImageNet(root=os.path.join(root, 'ImageNet12'), split='train' if train else 'val',
                             **kw)
    return dset


imagenet_classes = [_[0] for _ in _imagenet_getter().classes]


def get_dataset(dataset='mnist', 
                transformer='default',
                data_augmentation=[],
                conf_file='data/sets.ini',
                getters=getters,
                getter_adapters=getter_adapters
                ):

    parsed_config = configparser.ConfigParser()
    parsed_config.read(conf_file)

    dataset = dataset.lower()
    rotated = dataset.endswith('90')
    if rotated:
        dataset = dataset[:-2]
        rotation = transforms.Lambda(lambda img: transforms.functional.rotate(img, 90))

    target_transform = None
    parent_set, heldout_classes = get_heldout_classes_by_name(dataset)
    
    if heldout_classes:
        dataset = parent_set
        C = get_shape_by_name(parent_set)[-1]
        heldin = [_ for _ in range(C) if _ not in heldout_classes]
        d = {c: i for (i, c) in enumerate(heldin)}
        d.update({_: -1 for _ in heldout_classes})
        target_transform =  d.get

    if transformer == 'default':
        transformer = set_dict[dataset]['default']

    transform = transformers[transformer][dataset]

    # shape = set_dict[dataset]['shape']
    # same_size = [s for s in set_dict if set_dict[s]['shape'] == shape]
    # same_size.remove(dataset)
    same_size = get_same_size_by_name(get_name_by_heldout_classes(dataset, *heldout_classes))

    train_transforms = []

    for t in data_augmentation:
        if t == 'flip':
            t_ = transforms.RandomHorizontalFlip()

        if t == 'crop':
            size = set_dict[dataset]['shape'][1:]
            padding = 0 if 'imagenet' in dataset else size[0] // 8
            t_ = transforms.RandomCrop(size, padding=padding, padding_mode='edge')

        train_transforms.append(t_)

    train_transforms.append(transform)
    train_transform = transforms.Compose(train_transforms)

    getter = set_dict[dataset]['getter']
    if rotated:
        getter = modify_getter(getter, pretransform=rotation)

    with suppress_stdout(log=True):
        trainset = getter(root=root, train=True,
                          download=True,
                          target_transform=target_transform,
                          transform=train_transform)

        testset = getter(root=root, train=False,
                         download=True,
                         target_transform=target_transform,
                         transform=transform)

        returned_sets = (trainset, testset)
        
    for s in returned_sets:
        if s is not None:
            s.name = dataset + ('90' if rotated else '')
            s.same_size = same_size
            s.transformer = transformer

            # if not hasattr(s, 'classes'):
            C = set_dict[dataset]['labels']
            s.classes = set_dict[dataset].get('classes', [str(i) for i in range(C)])
                
            s.heldout = []
            if heldout_classes:
                s.heldout = heldout_classes
                s.classes = [c for (i, c) in enumerate(s.classes) if i not in heldout_classes]
                if len(heldout_classes) < C / 2:
                    s.name = s.name + '-' + '-'.join(str(_) for _ in heldout_classes)
                else:
                    s.name = s.name + '+' + '+'.join(str(_) for _ in range(C) if _ not in heldout_classes)

            if s.target_transform:
                y = torch.tensor([s.target_transform(int(_)) for _ in s.targets], dtype=int)
                s.data = s.data[y >= 0]
                for attr in ('targets', 'labels'):
                    if hasattr(s, attr):
                        labels = getattr(s, attr)
                        if isinstance(labels, torch.Tensor):
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
    p = transformers['pad'][set_name].transforms[0].padding
    if len(shape)==3:
        return (shape[0], shape[1] + 2 * p, shape[2] + 2 * p), num_labels


def get_same_size_by_name(set_name, rotated=False):

    if set_name.endswith('-?'):
        return [set_name[:-2] + '+?']

    if set_name.endswith('90'):
        return get_same_size_by_name(set_name[:-2], rotated=True)

    parent_set, heldout = get_heldout_classes_by_name(set_name)
    if heldout:
        C = get_shape_by_name(parent_set)[-1]
        new_heldout = [_ for _ in range(C) if _ not in heldout]
        return [get_name_by_heldout_classes(parent_set, *new_heldout)]
        
    if set_name not in set_dict:
        return []

    shape, _ = get_shape_by_name(set_name)
    same_size = [s for s in set_dict if set_dict[s]['shape'] == shape]
    if not rotated:
        same_size.remove(set_name)
        same_size.append(set_name + '90')
    
    return same_size


def get_classes_by_name(dataset):

    if dataset.endswith('90'):
        return [_ + '90' for _ in get_classes_by_name(dataset[:-2])]
    parent_set, ho = get_heldout_classes_by_name(dataset)

    parent_classes = set_dict[parent_set].get('classes', [dataset.upper()])

    return [_ for i, _ in enumerate(parent_classes) if i not in ho]


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
 

def show_images(imageset, shuffle=True, num=4, **kw):

    batch_size = num
    loader = torch.utils.data.DataLoader(imageset,
                                         shuffle=shuffle,
                                         batch_size=batch_size)

    x, y = next(iter(loader))
    
    npimages = torchvision.utils.make_grid(x[:num]).numpy().transpose(1, 2, 0)
    labels = y
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

    import time
    
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug('Going to build dataset')
    t0 = time.time()
    # dset = _imagenet_getter(transform=transforms.ToTensor())
    train, test = get_dataset('imagenet12', data_augmentation=['crop'])
    logging.debug('Built in {:.1f}s'.format(time.time() - t0))

    show_images(train, num=32)
    x, y = get_batch(train)
    
    print(*x.shape)

    torch.utils.data.dataset.Subset
