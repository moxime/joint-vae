from torchvision import datasets, transforms
import torchvision
import torch
import matplotlib.pyplot as plt

def choose_device(device=None):
    """

    if device is None, returns cuda or cpu if not available.
    else returns device
    """    

    if device is None:

        has_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if has_cuda else 'cpu')

    return device


simple_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Lambda(lambda x: x
                                                         / 255.0)])
simple_transform = transforms.ToTensor()

cifar_means = (0.4914, 0.4822, 0.4465)
cifar_stds = (0.2023, 0.1994, 0.2010)
cifar_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(cifar_means,
                                                           cifar_stds)])

pad_transform = transforms.Compose([transforms.Pad(2), simple_transform])


def get_dataset(dataset='MNIST', root='./data', ood=None, transformer='default'):

    default_transform = transformer == 'default'
    dataset = dataset.lower()
    
    if dataset == 'mnist':

        getter = datasets.MNIST
        if default_transform:
            transform = simple_transform
        elif transformer == 'simple':
            transform = simple_transform

    if dataset == 'fashion':

        getter = datasets.FashionMNIST
        if default_transform:
            transform = simple_transform
        elif transformer == 'simple':
            transform = simple_transform
        elif transformer == 'pad':

            transform = pad_transform

    if dataset == 'svhn':

        def getter(train=True, **kw):
            return datasets.SVHN(split='train' if train else 'test', **kw)

        if default_transform:
            transform = simple_transform
        elif transformer == 'simple':
            transform = simple_transform

    if dataset == 'cifar10':
        
        getter = datasets.CIFAR10
        # transform = cifar_transform if default_transform else transform
        if default_transform:
            transform = simple_transform
        elif transformer == 'simple':
            transform = simple_transform
        elif transformer == 'normal':
            transform = cifar_transform


    trainset = getter(root=root, train=True,
                      download=True,
                      transform=transform)

    testset = getter(root=root, train=False,
                     download=True,
                     transform=transform)

    trainset.name = dataset
    testset.name = dataset
    
    return trainset, testset


def get_mnist(**kw):

    return get_dataset(dataset='MNIST', **kw)


def get_fashion_mnist(**kw):

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
    num_labels = len(torch.unique(data[1]))

    return tuple(data[0][0].shape)

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

    
