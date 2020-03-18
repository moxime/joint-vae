from torchvision import datasets, transforms
import torch


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


def get_dataset(dataset='MNIST', root='./data', ood=None):

    if dataset == 'MNIST':

        getter = datasets.MNIST
        transform = simple_transform

    if dataset == 'fashion':
        getter = datasets.FashionMNIST
        transform = simple_transform

    if dataset == 'svhn':

        def getter(train=True, **kw):
            return datasets.SVHN(split='train' if train else 'test', **kw)
        transform = simple_transform

    if dataset == 'cifar10':
        getter = datasets.CIFAR10
        transform = simple_transform
        
    trainset = getter(root=root, train=True,
                          download=True,
                          transform=transform)

    testset = getter(root=root, train=False,
                         download=True,
                         transform=transform)

    return trainset, testset


def get_mnist(**kw):

    return get_dataset(dataset='MNIST', **kw)


def get_fashion_mnist(**kw):

    return get_dataset(dataset='fashion', **kw)

def get_svhn(**kw):

    return get_dataset(dataset='svhn', **kw)

def get_cifar10(**kw):

    return get_dataset(dataset='cifar10', **kw)

def get_batch(theset, shuffle=True, batch_size=100, device=None):

    loader = torch.utils.data.DataLoader(theset,
                                         shuffle=shuffle,
                                         batch_size=batch_size)

    data = next(iter(loader))

    device = choose_device(device)
    
    return data[0].to(device), data[1].to(device)
