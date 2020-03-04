from torchvision import datasets, transforms


simple_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Lambda(lambda x: x
                                                         / 255.0)])
simple_transform = transforms.ToTensor()


def get_dataset(dataset='MNIST', root='./data', ood=None):

    if dataset == 'MNIST':

        set_getter = datasets.MNIST
        transform = simple_transform

    if dataset == 'fashion':
        set_getter = datasets.FashionMNIST
        transform = simple_transform
        
    trainset = set_getter(root=root, train=True,
                          download=True,
                          transform=transform)

    testset = set_getter(root=root, train=False,
                         download=True,
                         transform=transform)

    return trainset, testset


def get_mnist(**kw):

    return get_dataset(dataset='MNIST', **kw)


def get_fashion_mnist(**kw):

    return get_dataset(dataset='fashion', **kw)

