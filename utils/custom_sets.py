import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class ConstantDataset(Dataset):

    def __init__(self, shape, n=10000, transform=None, target_transform=None, download=False):

        self._shape = shape
        self._len = n
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self._len

    def _create_image(self, idx):
        color = torch.rand(self._shape[0], 1, 1)
        image = color.expand(self._shape)
        label = 0
        return image, label

    def __getitem__(self, idx):

        image, label = self._create_image(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class UniformDataset(ConstantDataset):

    def _create_image(self, idx):

        label = 0
        image = torch.rand(self._shape)

        return image, label


class FromNumpy(Dataset):

    def __init__(self, root='data/foo', split='test', transform=None, target_transform=None, download=False):

        data_dir = os.path.join(root, split)
        files_in_root = [_ for _ in os.listdir(data_dir) if _.endswith('.npy')]

        assert len(files_in_root) <= 1, '{} npy file in {}'.format(len(files_in_root), root)

        self.transform = transform
        self.target_transform = target_transform

        if len(files_in_root):
            self.data = np.load(os.path.join(data_dir, files_in_root[0]))

        else:
            self.data = np.ndarray(0)

        self._len = len(self.data)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        image, label = self.data[idx], 0

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


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


class ImageListDataset(datasets.ImageFolder):
    def __init__(self, root, split='test', *a, img_file='imglist.txt', **kw):

        self.root = os.path.join(root, split)
        try:
            super().__init__(self.root, *a, **kw)
        except FileNotFoundError:
            super(datasets.DatasetFolder, self).__init__(self.root, *a, **kw)
            self.samples = []
            self.classes = []
            self.targets = []
        logging.debug('Creating dataset in folder {} based on images listed in {}'.format(root, img_file))

        self._img_file = os.path.join(root, split, img_file)
        if not os.path.exists(self._img_file):
            self._img_file = None

        if self._img_file:
            self.samples = []
            self.targets = []
            with open(self._img_file) as f:
                for line in f:
                    if not line.startswith('#'):
                        label_name = os.path.split(line.strip())[0]
                        label = self.class_to_idx[label_name]
                        node = os.path.join(self.root, line.strip())
                        self.samples.append((node, label))
                        self.targets.append(label)


def create_file(root, txt_file, prefix='cifar100/test/'):

    dest_file = os.path.join(root, prefix, 'imglist.txt')
    original_file = os.path.join(root, txt_file)

    with open(original_file) as o_f:
        with open(dest_file, 'w') as d_f:

            for l in o_f:

                impath = l.strip().split()[0]
                if impath.startswith(prefix):
                    print(impath)
                    d_f.write(impath[len(prefix):])
                    d_f.write('\n')


if __name__ == '__main__':

    c = ImageListDataset('data/openood/cifar100')
    c_ = ImageListDataset('data/openood/cifar100', split='train')
    t = ImageListDataset('data/openood/tin', split='val')

    p = ImageListDataset('data/openood/places365')
