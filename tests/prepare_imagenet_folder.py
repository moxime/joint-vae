import os, shutil
import torch
from scipy.io import loadmat


def create_bash_for_symlinks(directory='data/ImageNet12'):

    meta_file = os.path.join(directory, 'meta.bin')

    classes = list(torch.load(meta_file)[0].keys())

    print('#!/bin/bash')
    for _ in classes:

        print('ln -s $1/{} .'.format(_))


def create_small_train_folder(directory='data/ImageNet12', target='tmp', source='train', num=2):

    source_directory = os.path.join(directory, source)

    nodes = [_ for _ in os.listdir(source_directory) if os.path.isdir(_)]

    for node in nodes:

        target_dir = os.path.join(directory, target, node)
        if not os.apth.exists(target_dir):
            os.makedirs(target_dir)

        images = [_ for _ in os.listdir(os.path.join(source_directory, node)) if _.endswith('.JPEG')]

        for image in images[:2]:
            image_path = os.path.join(source_directory, node, image)
            print('{} -> {}'.format(image_path, target_dir))
            shutil.copy(image_path, target_dir)

def create_validation_folder(directory='data/ImageNet12', target='val', source='tmp'):

    dev_kit_folder = 'ILSVRC2012_devkit_t12/data'
    
    meta_file = os.path.join(directory, dev_kit_folder, 'meta.mat')

    meta_data = loadmat(meta_file)

    synsets = meta_data['synsets']

    ids = {}
    for s in synsets:
        ids[s[0][0][0][0]] = s[0][1][0]
        
    ground_truth_file = os.path.join(directory, dev_kit_folder, 'ILSVRC2012_validation_ground_truth.txt')

    source_dir = os.path.join(directory, source)

    image_files = sorted([_ for _ in os.listdir(source_dir) if _.endswith('.JPEG')])

    nodes_by_file = {}
    images_not_found = []
    with open(ground_truth_file) as f:
        for i, line in enumerate(f):

            id = int(line.rstrip())
            node = ids[id]
            image = 'ILSVRC2012_val_{:08d}.JPEG'.format(i + 1)

            if image in image_files:
                nodes_by_file[image] = node
                # print(image, node)
            else:
                images_not_found.append(image)
                
    if images_not_found:
        print('*** IMAGE NOT FOUND ***')
        print(','.join(images_not_found))

    for image, node in nodes_by_file.items():

        tdir = os.path.join(directory, target, node)
        if not os.path.exists(tdir):
            os.makedirs(tdir)

        src_file = os.path.join(source_dir, image)
        print('{} -> {}'.format(src_file, tdir))
        shutil.copy(src_file, tdir)
            
    return nodes_by_file

# m = create_validation_folder()
create_small_train_folder()
