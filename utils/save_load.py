import os
import pickle
import json
from tensorflow.keras.models import load_model


def get_path(dir_name, file_name, create_dir=True):

    dir_path = os.path.realpath(dir_name)
    if not os.path.exists(dir_path) and create_dir:
        os.makedirs(dir_path)

    return full_path(dir_name, file_name)


def save_net(net, dir_name, file_name):

    dir_path = os.path.realpath(dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = full_path(dir_name, file_name)
    net.save(file_path)

    
def load_net(dir_name, file_name):

    file_path = full_path(dir_name, file_name)
    return load_model(file_path)
    

def save_json(d, dir_name, file_name, create_dir=True):

    p = get_path(dir_name, file_name, create_dir)

    with open(p, 'w') as f:
        json.dump(d, f)


def load_json(dir_name, file_name):

    p = get_path(dir_name, file_name, create_dir=False)

    with open(p, 'rb') as f:
        return json.load(f)
    

def save_object(o, dir_name, file_name):
    dir_path = os.path.realpath(dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, file_name), 'wb') as f:
        pickle.dump(o, f)

            
def load_object(dir_path, file_name):

    file_path = full_path(dir_path, file_name)

    with open(file_path, 'rb') as f:
        return pickle.load(f)


def full_path(dir_name, file_name):

    return os.path.join(os.path.realpath(dir_name), file_name)
