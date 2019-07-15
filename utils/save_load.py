import os
import pickle
from tensorflow.keras.models import load_model






def save_net(net, dir_name, file_name):

    dir_path = os.path.realpath(dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = full_path(dir_name, file_name)
    net.save(file_path)

    
def load_net(dir_name, file_name):

    file_path = full_path(dir_name, file_name)
    return load_model(file_path)
    
    
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
