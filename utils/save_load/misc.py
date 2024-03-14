import os
import json
import logging


def get_path(dir_name, file_name, create_dir=True):

    dir_path = os.path.realpath(dir_name)
    if not os.path.exists(dir_path) and create_dir:
        os.makedirs(dir_path)

    return os.path.join(dir_name, file_name)


def job_to_str(number, string, formats={int: '{:06d}'}):
    job_format = formats.get(type(number), '{}')
    return string.replace('%j', job_format.format(number))


def create_file_for_job(number, directory, filename, mode='w'):

    directory = job_to_str(number, directory)

    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)

    return open(filepath, mode)


def save_json(d, dir_name, file_name, create_dir=True):

    p = get_path(dir_name, file_name, create_dir)

    with open(p, 'w') as f:
        json.dump(d, f)


def load_json(dir_name, file_name, presumed_type=str):

    p = get_path(dir_name, file_name, create_dir=False)

    # logging.debug('*** %s', p)
    with open(p, 'rb') as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError:
            logging.error('Corrupted file\n%s', p)
            with open('/tmp/corrupted', 'a') as f:
                f.write(p + '\n')
            return {}
    d_ = {}
    for k in d:
        try:
            k_ = presumed_type(k)
        except ValueError:
            k_ = k
        d_[k_] = d[k]

    return d_
