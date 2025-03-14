import os
import json
import logging
import time


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


def load_json(dir_name, file_name, presumed_type=str, attempt=0, max_attempt=10, wait=0.1):

    p = get_path(dir_name, file_name, create_dir=False)

    with open(p, 'rb') as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError as e:
            if attempt < max_attempt:
                logging.warning('Corrupted file, attempt {}'.format(attempt + 1))
                time.sleep(wait)
                return load_json(dir_name, file_name, presumed_type=presumed_type,
                                 attempt=attempt + 1, max_attempt=10, wait=wait)
            else:
                logging.error('Corrupted file\n%s', p)
                with open('/tmp/corrupted', 'a') as f:
                    f.write(p + '\n')
                raise e
    d_ = {}
    if isinstance(d, str):
        print(dir_name, file_name)
    for k in d:
        try:
            k_ = presumed_type(k)
        except ValueError:
            k_ = k
        d_[k_] = d[k]

    return d_
