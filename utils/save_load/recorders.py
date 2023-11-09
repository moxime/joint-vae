import os

import logging

import numpy as np

import torch
import re


class LossRecorder:

    def __init__(self,
                 batch_size,
                 num_batch=0,
                 device=None,
                 **tensors):

        self.last_batch_size = None
        self._seed = None

        self._num_batch = 0
        self._samples = 0

        self.batch_size = batch_size
        self.reset()

        self._tensors = {}

        self.device = device

        if tensors:
            self._create_tensors(num_batch, device=device, **tensors)

    def _create_tensors(self, num_batch, device=None, **tensors):

        assert not self._tensors
        self._num_batch = num_batch
        self._samples = num_batch * self.batch_size

        if not device and not self.device:
            device = next(iter(tensors.values())).device

        self.device = device

        for k, t in tensors.items():
            shape = t.shape[:-1] + (self._samples,)
            self._tensors[k] = torch.zeros(shape,
                                           dtype=t.dtype,
                                           device=self.device)
        self.last_batch_size = self.batch_size

    def to(self, device):

        for t in self._tensors:
            self._tensors[t] = self._tensors[t].to(device)

    def reset(self, seed=False):

        self._recorded_batches = 0
        if self._seed is None or seed:
            self._seed = np.random.randint(1, int(1e8))
        self.last_batch_size = self.batch_size
        return

    def init_seed_for_dataloader(self):

        self._initial_seed = torch.seed()
        seed = self._seed
        torch.manual_seed(seed)

    def restore_seed(self):
        torch.manual_seed(self._initial_seed)

    def keys(self):
        return self._tensors.keys()

    def __len__(self):
        return self._recorded_batches

    def __repr__(self):
        return ('Recorder for '
                + ' '.join([str(k) for k in self.keys()]))

    def __getitem__(self, k):

        end = (len(self) - 1) * self.batch_size + self.last_batch_size
        return self._tensors[k][..., 0:end]

    def __iter__(self):

        return iter(self._tensors)

    def save(self, file_path, cut=True):
        """dict_ = self.__dict__.copy()
        tensors = dict.pop('_tensors')
        """

        if cut:
            self.num_batch = len(self)
            t = self._tensors
            for k in t:
                end = (self.num_batch - 1) * self.batch_size + self.last_batch_size
                t[k] = t[k][..., 0:end]

        torch.save(self.__dict__, file_path)

    @classmethod
    def load(cls, file_path, device=None, **kw):

        if 'map_location' not in kw and not torch.cuda.is_available():
            kw['map_location'] = torch.device('cpu')

        dict_of_params = torch.load(file_path, **kw)
        num_batch = dict_of_params['_num_batch']
        batch_size = dict_of_params['batch_size']
        tensors = dict_of_params['_tensors']

        r = LossRecorder(batch_size, num_batch, **tensors)

        for k in ('_seed', '_tensors', '_recorded_batches'):
            setattr(r, k, dict_of_params[k])

        for k in dict_of_params:
            if not k.startswith('_'):
                setattr(r, k, dict_of_params[k])

        # retro-compatibility
        if 'last_batch_size' in dict_of_params and isinstance(r.last_batch_size, dict):
            last_batch_size = next(iter(r.last_batch_size.values()))
            r.last_batch_size = last_batch_size

        if device:
            for k in r._tensors:
                if r._tensors[k].device != device:
                    r._tensors[k] = r._tensors[k].to('cpu')
        return r

    @classmethod
    def loadall(cls, dir_path, *w, file_name='record-{w}.pth', output='recorders', **kw):
        r"""
        If w is empty will find all recorders in directory
        if output is 'recorders' return recorders, if 'paths' return full paths

        """

        def outputs(p): return LossRecorder.load(path, **kw) if output.startswith('record') else p

        r = {}

        if not w:

            pattern = file_name.replace('.', '\.')
            pattern = pattern.replace('{w}', '(?P<name>.+)')

            for f in os.listdir(dir_path):
                regexp_match = re.match(pattern, f)
                if regexp_match:
                    path = os.path.join(dir_path, f)
                    r[regexp_match.group('name')] = outputs(path)

        for word in w:
            f = file_name.format(w=word)
            path = os.path.join(dir_path, f)
            if os.path.exists(path):
                r[word] = outputs(path)
            else:
                logging.warning(f'{f} not found')

        return r

    def copy(self, device=None):
        new_record = type(self)(self.batch_size)
        logging.debug('New recorder created')
        for i in range(len(self)):
            new_record.append_batch(**self.get_batch(i, device=device))
        return new_record

    def merge(self, other):

        assert isinstance(other, type(self))

        samples_to_be_added = other.recorded_samples
        batches_to_add = samples_to_be_added // self.batch_size + 1
        self.num_batch = len(self) + batches_to_add

        start = self.recorded_samples
        end = start + other.recorded_samples

        common_k = set.intersection(set(self), set(other))

        for k in common_k:
            self._tensors[k][..., start:end] = other[k]

        for k in [_ for _ in self if _ not in common_k]:
            self._tensors.pop(k)

        self.last_batch_size = (end - 1) % self.batch_size + 1
        self._recorded_batches = (end - 1) // self.batch_size + 1

    @property
    def recorded_samples(self):
        return (len(self) - 1) * self.batch_size + self.last_batch_size

    @property
    def num_batch(self):
        return self._num_batch

    @num_batch.setter
    def num_batch(self, n):

        if not self._tensors:
            return

        first_tensor = next(iter(self._tensors.values()))
        height = first_tensor.shape[-1]
        n_sample = n * self.batch_size

        if n_sample > height:
            d_h = n_sample - height
            for k in self._tensors:

                t = self._tensors[k]
                # print('sl353:', 'rec', self.device, k, t.device)
                z = torch.zeros(t.shape[:-1] + (d_h,),
                                dtype=t.dtype,
                                device=self.device)
                self._tensors[k] = torch.cat([t, z], axis=-1)

        self._num_batch = n
        self._samples = n * self.batch_size
        self._recorded_batches = min(n, self._recorded_batches)

    def has_batch(self, number, only_full=False):
        r""" number starts at 0
        """
        if number == len(self) - 1:
            return not only_full or self.last_batch_size == self.batch_size
        return number < self._recorded_batches

    def get_batch(self, i, *which, device=None, force_dict=False):

        if not which:
            if not self.keys():
                raise KeyError('empty recorder')
            return self.get_batch(i, *self.keys(), force_dict=True)

        if len(which) > 1 or force_dict:
            return {w: self.get_batch(i, w) for w in which}

        if not self.has_batch(i):
            raise IndexError(f'{i} >= {len(self)}')

        start = i * self.batch_size

        w = which[0]
        if i == len(self) - 1:
            end = start + self.last_batch_size
        else:
            end = start + self.batch_size

        t = self._tensors[w]
        if device:
            t = t.to(device)

        return t[..., start:end]

    def append_batch(self, extend=True, **tensors):

        if not self._tensors:
            self._create_tensors(1, **tensors)

        start = self._recorded_batches * self.batch_size
        end = start + self.batch_size

        if end > self._samples:
            if extend:
                self.num_batch *= 2
            else:
                raise IndexError

        batch_sizes = set(tensors[k].shape[-1] for k in tensors)
        assert len(batch_sizes) == 1, 'all batches have to be of same size'
        batch_size = batch_sizes.pop()
        assert batch_size <= self.batch_size, 'appended batch to large'
        assert self.last_batch_size == self.batch_size
        self.last_batch_size = batch_size
        end = start + batch_size

        for k in tensors:
            if k not in self.keys():
                raise KeyError(k)
            self._tensors[k][..., start:end] = tensors[k]

        self._recorded_batches += 1
