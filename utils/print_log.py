import time
import numpy as np
import logging
import torch
import pandas as pd
import sys
import re
import functools
from contextlib import contextmanager
import os


def harddebug(*a):
    if harddebug.level:
        print(harddebug.pre, *a)


harddebug.level = 0


def printdebug(d=True):
    def wrapper(func):
        @functools.wraps(func)
        def wrapped(*a, **kw):
            harddebug.level = d
            harddebug.pre = '*** ' + func.__name__ + ' ***'
            out = func(*a, **kw)
            harddebug.level = False
            return out
        return wrapped
    return wrapper


@contextmanager
def turnoff_debug(turnoff=True, logger=None, type_of_stream=os.sys.stderr):
    logger = logger or logging.getLogger()
    handlers = logger.handlers
    logging_levels = {_: _.level for _ in handlers}
    if turnoff:
        for h in handlers:
            if h.stream == type_of_stream or not type_of_stream:
                h.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for h in handlers:
            h.setLevel(logging_levels[h])


class EpochOutput:

    EVERY_BATCH = 20
    END_OF_EPOCH = 10
    END_OF_SET = 0

    CELL_WIDTH = 9

    unix_tags = [('%B', '\033[1m'),
                 ('%b', '\033[0m'),
                 ('%I', '\033[3m'),
                 ('%i', '\033[0m')]

    no_style_tags = [('%B', ''),
                     ('%b', ''),
                     ('%I', ''),
                     ('%i', '')]

    cell_formats = {'metrics': '{{{sep}{c}.2e}}'.format(sep=':', c=CELL_WIDTH),
                    'losses': '{{{sep}{c}.2e}}'.format(sep=':', c=CELL_WIDTH),
                    'accuracy': '{{{sep}{c}.2%}}'.format(sep=':', c=CELL_WIDTH),
                    'dB': '{{{sep}{c}.1f}} dB'.format(sep=':', c=CELL_WIDTH - 3),
                    'odin': '{{{sep}>{c}}}'.format(sep=':', c=CELL_WIDTH),
                    'text': '{{{sep}>{c}}}'.format(sep=':', c=CELL_WIDTH),
                    }

    def __init__(self):

        stdout = {'stream': sys.stdout, 'when': self.EVERY_BATCH, 'tags': self.unix_tags}
        self.streams = [stdout]
        self.files = []
        self.last_row = []

    def format_encoding(self, string, tags=no_style_tags):

        s = string
        for pair in tags:
            s = s.replace(*pair)

        return s

    def add_file(self, path, when=EVERY_BATCH, tags=unix_tags):

        self.files.append({'path': path, 'when': when, 'tags': tags})

    def write(self, string, when=END_OF_EPOCH):

        for stream in self.streams:
            if stream['when'] >= when:
                stream['stream'].write(self.format_encoding(string, stream['tags']))

        for f in self.files:
            if f['when'] >= when:
                with open(f['path'], 'a') as f_:
                    f_.write(self.format_encoding(string, f['tags']))
                    f_.flush()

    def result_col(self, title, default_format='text', header=False, masked=[],
                   sep=' | ',
                   **kv):

        default_cell_format = self.cell_formats.get(default_format) or self.cell_formats['text']
        col_width = self.CELL_WIDTH

        if header:
            h1 = sep.join('{k:_^{w}}'.format(k=k, w=col_width) for k in kv if k not in masked)
            h0 = '{t:_^{w}}'.format(t=title, w=len(h1))
            return h0 if header == 2 else h1
        else:
            return sep.join(self.cell_formats.get(k, default_cell_format).format(kv[k])
                            for k in kv if k not in masked)

    def result_row(self, header=False, masked=[], sep=' | ', double_sep=' || ', **kvs):

        self.last_row = [*kvs]

        return double_sep.join(self.result_col(k, header=header,
                                               masked=masked,
                                               default_format=k,
                                               sep=sep,
                                               **kvs[k])
                               for k in kvs)

    def results(self, batch, batches, epoch, epochs,
                loss_components=None,
                masked_components=['z_logdet', 'z_mahala', 'z_tr_inv_cov'],
                best_of={'odin': -1},
                time_per_i=0,
                batch_size=100,
                preambule='',
                end_of_epoch='\n',
                **kvs,
                ):

        if preambule == 'train':
            preambule = '%I' + preambule
            end_of_format = '%i'
            sep = '%i' + ' | ' + '%I'
            double_sep = '%i' + ' || ' + '%I'
        else:
            end_of_format = ''
            sep = ' | '
            double_sep = ' || '

        kepts_kvs = {'': {'epoch': '{}/{}'.format(epoch, epochs) if preambule.lower() == 'train' else '',
                          '': preambule}}

        for title in kvs:
            done_best = []
            kept_kvs[title] = {}
            for k in kvs[title]:
                if k in masked_components:
                    continue
                if title == 'metrics' and '-a-' in k or k.endswith('-2s'):
                    continue
                for best_k in best_of:
                    if k.startswith(best_k) and best_k not in done_best:
                        kept_kvs[title][best_k] = _s * max(_s * kvs[title][_]
                                                           for _ in kvs[title] if _.startswith(best_k))
                        done_best.append(best_k)
                    continue
                kept_kvs[title][k] = kvs[title][k]

        if time_per_i > 0:
            time_per_i = Time(time_per_i)
            kept_kvs['time']['/i'] = time_per_i

            if batch < batches - 1:
                kept_kvs['time']['eta'] = time_per_i * batch_size * (batches - batch)
            else:
                kept_kvs['time']['total'] = time_per_i * batch_size * batches

        if not batch:
            if self.last_row != [*kept_kvs]:
                line = '\r' + self.result_row(header=2, sep=sep, double_sep=double_sep, **kept_kvs)
                self.write(line + '\n', when=self.END_OF_EPOCH)
            line = '\r' + self.result_row(header=1, sep=sep, double_sep=double_sep, **kept_kvs)
            self.write(line + '\n', when=self.END_OF_EPOCH)

        line = '\r' + self.result_row(header=False, sep=sep, double_sep=double_sep, **kept_kvs)
        self.write(line, when=self.EVERY_BATCH)

        if batch == batches - 1:
            self.write(line + '\n', when=self.END_OF_EPOCH)


def texify_str(s, num=False, space=None, underscore=None, verbatim=False):
    if type(s) != str:
        return s
    try:
        float(s)
        return s
    except ValueError:
        pass
    s = s.replace('->', '\\ensuremath{\\to{}}')
    if space:
        s = s.replace(' ', space)
    if underscore:
        s = s.replace('_', underscore)
    if not num:
        return s
    return re.sub(r'[-+]?\d*\.\d+', r'\\num{\g<0>}', s)


class Time(float):

    def __str__(self, max=2):

        t = self

        units = ['d', 'h', 'm', 's', 'ms', '\u03bcs', 'ns']
        qs = [24 * 3600, 3600, 60, 1, 1e-3, 1e-6, 1e-9]

        if t == 0:
            return '0s'

        str = '-' if t < 0 else ''
        t = abs(t)

        orig_t = t
        for i, (unit, q) in enumerate(zip(units, qs)):

            n = int(t / q)
            if n:
                str += f'{n}{unit}'
            t -= q * n

            if t <= orig_t / 100:
                break

        return str

    def __add__(self, t_):

        return Time(float(self) + float(t_))

    def __neg__(self):

        return Time(-float(self))

    def __sub__(self, t):

        return self + (-t)

    def __mul__(self, k):

        return Time(float(self) * k)

    def __truediv__(self, k):

        return Time(float(self) / k)

    def __format__(self, *a, **k):

        return str(self).__format__(*a, **k)


def text_dataviz(start, end, *marks, N=100, min_val=None, max_val=None, default='-'):

    if min_val is None:
        min_val = start
    if max_val is None:
        max_val = end

    width = max_val - min_val

    def f2i(x):
        i = int(N * (x - min_val) / width)
        return max(min(i, N - 1), 0)

    start_ = f2i(start)
    end_ = f2i(end)

    chars = {_: (default if (start_ <= _ <= end_) else ' ') for _ in range(N)}

    for m in marks:
        chars[f2i(m[0])] = m[1]

    return ''.join([chars[_] for _ in range(N)])


def timerun(func):
    """ Calculate the execution time of a method and return it back"""

    @ functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start

        logger.debug(f"Duration of {func.__name__} function was {duration}.")
        return result
    return wrapper


if __name__ == '__main__':

    kvs = {'': {'epoch': '400/600', '': 'VALID'},
           'losses': {'total': -1.2332e-1, 'kl': 32.5454e-2, 'dB': -52.22325, 'odin': 'yes!'},
           'accuracy': {'iws': 0.84846, 'iws-2': 0.987, 'foo': 0.6545},
           'time': {'/i': Time(122.1511e-6), 'eta': Time(617.65154)}
           }

    o = EpochOutput()
    for header in (2, 1, False, False, False, False):
        print(o.result_row(header=header, masked=['foo'], **kvs))
