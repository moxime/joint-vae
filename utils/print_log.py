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

    CELL_WIDTH = 11

    unix_tags = [('%B', '\033[1m'),
                 ('%b', '\033[0m'),
                 ('%I', '\033[3m'),
                 ('%i', '\033[0m')]

    no_style_tags = [('%B', ''),
                     ('%b', ''),
                     ('%I', ''),
                     ('%i', '')]

    def __init__(self):

        stdout = {'stream': sys.stdout, 'when': self.EVERY_BATCH, 'tags': self.unix_tags}
        self.streams = [stdout]
        self.files = []

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

    def results(self, i, per_epoch, epoch, epochs,
                loss_components=None,
                masked_loss_components=['z_logdet', 'z_mahala', 'z_tr_inv_cov'],
                losses={},
                acc_methods=(),
                accuracies={},
                metrics=(),
                measures={},
                best_of={'odin': -1},
                time_per_i=0, batch_size=100,
                preambule='',
                end_of_epoch='\n'):

        if loss_components:
            loss_components = [_ for _ in loss_components if _ not in masked_loss_components]
        if preambule == 'train':
            preambule = '%I' + preambule
            end_of_format = '%i'
            sep = '%i' + '|' + '%I'
            double_sep = '%i' + '||' + '%I'
        else:
            end_of_format = ''
            sep = '|'
            double_sep = '||'

        K_epochs = 5
        cell_width = self.CELL_WIDTH
        K_preambule = 9

        num_format = {'default': '{' + f':{cell_width-1}.2e' + '} ',
                      'odin': '{' + f':>{cell_width}' + '}',
                      'dB': '{' + f':{cell_width-4}.1f' + '} dB '}

        kept_metrics = {}
        for k in metrics:
            if '-a-' not in k and not k.endswith('-2s'):
                kept = True
                for k_ in best_of:
                    if k.startswith(k_):
                        kept_metrics[k_] = None
                        kept = False
                if kept:
                    kept_metrics[k] = None
        metrics = list(kept_metrics)

        kept_accuracies = {}
        kept_methods = {}
        which_is_best = {}

        for k in acc_methods:
            kept = True
            for k_ in best_of:
                if k.startswith(k_):
                    kept_methods[k_] = None
                    _s = best_of[k_]
                    # ex _s = -1 ; acc = 29 k_acc = inf
                    # ex _s = -1 ; acc = inf k_acc = 29
                    # ex _s = 1 ; acc = 29 k_acc = -inf
                    if _s * accuracies.get(k, -_s * np.inf) > _s * kept_accuracies.get(k_, -_s * np.inf):
                        # print('***     best {:16} {:4.1f}'.format(k, 100 * accuracies.get(k, -_s * np.inf)))
                        kept_accuracies[k_] = accuracies[k]
                        measures[k_] = k[len(k_) + 1:]
                    else:
                        pass
                        # print('*** not best {:16} {:4.1f}'.format(k, 100 * accuracies.get(k, -_s * np.inf)))
                    kept = False

            if kept:
                kept_methods[k] = None
                if k in accuracies:
                    kept_accuracies[k] = accuracies[k]

        acc_methods = list(kept_methods)
        accuracies = kept_accuracies

        # print('*** acc_m', *acc_methods)
        if epoch == -2:
            i = per_epoch - 1
            preambule = f'{"epoch":_^{2 * K_epochs}}{preambule:_>{K_preambule}}_'
            if loss_components:
                length = len(sep.join(f'{k:^{cell_width}}' for k in loss_components))
                loss_str = f'{"losses":_^{length}}'
            else:
                loss_str = ''
            if metrics:
                length = len(sep.join(f'{k:^{cell_width}}' for k in metrics))
                metrics_str = f'{"metrics":_^{length}}'
            else:
                metrics_str = ''

        elif epoch == -1:
            i = per_epoch - 1
            preambule = f'{" ":^{2 * K_epochs}}{preambule:>{K_preambule}} '

            if loss_components:
                length = len(sep.join(f'{k:^{cell_width}}' for k in loss_components))
                loss_str = sep.join(f'{k:^{cell_width}}' for k in loss_components)
            else:
                loss_str = ''
            if metrics:
                length = len(sep.join(f'{k:^{cell_width}}' for k in metrics))
                metrics_str = sep.join(f'{k:^{cell_width}}' for k in metrics)
            else:
                metrics_str = ''
        else:
            if epoch:
                preambule = f'{epoch:{K_epochs}d}/{epochs:<{K_epochs}d} {preambule:>{K_preambule}} '
            else:
                preambule = f'{preambule:>{K_preambule + 2 * K_epochs}} '

            if loss_components:

                if losses:
                    formatted = {k: num_format.get(k, num_format['default'])
                                 for k in loss_components}
                    value = {k: losses.get(k, np.nan)
                             for k in loss_components}

                    loss_str = sep.join(formatted[k].format(value[k])
                                        for k in loss_components)
                else:
                    loss_str = sep.join(cell_width * ' ' for k in loss_components)
            else:
                loss_str = ''

            if metrics:
                if measures:
                    formatted = {k: num_format.get(k, num_format['default'])
                                 for k in metrics}
                    value = {k: measures.get(k, np.nan)
                             for k in metrics}

                    metrics_str = sep.join(formatted[k].format(value[k])
                                           for k in metrics)
                else:
                    metrics_str = sep.join(cell_width * ' ' for k in metrics)

            else:
                metrics_str = ''

        if epoch == -2:
            length = len(sep.join(f'{k:^9}' for k in acc_methods))
            acc_str = f'{"accuracy":_^{length}}'
        elif epoch == -1:
            acc_str = sep.join(f'{k:^9}' for k in acc_methods)
        elif accuracies:
            acc_str_ = []
            for k in acc_methods:
                acc_str_.append(f' {accuracies[k]:7.2%} ')
            acc_str = sep.join(acc_str_)
        else:
            acc_str = sep.join(9 * ' ' for k in acc_methods)

        if time_per_i > 0:
            time_per_i = Time(time_per_i)
            if i < per_epoch - 1:
                eta_str = f' {time_per_i / batch_size:>9}/i'
                eta_str += f'   eta: {time_per_i * (per_epoch - i):<9}'
            else:
                eta_str = f' {time_per_i / batch_size:>9}/i'
                eta_str += f' total: {time_per_i * per_epoch:<9}'

        else:
            eta_str = ' '

        strings = [s for s in [preambule,
                               loss_str,
                               metrics_str,
                               acc_str + end_of_format,
                               eta_str + end_of_format] if s]

        line = '\r' + double_sep.join(strings) + ' '
        self.write(line, when=self.EVERY_BATCH)

        if i == per_epoch - 1:
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

        units = ['d', 'h', 'm', 's', 'ms', 'mus', 'ns']
        qs = [24 * 3600, 3600, 60, 1, 1e-3, 1e-6, 1e-9]

        if t == 0:
            return '0s'

        str = '-' if t < 0 else ''
        t = abs(t)

        break_loop = False

        coarser = None
        for i, (unit, q) in enumerate(zip(units, qs)):

            if coarser is not None and i - coarser >= max - 1:
                n = int(np.round(t / q))
                break_loop = True
            else:
                n = int(t / q)
            if n:
                str += f'{n}{unit}'
                if coarser is None:
                    coarser = i
            t -= q * n
            if break_loop:
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
        return max(min(i, N-1), 0)

    start_ = f2i(start)
    end_ = f2i(end)

    chars = {_: (default if (start_ <= _ <= end_) else ' ') for _ in range(N)}

    for m in marks:
        chars[f2i(m[0])] = m[1]

    return ''.join([chars[_] for _ in range(N)])


def timerun(func):
    """ Calculate the execution time of a method and return it back"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start

        logger.debug(f"Duration of {func.__name__} function was {duration}.")
        return result
    return wrapper


if __name__ == '__main__':
    pass
