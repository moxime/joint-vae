import os
import sys
from collections import OrderedDict
from utils.save_load import create_file_for_job as create_file
from utils.tables import create_printout
from module.optimizers import Optimizer
import torch
import utils.torch_load as torchdl
import string
import pandas as pd
import numpy as np
from utils.print_log import texify_str
from datetime import datetime
from utils.parameters import DEFAULT_RESULTS_DIR


def bold_best_values(data, value, format_string='{:.1f}', prec=1, highlight='\\bfseries ', max_value=99.9):

    if round(data, prec) == round(value, prec):
        return highlight + format_string.format(min(data, max_value))
    return format_string.format(min(data, max_value))


def tex_architecture(net_dict, filename='arch.tex',
                     directory=os.path.join(DEFAULT_RESULTS_DIR, '%j'),
                     stdout=False,):

    net = net_dict['net']
    epoch = net_dict['epoch']
    f = create_file(net.job_number, directory, filename) if filename else None
    printout = create_printout(file_id=f, std=stdout)
    arch = net.architecture
    empty_optimizer = Optimizer([torch.nn.Parameter(torch.Tensor())], **net.training_parameters['optimizer'])
    oftype = net.architecture['type']
    latent_prior_means = net.architecture['prior']['init_mean'] if oftype == 'cvae' else 0
    beta = net.training_parameters['beta']
    trainset = net.training_parameters['set']
    sigmabeta = r'\ensuremath\sigma=' + f'{net.sigma}'.upper()
    if net.sigma.is_rmse:
        sigmabeta += f' (\\ensuremath\\beta=\\num{{{beta}}})'

    parent_set, heldout = torchdl.get_heldout_classes_by_name(trainset)
    parent_classes = torchdl.dataset_properties()[parent_set]['classes']
    classes = [c for (i, c) in enumerate(parent_classes) if i not in heldout]
    ood_results = net.ood_results.get(epoch, {})
    ood_sets = list(ood_results)
    if trainset in ood_sets:
        ood_sets.remove(trainset)
    exported_values = dict(
        oftype=oftype,
        dataset=trainset,
        numclasses=arch['num_labels'],
        classes=','.join(classes),
        oodsets=','.join(ood_sets),
        allsets=','.join([trainset, *ood_sets]),
        allsetssepcorrect=','.join(['correct', 'incorrect', *ood_sets]),
        noodsets=len(ood_results),
        texoodsets=', '.join(['\\' + o.rstrip(string.digits) for o in ood_results.keys()]),
        epochs=net.train_history['epochs'],
        arch=net.print_architecture(excludes='type', sigma=True, sampling=True),
        archcode=net_dict['arch_code'],
        option=net.option_vector('-', '--'),
        K=arch['latent_dim'],
        L=net.training_parameters['latent_sampling'],
        encoder='-'.join(str(w) for w in arch['encoder']),
        encoderdepth=len(arch['encoder']),
        decoder='-'.join(str(w) for w in arch['decoder']),
        decoderdepth=len(arch['decoder']),
        features=arch.get('features') or 'none',
        sigma='{:x}'.format(net.sigma),
        beta=beta,
        prior_means=latent_prior_means,
        optimizer='{:3x}'.format(empty_optimizer),
        betasigma=sigmabeta,
    )

    for cmd, k in exported_values.items():
        printout(f'\def\\net{cmd}{{{k}}}')

    history = net.train_history

    for _s in ('train', 'test'):
        for _w in ('loss', 'measures', 'accuracy'):
            _b = f'{_s}_{_w}' in history
            printout(f'\{_s}{_w}{_b}'.lower())


def texify_test_results(net,
                        directory=os.path.join(DEFAULT_RESULTS_DIR, '%j'),
                        filename='res.tex',
                        which='all',
                        tpr=[0.95, 'auc'],
                        method='first',
                        stdout=False):
    """ 
    which: 'ood' or 'test' or 'all'
    method: 'first' or 'all' or a specific method (default, first)

    """
    def _pcf(x):
        if x is None:
            return '-'
        return f'{100 * x:5.2f}'

    if filename:
        f = create_file(net['job'], directory, filename)
    else:
        f = None

    printout = create_printout(file_id=f, std=stdout)

    show_ood = which in ('all', 'ood')
    show_test = which in ('all', 'test')
    all_methods = method == 'all'

    ood_methods = net['net'].ood_methods
    accuracies = net['accuracies']

    if not accuracies:
        printout('no result')
        return

    if not net['ood_fpr']:
        show_ood = False
    elif not list(net['ood_fpr'].values())[0]:
        show_ood = False

    header = dict()

    if show_test:
        header[net['set']] = len(accuracies) - 1 if all_methods else 1
    if show_ood:
        ood_sets = list(net['ood_fprs'])
        if not all_methods:
            ood_methods = ood_methods[:1]
        for dataset in net['ood_fprs']:
            fprs = net['ood_fprs'][dataset]
            header[dataset] = len(tpr) * ((len(fprs) - 1) if all_methods else 1)

    n_cols = sum(c for c in header.values())
    col_style = 'l'
    printout('\\begin{tabular}')
    printout(f'{{{col_style * n_cols}}}')
    printout('\\toprule')
    printout(' & '.join(f'\\multicolumn{cols}c{{{dataset}}}'
                        for dataset, cols in header.items()))
    printout('\\\\ \\midrule')
    if all_methods:
        if show_test:
            printout(' & '.join(list(accuracies)[:-1]), end='& ' if show_ood else '\n')
        if show_ood:
            printout(' & '.join(
                ' & '.join(f'\\multicolumn{len(tpr)}c{{{_}}}' for _ in ood_methods)
                for s in ood_sets))
        printout('\\\\')
    if show_ood and len(tpr) > 1:
        printout('    &' * header[net['set']], end=' ')
        printout(' & '.join(' & '.join(' & '.join(str(t) for t in tpr)
                                       for _ in range(header[dataset] // len(tpr)))
                            for dataset in ood_sets))
        printout('\\\\ \\midrule')
    if show_test:
        acc = list(accuracies.values())[:-1] if all_methods else [accuracies['first']]
        printout(' & '.join(_pcf(a) for a in acc), end=' & ' if show_ood else '\n')
    if show_ood:
        ood_ = []
        for dataset in net['ood_fprs']:
            if all_methods:
                fprs = list(net['ood_fprs'][dataset].values())[:-1]
            else:
                fprs = [net['ood_fprs'][dataset].get('first', None)]
                # print('*** fprs', *fprs)
            ood_.append(' & '.join(' & '.join((_pcf(m.get(t, None)) if m is not None else '-')
                                              for t in tpr) for m in fprs))
        printout(' & '.join(ood_))

    printout('\\\\ \\bottomrule')
    printout('\\end{tabular}')


def texify_test_results_df(df, dataset, tex_file, tab_file, tab_code=None):

    datasets = torchdl.get_same_size_by_name(dataset)
    datasets.append(dataset)

    replacement_dict = {'sigma': r'$\sigma$',
                        'optim_str': 'Optim',
                        'auc': r'\acron{auc}',
                        'measures': '',
                        'rmse': r'\acron{rmse}',
                        'rate': '',
                        }

    def _r(w, macros=datasets, rdict=replacement_dict):

        if w in macros:
            return f'\\{w.rstrip(string.digits)}'

        try:
            float(w)
            return f'\\num{{{w}}}'
        except ValueError:
            pass
        return replacement_dict.get(w, w.replace('_', ' '))

    cols = df.columns

    tex_cols = pd.MultiIndex.from_tuples([tuple(_r(w) for w in c) for c in cols])

    tab_cols = ['-'.join([str(c) for c in col if c]).replace('_', '-').replace('~', '^') for col in cols]

    oodsets = [c[:-4] for c in tab_cols if c.endswith('-auc')]

    # print(cols, tab_cols)
    # return tab_cols

    to_string_args = dict(sparsify=False, index=False)

    tab_df = df.copy()
    tab_df.columns = tab_cols
    tab_df = tab_df.reset_index()
    # print('*** tab_df.cols', *tab_df.columns)
    tab_df.columns = [texify_str(c, underscore='-') for c in tab_df.columns]
    tab_df = tab_df.map(lambda x: texify_str(x, space='-', num=True))

    if 'job' in tab_df.columns:
        tab_df = tab_df.set_index('job').reset_index()

    levels = df.columns.nlevels

    if tex_file:
        with open(tex_file, 'w') as f:

            f.write(f'% Generated on {datetime.now()}\n')
            f.write(f'\\def\\setname{{{dataset}}}\n')
            f.write(f'\\def\\testcolumn{{{dataset}-rate}}\n')

            f.write(r'\def\oodsets{')
            f.write(','.join(oodsets))
            f.write(r'}')
            f.write('\n')

            if oodsets:
                f.write(r'\def\oodset{')
                f.write(oodsets[0])
                f.write(r'}')
                f.write('\n')

            f.write(r'\def\noodsets{')
            f.write(str(len(oodsets)))
            f.write(r'}')
            f.write('\n')

            # colors = ['green', 'magenta', 'cyan']
            # f.write(r'\pgfplotscreateplotcyclelist{my colors}{')
            # f.write(','.join([f'{{color={c}}}' for c in colors[:len(oodsets)]]))
            # f.write(r'}')
            # f.write('\n')

            done_col = [c for c in tab_df if c.endswith('done')]
            if done_col:
                total_epochs = tab_df[done_col[0]].sum()
                f.write(r'\def\totalepochs{')
                f.write(f'{total_epochs}')
                f.write(r'}')
                f.write('\n')

            file_code = tab_code or hashlib.sha1(bytes(tab_file, 'utf-8')).hexdigest()[:6]
            f.write(r'\def\tabcode{')
            f.write(f'{file_code}')
            f.write(r'}')
            f.write('\n')

            if 'measures-dict-var' in tab_df:
                unique_dict_vars = tab_df['measures-dict-var'].unique()
                unique_dict_vars.sort()
                f.write(r'\def\tabdictvars{')
                f.write(','.join(str(a) for a in unique_dict_vars))
                f.write(r'}')
                f.write('\n')

            if 'sigma' in tab_df:
                unique_sigmas = sorted(tab_df['sigma'].unique(), key=lambda x: (str(type(x)), x))
                f.write(r'\def\tabsigmas{')
                f.write(','.join(str(a) for a in unique_sigmas))
                f.write(r'}')
                f.write('\n')

            f.write(f'\\pgfplotstableread{{{tab_file}}}{{\\testtab}}')
            f.write('\n')

            tab_cols = [_ for _ in tab_cols if not _.startswith('std-')]
            f.write(r'\def\typesetwithmeasures#1{\pgfplotstabletypeset[columns={')
            f.write(','.join(tab_cols))
            # f.write('job,type')
            f.write(r'},')
            f.write(r'#1')
            f.write(r']{\testtab}}')
            f.write('\n')

            tab_cols_wo_measures = [c for c in tab_cols if 'measures-' not in c]
            f.write(r'\def\typeset#1{\pgfplotstabletypeset[columns={')
            f.write(','.join(tab_cols_wo_measures))
            # f.write('job,type')
            f.write(r'},')
            f.write(r'#1')
            f.write(r']{\testtab}}')
            f.write('\n')

    if tab_file:
        with open(tab_file, 'w') as f:
            tab_df.to_string(buf=f, **to_string_args)


def pgfplotstable_preambule(df, dataset, file, mode='a'):
    replacement_dict = {'rmse': 'RMSE'}

    def _r(s, f=string.capwords):
        return replacement_dict.get(s, f(s))

    oodsets = torchdl.get_same_size_by_name(dataset)

    with open(file, mode) as f:
        f.write('\pgfplotstableset{%\n')
        cols = {c: {} for c in df.columns}
        for c in df.columns:
            if c.startswith('measures'):
                cols[c] = {'style': 'sci num',
                           'name': ' '.join(_r(w) for w in c.split('-')[1:])}
            elif c.startwith(dataset):
                cols[c] = {'style': 'fixed num',
                           'name': '\\' + dataset.rstrip(string.digits)}
            elif c.startswith(tuple(oodsets)):
                w_ = c.split('-')
                w_[0] = '\\' + w[0]
                for i, w in enumerate(w_[1:]):
                    try:
                        float(w)
                        w_[i + 1] = '@FPR=' + w
                    except ValueError:
                        pass

                cols[c] = {'style': 'fixed num',
                           'name': ' '.join(w_)}


tex_faces = {'it': r'\itshape ', 'bf': r'\bfseries '}


def tex_command(command, *args):

    c = r'\{}'.format(command)

    for a in args:
        c += '{'
        c += str(a)
        c += '}'

    return c


def tabular_env(formats, col_seps, env='tabular', reduce_space=True):
    """
    formats is a list of (n, f), n is int, f is str (eg l, S[table-format=2.1])...
    """
    col_seps_tex = [''] * (len(formats) + 1)

    for i in range(len(formats) + 1):
        col_seps_tex[i] = '@{' + col_seps[i] + '}' if col_seps[i] else ''

    for i in (0, -1):
        col_seps_tex[i] = '@{' + col_seps[i] + '}'

    col_formats = '%\n'
    for f, s in zip(formats, col_seps_tex):
        col_formats += s + f + '%\n'
    col_formats += col_seps_tex[-1] + '%\n'

    if env:
        begin_env = tex_command('begin', env, col_formats)
        end_env = tex_command('end', env)
        return begin_env, end_env

    return col_formats


def tabular_rule(where, start=0, end=-1, tab_width=None):

    if where == 'top':
        return r'\toprule' + '\n'
    if where == 'bottom':
        return r'\\\bottomrule' + '\n'
    if where == 'mid':
        if not start and (end == -1 or end == tab_width - 1):
            return r'\midrule' + '\n'
        if end == -1:
            end = tab_width - 1

        border = ''
        if start:
            border += 'l'
        if end < tab_width - 1:
            border += 'r'
        border = '({})'.format(border)
        return r'\cmidrule{}{{{}-{}}}'.format(border, start + 1, end + 1) + '\n'


def tabular_row(*a, end='\n'):

    return ' & '.join(str(_) for _ in a) + r'\\' + end


def tabular_multicol(width, cell_format, s):

    return tex_command('multicolumn', width, cell_format, s)


class TexCell(object):

    def __init__(self, a, width=1, multicol_format=None, formatter='{}', na_rep='na', face=None):

        assert width == 1 or multicol_format

        self._value = a
        self._multicol = multicol_format
        self._width = width
        self._formatted_str = formatter
        self.na_rep = na_rep
        self._face = face

    @property
    def value(self):
        return self._value

    @property
    def width(self):
        return self._width

    @property
    def face(self):
        return self._face

    @face.setter
    def face(self, f):
        assert f in tex_faces
        self._face = f

    def __eq__(self, other):

        return self._value == other

    def __repr__(self):

        of_width = 'of width {} '.format(self.width) if self.width > 1 else ''
        return 'cell {}containing {}'.format(of_width, self._formatted_str.format(self._value))

    def __str__(self):

        # try:
        #     is_nan = np.isnan(self._value)
        #     if is_nan:
        #         return self.na_rep
        # except TypeError:
        #     pass
        if self._value is None:
            return self.na_rep

        return self._formatted_str.format(self._value)

    def __format__(self, spec):

        s = ''
        tex = spec.endswith('x')
        if tex:
            tex = True
            spec = spec[:-1]

        if self._multicol and tex:
            s += r'\multicolumn{{{}}}{{{}}}'.format(self.width, self._multicol)
            s += '{'

        if tex and self.face:
            s += tex_faces.get(self._face, '') + ' '  # '{'

        s += str(self).__format__(spec)

        if tex and self.face:
            s += ''  # '}'
        if self._multicol and tex:
            s += '}'

        return s


class TexRow(list):

    def __init__(self, *a, col_format=[]):

        super().__init__(*a)
        self._col_formats = col_format

    def __len__(self):

        return sum(_.width for _ in self)

    def __str__(self):

        return ' '.join(str(_) for _ in self)

    def render(self, spec='x', prev_row_for_sparse=[]):

        while len(prev_row_for_sparse) < len(self):
            prev_row_for_sparse.append('xxxfooxxx')

        tex = spec.endswith('x')
        if tex:
            spec = spec[-1]

        sep = '& ' if tex else ' '

        if '-' in spec:
            row_str = ''
            str_w = [int(_) for _ in spec.plits('-')]
            i0 = 0
            for i, c in enumerate(self):
                c_w = sum(str_w[i0:i0 + c.width + 1])
                # c_f = ''
                row_str += '{:{}}'.format(c, c_w)

        return sep.join(c.__format__(spec) if c != prev_row_for_sparse[i] else '' for i, c in enumerate(self))

    def __format__(self, spec):

        print('**** format spec:', spec)
        return self.render(spec)


class TexTab(object):

    def __init__(self, *col_format, environment='tabular', float_format='{}',
                 sparse_index_width=0,
                 na_rep='--',
                 multicol_format='c'):

        try:
            float_format.format(4.54)
        except (IndexError, ValueError):
            raise ValueError(float_format + ' is not a valid float format')

        self._env = environment
        self._col_format = col_format

        self._col_sep = ['' for _ in col_format] + ['']

        self._has_to_be_float = [_.startswith('s') for _ in col_format]

        self.width = len(col_format)

        self._rows = OrderedDict()
        self._rules = {'top': True, 'bottom': True, 'mid': {}}

        self._comments = {}

        self.na_rep = na_rep
        self.float_format = float_format
        self.default_multicol_format = multicol_format

        self._sparse_index_witdth = sparse_index_width

    def __repr__(self):

        return 'TeX tab of format {} with {} rows'.format(self._col_format, len(self._rows))

    def __str__(self):

        return '\n'.join(str(self[_]) for _ in self)

    def __len__(self):

        return len(self._rows)

    def __iter__(self):

        return self._rows.__iter__()

    def __getitem__(self, row):

        return self._rows[row]

    @classmethod
    def _next_row(cls, row_id):

        if row_id is None:
            return 0

        if isinstance(row_id, int):
            return row_id + 1

        if isinstance(row_id, str):
            splitted_id = row_id.split('-')
            try:
                k = int(splitted_id[-1])
                splitted_id[-1] = str(k + 1)
            except ValueError:
                splitted_id.append('1')

            return '-'.join(splitted_id)

    def _new_row(self, row_id=None):
        """
        if row_id = -1: will throw an error
        """
        # assert row_id != -1, 'Please choose a row_id that is not {}'.format(row_id)

        if row_id is None or row_id in self._rows:
            return self._new_row(self._next_row(row_id))

        else:
            self._rows[row_id] = TexRow(col_format=self._col_format)
            return row_id

    def _make_cell(self, a, width=1, multicol_format=None,
                   formatter=None, has_to_be_float=None, face=None, seps=('', '')):

        try:
            float(a)
            is_float = not np.isnan(a)
            if np.isnan(a):
                a = None
        except (ValueError, TypeError):
            is_float = False

        is_multicol = width > 1 or multicol_format or (has_to_be_float and not is_float) or a is None
        multicol_format = multicol_format or self.default_multicol_format

        multicol_format = (('@{}' if seps[0] else '') +
                           multicol_format +
                           ('@{{{}}}'.format(seps[1]) if seps[1] else ''))

        return TexCell(a, width=width,
                       multicol_format=multicol_format if is_multicol else None,
                       na_rep=self.na_rep,
                       face=face,
                       formatter=formatter or (self.float_format if is_float else '{}'))

    def get(self, *a, **kw):

        return self._rows.get(*a, **kw)

    def add_col_sep(self, before_col, sep=''):

        self._col_sep[before_col] = sep

    def render(self, io=sys.stdout, robustify=True):

        for comment in self._comments.get(None, []):
            io.write(comment)
            io.write('\n')

        if robustify:
            for _ in tex_faces:
                io.write(r'\robustify' + tex_faces[_] + '\n')

        col_formats = []
        for i, f in enumerate(self._col_format):
            if f.startswith('s'):
                col_formats.append('S[table-format={}]'.format(f[1:]))
            else:
                col_formats.append(f)

        begin_env, end_env = tabular_env(col_formats, self._col_sep, env=self._env)

        io.write(begin_env)
        io.write('\n')

        body = ''
        prev_row_for_sparse = []
        for r in self:
            if r in self._rules['mid']:
                for (start, end) in self._rules['mid'][r]:
                    body += tabular_rule('mid', start=start, end=end, tab_width=self.width)

            body += self[r].render('x', prev_row_for_sparse=prev_row_for_sparse)
            prev_row_for_sparse = self[r][:self._sparse_index_witdth]

            body += '\\\\\n'

            for comment in self._comments.get(r, []):
                body += comment
                body += '\n'

        bottom = tabular_rule('bottom') if self._rules['bottom'] else ''
        top = tabular_rule('top') if self._rules['top'] else ''

        io.write(top)
        io.write(body[:-3])
        io.write('\n')
        io.write(bottom)
        io.write(end_env)
        io.write('\n')

        for comment in self._comments.get(-1, []):
            io.write(comment)
            io.write('\n')

    def append_cell(self, a, row=None, width=1, multicol_format=None, formatter=None, face=None):
        """if row is None will create a new row"""

        if row not in self:
            row = self._new_row(row)

        row_width = len(self[row])
        if row_width + width > self.width:
            raise IndexError('row {} already full'.format(row))

        has_to_be_float = self._has_to_be_float[row_width]
        seps = (self._col_sep[row_width], self._col_sep[row_width + width])
        self[row].append(self._make_cell(a, width=width,
                                         multicol_format=multicol_format,
                                         face=face,
                                         formatter=formatter,
                                         has_to_be_float=has_to_be_float,
                                         seps=seps))
        return row

    def add_midrule(self, row, start=0, end=-1, after=False):

        assert row in self._rows

        if after:
            row_list = list(self._rows)
            row = row_list[row_list.index(row) + 1]

        if end == -1:
            end = self.width - 1

        if start > end or end > self.width - 1:
            raise IndexError(max(start, end))

        if row not in self._rules['mid']:
            self._rules['mid'][row] = []

        rules = self._rules['mid'][row]

        if not rules or end < rules[0][0]:
            rules.insert(0, (start, end))
            return

        for i, (s, e) in enumerate(rules):
            if start > e:
                rules.insert(i + 1, (start, end))
                break
        else:
            raise IndexError('Can\'t insert midrule {} -- {} '
                             'with already existing {} -- {}'.format(start, end, s, e))

    def comment(self, s, row=None):
        """ if row is None, will be added before header, if ==-1, added after footer """

        if row not in self._comments:
            self._comments[row] = []

        self._comments[row].append('% ' + s.strip('\n'))

    @classmethod
    def from_dataframe(df):
        raise NotImplementedError('TexTab.from_dataframe(df)')


if __name__ == '__main__':
    from numpy import nan

    tab = TexTab('l', 'r', 's3.1', 's3.1', float_format='{:.3f}', sparse_index_width=1, na_rep='BOGUS')
    tab.add_col_sep(2, '/')
    tab.append_cell('', row=0)
    tab.append_cell(None, row=0)
    tab.append_cell('fg', row=0)
    tab.append_cell(nan)
    tab.append_cell(None, row=1, face='it')
    tab.append_cell('fr', width=2, row=1)
    tab.append_cell('fr', width=2, row=2)
    tab.append_cell('fr', width=2, row=3)
    tab.append_cell(1, row=3)
    tab.add_midrule(1, start=1)
    # tab.add_midrule(2, before=True)

    # tab.comment('Generated')
    # tab.comment('Does that work?', row=1)
    # tab.comment('Does that work?', row=1)
    # tab.comment('Last comment', row=-1)
    # tab.comment('Last comment', row=-1)

    print('\n**\n' * 5)
    tab.render()
