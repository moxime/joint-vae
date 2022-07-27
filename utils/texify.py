import os, sys
from collections import OrderedDict
from utils.save_load import create_file_for_job as create_file
from utils.tables import create_printout
from module.optimizers import Optimizer
import torch
import utils.torch_load as torchdl
import string
import pandas as pd
from utils.print_log import texify_str
from datetime import datetime
from utils.parameters import DEFAULT_RESULTS_DIR


def tex_architecture(net_dict, filename='arch.tex', directory=os.path.join(DEFAULT_RESULTS_DIR, '%j'), stdout=False,):

    net = net_dict['net']
    epoch = net_dict['epoch']
    f = create_file(net.job_number, directory, filename) if filename else None
    printout = create_printout(file_id=f, std=stdout)
    arch = net.architecture
    empty_optimizer = Optimizer([torch.nn.Parameter(torch.Tensor())], **net.training_parameters['optim'])
    oftype = net.architecture['type']
    dict_var = net.training_parameters['dictionary_variance'] if oftype == 'cvae' else 0
    beta = net.training_parameters['beta']
    trainset = net.training_parameters['set']
    sigmabeta = r'\ensuremath\sigma=' +f'{net.sigma}'.upper()
    if net.sigma.is_rmse:
        sigmabeta += f' (\\ensuremath\\beta=\\num{{{beta}}})'

    parent_set, heldout = torchdl.get_heldout_classes_by_name(trainset)
    parent_classes = torchdl.dataset_properties()[parent_set]['classes']
    classes = [c for (i, c) in enumerate(parent_classes) if i not in heldout]
    ood_results = net.ood_results.get(epoch, {})
    exported_values = dict(
        oftype=oftype,
        dataset=trainset,
        numclasses=arch['labels'],
        classes=','.join(classes),
        oodsets=','.join(ood_results.keys()),
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
        features=arch.get('features', {}).get('name', 'none'),
        sigma='{:x}'.format(net.sigma),
        beta=beta,
        dictvar=dict_var,
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
    else: f = None
    
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

    tab_cols = ['-'.join([str(c) for c in col if c]).replace('_', '-') for col in cols] 


    oodsets = [c[:-4] for c in tab_cols if c.endswith('-auc')]
    
    # print(cols, tab_cols)
    # return tab_cols
    
    to_string_args = dict(sparsify=False, index=False)

    tab_df = df.copy()
    tab_df.columns = tab_cols
    tab_df = tab_df.reset_index()
    # print('*** tab_df.cols', *tab_df.columns)
    tab_df.columns = [texify_str(c, underscore='-') for c in tab_df.columns]
    tab_df = tab_df.applymap(lambda x: texify_str(x, space='-', num=True))
    
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
                           'name': ' '.join(_r(w) for w in  c.split('-')[1:])}
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


def tex_command(command, *args):

    c = r'\{}'.format(command)

    for a in args:
        c += '{'
        c += str(a)
        c += '}'

    return c


def tabular_env(*formats, env='tabular', reduce_space=True):
    """
    formats is a list of (n, f), n is int, f is str (eg l, S[table-format=2.1])...
    """

    col_formats = ''
    if reduce_space:
        col_formats += '@{}%\n'
    for n, f in formats:
        col_formats += f * n
    if reduce_space:
        col_formats += '%\n@{}'

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

    def __init__(self, a, width=1, multicol_format=None, formatter='{}', na_rep='na'):

        assert width == 1 or multicol_format

        self._value = a
        self._multicol = multicol_format
        self._width = width
        self._format = formatter
        self.na_rep = na_rep

    @property
    def value(self):
        return self._value

    @property
    def width(self):
        return self._width
        
    def __repr__(self):

        of_width = 'of width {} '.format(self.width) if self.width > 1 else ''
        return 'cell {}containing {}'.format(of_width, self._format.format(self._value))

    def __str__(self):

        if self._value is None:
            return self.na_rep
        
        return self._format.format(self._value)

    def __format__(self, spec):

        s = ''
        tex = spec.endswith('x')
        if tex:
            tex = True
            spec = spec[:-1]
        
        if self._multicol and tex:
            s += r'\multicolumn{{{}}}{{{}}}'.format(self.width, self._multicol)
            s += '{'

        s += str(self).__format__(spec)
            
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

    def __format__(self, spec):

        tex = spec.endswith('x')
        if tex:
            spec = spec[-1]
            
        sep = '& ' if tex else ' '

        if '-' in spec:
            row_str = ''
            str_w = [int(_) for _ in spec.plits('-')]
            i0 = 0
            for i, c in enumarate(self):
                c_w = sum(str_w[i0:i0 + c.width + 1])
                # c_f = ''
                row_str += '{:{}}'.format(c, c_w)
        
        return sep.join(_.__format__(spec) for _ in self)
    
    
class TexTab(object):

    def __init__(self, *col_format, environment='tabular', float_format='{}'.format,
                 na_rep='--',
                 multicol_format='c'):

        self._env = environment
        self._col_format = col_format

        self._col_sep = ['' for _ in col_format]

        self._has_to_be_float = [_.startswith('f') for _ in col_format]
        
        self.width = len(col_format)
        
        self._rows = OrderedDict()
        self._rules = {'top': True, 'bottom': True, 'mid': {}}

        self.na_rep = na_rep
        self.float_format = float_format
        self.default_multicol_format = multicol_format
        
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
            return row_id+1

        if isinstance(row_id, str):
            splitted_id = row_id.split('-')
            try:
                k = int(splitted_id[-1])
                splitted_id[-1] = str(k + 1)
            except ValueError:
                splitted_id.append('1')
                
            return '-'.join(splitted_id)

    def _new_row(self, row_id=None):

        if row_id is None or row_id in self._rows:
            return self._new_row(self._next_row(row_id))

        else:
            self._rows[row_id] = TexRow(col_format=self._col_format)
            return row_id
        
    def _make_cell(self, a, width=1, multicol_format=None, formatter=None, has_to_be_float=None):

        try:
            float(a)
            is_float = True
        except (ValueError, TypeError):
            is_float = False

        is_multicol = width > 1 or multicol_format or (has_to_be_float and not is_float) or a is None
        multicol_format = multicol_format or self.default_multicol_format

        return TexCell(a, width=width,
                       multicol_format= multicol_format if is_multicol else None,
                       na_rep=self.na_rep,
                       formatter=formatter or (self.float_format if is_float else '{}'))

    def get(self, *a, **kw):

        return self._rows.get(*a, **kw)
    
    def add_col_sep(self, before_col, sep=''):

        self._col_sep[before_col] = '@{{{}}}'.format(sep)
        
    def render(self, io=sys.stdout):

        col_format = ''
        for f, s in zip(self._col_format, self._col_sep):
            if f.startswith('s'):
                col_format += s + 'S[table-format={}]%\n'.format(f[1:])
            else:
                col_format += s + f + '%\n'

        begin_env, end_env = tabular_env((1, col_format), env=self._env)
        
        io.write(begin_env)
        io.write('\n')

        body = ''
        for r in self:
            if r in self._rules['mid']:
                for (start, end) in self._rules['mid'][r]:
                    body += tabular_rule('mid', start=start, end=end, tab_width=self.width)

            body += '{:x}'.format(self[r])
            body += '\\\\\n'

        bottom = tabular_rule('bottom') if self._rules['bottom'] else ''
        top = tabular_rule('top') if self._rules['top'] else ''

        io.write(top)
        io.write(body[:-3])
        io.write('\n')
        io.write(bottom)
        io.write(end_env)
        io.write('\n')
            
    def append_cell(self, a, row=None, width=1, multicol_format=None, formatter=None):
        """if row is None will create a new row"""

        if row not in self:
            row = self._new_row(row)

        row_width = len(self[row])
        if row_width + width > self.width:
            raise IndexError('row {} already full'.format(row))

        has_to_be_float = self._has_to_be_float[row_width]
        self[row].append(self._make_cell(a, width=width,
                                         multicol_format=multicol_format,
                                         formatter=formatter,
                                         has_to_be_float=has_to_be_float))
        return row

    def add_midrule(self, row, start=0, end=-1):

        assert row in self._rows
        if end == -1:
            end = self.width - 1

        if start > end or end > self.width - 1:
            raise IndexError(max(start, end))
        
        if row not in self._rules['mid']:
            self._rules['mid'][row] = []

        rules = self._rules['mid'][row]
        
        if not rules or end < rules[0][0]:
            print('*** {}--{} inserted at beginning'.format(start, end))
            rules.insert(0, (start, end))
            return
        
        for i, (s,e) in enumerate(rules):
            if start > e:
                print('*** {}--{} inserted after {} -- {}'.format(start, end, s, e)) 
                rules.insert(i + 1, (start, end))
                break
        else:
            raise IndexError('Can\'t insert midrule {} -- {} '
                             'with already existing {} -- {}'.format(start, end, s, e))
                
