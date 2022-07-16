import os
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
  
    begin_env = tex_command('begin', env, col_formats) 

    end_env = tex_command('end', env)
    
    return begin_env, end_env
    

def tabular_rule(where, start=1, end=-1, tab_width=None):

    if where == 'top':
        return r'\toprule' + '\n'
    if where == 'bottom':
        return r'\bottomrule' + '\n'
    if where == 'mid':
        if start == 1 and (end == -1 or end == tab_width):
            return r'\midrule' + '\n'
        if end == -1:
            end == tab_width
        border = ''
        if start > 1:
            border += 'l'
        if end < tab_width:
            border += 'r'
        border = '({})'.format(border)
        return r'\cmidrule{}{{{}-{}}}'.format(border, start, end) + '\n'


def tabular_row(*a, end='\n'):

    return ' & '.join(str(_) for _ in a) + r'\\' + end
    
def tabular_multicol(width, cell_format, s):

    return tex_command('multicolumn', width, cell_format, s)

