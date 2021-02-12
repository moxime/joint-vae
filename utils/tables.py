from cvae import ClassificationVariationalNetwork as Net
from utils.save_load import find_by_job_number
from sys import stdout
import os.path
import functools
from utils.save_load import create_file_for_job as create_file

def printout(s='', file_id=None, std=True, end='\n'):
    if file_id:
        file_id.write(s + end)
    if std:
        stdout.write(s + end)

        
def create_printout(file_id=None, std=True):
    return functools.partial(printout, file_id=file_id, std=std) 

    
def tex_architecture(net, filename='arch.tex', directory='results/%j',
                     type_of_net = 'nettype',
                     architecture = 'netarch',
                     dataset = 'trainset',
                     epochs = 'epochs',
                     stdout=False):

    f = create_file(net.job_number, directory, filename) if filename else None
    printout = create_printout(file_id=f, std=stdout)

    type_of_net_ = net.architecture['type']
    dataset_ = net.training['set']
    epochs_ = net.train_history['epochs']
    architecture_ = net.print_architecture(excludes='type', sigma=True, sampling=True)
    for cmd, k in zip((type_of_net, architecture, dataset, epochs),
                      (type_of_net_, architecture_, dataset_, epochs_)):
        printout(f'\def\{cmd}{{{k}}}')

    history = net.train_history

    for _s in ('train', 'test'):
        for _w in ('loss', 'measures', 'accuracy'):
            _b = f'{_s}_{_w}' in history
            printout(f'\{_s}{_w}{_b}'.lower())
    
        
def export_losses(net, which='loss',
                  directory='results/%j',
                  filename='losses.tab',
                  col_width=0, stdout=False):
    """ which is either 'loss' or 'measures' or 'all'

    """
    f = create_file(net.job_number, directory, filename)
    printout = create_printout(file_id=f, std=stdout)

    history = net.train_history

    sets = ['train', 'test']

    if type(which) == str:
        which = ['loss', 'measures', 'accuracy'] if which=='all' else [which]

    entries = [f'{s}_{w}' for w in which for s in sets] 

    epochs = history['epochs']
    columns = {'epochs': [e + 1 for e in range(epochs)]}
    
    for entry in entries:
        if history.get(entry, []):
            for k in history[entry][0].keys():
                columns[f'{entry}_{k}'.replace('_', '-')] = [v[k] for v in history[entry]]
            
    col_width = max(col_width, 7)
    col_width = {c: max(len(c), col_width) for c in columns}
                    
    type_of_net = net.architecture['type']
    arch = net.print_architecture(excludes=['type'])
    training_set = net.training['set']
    printout(f'# {type_of_net} {arch} for {training_set}')

    for c in columns:
        printout(f'  {c:>{col_width[c]}}', end='')

    printout()

    for epoch in range(epochs):
        for c in columns:
            printout(f'  {columns[c][epoch]:{col_width[c]}.6g}', end='')

        printout()
        
    f.close()
    
if __name__ == '__main__':

    j_ = [37]
    j_ = [107984, 37]
    j_ = [_ for _ in range(109000, 110000)]
    nets = find_by_job_number('jobs', *j_, load_net=False)

    stdout = False
    for n in nets.values():
        net = n['net']
        print(net.job_number)
        export_losses(net, which='all', col_width=0, stdout=stdout)
        tex_architecture(net, stdout=True) #, filename=None)
