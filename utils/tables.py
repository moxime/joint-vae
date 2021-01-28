from cvae import ClassificationVariationalNetwork as Net
from utils.save_load import find_by_job_number
from sys import stdout
import os.path
import functools

def printout(s='', file_id=None, std=True, end='\n'):
    if file_id:
        file_id.write(s + end)
    if std:
        stdout.write(s + end)

        
def create_printout(file_id=None, std=True):
    return functools.partial(printout, file_id=file_id, std=std) 


def create_file(number, directory, filename, mode='w'):
    format_ = {int: '06d'}
    job_format = format_.get(type(number), '')
    directory = directory.replace('%j', f'{number:{job_format}}')

    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)

    return open(filepath, mode)

    
def tex_architecture(net, filename='arch.tex', directory='results/%j',
                     type_of_net = 'nettype',
                     architecture = 'netarch',
                     dataset = 'trainset',
                     stdout=False):

    f = create_file(net.job_number, directory, filename) if filename else None
    printout = create_printout(file_id=f, std=stdout)

    type_of_net_ = net.architecture['type']
    dataset_ = net.training['set']
    architecture_ = net.print_architecture(excludes='type')
    for cmd, k in zip((type_of_net, architecture, dataset),
                      (type_of_net_, architecture_, dataset_)):
        printout(f'\def\{cmd}{{{k}}}')
        
def export_losses(net, which='loss', directory='results/%j', filename='losses.tab', col_width=0, stdout=False):
    """ which is either 'loss' or 'measures' or 'all'

    """
    f = create_file(net.job_number, directory, filename)

    printout = create_printout(file_id=f, std=stdout)
    
    which = ['loss', 'measures'] if which=='all' else [which]
    history = net.train_history
    
    train_ = {w: history[f'train_{w}'] for w in which}
    test_ = {w: history[f'test_{w}'] for w in which}
    
    epochs = history['epochs']

    train_k = {w: {k: f'train_{w}_{k}' for k in train_[w][0].keys()} for w in which}
    test_k = {w: {k: f'test_{w}_{k}' for k in test_[w][0].keys()} for w in which}

    if not col_width:
        for w in which:
            for keys in (train_k[w], test_k[w]):
                for k in keys:
                    col_width = max(col_width, len(keys[k])+1)

    type_of_net = net.architecture['type']
    arch = net.print_architecture(excludes=['type'])
    training_set = net.training['set']
    printout(f'# {type_of_net} {arch} for  {training_set}')
                    
    for w in which:
        for keys in (train_k[w], test_k[w]):
            for k in keys:
                    printout(f'{keys[k]:{col_width}}', end='')

    printout()

    for epoch in range(epochs):
        for w in which:
            for keys, val_ in zip((train_k[w], test_k[w]), (train_[w][epoch], test_[w][epoch])):
                for k in keys:
                    printout(f'{val_[k]:{col_width}g}', end='')
        printout()

    f.close()
    
if __name__ == '__main__':


    net = find_by_job_number('jobs', 108367, load_net=False)[108367]['net']

    export_losses(net, which='all', col_width=0, stdout=True)
    tex_architecture(net, stdout=True) #, filename=None)
