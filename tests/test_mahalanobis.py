from cvae import ClassificationVariationalNetwork as M
import sys
import os
import argparse
import logging
from utils.parameters import parse_filters
from utils.filters import DictOfListsOfParamFilters
from utils.save_load import load_json, needed_remote_files, LossRecorder
from utils.torch_load import get_classes_by_name

rmodels = load_json('jobs', 'models-home.json')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--last', default=0, type=int)
    parser.add_argument('-v', action='count', default=1)

    args_from_file = ('--dataset cifar10 '
                      '--type cvae '
                      '--gamma 500 '
                      '--sigma-train coded '
                      '--coder-dict learned '
                      '--last 1'
                      # '--job-num 149127'
                      ).split()

    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)

    logging.getLogger().setLevel(40 - 10 * args.v)
    
    filter_parser = parse_filters()
    filter_args = filter_parser.parse_args(ra)

    filters = DictOfListsOfParamFilters()

    for _ in filter_args.__dict__:
        filters.add(_, filter_args.__dict__[_])

    mdirs = [_ for _ in rmodels if filters.filter(rmodels[_])][-args.last:]

    total_models = len(mdirs)
    with open('/tmp/files', 'w') as f:

        for mdir, sdir in needed_remote_files(*mdirs, epoch='last', which_rec='ind', state=True):
            if mdir in mdirs:
                mdirs.remove(mdir)
            f.write(sdir + '\n')

    print(len(mdirs), 'complete model' + ('s' if len(mdirs) > 1 else ''), 'over', total_models)
    
    if not mdirs:
        logging.warning('Exiting, load files')
        logging.warning('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')

    for mdir in mdirs:

        model = M.load(mdir, load_state=True)
        testset = model.training_parameters['set']

        oodsets = []
        
        epoch = 'last'

        epoch_str = '{:0>4}'.format(epoch)
        
        print('__', model.job_number, testset,
              '@',  model.training_parameters.get('early-min-loss'))

        record_dir = os.path.join(mdir, 'samples', epoch_str)
        recorders = LossRecorder.loadall(record_dir)

        is_testset = True

        classes_ = get_classes_by_name(testset)  # + ['OOD']

        
