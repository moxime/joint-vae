import os
import sys
import argparse
from utils.save_load import load_json, needed_remote_files, develop_starred_methods, LossRecorder
from utils.parameters import create_filter_parser
import numpy as np
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters
import logging
from utils.parameters import gethostname
import pandas as pd
from utils.roc_curves import fpr_at_tpr

parser = argparse.ArgumentParser()

parser.add_argument('--last', default=0, type=int)
# parser.add_argument('--metrics', nargs='*', default=['zdist'])
parser.add_argument('--metrics', default='zdist')
parser.add_argument('--compute', action='store_true')
parser.add_argument('--soft', action='store_true')
parser.add_argument('--by-classes', action='store_true')
parser.add_argument('-v', action='count', default=0)
parser.add_argument('--result-dir', default='/tmp')
parser.add_argument('--when', default='last')
parser.add_argument('--kept-temps', nargs='*', type=int, default=[])

rmodels = load_json('jobs', 'models-{}.json'.format(gethostname()))

col_width = 10
str_col_width = '13.13'
flt_col_width = '5.1f'

tpr = 0.95

np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':

    args_from_file = ('--dataset mnist '
                      '--type cvae '
                      # '--gamma 500 '
                      '--sigma-train learned '
                      '--learned-prior-mean true '
                      '--latent-prior-variance scalar '
                      '--last 1 '
                      '--kept-temps 1 2 5 10 20 50 100 '
                      '-vv '
                      ).split()

    # args_from_file = '--job-num 149432'.split()
    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)
    wanted = args.when
    kept_temps = [np.nan] + args.kept_temps

    logging.getLogger().setLevel(40 - 10 * args.v)
    
    metrics_for_mis = args.metrics

    filter_parser = create_filter_parser()
    filter_args = filter_parser.parse_args(ra)

    filters = DictOfListsOfParamFilters()

    for _ in filter_args.__dict__:
        filters.add(_, filter_args.__dict__[_])

    mdirs = [_ for _ in rmodels if filters.filter(rmodels[_])][-args.last:]

    total_models = len(mdirs)
    with open('/tmp/files', 'w') as f:

        for mdir, sdir in needed_remote_files(*mdirs, epoch=wanted, which_rec='ind', state=False):
            if mdir in mdirs:
                mdirs.remove(mdir)
            f.write(sdir + '\n')

    print(len(mdirs), 'complete model' + ('s' if len(mdirs) > 1 else ''), 'over', total_models)
    
    if not mdirs:
        logging.warning('Exiting, load files')
        logging.warning('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')

    for mdir in mdirs:
        
        model = M.load(mdir, load_state=False)
        testset = model.training_parameters['set']

        if args.compute:
            model.misclassification_detection_rate(predict_methods='first', epoch=args.when,
                                                   misclass_methods='all',
                                                   # misclass_methods='softmahala-20',
                                                   # from_where=('recorders')
                                                   )
            model.save(model.saved_dir)

        predict_method = model.predict_methods[0]

        if args.when == 'min-loss':
            epoch = model.training_parameters.get('early-min-loss', 'last')

        if args.when == 'last' or epoch == 'last':
            epoch = max(model.testing)
            
        results = model.testing[epoch][predict_method]

        misclass_methods = {_: None for _ in develop_starred_methods(model.misclass_methods, model.methods_params)}

        acc = results['accuracy']
        print('*** {} accuracy ({}) {:.2%}%'.format(model.job_number, predict_method, acc)) 
        fpr = []
        P = []
        for m in misclass_methods:
            T = None
            e = None
            m_ = m.split('-')
            if len(m_) >= 2:
                T = float(m_[1])
            if len(m_) == 3:
                e = float(m_[2])
            misclass_methods[m] = (m_[0], T, e)

            fpr.append(100 * fpr_at_tpr(results[m]['fpr'], results[m]['tpr'], tpr))
            P.append(100 * fpr_at_tpr(results[m]['precision'], results[m]['tpr'], tpr))
            
        idx = pd.MultiIndex.from_tuples(misclass_methods.values(), name=['method', 'temp', 'eps'])
        rename = {'softzdist': 'soft-euclid', 'zdist': 'euclid'}
        
        df = pd.DataFrame.from_records({'prec': P, 'fpr': fpr}, index=idx)

        df['dP'] = df['prec'] - 100 * acc

        i_E = df.index.get_level_values('eps').isin([np.nan, 0])

        drop_eps = False
        if not len(idx.levels[2]):
            df.reset_index('eps', drop=True, inplace=True)
            drop_eps = True

        index_rename = {'zdist': 'euclid', 'mahala': 'mahala', 'kl': 'KL', 'iws': 'IWS'}
        index_rename.update({'soft' + k: v for k, v in index_rename.items()})
        df.rename(columns={'fpr': 'FPR', 'prec': 'P'}, index=index_rename, inplace=True)
        df.index = df.index.rename({'temp': 'T'})
        
        i_T = df.index.get_level_values('T').isin(kept_temps)

        i_ = i_T & i_E

        if not drop_eps:
            df.reset_index('eps', drop=True, inplace=True)
        
        _f = dict(float_format='{:.1f}'.format, na_rep='')
        
        df = df[i_]

        df.sort_index(level='T', na_position='first', inplace=True)

        index_sort = {_: i/1000 for i, _ in enumerate(set(df.index.get_level_values('method')))}

        index_sort['hyz'] = 10
        index_sort['euclid'] = 20
        index_sort['mahala'] = 30
        index_sort['fisher_rao'] = 40
        index_sort['KL'] = 50
        index_sort['kl_rec'] = 45
        index_sort['baseline'] = 15
        
        def sorting_callable(idx):
            return pd.Index([index_sort[_] for _ in idx], name=idx.name)
        
        df.sort_index(level='method', inplace=True, key=sorting_callable)
        
        print(df.to_string(**_f))

        rfile = os.path.join(args.result_dir, 'misclass-{}'.format(model.job_number))
        df.to_latex(buf=rfile, **_f, formatters={'dP': '{:+.2f}'.format})
        
        r = LossRecorder.loadall(os.path.join(mdir, 'samples', '{:04d}'.format(epoch)))[testset]._tensors
                                         
