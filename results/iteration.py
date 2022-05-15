import os
import sys
import argparse
from utils.save_load import load_json, needed_remote_files, develop_starred_methods, LossRecorder
from utils.parameters import parse_filters
import numpy as np
from cvae import ClassificationVariationalNetwork as M
from utils.filters import DictOfListsOfParamFilters
import logging
from utils.parameters import gethostname
import pandas as pd
from utils.roc_curves import fpr_at_tpr
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import torch
from torch.nn.functional import one_hot
from scipy.stats import mode
from module.iteration import IteratedModels

parser = argparse.ArgumentParser()

parser.add_argument('--jobs', '-j', nargs='+', type=int, default=[])
parser.add_argument('-v', action='count', default=0)
parser.add_argument('--result-dir', default='/tmp')
parser.add_argument('--when', default='last')
parser.add_argument('--plot', nargs='?', const='p')
parser.add_argument('--tex', nargs='?', default=None, const='/tmp/r.tex')
parser.add_argument('--job-dir', default='./jobs')


if __name__ == '__main__':

    args_from_file = ('-vvvv '
                      '--tex '
                      '--jobs 193080 193082'
                      ).split()

    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)
    rmodels = load_json(args.job_dir, 'models-{}.json'.format(gethostname()))
    wanted = args.when
    
    logging.getLogger().setLevel(40 - 10 * args.v)

    if len(args.jobs) < 2:
        logging.error('At least two jobs (%d provided)', len(args.jobs))
        sys.exit(1)
    
    mdirs = [_ for _ in rmodels if rmodels[_]['job'] in args.jobs]

    if len(mdirs) < len(args.jobs):
        logging.error('Jobs not found')
        sys.exit(1)

    if len(set(rmodels[_]['set'] for _ in mdirs)) > 1:
        logging.error('Not all jobs trained on the same set')
        sys.exit(1)

    total_models = len(mdirs)
    logging.info('{} models found'.format(total_models))
    removed = False
    
    with open('/tmp/files', 'w') as f:

        for mdir, sdir in needed_remote_files(*mdirs, epoch=wanted, which_rec=None, state=True):
            logging.debug('{} for {}'.format(sdir[-30:], wanted))
            if mdir in mdirs:
                mdirs.remove(mdir)
                removed = True
                logging.info('{} is removed (files not found)'.format(mdir.split('/')[-1]))
            f.write(sdir + '\n')

    logging.info((len(mdirs), 'complete model' + ('s' if len(mdirs) > 1 else ''), 'over', total_models))
    
    if removed:
        logging.error('Exiting, load files')
        logging.error('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')
        logging.error(' Or: %s', '$ . /tmp/rsync-files remote:dir/joint-vae')
        with open('/tmp/rsync-files', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('rsync -avP --files-from=/tmp/files $1 .\n')
        sys.exit(1)

    models = [M.load(d, load_state=True) for d in mdirs]
    model = IteratedModels(*models)

    print(model.training_parameters)

    model = IteratedModels.load('/tmp/m')

    x = torch.randn(100, 3, 32, 32).to('cuda')

    out = model.evaluate(x)
    
