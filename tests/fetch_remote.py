from utils.save_load import needed_remote_files, load_json
import argparse
import logging
import torch
from utils.print_log import turnoff_debug
import sys
from utils.parameters import gethostname
from cvae import ClassificationVariationalNetwork as M

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', '-j', nargs='+', type=int, default=[])
    parser.add_argument('-v', action='count', default=0)
    parser.add_argument('--result-dir', default='/tmp')
    parser.add_argument('--when', default='last')
    parser.add_argument('--job-dir', default='./jobs')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-batch', type=int, default=10)
    parser.add_argument('--device', default='cuda')

    args_from_file = ('-vvvv '
                      '--jobs 311396'
                      ).split()

    exec_from_file = len(sys.argv) <= 1
    args = parser.parse_args(args_from_file if exec_from_file else None)

    rmodels = load_json(args.job_dir, 'models-{}.json'.format(gethostname()))
    wanted = args.when

    logging.getLogger().setLevel(40 - 10 * args.v)

    mdirs = [_ for _ in rmodels if rmodels[_]['job'] in args.jobs]

    if len(mdirs) < len(args.jobs):
        logging.error('Jobs not found')
        sys.exit(1)

    total_models = len(mdirs)
    logging.info('{} models found'.format(total_models))
    removed = False

    with open('/tmp/files', 'w') as f:

        for mdir, sdir in needed_remote_files(*mdirs, epoch=wanted, which_rec='all', missing_file_stream=f):
            logging.debug('{} for {}'.format(sdir[-30:], wanted))
            if mdir in mdirs:
                mdirs.remove(mdir)
                removed = True
                logging.info('{} is removed (files not found)'.format(mdir.split('/')[-1]))
            f.write(sdir + '\n')

    logging.info('{} model{} over {}'.format(len(mdirs), 's' if len(mdirs) > 1 else '', total_models))

    if removed:
        logging.error('Exiting, load files')
        logging.error('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')
        logging.error(' Or: %s', '$ . /tmp/rsync-files remote:dir/joint-vae')
        with open('/tmp/rsync-files', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('rsync -avP --files-from=/tmp/files $1 .\n')
        if exec_from_file:
            raise KeyboardInterrupt
        else:
            sys.exit(1)

    # models = [M.load(d, load_state=True) for d in mdirs]

    # for _ in mdirs:

    #     model = M.load(_, load_state=True)
    #     dset = rmodels[_]['set']
    #     job = rmodels[_]['job']
    #     print(job)
