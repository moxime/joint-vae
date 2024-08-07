from utils.save_load import needed_remote_files, load_json, LossRecorder
import argparse
import logging
import torch
from utils.print_log import turnoff_debug
import sys
import os
from utils.parameters import gethostname
from cvae import ClassificationVariationalNetwork as M
import utils.torch_load as tl

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
    parser.add_argument('--accuracy', action='store_true')
    parser.add_argument('--ood', action='store_true')

    args_from_file = ('-vvvv '
                      '--jobs 193080 193082'
                      ).split()

    args = parser.parse_args(None if len(sys.argv) > 1 else args_from_file)

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

        opt = dict(which_rec='none', state=True)

        for mdir, sdir in needed_remote_files(*mdirs, epoch=wanted, **opt):
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
        sys.exit(1)

    models = [M.load(d, load_state=True) for d in mdirs]

    device = args.device if torch.cuda.is_available() else 'cpu'

    for _ in mdirs:

        model = M.load(_, load_state=True)
        model.to(device)

        dset = rmodels[_]['set']

        all_sets = tl.get_same_size_by_name(dset)
        all_sets.append(dset)

        job = rmodels[_]['job']

        num_batch = args.num_batch
        batch_size = args.batch_size

        recorders = {_: LossRecorder(batch_size) for _ in all_sets}

        _s = '*** Computing accuracy for {} on model # {} with {} images on {}'
        print(_s.format(dset, job,
                        num_batch * batch_size,
                        next(model.parameters()).device))

        sample_dir = os.path.join('/tmp/reload/samples', str(job))
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        with turnoff_debug():
            with torch.no_grad():

                print('ACCURACY')
                if args.accuracy:
                    acc = model.accuracy(batch_size=args.batch_size,
                                         num_batch=args.num_batch,
                                         sample_dirs=[sample_dir],
                                         from_where=('compute'),
                                         print_result='ACC',
                                         update_self_testing=False,
                                         )

                print('OOD DETECTION')
                if args.ood:
                    ood = model.ood_detection_rates(batch_size=batch_size,
                                                    num_batch=num_batch,
                                                    print_result='OOD',
                                                    sample_dirs=[sample_dir],
                                                    update_self_ood=False,
                                                    recorders=recorders,
                                                    from_where=('compute'))
