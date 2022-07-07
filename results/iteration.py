import os
import sys
import argparse
from utils.save_load import load_json, needed_remote_files, develop_starred_methods, LossRecorder
from utils.torch_load import get_dataset, get_same_size_by_name
import numpy as np
from cvae import ClassificationVariationalNetwork as M
import logging
from utils.parameters import gethostname
import torch
from module.iteration import IteratedModels, iterate_with_prior
import time
parser = argparse.ArgumentParser()

parser.add_argument('--jobs', '-j', nargs='+', type=int, default=[])
parser.add_argument('-v', action='count', default=0)
parser.add_argument('--result-dir', default='/tmp')
parser.add_argument('--when', default='last')
parser.add_argument('--prior', action='store_true')
parser.add_argument('--plot', nargs='?', const='p')
parser.add_argument('--tex', nargs='?', default=None, const='/tmp/r.tex')
parser.add_argument('--job-dir', default='./jobs')

parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-batch', type=int, default=int(1e6))
parser.add_argument('--device', default='cuda')

parser.add_argument('--saved-samples-per-batch', type=int, default=2)

if __name__ == '__main__':

    args_from_file = ('-vvvv '
                      '--prior '
                      #                      '--jobs 193080 193082'
                      '--jobs 169381'
                      ).split()

    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)
    rmodels = load_json(args.job_dir, 'models-{}.json'.format(gethostname()))
    wanted = args.when
    
    logging.getLogger().setLevel(40 - 10 * args.v)

    if len(args.jobs) < 2:
        logging.error('At least two jobs (%d provided)', len(args.jobs))
        sys.exit(1)
    
    mdirs_ = {rmodels[_]['job']: _  for _ in rmodels if rmodels[_]['job'] in args.jobs}
    
    if len(mdirs_) < len(args.jobs):
        logging.error('Jobs not found')
        sys.exit(1)

    mdirs = [mdirs_[j] for j in args.jobs]
        
    if len(set(rmodels[_]['set'] for _ in mdirs)) > 1:
        logging.error('Not all jobs trained on the same set')
        sys.exit(1)

    total_models = len(mdirs)
    logging.info('{} models found'.format(total_models))
    removed = False
    
    with open('/tmp/files', 'w') as f:

        opt = dict(which_rec='none', state=True) if not args.prior else dict(which_rec='ind')
        
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

    if args.prior:
        kept_testset = None
        list_of_logp_x_y = []
        for mdir in mdirs:
            model = M.load(mdir, load_net=False)
            testset = model.training_parameters['set']
            if kept_testset and testset != kept_testset:
                continue
            else:
                kept_testset = testset

            if args.when == 'min-loss':
                epoch = model.training_parameters.get('early-min-loss', 'last')

            if args.when == 'last' or epoch == 'last':
                epoch = max(model.testing)

            recorders = LossRecorder.loadall(os.path.join(mdir, 'samples', '{:04d}'.format(epoch)), device='cpu')

            rec = recorders[testset]
            list_of_logp_x_y.append(rec._tensors['iws'].softmax(0))
            y_true = recorders[kept_testset]._tensors['y_true']
            sets = [*recorders.keys()]

            # exclude rotated set
            oodsets = [_ for _ in sets if not _.startswith(kept_testset)]
            # sets = [kept_testset, 'lsunr']  # + sets
            sets = [kept_testset] + oodsets

        logp_x_y = torch.stack(list_of_logp_x_y)
        p_y_x = iterate_with_prior(logp_x_y)

    else:

        models = [M.load(d, load_state=True) for d in mdirs]
        model = IteratedModels(*models)

        device = args.device

        model.to(device)
        
        testset = model.training_parameters['set']
        allsets = [testset]
        allsets.extend(get_same_size_by_name(testset))

        transformer = model.training_parameters['transformer']

        for s in allsets:
            logging.info('Working on {}'.format(s))

            _, dset = get_dataset(s, transformer=transformer, splits=['test'])

            dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False)

            recorder = LossRecorder(args.batch_size)

            t0 = time.time()

            n = min(args.num_batch, len(dataloader))

            samples = {'x_': [], 'x': [], 'y': [], 'losses': []}
            
            for i, (x, y) in enumerate(dataloader):

                if i >= args.num_batch:
                    break
                if i:
                    ti = time.time()
                    t_per_i = (ti - t0) / i
                    eta = (n - i) * t_per_i

                else:
                    eta = 0

                eta_m = int(eta / 60)
                eta_s = eta - 60 * eta_m

                print('\r{:4}/{} -- eta: {:.0f}m{:.0f}s   '.format(i + 1, n, eta_m, eta_s), end='')

                x = x.to(device)
                y = y.to(device)

                with torch.no_grad():
                    x_, y_, losses, measures = model.evaluate(x)

                losses.update(y_true=y, logits=y_.permute(0, 2, 1))
                recorder.append_batch(**losses)

                n_samples = args.saved_samples_per_batch
                samples['x'].append(x[:n_samples])
                samples['x_'].append(x_[:, :2, :n_samples])
                samples['losses'].append({k: losses[k][..., :n_samples] for k in losses})
                samples['y'].append(y[:n_samples])
                for _ in ('x', 'x_', 'y'):
                    print('***', _, *samples[_][-1].shape)

                for _ in samples['losses'][-1]:
                    print('***', _, *samples['losses'][_][-1].shape)


            print()
            model.save()
            recorder.save(os.path.join(model.saved_dir, 'record-{}.pth'.format(s)))

        logging.info('Model saved in %s', model.saved_dir)

