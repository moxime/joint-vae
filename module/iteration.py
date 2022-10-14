import os
import sys
import torch
from cvae import ClassificationVariationalNetwork as M
from utils import save_load
import logging
import argparse
from utils.save_load import load_json, needed_remote_files, develop_starred_methods, LossRecorder
from utils.torch_load import get_dataset, get_same_size_by_name
import numpy as np
import logging
from utils.parameters import gethostname
import time
from torchvision.utils import save_image


class IteratedModels(M):

    def __new__(cls, *models):

        m = models[-1].copy()
        m.__class__ = cls
        
        return m
    
    def __init__(self, *models):

        assert len(models) > 1 or True # DEBUG
        self._modules = {str(_): m for _, m in enumerate(models)}
        self._models = models
        self.predict_methods = ['iter']
        self.ood_results = {}
        self.testing = {}
        # self.training_parameters = self._models.training_parameters
        
    def __len__(self):
        return len(self._models)

    def to(self, device):
        for m in self._models:
            m.to(device)
    
    def save(self, dir_name=None):
        if dir_name is None:
            trainset = self.training_parameters['set']
            dir_name = os.path.join('iterated-jobs', trainset, '-'.join(str(_.job_number) for _ in self._models))
        architecture = {_: m.saved_dir for _, m in enumerate(self._models)}
            
        save_load.save_json(architecture, dir_name, 'params.json')
        save_load.save_json(self.testing, dir_name, 'test.json')
        save_load.save_json(self.ood_results, dir_name, 'ood.json')
        self.saved_dir = dir_name
        
    @classmethod
    def load(cls, dir_name, *a, **kw):

        architecture = save_load.load_json(dir_name, 'params.json')
        models = [architecture[str(_)] for _ in range(len(architecture))]

        m = cls(*[M.load(_, *a, **kw) for _ in models])

        try:
            m.testing = save_load.load_json(dir_name, 'test.json', presumed_type=int)
        except(FileNotFoundError):
            pass

        try:
            m.ood_results = save_load.load_json(dir_name, 'ood.json', presumed_type=int)
        except(FileNotFoundError):
            pass

        m.saved_dir = dir_name
        
        return m

    def evaluate(self, x,
                 y=None,
                 z_output=False,
                 **kw):

        input = {'x': x, 'y': y, 'z_output': z_output}

        x_ = []
        y_ = []
        losses_ = []
        measures_ = []

        mse_ = []

        for m in self._models:

            out = m.evaluate(**input, **kw)
            input['x'] = out[0][1]
            input['y'] = out[1].argmax(-1) if y else None

            """
            for k in 'xy':
                print('***', k, ':', *input[k].shape if input[k] != None else 'None')

            for k in out[2]:
                print('*** losses', k, ':', *out[2][k].shape)

            for k in out[3]:
                print('*** meas', k, ':', type(out[3][k]))
            """

            x_.append(out[0])
            y_.append(out[1])
            losses_.append(out[2])
            measures_.append(out[3])

        x_ = torch.stack(x_)
        y_ = torch.stack(y_)

        input_dims = tuple([0] + [_ - self.input_dim for _ in range(self.input_dim)])

        for i in range(len(x_) + 1):
            for j in range(i):

                x_i = x_[i - 1][1:]
                x_j = x.unsqueeze(0) if not j else x_[j - 1][1:]

                mse_.append((x_i - x_j).pow(2).mean(input_dims))

        output_losses = {}
        output_measures = {}

        for k in losses_[0]:
            output_losses[k] = torch.stack([_[k] for _ in losses_])

        for k in measures_[0]:
            output_measures[k] = torch.tensor([_[k] for _ in measures_])

        output_losses['mse'] = torch.stack(mse_)
            
        return x_, y_, output_losses, output_measures

    def predict_after_evaluate(self, logits, losses, method='iter'):

        if method == 'iter':
            return logits[-1].max(axis=0)
        
        return self._models[-1].predict_after_evaluate(logits[-1], losses[-1], **kw)

    def batch_dist_measures(self):
        pass


def iterate_with_prior(logp_x_y):

    """Args:

    -- p_x_y is a tensor of dim MxCxN with M the number of models, C
    the number of labels and N the number of sample

    """

    M, C, N = logp_x_y.shape
    prior = torch.ones(C, N, device=logp_x_y.device) / C
    posterior = torch.zeros_like(logp_x_y)

    for i in range(M):

        joint = logp_x_y[i] * prior
        p_x = joint.sum(0, keepdim=True)
        # print('***', prior.shape, joint.shape, p_x.shape)
        posterior[i] = joint / p_x
        prior = posterior[i]

    return posterior


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--jobs', '-j', nargs='+', type=int, default=[])
    parser.add_argument('-v', action='count', default=0)
    parser.add_argument('--when', default='last')

    parser.add_argument('--plot', nargs='?', const='p')
    parser.add_argument('--tex', nargs='?', default=None, const='/tmp/r.tex')
    parser.add_argument('--job-dir', default='./jobs')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-batch', type=int, default=int(1e6))
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--saved-samples-per-batch', type=int, default=2)

    args_from_file = ('-vvvv '
                      #                      '--jobs 193080 193082'
                      '--jobs 169381'
                      ).split()

    args, ra = parser.parse_known_args(None if len(sys.argv) > 1 else args_from_file)
    rmodels = load_json(args.job_dir, 'models-{}.json'.format(gethostname()))
    wanted = args.when
    
    logging.getLogger().setLevel(40 - 10 * args.v)

    if len(args.jobs) < 2 and False: # DEBUG
        logging.error('At least two jobs (%d provided)', len(args.jobs))
        sys.exit(1)
    
    mdirs_ = {rmodels[_]['job']: _  for _ in rmodels if rmodels[_]['job'] in args.jobs}
    
    if len(mdirs_) < len(set(args.jobs)):
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
    model = IteratedModels(*models)

    device = args.device

    model.to(device)

    logging.debug('Model sent to {} (device wanted: {})'.format(next(iter(model.parameters())), device))
    
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

            # print('***', 'x:', *x.shape, 'x_:', *x_.shape)
            # for k in losses:
            #     print('***', k, *losses[k].shape)
            n_samples = args.saved_samples_per_batch
            concat_dim = {'x': 0, 'x_': 1, 'y': 0}
            samples['x'].append(x[:n_samples].to('cpu'))
            samples['x_'].append(x_[:, 0, :n_samples].to('cpu'))
            samples['y'].append(y[:n_samples].to('cpu'))

            save_image(x_[0, 0, 0].to('cpu'), f'/tmp/iter/out_{i}.png')
            
        for k in ('x', 'x_', 'y'):
            samples[k] = torch.cat(samples[k], dim=concat_dim[k])

        print()
        model.save()
        recorder.save(os.path.join(model.saved_dir, 'record-{}.pth'.format(s)))
        f = os.path.join(model.saved_dir, 'sample-{}.pth'.format(s))
        torch.save(samples, f)

    logging.info('Model saved in %s', model.saved_dir)


