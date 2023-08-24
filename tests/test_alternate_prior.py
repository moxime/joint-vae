import sys
import logging
import numpy as np
import torch
from module.wim import WIMVariationalNetwork as W
from utils.save_load import find_by_job_number
from utils.torch_load import get_batch, get_dataset, get_same_size_by_name
import argparse

job = 317036

batch = 0
batch = 5

prior = '--prior tilted'
prior = ''

argv = '--batch {} {} {}'.format(batch, prior, job).split()


class EndOfScript(Exception):
    pass


def quiet_kook(kind, message, traceback):
    if kind is EndOfScript:
        logging.warning('End of script {}'.format(message))
    else:
        sys.__excepthook__(kind, message, traceback)


sys.excepthook = quiet_kook


def end_of_script(b, msg=''):
    if b:
        raise EndOfScript(msg)


if __name__ == '__main__':

    logging.getLogger().setLevel(20)

    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=int)
    parser.add_argument('--prior')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--device')

    args = parser.parse_args(None if sys.argv[0] else argv)

    d = find_by_job_number(args.job, load_net=False)['dir']

    m = W.load(d, load_state=True, load_net=True)

    alternate_prior = m.encoder.prior.params

    if args.prior:
        alternate_prior['distribution'] = args.prior

    m.set_alternate_prior(alternate_prior)

    logging.info('From {} to {}'.format(m.original_prior(), m.alternate_prior()))
    logging.info('{:.4} -> {:.4}'.format(m.original_prior().mean.var(0).mean(),
                                         m.alternate_prior().mean.var(0).mean()))

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    testset = m.training_parameters['set']
    oodsets = get_same_size_by_name(testset)

    _, testset = get_dataset(testset)
    oodsets = [get_dataset(o, splits=['test'])[1] for o in oodsets]

    end_of_script(not args.batch)

    x = {}

    x[testset.name], y = get_batch(testset, batch_size=args.batch)
    for oodset in oodsets:
        x[oodset.name], y = get_batch(oodset, batch_size=args.batch)

    m.to(device)
    m.eval()

    m.original_prior()

    priors = ('original', 'alternate')
    losses_k = ('total', 'kl', 'zdist')
    losses = {_: {} for _ in priors}

    for p in priors:
        for s in x:
            logging.info('Computing {} with {} prior'.format(s, p))
            _, _, loss, _ = m.evaluate(x[s].to(device))
            losses[p][s] = {_: loss[_].min(0)[0].mean() for _ in losses_k}
        m.alternate_prior()

    for p in priors:
        print('===={}===='.format(p))
        for s in x:
            print('===', s)
            print(' | ' .join('{:6}: {:9.4e}'.format(k, v) for k, v in losses[p][s].items()))

    print('===Diff===')

    for k in losses_k:
        for s in x:
            print(k, s, '{:.4}'.format(losses['alternate'][s][k] - losses['original'][s][k]))
