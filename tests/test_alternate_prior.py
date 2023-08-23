import logging
import torch
from module.wim import WIMVariationalNetwork as W
from utils.save_load import find_by_job_number
from utils.torch_load import get_batch, get_dataset
import argparse

if __name__ == '__main__':

    logging.getLogger().setLevel(20)

    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=int)
    parser.add_argument('--batch', type=int, default=100)
    args = parser.parse_args()

    d = find_by_job_number(args.job, load_net=False)['dir']

    m = W.load(d, load_state=True, load_net=True)

    alternate_prior = m.encoder.prior.params

    m.set_alternate_prior(alternate_prior)

    logging.info('From {} to {}'.format(m.original_prior(), m.alternate_prior()))
    logging.info('{:.4} -> {:.4}'.format(m.original_prior().mean.var(0).mean(),
                                         m.alternate_prior().mean.var(0).mean()))

    device = 'cuda' if torch.cuda.is_available else 'cpu'

    _, dset = get_dataset(m.training_parameters['set'])

    x, y = get_batch(dset, batch_size=args.batch)

    m.to(device)
    m.eval()

    m.original_prior()
    _, _, losses, _ = m.evaluate(x)

    print(*losses['total'].shape)

    print('original')

    print('total: {:.4}, kl: {:.4}'.format(*(losses[_].min(0)[0].mean() for _ in ('total', 'kl'))))

    m.alternate_prior()
    _, _, losses, _ = m.evaluate(x.to(device))

    print('alternate')

    print('total: {:.4}, kl: {:.4}'.format(*(losses[_].min(0)[0].mean() for _ in ('total', 'kl'))))
