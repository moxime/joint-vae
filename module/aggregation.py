import torch
from torch.nn.functional import one_hot

TEMPS = [None, 1, 5]
NAN_TEMPS = [None, -1, 0]


def log_mean_exp(*tensors, normalize=False):

    t = torch.cat([_.unsqueeze(0) for _ in tensors])

    tref = t.max(axis=0)[0]
    dt = t - tref

    return (dt.exp().mean(axis=0).log() + tref).squeeze(0)


def posterior(logits, axis=0, temps=TEMPS):

    posterior = {}
    nan_temps = [_ for _ in temps if _ in NAN_TEMPS]

    for _ in nan_temps:
        posterior[_] = logits.clone()

    posterior.update({t: (logits / t).softmax(0) for t in temps if t not in nan_temps})

    return posterior


def joint_posterior(*zdist, axis=0, temps=TEMPS):

    z = torch.stack(zdist).sum(0)
    return posterior(-z / 2, axis=axis, temps=temps)


def mean_posterior(*p_x_y, axis=0, temps=TEMPS):

    mean_p_x_y = log_mean_exp(*p_x_y)
    return posterior(mean_p_x_y, axis=axis, temps=temps)


def voting_posterior(*y, temps=[None]):

    one_hot_ = [one_hot(_).T for _ in y]

    p_y_x = sum(one_hot_) / len(y)

    return {t: p_y_x for t in temps}


def latent_mutual_info(m1, m2, x, y):

    assert m1.is_cvae
    assert m2.is_cvae
    assert m1.input_shape == m2.input_shape

    models = {0: m1, 1: m2}

    z = {}
    muz = {}
    y_ = {}
    posterior = {}

    for _ in models:

        m = models[_]
        outs = m.forward(x)
        z[_] = outs[-1]
        muz[_] = outs[-2]

        my = m.encoder.prior.mean.rename('y', 'K')


if __name__ == '__main__':

    import sys
    import time
    import logging
    import warnings
    import argparse
    from utils.torch_load import get_dataset
    from utils.save_load import find_by_job_number, needed_remote_files
    from cvae import ClassificationVariationalNetwork as M

    parser = argparse.ArgumentParser()
    parser.add_argument('jobs', nargs=2, type=int)
    parser.add_argument('-v', action='count', default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', '-M', default=256, type=int)
    parser.add_argument('-N', default=2000, type=int)

    args_from_file = '-vv 226397 226180 -N 2000 -M 64'.split()

    args = parser.parse_args(None if len(sys.argv) > 1 else args_from_file)

    logging.getLogger().setLevel(40 - 10 * args.v)
    warnings.filterwarnings("ignore", category=UserWarning)

    jobs = args.jobs
    device = args.device

    models = find_by_job_number(*jobs, load_state=False, load_net=True)

    assert len(models) == 2

    params = {}

    for k in ('set', 'transformer'):
        params[k] = models[jobs[0]][k]
        logging.info('{:12}: {}'.format(k, params[k]))
        assert params[k] == models[jobs[1]][k]

    mdirs = [models[_]['dir'] for _ in models]

    removed = False
    with open('/tmp/files', 'w') as f:
        for mdir, sdir in needed_remote_files(*mdirs, which_rec='none', state=True, missing_file_stream=f):
            logging.debug('{} for {}'.format(sdir[-30:], 'last'))
            if mdir in mdirs:
                removed = True
                logging.info('{} is removed (files not found)'.format(mdir.split('/')[-1]))

    if removed:
        logging.error('Exiting, load files')
        logging.error('E.g: %s', '$ rsync -avP --files-from=/tmp/files remote:dir/joint-vae .')
        logging.error(' Or: %s', '$ . /tmp/rsync-files remote:dir/joint-vae')
        with open('/tmp/rsync-files', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('rsync -avP --files-from=/tmp/files $1 .\n')
        sys.exit(1)

    m_ = [M.load(mdir, load_state=True) for mdir in mdirs]
    for _ in m_:
        _.to(args.device)

    _, dataset = get_dataset(params['set'], transformer=params['transformer'], splits=['test'])  #

    loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                         batch_size=args.batch_size)

    n = 0
    t0 = time.time()
    for x, y in loader:

        n += len(x)

        if n >= args.N:
            break
        out = latent_mutual_info(*m_, x.to(device), y.to(device))

        t1 = time.time()
        t_per_i = (t1 - t0) / n
        eta = (args.N - n) * t_per_i
        print('{:4}/{:4} -- {:.3f} ms/i -- eta {:.0f}s   '.format(n, args.N, 1000 * t_per_i, eta), end='\r')

    else:
        logging.warning('Only {} images processed'.format(n))

    print('Processed {} images in {:.0f}s ({:.0f}ms/i)'.format(n, t1 - t0, 1000 * (t1 - t0) / n))
