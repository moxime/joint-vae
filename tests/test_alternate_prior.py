import sys
import logging
import numpy as np
import torch
from module.wim import WIMVariationalNetwork as W
from utils.save_load import find_by_job_number
from utils.torch_load import get_batch, get_dataset, get_same_size_by_name
import argparse

job = 339798
job = 339814

batch = 0
batch = 7

prior = '--prior tilted'
prior = ''

show = ''
show = '--show'

argv = '--batch {} {} {} {}'.format(batch, prior, job, show).split()


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

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    logging.getLogger().setLevel(20)

    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=int)
    parser.add_argument('--job-dir', '-J', default='./jobs')
    parser.add_argument('--prior')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--device')
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args(None if sys.argv[0] else argv)

    d = find_by_job_number(args.job, job_dir=args.job_dir, load_net=False)['dir']

    m = W.load(d, load_state=True, load_net=True)

    alternate_prior = m.encoder.prior.params

    if args.prior:
        alternate_prior['distribution'] = args.prior

    try:
        m.alternate_prior = True
        _s = 'Had already an alternate prior finetune with {}, ignoring --prior param'
        logging.info(_s.format(m.wim_params.get('sets', 'unknown set')))
    except AttributeError:
        m.set_alternate_prior(alternate_prior)

    with m.original_prior as p1:
        with m.alternate_prior as p2:
            _s = 'From {} ({:.4}) to {} ({:.4})'
            logging.info(_s.format(p1, p1.mean.var(0).mean(), p2, p2.mean.var(0).mean()))

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    testset = m.training_parameters['set']
    oodsets = get_same_size_by_name(testset)

    if device == 'cpu' or True:
        oodsets = m.wim_params.get('sets', oodsets[:1])

    _, testset = get_dataset(testset)
    oodsets = [get_dataset(o, splits=['test'])[1] for o in oodsets]

    end_of_script(not args.batch)

    x = {}

    x[testset.name], y = get_batch(testset, batch_size=args.batch)
    for oodset in oodsets:
        x[oodset.name], y = get_batch(oodset, batch_size=args.batch)

    m.to(device)
    m.eval()

    priors = ('original', 'alternate')
    losses_k = ('total', 'kl', 'zdist')
    losses = {_: {} for _ in priors}
    mus = {_: {} for _ in priors}

    m.original_prior = True
    for p in priors:
        for s in x:
            logging.info('Computing {} with {} prior'.format(s, p))
            with torch.no_grad():
                _, _, loss, _, mu, _, _ = m.evaluate(x[s].to(device), z_output=True)
            mus[p][s] = mu.detach().cpu().numpy()
            print('*** Y in losses;', m.losses_might_be_computed_for_each_class)
            for _ in losses_k:
                print('***', _, *loss[_].shape)
            if m.losses_might_be_computed_for_each_class:
                losses[p][s] = {_: loss[_].min(0)[0].mean() for _ in losses_k}
            else:
                losses[p][s] = {_: loss[_].mean() for _ in losses_k}
        m.alternate_prior = True

    for p in priors:
        print('===={}===='.format(p))
        for s in x:
            print('===', s)
            print(' | ' .join('{:6}: {:9.4e}'.format(k, v) for k, v in losses[p][s].items()))

    print('===Diff===')

    for k in losses_k:
        for s in x:
            print(k, s, '{:.4}'.format(losses['alternate'][s][k] - losses['original'][s][k]))

    if args.show:

        pca = PCA(n_components=2)

        sets = list(x)[1:]
        sets = list(x)[:1] + m.wim_params.get('sets', sets)

        mu_ = np.vstack(list(mus['original'][s] for s in sets))

        pca.fit(mu_)

        for s in sets:
            z = pca.fit_transform(mus['original'][s])
            plt.scatter(z[:, 0], z[:, 1])

        plt.legend(sets)
        plt.show()
