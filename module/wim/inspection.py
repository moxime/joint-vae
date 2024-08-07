import sys
import os
from os import path
import logging

import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
from scipy.io import loadmat

from utils.print_log import EpochOutput, turnoff_debug

from utils.save_load import model_subdir, SampleRecorder, find_by_job_number

from utils.torch_load import get_dataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def estimate_y(mu, centroids):

    mu_ = mu.unsqueeze(1)

    centroids_ = centroids.expand(len(mu_), -1, -1)

    zdist = (mu_ - centroids_).norm(dim=-1)

    return zdist.argmin(axis=-1)


def dmu(mu, centroids, y=None):

    assert y is not None or centroids.ndim == 1

    if y is None:
        return mu - centroids.unsqueeze(0)

    m_ = centroids.index_select(0, y)

    return mu - m_


def to_mat(sample_recorders_pre, sample_recorders_ft, tset, matfile):

    samples_ft = sample_recorders_ft[dset].split('y', 'mu', keep=True)
    samples_ft._aux = sample_recorders_ft[dset]._aux

    samples_pre = sample_recorders_pre[dset].split('y', 'mu', keep=True)

    y_ft = samples_ft['y']
    y_pre = samples_pre['y']

    print('All y are equal: {}'.format(all(y_pre == y_ft)))

    oodsets = sorted(sample_recorders_pre)
    oodsets.remove(tset)

    for i_s, s in enumerate(oodsets):

        print('Collecting', s)
        y_s = -i_s - 1

        y_shape = sample_recorders_ft[s]._tensors['y'].shape
        sample_recorders_ft[s]._tensors['y'] = y_s * torch.ones(y_shape, dtype=int)

        samples_ft.merge(sample_recorders_ft[s])
        samples_pre.merge(sample_recorders_pre[s])

    mu_pre = samples_pre['mu']

    centroids = samples_ft._aux['centroids']

    print('Estimating y~')
    y_est = estimate_y(mu_pre, centroids)

    samples_ft._aux['alternate'] = samples_ft._aux['alternate'].unsqueeze(0)

    samples_ft._tensors['y_est'] = y_est
    y = samples_ft['y']

    samples_ft._tensors['mu_pre'] = mu_pre

    print('Acc: {:.1%}'.format((y[y >= 0] == y_est[y >= 0]).float().mean()))

    samples_ft.to_mat(matfile, oned_as='column')

    s_ = loadmat(matfile)
    print(matfile)
    for _ in s_:
        if not _.startswith('__'):
            print(_, *s_[_].shape)

    pass


def proj2d(sample_recorders_pre, sample_recorders_ft, tset,
           include_alternate=False, N=60, N_=10, Model=TSNE, csv_file=None):

    centroids = sample_recorders_pre[tset]._aux['centroids'].numpy()
    alternate = sample_recorders_pre[tset]._aux['alternate'].numpy()

    #    print('***', *centroids.shape, '***', *alternate.shape)

    mu = np.ndarray((0, centroids.shape[-1]))
    y = np.ndarray((0, 1), dtype=str)
    classes = get_dataset(tset)[1].classes
    y_centroids = np.expand_dims(np.array(classes), 1)
    if include_alternate:
        centroids = np.vstack([centroids, alternate])
        y_centroids = np.vstack([y_centroids, np.array([['ood']])])

    for sample_recorders in (sample_recorders_pre, sample_recorders_ft):
        for _ in sample_recorders:
            _N = N if _.startswith(tset) else N // 10
            mu = np.vstack([mu, *([centroids] * N_), sample_recorders[_]['mu'][:_N]])
            if _ == tset:
                y_batch = sample_recorders[_]['y'][:_N]
                c_batch = np.expand_dims(y_centroids.take(y_batch), 1)
            else:
                c_batch = np.zeros((_N, 1), object)
                c_batch[:] = _
            y = np.vstack([y, *([y_centroids] * N_), c_batch])
            # print(_, mu.shape)

    print('Mu of shape', *mu.shape)

    model = Model(2)
    mu = model.fit_transform(mu)

    mu_ = {}
    y_ = {}
    start = 0

    for sample_recorders, suffix in zip((sample_recorders_pre, sample_recorders_ft), ('pre', 'ft')):
        for _ in sample_recorders:
            _N = N if _.startswith(tset) else N // 10

            mu_['centroids'] = mu[start:start + 10]
            y_['centroids'] = classes
            if include_alternate:
                mu_['alternate'] = mu[start + 10:start + 11]
                y_['alternate'] = ['ood']
            start += N_ * len(centroids)
            # print(_, start, start + N)
            k = '{}-{}'.format(_, suffix)
            mu_[k] = mu[start:start + _N]
            y_[k] = y[start:start + _N].squeeze(1)
            # print('***', k, mu_[k].shape, y_[k].shape, y_[k])
            start += _N

    if csv_file:

        with open(csv_file, 'w') as f:

            print(','.join(['x1', 'x2', 'y', 'set', 'dist', 'ft']), file=f)
            for k in mu_:
                dset = k.split('-')[0]
                try:
                    ft = k.split('-')[1]
                    dist = 'ind' if dset == tset else 'ood'
                except IndexError:
                    if dset == 'centroids':
                        dist = 'ind'
                        ft = 'both'
                    else:
                        dset = 'alt'
                        dist = 'ood'
                        ft = 'both'
                for (x1, x2), y in zip(mu_[k], y_[k]):
                    print('{x1},{x2},{y},{s},{d},{ft}'.format(x1=x1, x2=x2, y=y, s=dset, d=dist, ft=ft), file=f)

    return mu_, y_


def plot2d(mu2d, dset, ax=None):

    marker = {_: ',' for _ in mu2d}
    marker['centroids'] = 'x'
    color = {}
    color['centroids'] = 'b'
    for _ in mu2d:
        if _.startswith(dset):
            color[_] = 'b'
    if 'alternate' in marker:
        marker['alternate'] = 'x'
        color['alternate'] = 'r'

    oodsets = {_ for _ in mu2d if _ not in color}
    color.update({_: __ for _, __ in zip(oodsets, 'rrrrr')})

    ax = ax or plt.gca()

    for _ in mu2d:

        facecolor = color[_] if marker[_] == 'x' else 'none'
        ax.scatter(mu2d[_][:, 0], mu2d[_][:, 1], marker=marker[_],
                   edgecolors=color[_], facecolors=facecolor)

    return ax


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--array-dir', default='wim-arrays-inspection')
    parser.add_argument('jobs', nargs='*', type=int)
    parser.add_argument('--mat', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--pca', action='store_const', dest='model', const=PCA, default=PCA)
    parser.add_argument('--tsne', action='store_const', dest='model', const=TSNE)
    parser.add_argument('-N', type=int, default=10)
    parser.add_argument('--csv')

    j = 655755
    j = 660655

    argv = '{} --plot --pca -N 50'.format(j).split()
    argv = '{} --plot --tsne -N 10 --csv /dev/stdout'.format(j).split()

    argv = None if sys.argv[0] else argv
    args = parser.parse_args(argv)

    if argv:
        print(args)

    models = find_by_job_number(*args.jobs, job_dir=args.array_dir, force_dict=True)

    for j in models:

        print('{:=^50}'.format(j))

        mdict = models[j]
        dset = mdict['set']

        rdir = model_subdir(mdict, 'samples', '{:04}'.format(mdict['done']))

        sample_recorders_ft = SampleRecorder.loadall(rdir)
        oodsets = sorted(sample_recorders_ft)
        oodsets.remove(dset)
        print('sets', dset, *oodsets)

        rdir = path.join(rdir, 'init')
        sample_recorders_pre = SampleRecorder.loadall(rdir)

        if args.mat:
            to_mat(sample_recorders_pre, sample_recorders_ft, dset, '/tmp/s-{}.mat'.format(j))

        dset, _ = get_dataset(dset)

        for s in oodsets:

            y = sample_recorders_pre[s]['y']
            y_est = estimate_y(sample_recorders_pre[s]['mu'],
                               sample_recorders_pre[s]._aux['centroids'])

            y_, c = np.unique(y_est, return_counts=True)

            print('{:=^20}'.format(s))
            i_ = np.argsort(c)[::-1]
            for i in i_:
                if c[i] > 5 * sum(c) / 100:
                    print('{:10}: {:4.0%}'.format(dset.classes[y_[i]], c[i] / sum(c)))

        dset = dset.name

        if args.plot or args.csv:
            mu2d, y2d = proj2d(sample_recorders_pre, sample_recorders_ft,
                               dset, N=args.N, N_=args.N // 10,
                               include_alternate=True, Model=args.model,
                               csv_file=args.csv)

            mu2d_pre = {_: mu2d[_] for _ in mu2d if not _.endswith('-ft')}
            mu2d_ft = {_: mu2d[_] for _ in mu2d if not _.endswith('-pre')}
            mu2d_pre.pop('alternate')

            print('=== PRE ===')
            for _ in mu2d_pre:
                print(_, *mu2d_pre[_].shape)

            print('=== FT ===')
            for _ in mu2d_ft:
                print(_, *mu2d_ft[_].shape)

            figure = plt.figure()
            plot2d(mu2d_pre, dset, ax=figure.gca())
            figure.show()

            figure = plt.figure()
            plot2d(mu2d_ft, dset, ax=figure.gca())
            figure.show()

            input()

        centroids = sample_recorders_pre[dset]._aux['centroids']
        centroids_ = sample_recorders_pre[dset]._aux['alternate']

        d_ = {'pre': {}, 'ft': {}}
        for _ in [dset, *oodsets]:
            mu_ = sample_recorders_ft[_]['mu']
            d_['ft'][_] = dmu(mu_, centroids_)

        mu = sample_recorders_pre[dset]['mu']
        y = sample_recorders_pre[dset]['y']
        d_['pre'][dset] = dmu(mu, centroids, y=y)
