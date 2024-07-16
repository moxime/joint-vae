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


def proj2d(sample_recorders, tset, include_alternate=False, N=60, N_=10, Model=TSNE):

    centroids = sample_recorders[tset]._aux['centroids'].numpy()
    alternate = sample_recorders[tset]._aux['alternate'].numpy()

    #    print('***', *centroids.shape, '***', *alternate.shape)

    if include_alternate:
        centroids = np.vstack([centroids, alternate])

    y = np.ndarray(0, dtype=int)

    mu = np.ndarray((0, centroids.shape[-1]))
    for _ in sample_recorders:
        mu = np.vstack([mu, *([centroids] * N_), sample_recorders[_]['mu'][:N]])
        # print(_, mu.shape)

    print('Mu of shape', *mu.shape)

    model = Model(2)
    mu = model.fit_transform(mu)

    mu_ = {}
    start = 0

    for _ in sample_recorders:
        mu_['centroids'] = mu[start:start + 10]
        if include_alternate:
            mu_['alternate'] = mu[start + 10:start + 11]
        start += N_ * len(centroids)
        # print(_, start, start + N)
        mu_[_] = mu[start:start + N]
        start += N

    return mu_


def plot2d(mu2d, dset, ax=None):

    marker = {_: ',' for _ in mu2d}
    marker['centroids'] = 'x'
    color = {}
    color['centroids'] = 'b'
    color[dset] = 'b'
    if 'alternate' in marker:
        marker['alternate'] = 'x'
        color['alternate'] = 'r'

    oodsets = {_ for _ in mu2d if _ not in color}
    color.update({_: __ for _, __ in zip(oodsets, 'rgcy')})

    ax = ax or plt.gca()

    for _ in mu2d:

        facecolor = color[_] if marker[_] == 'x' else 'none'
        ax.scatter(mu2d[_][:, 0], mu2d[_][:, 1], marker=marker[_],
                   edgecolors=color[_], facecolors=facecolor)

    return ax


if __name__ == '__main__':

    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument('--array-dir', default='wim-arrays-inspection')
    parser.add_argument('jobs', nargs='*', type=int)
    parser.add_argument('--mat', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--pca', action='store_const', dest='model', const=PCA, default=PCA)
    parser.add_argument('--tsne', action='store_const', dest='model', const=TSNE)
    parser.add_argument('-N', type=int, default=10)

    j = 660655
    j = 655755

    argv = '{}'.format(j).split()

    argv = None if sys.argv[0] else argv
    args = parser.parse_args(argv)

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

        if args.plot:
            mu2d_pre = proj2d(sample_recorders_pre, dset, N=args.N, include_alternate=False, Model=args.model)
            mu2d_ft = proj2d(sample_recorders_ft, dset, N=args.N, include_alternate=True, Model=args.model)

            for _ in mu2d_pre:
                print(_, *mu2d_pre[_].shape)

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
