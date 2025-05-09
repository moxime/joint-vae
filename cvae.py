import sys
import logging
import errno
import copy
import torch
import torch.utils.data
from torch import nn
from module.optimizers import Optimizer
from torch.nn import functional as F
from module.losses import x_loss, mse_loss, categorical_loss
from utils.save_load import LossRecorder, available_results, develop_starred_methods, MissingKeys
from utils.save_load import DeletedModelError, NoModelError, StateFileNotFoundError
from utils.misc import make_list
from module.vae_layers import Encoder, Classifier, Sigma, build_de_conv_layers, find_input_shape
from module.vae_layers import onehot_encoding
import tempfile
import shutil
import random

import utils.torch_load as torchdl
from utils.torch_load import choose_device, collate
from utils import save_load
import numpy as np

from utils.roc_curves import roc_curve, fpr_at_tpr
# from sklearn.metrics import auc, roc_curve

from utils.print_log import EpochOutput

from utils.parameters import get_args

from utils.signaling import SIGHandler

import os.path
import time

import re

DEFAULT_ACTIVATION = 'relu'
# DEFAULT_OUTPUT_ACTIVATION = 'sigmoid'
DEFAULT_OUTPUT_ACTIVATION = 'linear'
DEFAULT_LATENT_SAMPLING = 100

VERSION = 2.

activation_layers = {'linear': nn.Identity,
                     'sigmoid': nn.Sigmoid,
                     'relu': nn.ReLU,
                     'leaky': nn.LeakyReLU,
                     }


class EmptyShell(object):

    def __init__(self):

        super().__init__()


class ClassificationVariationalNetwork(nn.Module):
    r"""Creates a network

    X -- Features ---- Encoder -- Latent var Z -- Decoder -- Upsampler -- Ẍ
                    /                           \
                 Y_/                             \-- Classfier-- Ŷ

    Args:

    input_shape: the shape of the image (3, 32, 32), (1, 28, 28)
    ...

    num_labels: the number of labels.

    features (default None): type of features extractors from vgg
    convolutional encoder (possibilities are 'vgg11', 'vgg16'...

    pretrained_features (default None) .pth file in which the
    weights of the pretained features extractor are stored.

    """

    loss_components_per_type = {'jvae': ('cross_x', 'kl', 'cross_y', 'total'),
                                'cvae': ('cross_x', 'kl', 'total', 'zdist', 'var_kl', 'dzdist', 'iws',
                                         'sigma', 'wmse',
                                         'z_logdet', 'z_tr_inv_cov'),
                                'xvae': ('cross_x', 'kl', 'total', 'zdist', 'iws'),
                                'vae': ('cross_x', 'kl', 'zdist', 'var_kl', 'total', 'iws'),
                                'vib': ('cross_y', 'kl', 'total')}

    predict_methods_per_type = {'jvae': ['loss', 'esty'],
                                # 'cvae': ('closest', 'iws'),
                                'cvae': ['iws', 'closest'],
                                'xvae': ['loss', 'closest'],
                                'vae': [],
                                'vib': ['esty']}

    metrics_per_type = {'jvae': ['rmse', 'dB', 'sigma'],
                        'cvae': ['rmse', 'dB', 'd-mind', 'ld-norm', 'sigma'],
                        'xvae': ['rmse', 'dB', 'zdist', 'd-mind', 'ld-norm', 'sigma'],
                        'vae': ['rmse', 'dB', 'sigma'],
                        'vib': ['sigma', ]}

    ood_methods_per_type = {'cvae': ['iws-2s', 'iws-a-1-1', 'iws-a-4-1',
                                     'iws', 'mse', 'elbo', 'soft', 'elbo-2s', 'elbo-a-1-1', 'elbo-a-4-1', 'zdist'],
                            'xvae': ['max', 'mean', 'std'],  # , 'mag', 'IYx'],
                            'jvae': ['max', 'sum', 'std'],  # 'mag'],
                            'vae': ['iws', 'iws-2s', 'iws-a-1-1', 'iws-a-4-1',
                                    'elbo', 'elbo-2s', 'elbo-a-1-1', 'elbo-a-4-1',
                                    'zdist'],
                            'vib': ['odin*', 'baseline', 'logits']}

    misclass_methods_per_type = {'cvae': ['softkl*', 'iws', 'softiws*', 'kl', 'max',
                                          'zdist',
                                          'softzdist*', 'baseline*', 'hyz'],
                                 'xvae': [],
                                 'jvae': [],
                                 'vae': [],
                                 'vib': ['odin*', 'baseline', 'logits', 'hyz']}

    ODIN_TEMPS = [_ * 10 ** i for i in (0, 1, 2) for _ in (1, 2, 5)] + [1000]
    ODIN_EPS = [_ / 20 * 0.004 for _ in range(21)]
    # ODIN_EPS = [0]

    odin_params = []
    for T in ODIN_TEMPS:
        for eps in ODIN_EPS:
            odin_params.append('odin-{:.0f}-{:.4f}'.format(T, eps))
    methods_params = {}
    for k in ['soft' + _ for _ in ['kl', 'zdist']] + ['baseline']:
        methods_params[k] = []
        for _ in ODIN_TEMPS:
            methods_params[k].append(f'{k}-{_:.0f}')
    methods_params.update({'odin': odin_params})

    def __init__(self,
                 input_shape,
                 num_labels,
                 type='cvae',  # or 'vib' or cvae or vae
                 y_is_coded=False,
                 output_distribution='gaussian',
                 job_number=0,
                 features=None,
                 pretrained_features=None,
                 batch_norm=False,
                 dropout=False,
                 encoder=[36],
                 latent_dim=32,
                 prior={},  # default scalar gaussian
                 beta=1.,
                 gamma=0.,
                 decoder=[36],
                 upsampler=None,
                 pretrained_upsampler=None,
                 classifier=[36],
                 name='joint-vae',
                 activation=DEFAULT_ACTIVATION,
                 latent_sampling=DEFAULT_LATENT_SAMPLING,
                 # if none will be the same as (train) latent_sampling
                 test_latent_sampling=None,
                 encoder_forced_variance=False,
                 output_activation=DEFAULT_OUTPUT_ACTIVATION,
                 sigma={'value': 1},
                 optimizer={},
                 shadow=False,
                 representation='rgb',
                 version=VERSION,
                 * args, **kw):

        super().__init__(*args, **kw)
        self.name = name

        self.job_number = job_number

        assert type in ('jvae', 'cvae', 'xvae', 'vib', 'vae')
        self.type = type

        self.loss_components = self.loss_components_per_type[self.type]

        self.metrics = self.metrics_per_type[self.type]

        self._test_losses = {}
        self._test_measures = {}

        self.predict_methods = self.predict_methods_per_type[self.type].copy()
        self.ood_methods = self.ood_methods_per_type[self.type].copy()
        self.misclass_methods = self.misclass_methods_per_type[self.type].copy()

        self.is_jvae = type == 'jvae'
        self.is_vib = type == 'vib'
        self.is_vae = type == 'vae'
        self.is_cvae = type == 'cvae'
        self.is_xvae = type == 'xvae'

        assert not (y_is_coded and (self.is_vib or self.is_vae))
        self.y_is_coded = y_is_coded
        # self.y_is_decoded = self.is_vib or self.is_jvae
        self.y_is_decoded = True
        if self.is_cvae or self.is_vae:
            self.y_is_decoded = gamma

        self.x_is_generated = not self.is_vib

        self.output_distribution = output_distribution if self.x_is_generated else None

        self.losses_might_be_computed_for_each_class = not self.is_vae and not self.is_vib

        logging.debug('y is%s coded', '' if self.y_is_coded else ' not')

        self._measures = {}

        if self.y_is_decoded:
            self.classifier_type = 'linear'
            if self.is_cvae and classifier and isinstance(classifier[0], str):
                assert classifier[0] in ('softmax',)
                self.classifier_type = classifier[0]
        else:
            self.classifier_type = None
            classifier = []

        logging.debug('Claasifier type: {}'.format(self.classifier_type))

        if not self.x_is_generated:
            decoder = []
            upsampler = None

        if self.y_is_decoded and 'esty' not in self.predict_methods:
            self.predict_methods = self.predict_methods + ['esty']

        if self.y_is_decoded and 'cross_y' not in self.loss_components:
            self.loss_components += ('cross_y',)

        # no upsampler if no features
        assert (not upsampler or features)

        if not features:
            batch_norm = False
        else:
            batch_norm_encoder = (batch_norm == 'encoder' or
                                  batch_norm == 'both')
            batch_norm_decoder = batch_norm == 'both'
        if features:
            logging.debug('Building features')

            self.features = build_de_conv_layers(input_shape, features,
                                                 activation=activation,
                                                 batch_norm=batch_norm_encoder,
                                                 pretrained_dict=pretrained_features)

            encoder_input_shape = self.features.output_shape
            logging.debug('Features built')

        else:
            encoder_input_shape = input_shape
            self.features = None

        self.trained = 0
        # print('*** sigma of type', type(sigma))
        if isinstance(sigma, Sigma):
            self.sigma = sigma
        elif isinstance(sigma, dict):
            self.sigma = Sigma(**sigma)
        else:
            self.sigma = Sigma(value=sigma)

        if not test_latent_sampling:
            test_latent_sampling = latent_sampling

        self.beta = beta

        self.gamma = gamma if self.y_is_decoded else None
        logging.debug(f'Gamma: {self.gamma}')

        if self.type in ('cvae', 'xvae'):
            prior['num_priors'] = num_labels

        sampling = latent_sampling > 1 or beta > 0

        self.encoder = Encoder(encoder_input_shape, num_labels,
                               intermediate_dims=encoder,
                               latent_dim=latent_dim,
                               y_is_coded=self.y_is_coded,
                               dropout=dropout,
                               sigma_output_dim=self.sigma.output_dim if self.sigma.coded else 0,
                               forced_variance=encoder_forced_variance,
                               sampling_size=latent_sampling,
                               prior=prior,
                               activation=activation, sampling=sampling)

        activation_layer = activation_layers[activation]()

        if self.x_is_generated:
            decoder_layers = []
            input_dim = latent_dim
            for output_dim in decoder:
                decoder_layers += [nn.Linear(input_dim, output_dim),
                                   activation_layer]
                if dropout:
                    decoder_layers.append(nn.Dropout(p=dropout))
                input_dim = output_dim

            self.decoder = nn.Sequential(*decoder_layers)

            imager_input_dim = input_dim
            if upsampler:
                imager_input_shape = find_input_shape(upsampler, input_shape[1:])
                f = imager_input_shape[0] * imager_input_shape[1]
                e = 'Could not go from {} to *, {} {}'.format(imager_input_dim, *imager_input_shape)
                assert not imager_input_dim % f, e
                imager_input_dim = (imager_input_dim // f, *imager_input_shape)
                logging.debug('New imager input shape: {}'.format(imager_input_dim))
                self.imager = build_de_conv_layers(imager_input_dim, upsampler,
                                                   batch_norm=batch_norm_decoder,
                                                   activation=activation,
                                                   output_activation=output_activation,
                                                   output_distribution=self.output_distribution,
                                                   pretrained_dict=pretrained_upsampler,
                                                   where='output')

            else:
                f = 1 if self.output_distribution == 'gaussian' else 256
                upsampler = None
                activation_layer = activation_layers[output_activation]()
                self.imager = nn.Sequential(nn.Linear(imager_input_dim,
                                                      f * np.prod(input_shape)),
                                            activation_layer)
                self.imager.input_shape = (imager_input_dim,)

        if self.classifier_type in ('linear', None):
            self.classifier = Classifier(latent_dim, num_labels,
                                         classifier,
                                         activation=activation)

        self.input_shape = tuple(input_shape)
        self.num_labels = num_labels
        self.input_dim = len(input_shape)

        self.training_parameters = {}  #

        self.batch_norm = batch_norm
        self.dropout = dropout

        self._sizes_of_layers = [input_shape, num_labels,
                                 encoder, latent_dim,
                                 decoder,
                                 upsampler,
                                 classifier]

        self.architecture = {'input_shape': input_shape,
                             'num_labels': num_labels,
                             'output_distribution': self.output_distribution,
                             'type': type,
                             'representation': representation,
                             # 'features': features_arch,
                             'encoder': encoder,
                             'batch_norm': batch_norm,
                             'dropout': dropout,
                             'activation': activation,
                             'encoder_forced_variance': self.encoder.forced_variance,
                             'latent_dim': latent_dim,
                             'test_latent_sampling': test_latent_sampling,
                             'prior': self.encoder.prior.params,
                             'decoder': decoder,
                             'upsampler': upsampler,
                             'classifier': classifier,
                             'output_activation': output_activation,
                             'version': VERSION,
                             }

        self.depth = (len(encoder)
                      + len(decoder)
                      + len(classifier) if self.classifier_type == 'linear' else 0)

        self.width = (sum(encoder)
                      + sum(decoder)
                      + sum(classifier) if self.classifier_type == 'linear' else 0)

        if features:
            self.architecture['features'] = self.features.name

        self.training_parameters = {
            'sigma': self.sigma.params,
            'beta': self.beta,
            'gamma': self.gamma,
            'latent_sampling': latent_sampling,
            'set': None,
            'data_augmentation': [],
            'pretrained_features': getattr(pretrained_features, 'name', None),
            'pretrained_upsampler': getattr(pretrained_upsampler, 'name', None),
            'epochs': 0,
            'batch_size': None,
            'fine_tuning': []}

        self.testing = {0: {m: {'n': 0, 'epochs': 0, 'accuracy': 0}
                            for m in self.predict_methods}}
        # self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

        self.ood_results = {}

        self.optimizer = Optimizer(self.parameters(), **optimizer)
        self.training_parameters['optimizer'] = self.optimizer.params

        self.train_history = {'epochs': 0}

        self.latent_dim = latent_dim
        self.latent_sampling = latent_sampling
        self._latent_samplings = {
            'train': latent_sampling, 'eval': test_latent_sampling}
        self.encoder_layer_sizes = encoder
        self.decoder_layer_sizes = decoder
        self.classifier_layer_sizes = classifier
        self.upsampler = upsampler
        self.activation = activation
        self.output_activation = output_activation

        self.z_output = False

        self.eval()

    def train(self, *a, **k):
        state = 'train' if self.training else 'eval'
        super().train(*a, **k)
        new_state = 'train' if self.training else 'eval'
        logging.debug(f'Going from {state} to {new_state}')
        self.latent_sampling = self._latent_samplings[new_state]

    def forward(self, x, y=None, x_features=None, **kw):
        """inputs: x, y where x, and y are tensors sharing first dims.

        - x is of size N1x...xNgxD1x..xDt
        - y is of size N1x....xNg(x1)

        """

        if y is None and self.y_is_coded:
            raise ValueError('y is supposed to be an input of the net')

        if x.dim() == self.input_dim:
            batch_shape = (1,)
        else:
            batch_shape = x.shape[:-self.input_dim]

        f_shape = self.encoder.input_shape

        if not self.features:
            x_features = x

        if x_features is None:
            x_features = self.features(x.view(-1, *self.input_shape)).view(*batch_shape, *f_shape)

        return self.forward_from_features(x_features,
                                          None if y is None else y.view(
                                              *batch_shape),
                                          x, **kw)

    def forward_from_features(self, x_features, y, x,
                              z_output=True, sampling_epsilon_norm_out=False, sigma_out=False):

        batch_shape = x_features.shape
        batch_size = batch_shape[:-len(self.encoder.input_shape)]  # N1 x...xNg
        if self.output_distribution == 'gaussian':
            reco_batch_shape = tuple((*batch_size, *self.input_shape))
        else:
            reco_batch_shape = tuple((*batch_size, 256, *self.input_shape))

        x_ = x_features.view(*batch_size, -1)  # x_ of size N1x...xNgxD

        if y is None:
            y_onehot = None
        else:
            y_onehot = onehot_encoding(y, self.num_labels).float()

        """"print('**** cvae l. 370',
              'y:', *y.shape if y is not None else ('*',),
              'y_01:', *y_onehot.shape if y is not None else ('*',))
        """
        try:
            z_mean, z_log_var, z, sample_eps, sigma = self.encoder(x_, y_onehot)
            # z of size LxN1x...xNgxK
        except ValueError as e:
            dir_ = f'log/dump-{self.job_number}'
            self.save(dir_)
            x_p = os.path.join(dir_, 'x.pt')
            y_p = os.path.join(dir_, 'y.pt')
            torch.save(x, x_p)
            torch.save(y, y_p)
            logging.error('Error %s, net dumped in %s',
                          str(e), dir_)
            raise e

        if not self.is_vib:
            u = self.decoder(z)
            # x_output of size LxN1x...xKgxD
            x_ = self.imager(u.view(-1, *self.imager.input_shape))

        if self.classifier_type in ('linear', None):
            # y_output = self.classifier(z_mean.unsqueeze(0))  # for classification on the means
            y_output = self.classifier(z)  # for classification on z
        elif self.classifier_type == 'softmax':
            y_output = F.linear(z, self.encoder.prior.mean, self.encoder.prior.mean.pow(2).sum(-1) / 2)

        # y_output of size LxN1x...xKgxC
        # print('**** y_out', y_output.shape)

        if self.is_vib:
            out = (x,)
        else:
            # print('***', *x_.shape)
            out = (x_.view(self.latent_sampling + 1, *reco_batch_shape),)

        out += (y_output,)

        if z_output:
            out += (z_mean, z_log_var, z)

        if sampling_epsilon_norm_out:
            out += ((sample_eps ** 2).sum(-1),)

        if sigma_out:
            out += (sigma,)

        return out

    def evaluate(self, x,
                 y=None,
                 batch=0,
                 current_measures=None,
                 with_beta=False,  #
                 kl_var_weighting=1.,
                 gamma_weighting=1,
                 z_output=False,
                 **kw):
        """x input of size (N1, .. ,Ng, D1, D2,..., Dt)

        creates a x of size C * N1, ..., D1, ...., Dt)
        and a y of size C * N1 * ... * Ng

        ----- Returns

        x_ (C,N1,..., D1...) tensor,

        logits (C,N1,...,C) tensor,

        batch_losses dict of tensors

        mesures: dict of  tensor

        """
        y_in_input = y is not None
        x_repeated_along_classes = self.y_is_coded and not y_in_input
        losses_computed_for_each_class = (self.losses_might_be_computed_for_each_class
                                          and not y_in_input)

        y_is_built = losses_computed_for_each_class

        compute_iws = not self.training

        cross_y_weight = False
        if self.y_is_decoded:
            if self.is_cvae or self.is_vae:
                cross_y_weight = gamma_weighting * self.gamma if self.training else False
            else:
                cross_y_weight = gamma_weighting * self.gamma

        if not batch:
            # print('*** training:', self.training)
            mode = 'training' if self.training else 'eval'
            logging.debug(f'Evaluating model in {mode} mode with batch size {x.shape[0]} '
                          'y {}in input'.format('' if y_in_input else '*not* '))
            pass

        C = self.num_labels

        if self.features:
            f_shape = self.encoder.input_shape

            if x.dim() == self.input_dim:
                batch_shape = (1,)
            else:
                batch_shape = x.shape[:-self.input_dim]
            t = self.features(x.view(-1, *self.input_shape)).view(*batch_shape, *f_shape)

        else:
            t = x

        t_shape = t.shape

        y_shape = x.shape[:-len(self.input_shape)]

        if x_repeated_along_classes:
            # build a C* N1* N2* Ng *D1 * Dt tensor of input x_features
            t = t.expand(C, *t_shape)

        if y_is_built:
            # create a C * N1 * ... * Ng y tensor y[c,:,:,:...] = c
            y_shape_per_class = (1,) + y_shape
            y = torch.cat([c * torch.ones(y_shape_per_class,
                                          dtype=int,
                                          device=x.device)
                           for c in range(C)], dim=0)
            y_shape = y.shape

        y_in = y.view(y_shape) if self.y_is_coded else None

        o = self.forward(x, y=y_in, x_features=t,
                         sampling_epsilon_norm_out=True,
                         sigma_out=True,
                         **kw)

        x_reco, y_est, mu, log_var, z, eps_norm, sigma_coded = o
        # print('*** eps norm:', *eps_norm.shape)

        # print('*** cvae:472 logits:', 't:', *t.shape, 'x_:', *x_reco.shape)

        batch_quants = {}
        batch_losses = {}
        total_measures = {}

        if not current_measures:
            current_measures = {k: 0. for k in ('xpow', 'mse', 'dB',
                                                'imut-zy', 'd-mind',
                                                'ld-norm', 'var_kl',
                                                'zdist')}

        total_measures['sigma'] = self.sigma.value

        if self.x_is_generated:
            D = np.prod(self.input_shape)
            sigma_dims = D if self.sigma.per_dim else 1
            # print('****', sigma_dims)

            if self.sigma.coded:
                s_ = sigma_coded.view(-1, *self.sigma.output_dim)
                # print('****', *s_.shape, *sigma_coded.shape)
                self.sigma.update(v=s_)

            else:
                s_ = self.sigma

            if self.sigma.is_rmse or self.output_distribution == 'catgorical':
                sigma_ = 1.
                sigma2_ = 1.
                log_sigma = 0.
            else:
                sigma_ = s_.exp() if self.sigma.is_log else s_
                sigma2_ = sigma_ ** 2
                log_sigma = s_.squeeze() if self.sigma.is_log else s_.log().squeeze()

            if self.output_distribution == 'gaussian':
                weighted_mse_loss_sampling = mse_loss(x_reco[1:] / sigma_,
                                                      x / sigma_,
                                                      ndim=len(self.input_shape),
                                                      batch_mean=False)

            else:
                output_cross_entropy_sampling = categorical_loss(x_reco[1:], x, ndim=len(self.input_shape),
                                                                 batch_mean=False)
                weighted_mse_loss_sampling = mse_loss(x_reco[1:].argmax(-len(self.input_shape) - 1) / 255,
                                                      x,
                                                      ndim=len(self.input_shape),
                                                      batch_mean=False)

            if self.sigma.is_rmse:
                if not batch and False:
                    print('**** wmse', *weighted_mse_loss_sampling.shape,
                          '({})'.format(mode))
                sigma2_ = weighted_mse_loss_sampling.mean(0)
                sigma_ = sigma2_.sqrt()
                log_sigma = sigma_.log().squeeze()
                weighted_mse_loss_sampling = weighted_mse_loss_sampling / \
                    sigma2_.unsqueeze(0)

            batch_quants['wmse'] = weighted_mse_loss_sampling.mean(0)

            batch_quants['mse'] = batch_quants['wmse'] * sigma2_

            if compute_iws:

                if self.output_distribution == 'gaussian':
                    log_iws = -D / 2 * \
                        (weighted_mse_loss_sampling + 2 * log_sigma /
                         sigma_dims + np.log(2 * np.pi))
                else:
                    log_iws = - output_cross_entropy_sampling

                # print('*** log_iws shape', *log_iws.shape)
                # if log_iws.isinf().sum():
                #     logging.error('MSE INF')

            batch_quants['xpow'] = x.pow(2).mean().item()
            total_measures['xpow'] = (current_measures['xpow'] * batch
                                      + batch_quants['xpow']) / (batch + 1)

            mse = batch_quants['mse'].mean().item()
            total_measures['mse'] = (current_measures['mse'] * batch
                                     + mse) / (batch + 1)

            total_measures['rmse'] = np.sqrt(total_measures['mse'])
            snr = total_measures['xpow'] / total_measures['mse']
            total_measures['dB'] = 10 * np.log10(snr)

        dictionary = self.encoder.prior.mean if self.encoder.prior.conditional else None

        if not batch:
            logging.debug('warmup kl weight=%e', kl_var_weighting)

        debug_msg = ('mu ' + str(mu.shape) + 'var ' + str(log_var.shape) +
                     'y ' + ('None' if y is None else str(y.shape)))

        # logging.debug('*** TBR in cvae' + debug_msg)

        batch_kl_losses = self.encoder.prior.kl(mu, log_var,
                                                y=y if self.encoder.prior.conditional else None,
                                                var_weighting=kl_var_weighting,
                                                )

        zdist = batch_kl_losses['distance']

        var_kl = batch_kl_losses['var_kl']

        total_measures['zdist'] = (current_measures['zdist'] * batch +
                                   zdist.mean().item()) / (batch + 1)

        total_measures['var_kl'] = (current_measures['var_kl'] * batch +
                                    var_kl.mean().item()) / (batch + 1)

        batch_losses['kl'] = batch_kl_losses['kl']

        batch_losses['zdist'] = batch_kl_losses['distance']
        batch_losses['var_kl'] = batch_kl_losses['var_kl']

        if self.y_is_decoded:

            if y_is_built and not self.y_is_coded:
                y_in = None
            else:
                y_in = y

            batch_quants['cross_y'] = x_loss(y_in,
                                             y_est,
                                             batch_mean=False)

            # print('*** cvae:545 cross_y', *batch_quants['cross_y'].shape)

        batch_losses['total'] = torch.zeros_like(batch_losses['kl'])

        # logging.error('THIS LINE HAS BEEN REMOVED')
        if dictionary is not None:

            # batch_losses['zdist'] = 0
            dict_mean = dictionary.mean(0)
            zdist_to_mean = (mu - dict_mean).pow(2).sum(1)
            dict_norm_var = dictionary.pow(2).sum(
                1).mean(0) - dict_mean.pow(2).sum()
            batch_losses['dzdist'] = zdist_to_mean + dict_norm_var
            batch_quants['imut-zy'] = self.encoder.capacity()
            batch_quants['ld-norm'] = dictionary.pow(2).mean()
            batch_quants['d-mind'] = self.encoder.dict_min_distance()

            for k in ('ld-norm', 'imut-zy', 'd-mind'):
                # total_measures[k] = (current_measures[k] * batch +
                #                     batch_quants[k].item()) / (batch + 1)
                total_measures[k] = batch_quants[k].item()

        if self.x_is_generated:
            batch_wmse = batch_quants['wmse']
            D = np.prod(self.input_shape)

            if self.training:
                self.sigma.update(rmse=batch_quants['mse'].mean().sqrt())
                # if not batch: print('**** sigma', ' -- '.join(f'{k}:{v}' for k, v in self.sigma.params.items()))
                self.training_parameters['sigma'] = self.sigma.params

            if self.output_distribution == 'gaussian':
                batch_logpx = -D * (2 * log_sigma / sigma_dims +
                                    batch_wmse + np.log(2 * np.pi)) / 2

            else:
                batch_logpx = - output_cross_entropy_sampling.mean(0)

            # if not batch and True:
            #     print('**** SHAPES l749')
            #     print('log_sigma', *log_sigma.shape)
            #     print('sigma dims', sigma_dims)
            #     print('wmse', *batch_wmse.shape)
            #     print('log_px', *batch_logpx.shape)
            #     print('total', *batch_losses['total'] .shape)

            batch_losses['wmse'] = batch_wmse
            batch_losses['cross_x'] = - batch_logpx

            batch_losses['total'] += batch_losses['cross_x']

            if compute_iws:

                t0_p_z = time.time()

                y_for_sampling = None
                if self.encoder.prior.conditional:
                    y_for_sampling = torch.stack([y for _ in z[1:]])

                z_y = z[1:]
                if y_for_sampling is not None and z_y.ndim < y.ndim + 2:
                    z_y = torch.stack([z_y for _ in y], 1)

                # log_p_z_y = torch.ones_like(y_for_sampling)
                log_p_z_y = self.encoder.prior.log_density(z_y, y_for_sampling)
                t0_p_z = time.time() - t0_p_z

                if not batch and False:
                    print('**** SHAPES (batch of size', *x.shape, ')')
                    print('z_y', *z_y.shape)
                    print('y_for_sampling', *y_for_sampling.shape)
                    print('p_z_y', *log_p_z_y.shape)
                    print(
                        'time for z|y: {:.0f}us/i'.format(1e6 * t0_p_z / x.shape[0]))

                if log_iws.ndim < log_p_z_y.ndim:
                    log_iws = log_iws.unsqueeze(1)

                if not batch and False:
                    _t = log_iws[1, :, 0]
                    print('*** mse' +
                          ' - '.join(['{:.2e}'] * len(_t)).format(*_t))
                    # print('*** iws:', *iws.shape, 'eps', *eps_norm.shape)

                log_iws = log_iws + log_p_z_y

                if not batch and False:
                    _t = log_p_z_y[1, :, 0]
                    print('*** p_z' +
                          ' - '.join(['{:.2e}'] * len(_t)).format(*_t))
                    # print('*** iws:', *iws.shape, 'eps', *eps_norm.shape)

                if log_p_z_y.isinf().sum():
                    logging.error('P_Z_Y INF')

                K = log_var.shape[-1]
                log_inv_q_z_x = (eps_norm + log_var.sum(-1)) / \
                    2 + K / 2 * np.log(2 * np.pi)

                if log_inv_q_z_x.dim() < log_iws.dim():
                    log_inv_q_z_x = log_inv_q_z_x.unsqueeze(1)

                if not batch and False:
                    _t = log_inv_q_z_x[1, :, 0]
                    print('*** q_z' +
                          ' - '.join(['{:.2e}'] * len(_t)).format(*_t))

                log_iws = log_iws + log_inv_q_z_x

                if log_inv_q_z_x.isinf().sum():
                    logging.error('Q_Z_X INF')

                log_iws_remainder = log_iws.max(0)[0]

                dlog_iws = log_iws - log_iws_remainder

                if not batch and False:
                    _t = log_iws[1, :, 0]
                    print('*** log_iws' +
                          ' - '.join(['{:.2e}'] * len(_t)).format(*_t))
                    print('*** max: {:.2e}'.format(log_iws.max()))

                if not batch and False:
                    _t = dlog_iws[1, :, 0]
                    print('*** dlog_iws' +
                          ' - '.join(['{:.2e}'] * len(_t)).format(*_t))
                    print('*** max: {:.2e}'.format(dlog_iws.max()))

                iws = (dlog_iws).exp().mean(0) + log_iws_remainder

                if 'iws' in self.loss_components:
                    batch_losses['iws'] = iws

        if self.y_is_decoded:
            batch_losses['cross_y'] = batch_quants['cross_y']
            """ print('*** cvae:528', 'losses:',
                  'y', *batch_losses['cross_y'].shape,
                  'T', *batch_losses['total'].shape)
            """

            # print('*** cvae:602', 'T:', *batch_losses['total'].shape,
            #      'Xy:', *batch_losses['cross_y'].shape)

            if not batch:
                logging.debug('CE(y) weight: %.1e', cross_y_weight)
            if cross_y_weight:
                batch_losses['total'] = batch_losses['total'] + \
                    cross_y_weight * batch_losses['cross_y']

        if self.is_vib:
            beta = self.beta if with_beta else 1.
            if not batch:
                logging.debug(f'KL coef={beta} / {self.gamma}')
            # print('*** 612: T:', *batch_losses['total'].shape, 'kl', *batch_losses['kl'].shape)
            batch_losses['total'] += beta * batch_losses['kl']
        else:
            beta = self.beta if with_beta else 1.
            if not batch:
                logging.debug(f'KL coef={beta}')

            batch_losses['total'] += beta * batch_losses['kl']

        if not self.is_vib:
            pass
            # print('******* x_', x_reco.shape)
            # x_reco = x_reco.mean(0)

        # logging.debug('Losses computed')
        if self.is_cvae:
            y_est_out = y_est[1:].mean(0)
        else:
            y_est_out = y_est[1:].mean(0)
        out = (x_reco, y_est_out, batch_losses, total_measures)
        if z_output:
            out += (mu, log_var, z)
        return out

    def predict(self, x, method=None, **kw):
        """x input of size (N1, .. ,Ng, D1, D2,..., Dt)

        creates a x of size C * N1, ..., D1, ...., Dt)
        and a y of size C * N1 * ... * Ng

        - method: If 'mean'(default) output is of size N1 *...* and
        gives y predicted. If None output is C * N1 *... and gives
        p(y|x,y). If 'loss' returns the y which minimizes loss(x, y)

        """

        _, logits, batch_losses, measures = self.evaluate(x)

        if not method:
            method = self.predict_methods[0]
        # print('cvae l. 192', x.device, batch_losses.device)
        return self.predict_after_evaluate(logits, batch_losses, method=method)

    def predict_after_evaluate(self, logits, losses, method='default'):

        if method == 'default':
            method = self.predict_methods[0]
            # method = 'mean' if self.is_jvae else 'esty'

        if method is None:
            return F.softmax(logits, -1)

        if method == 'mean':
            return F.softmax(logits, -1).mean(0).argmax(-1)

        if method == 'loss':
            return losses['total'].argmin(0)

        if method == 'esty':
            # return F.softmax(logits, -1).argmax(-1)
            return logits.argmax(-1)

        if method == 'foo':
            # return F.softmax(logits, -1).argmax(-1)
            return logits.argmin(-1)

        if method == 'closest':
            return losses['zdist'].argmin(0)

        if method == 'iws':
            return losses['iws'].argmax(0)

        if method == 'already':
            return losses['y_est_already']

        raise ValueError(f'Unknown method {method}')

    def batch_dist_measures(self, logits, losses, methods, to_cpu=False):

        # print('*** cvae:865', *losses)
        # for k in losses:
        #    print(k, *losses[k].shape)
        dist_measures = {m: None for m in methods}
        # for m in methods:
        #    assert not m.startswith('odin') or m in odin_softmax

        C = self.num_labels

        loss = losses['total']

        logp = - loss
        # ref is max of logp
        logp_max = logp.max(axis=0)[0]
        d_logp = logp - logp_max

        if 'iws' in losses:
            iws = losses['iws']
        elif [_ for _ in methods if 'iws' in _]:
            logging.warning('Asking for iws not computed; will use elo')
            iws = -losses['total']
        if self.losses_might_be_computed_for_each_class:
            iws_max = iws.max(axis=0)[0]
            d_iws = iws - iws_max

        for m in methods:

            m_ = m
            if m.endswith('-2s'):
                m = m[:-3]

            if '-a-' in m:
                m = m.split('-')[0]

            if m == 'elbo':
                if not self.losses_might_be_computed_for_each_class:
                    measures = logp
                else:
                    measures = logp_max
            elif m == 'iws':
                if self.losses_might_be_computed_for_each_class:
                    measures = d_iws.exp().sum(axis=0).log() + iws_max
                    if not self.is_jvae:
                        measures += np.log(C)
                else:
                    measures = iws
            elif m == 'sum':
                measures = d_logp.exp().sum(axis=0).log() + logp_max
            elif m == 'max':
                measures = logp_max
            elif m == 'softiws':
                measures = (losses['iws']).softmax(0).max(axis=0)[0]
            elif m.startswith('softiws-'):
                T = float(m[8:])
                measures = (-losses['iws'] / T).softmax(0).max(axis=0)[0]
            elif m in ('soft', 'softkl'):
                # measures = logp.softmax(0).max(axis=0)[0]
                measures = (-losses['kl']).softmax(0).max(axis=0)[0]
            elif m.startswith('softkl-'):
                T = float(m[7:])
                measures = (- losses['kl'] / T).softmax(0).max(axis=0)[0]
            elif m in ('zdist', 'kl', 'fisher_rao', 'mahala', 'kl_rec'):
                if self.is_vae:
                    measures = -losses[m]
                else:
                    measures = (-losses[m]).max(axis=0)[0]
            elif m.startswith('soft') and '-' in m:
                T = float(m.split('-')[-1])
                k = m.split('-')[0][4:]
                measures = (-losses[k] / T).softmax(0).max(axis=0)[0]
            elif m == 'logits':
                measures = logits.max(axis=-1)[0]
            elif m.startswith('baseline'):
                if '-' in m:
                    T = float(m.split('-')[-1])
                else:
                    T = 1
                measures = (logits / T).softmax(-1).max(axis=-1)[0]
            elif m == 'mag':
                measures = logp_max - logp.median(axis=0)[0]
            elif m == 'std':
                measures = logp.std(axis=0)
            elif m == 'mean':
                measures = d_logp.exp().mean(axis=0).log() + logp_max
            elif m == 'nstd':
                measures = (d_logp.exp().std(axis=0).log()
                            - d_logp.exp().mean(axis=0).log()).exp().pow(2)
            elif m == 'hyz':
                p_y_z = logits.softmax(-1)
                measures = (p_y_z * p_y_z.log()).sum(-1)
            elif m == 'IYx':
                d_logp_x = d_logp.exp().mean(axis=0).log()

                measures = ((d_logp * (d_logp.exp())).sum(axis=0) / (C * d_logp_x.exp())
                            - d_logp_x)

            elif m == 'mse' and self.is_cvae:
                measures = -losses['cross_x']

            elif m == 'wmse' and self.is_cvae:
                measures = -losses['wmse']

            elif m.startswith('odin'):
                # print('odin losses:', *[_ for _ in losses if _.startswith('odin')])
                measures = losses[m]

            else:
                raise ValueError(f'{m} is an unknown ood method')

            dist_measures[m_] = measures.cpu() if to_cpu else measures

        return dist_measures

    def compute_max_batch_size(self, batch_size=4096, which='all', trials=2):
        if which == 'all':
            self.compute_max_batch_size(batch_size, which='train')
            self.compute_max_batch_size(batch_size, which='test')
            logging.debug('Max batch sizes: %d for train, %d for test.',
                          self.max_batch_sizes['train'],
                          self.max_batch_sizes['test'])
            return

        logging.debug('Computing max batch size for %s', which)
        if 'max_batch_sizes' not in self.training_parameters:
            self.training_parameters['max_batch_sizes'] = {}

        training = which == 'train'
        self.train(training)

        # print('***', batch_size, *self.input_shape, self.device)
        x = torch.randn(batch_size, *self.input_shape, device=self.device)
        y = torch.ones(batch_size, dtype=int,
                       device=self.device) if training else None

        while batch_size > 2:
            x = x[:batch_size]
            if y is not None:
                y = y[:batch_size]
            try:
                logging.debug('Trying batch size of %s for %s of job#%s.',
                              batch_size,
                              which,
                              self.job_number)
                if training:
                    logging.debug('Evaling net')
                    for _ in range(trials):
                        _, _, batch_losses, _ = self.evaluate(x, y=y)
                        logging.debug('Net evaled')
                        L = batch_losses['total'].mean()
                        logging.debug('Backwarding net')
                        L.backward()
                else:
                    with torch.no_grad():
                        for _ in range(trials):
                            self.evaluate(x, y=y)
                self.training_parameters['max_batch_sizes'][which] = batch_size // 2
                logging.debug('Found max batch size for %s : %s',
                              which, batch_size)
                self.eval()
                return batch_size // 2
            except RuntimeError as e:
                if 'CUDA' in str(e):
                    logging.debug('Batch size of %s too much for %s.',
                                  batch_size,
                                  which)
                    _s = str(e).split('\n')[0]
                    logging.debug(_s)
                    batch_size //= 2
                else:
                    raise (e)

    @ property
    def max_batch_sizes(self):
        return {'train': 32, 'test': 32}
        logging.debug('Calling max batch size')
        max_batch_sizes = self.training_parameters.get('max_batch_sizes', {})
        if max_batch_sizes:
            return max_batch_sizes
        self.compute_max_batch_size()
        return self.max_batch_sizes

    @ max_batch_sizes.setter
    def max_batch_sizes(self, v):
        assert 'train' in v
        assert 'test' in v
        self.training_parameters['max_batch_sizes'] = v

    @ property
    def test_losses(self):
        return self._test_losses

    @ test_losses.setter
    def test_losses(self, d):
        # if not d:
        #     print('*** test losses reset')
        # else:
        #     _s = '*** test losses set to: kl {:.4}, total {:.4}'
        #     print(_s.format(d.get('kl', np.nan), d.get('total', np.nan)))

        self._test_losses = d

    @ property
    def test_measures(self):
        return self._test_measures

    @ test_measures.setter
    def test_measures(self, d):
        # if not d:
        #     print('*** test measures reset')
        # else:
        #     print('*** test measures set:', *d)
        self._test_measures = d

    def accuracy(self, testset=None,
                 batch_size=100,
                 num_batch='all',
                 method='all',
                 print_result=False,
                 update_self_testing=True,
                 outputs=EpochOutput(),
                 sample_dirs=[],
                 recorder=None,
                 epoch='last',
                 from_where='all',
                 epoch_tolerance=0,
                 log=True):
        """return detection rate.
        method can be a list of methods

        """
        MAX_SAMPLE_SAVE = 200

        device = next(self.parameters()).device

        if not testset:
            testset_name = self.training_parameters['set']
            transformer = self.training_parameters['transformer']
            _, testset = torchdl.get_dataset(
                testset_name, transformer=transformer, splits=['test'])

        else:
            testset_name = testset.name

        if method == 'all':
            predict_methods = self.predict_methods
            only_one_method = False

        elif type(method) is str:
            predict_methods = [method]
            only_one_method = True
        else:
            predict_methods = method
            only_one_method = False

        shuffle = True

        if num_batch == 'all':
            num_batch = int(np.ceil(len(testset) / batch_size))
            shuffle = False

        if num_batch >= int(np.ceil(len(testset) / batch_size)):
            num_batch = int(np.ceil(len(testset) / batch_size))
            shuffle = False

        if epoch == 'last':
            epoch = self.trained

        froms = available_results(self, testset=testset_name,
                                  oodsets=[],
                                  predict_methods=predict_methods,
                                  wanted_epoch=epoch,
                                  epoch_tolerance=0,
                                  where=from_where,
                                  misclass_methods=[]).get(epoch)
        acc = {}
        if not froms:
            return acc

        if froms[testset_name]['where']['json']:
            for m in predict_methods:
                if froms[testset_name]['json'][m]:
                    acc[m] = self.testing[epoch][m]['accuracy']

        if not sum(froms[testset_name]['where'][_] for _ in ('recorders', 'compute')):
            return acc

        if froms[testset_name]['where']['recorders']:
            rec_dir = froms.pop('rec_dir')
            recorder = LossRecorder.loadall(
                rec_dir, testset_name, map_location=self.device)[testset_name]
            num_batch = len(recorder)
            batch_size = recorder.batch_size

        recorded = recorder is not None and len(recorder) >= num_batch
        recording = recorder is not None and len(recorder) < num_batch

        if recorded:
            logging.debug('Losses already recorded')
            num_batch = len(recorder)

        if recording:
            logging.debug('Recording session loss for accruacy')
            recorder.reset()
            recorder.num_batch = num_batch

        n_err = dict()
        mismatched = dict()
        acc = dict()
        for m in predict_methods:
            n_err[m] = 0
            mismatched[m] = []
        n = 0

        if recorder is not None:
            recorder.init_seed_for_dataloader()

        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 pin_memory=True,
                                                 num_workers=0,
                                                 collate_fn=collate,
                                                 shuffle=shuffle)
        test_iterator = iter(testloader)
        start = time.time()

        total_loss = {k: 0. for k in self.loss_components}
        mean_loss = total_loss.copy()

        current_measures = {}
        measures = {}

        logging.debug('Starting accuracy for {} batches of size {} for set of size {}'.format(num_batch,
                                                                                              batch_size,
                                                                                              len(testset)))
        for i in range(num_batch):

            # save_batch_as_sample = sample_file and i < sample_save // batch_size

            if not recorded:
                data = next(test_iterator)
                x_test, y_test = data[0].to(device), data[1].to(device)
                (x_, logits,
                 batch_losses, measures) = self.evaluate(x_test, batch=i,
                                                         current_measures=current_measures)

                current_measures = measures
            else:
                components = [k for k in recorder.keys() if k in self.loss_components]
                batch_losses = recorder.get_batch(i, *components)
                logits = recorder.get_batch(i, 'logits').T
                y_test = recorder.get_batch(i, 'y_true')

            y_pred = {}
            for m in predict_methods:
                y_pred[m] = self.predict_after_evaluate(logits,
                                                        batch_losses,
                                                        method=m)

            if recording:
                recorder.append_batch(
                    **batch_losses, y_true=y_test, logits=logits.T)

            ind = y_test.unsqueeze(0)
            for k in batch_losses:
                shape = '?'
                _s = 'x'.join([str(_) for _ in batch_losses[k].shape])
                if k in ('total', 'iws'):
                    shape = 'CxN' if self.losses_might_be_computed_for_each_class else 'N'
                elif k in ('zdist', 'kl'):
                    shape = 'CxN' if self.is_cvae or self.is_xvae else 'N'
                elif k in ('cross_y',):
                    shape = 'CxN'
                elif k in ('cross_x', 'var_kl', 'dzdist'):
                    shape = 'N'
                else:
                    logging.debug(f'{k} shape has not been anticipated: {_s}')

                if not i:
                    logging.debug(
                        f'Predicted shape for {k}: {shape}. Actual: {_s}')
                try:
                    if shape == 'CxNxC':
                        batch_loss_y = batch_losses[k].max(
                            -1)[0].gather(0, ind)
                    elif shape == 'CxN':
                        batch_loss_y = batch_losses[k].gather(0, ind)
                    else:
                        batch_loss_y = batch_losses[k]
                except RuntimeError:
                    logging.error(
                        f'{k} shape has been wrongly anticipated: {shape} in lieu of {_s}')
                    total_loss[k] = 0.0

                if k not in total_loss:
                    total_loss[k] = 0.0

                total_loss[k] += batch_loss_y.float().mean().item()
                mean_loss[k] = total_loss[k] / (i + 1)

            for m in predict_methods:
                n_err[m] += (y_pred[m] != y_test).sum().item()
                mismatched[m] += [torch.where(y_test != y_pred[m])[0]]
            n += len(y_test)
            time_per_i = (time.time() - start) / (i + 1)
            for m in predict_methods:
                acc[m] = 1 - n_err[m] / n

            if print_result:
                outputs.results(i, num_batch, 0, 0,
                                losses={_: mean_loss[_] for _ in self.loss_components},
                                metrics={_: measures.get(_, np.nan) for _ in self.metrics},
                                accuracy=acc,
                                time_per_i=time_per_i,
                                batch_size=batch_size,
                                preambule=print_result)

        self.test_losses = mean_loss
        if measures:
            self.test_measures = measures

        if recorder is not None:
            recorder.restore_seed()

        if recording:
            logging.debug('Saving examples in' + ', '.join(sample_dirs))

            for d in sample_dirs:

                f = os.path.join(d, f'record-{testset.name}.pth')
                logging.debug(f'Saving recorder in {f}')
                recorder.save(f)

        if not recorded:
            saved_dict = {
                'losses': {m: batch_losses[m][:MAX_SAMPLE_SAVE] for m in batch_losses},
                'measures': measures,
                'x': x_test[:MAX_SAMPLE_SAVE],
                'y': y_test[:MAX_SAMPLE_SAVE],
                'x_': x_[:MAX_SAMPLE_SAVE] if self.is_vib else x_.mean(0)[:MAX_SAMPLE_SAVE],
                'y_pred': {m: y_pred[m][:MAX_SAMPLE_SAVE] for m in y_pred},
            }
            if self.is_xvae or self.is_cvae:
                mu_y = self.encoder.prior.mean.index_select(0, y_test)
                saved_dict['mu_y'] = mu_y[:MAX_SAMPLE_SAVE]

            for d in sample_dirs:
                f = os.path.join(d, f'sample-{testset.name}.pth')
                torch.save(saved_dict, f)

        for m in predict_methods:
            mresults = self.testing.get(epoch)
            n_already = 0
            if mresults:
                n_already = self.testing[epoch].get(m, {'n': 0})['n']

            update_self_testing_method = (update_self_testing and
                                          n > n_already)
            if update_self_testing_method:
                if epoch not in self.testing:
                    self.testing[epoch] = {}
                if log:
                    logged = 'Updating accuracy %.3f%% for method %s (n=%s)'
                    logging.debug(logged,
                                  100 * acc[m],
                                  m, n)
                    # self.testing[m]['n'],
                    # self.testing[m]['epochs'],
                    # n, self.trained)

                self.testing[epoch][m] = {'n': n,
                                          'epochs': epoch,
                                          'sampling': self._latent_samplings['eval'],
                                          'accuracy': acc[m]}

            elif log:

                logging.debug(
                    'Accuracies not updated bc {}<={}'.format(n, n_already))

        return acc[m] if only_one_method else acc

    def ood_detection_rates(self, oodsets=None,
                            testset=None,
                            batch_size=100,
                            num_batch='all',
                            method='all',
                            print_result=False,
                            update_self_ood=True,
                            epoch='last',
                            outputs=EpochOutput(),
                            recorders=None,
                            from_where='all',
                            sample_dirs=[],
                            sample_recorders=None,
                            log=True):

        if epoch == 'last':
            epoch = self.trained

        if not testset:
            testset_name = self.training_parameters['set']
            transformer = self.training_parameters['transformer']
            _, testset = torchdl.get_dataset(
                testset_name, transformer=transformer, splits=['test'])

        if not method:
            return

        odin_parameters = [_ for _ in self.ood_methods if _.startswith('odin')]

        ood_methods = make_list(method, self.ood_methods)

        if oodsets is None:
            oodsets = [torchdl.get_dataset(n, transformer=testset.transformer, splits=['test'])[1]
                       for n in testset.same_size]
            logging.debug('Oodsets loaded: ' +
                          ' ; '.join(s.name for s in oodsets))

        _s = 'Will compute ood fprs and aucs with ind of length {} and oods {}'
        _l = ', '.join('{}:{}'.format(_.name, len(_)) for _ in oodsets)
        logging.info(_s.format(len(testset), _l))

        oodsets_names = [o.name for o in oodsets]
        all_set_names = [testset.name] + oodsets_names

        ood_methods_per_set = {s: ood_methods for s in all_set_names}
        all_ood_methods = ood_methods

        if recorders == {}:
            recorders = {n: LossRecorder(batch_size) for n in all_set_names}

        recorders = recorders or {n: None for n in all_set_names}
        sample_recorders = sample_recorders or {}

        max_num_batch = num_batch
        num_batch = {testset.name: int(np.ceil(len(testset) / batch_size))}
        batch_size = {testset.name: batch_size}
        for o in oodsets:
            num_batch[o.name] = int(np.ceil(len(o) / batch_size[testset.name]))
            batch_size[o.name] = batch_size[testset.name]

        shuffle = {s: False for s in all_set_names}
        recording = {}
        recorded = {}

        froms = available_results(self, oodsets=oodsets_names,
                                  predict_methods=None,
                                  ood_methods=ood_methods,
                                  wanted_epoch=epoch,
                                  epoch_tolerance=0,
                                  where=from_where,
                                  misclass_methods=[]).get(epoch)

        # print('*** cvae:1355\n' + '\n'.join('{:10}: {}'.format(_, froms[_]['where']) for _ in all_set_names))
        # print('all_sets  :', froms['all_sets'])

        ood_results = {o.name: {} for o in oodsets}

        if not froms:
            return ood_results

        for dset in oodsets_names:
            if froms[dset]['where']['json']:
                logging.debug('OOD FPR already computed for {}'.format(dset))
                ood_results[dset] = self.ood_results[epoch][dset]

        oodsets = [o for o in oodsets if froms[o.name]['where']
                   ['compute'] or froms[o.name]['where']['recorders']]

        logging.debug('Kept oodsets: {}'.format(','.join(_.name for _ in oodsets)))

        if froms['all_sets']['recorders']:
            rec_dir = froms.pop('rec_dir')

            loaded_recorders = LossRecorder.loadall(rec_dir, *all_set_names, map_location=self.device)

            for dset in all_set_names:
                if froms[dset]['where']['compute']:
                    recorders[dset] = LossRecorder(batch_size[dset])
                    logging.debug('{} will be computed'.format(dset))
                elif froms[dset]['where']['recorders']:
                    recorders[dset] = loaded_recorders[dset]
                    num_batch[dset] = len(recorders[dset])
                    batch_size[dset] = recorders[dset].batch_size
                    logging.debug('{} recorder available'.format(dset))
                    ood_methods_per_set[dset] = [m for m in ood_methods if froms[dset]['recorders'].get(m)]
                elif dset != testset.name:
                    recorders.pop(dset, None)
                    logging.debug('{} will be discarded'.format(dset))

            for s in recorders:  #
                logging.debug('OOD methods for {}: '.format(s) +
                              '-'.join(ood_methods_per_set[s]))
            all_ood_methods = [m for m in ood_methods if any(
                [m in ood_methods_per_set[s] for s in recorders])]
            # print('*** ood methods', *ood_methods)
            ood_methods_per_set[testset.name] = [m for m in
                                                 ood_methods if m in all_ood_methods]

            oodsets = [o for o in oodsets if o.name in recorders]
            all_set_names = [s for s in all_set_names if s in recorders]

        for s in all_set_names:
            if type(max_num_batch) is int:
                shuffle[s] = num_batch[s] > max_num_batch
                num_batch[s] = min(num_batch[s], max_num_batch)
            recording[s] = recorders[s] is not None and len(
                recorders[s]) < num_batch[s]
            recorded[s] = recorders[s] is not None and len(
                recorders[s]) >= num_batch[s]
            if recorded[s]:
                logging.debug(
                    'Losses already computed for %s %s', s, recorders[s])
            if recording[s]:
                recorders[s].reset()
                recorders[s].num_batch = num_batch[s]
                logging.debug('Recording session for %s %s', s, recorders[s])

        device = next(self.parameters()).device

        if oodsets:

            s = testset.name
            _s = '{} measures for {}'.format('Recovering for recorder ' if recorded[s] else 'Computing', s)
            logging.debug(_s)

            ind_measures = {m: np.ndarray(0)
                            for m in ood_methods}

            if recorders[s] is not None:
                recorders[s].init_seed_for_dataloader()

            loader = torch.utils.data.DataLoader(testset,
                                                 shuffle=shuffle[s],
                                                 num_workers=0,
                                                 collate_fn=collate,
                                                 batch_size=batch_size[testset.name])

            t_0 = time.time()

            test_iterator = iter(loader)
            # test_set
            _test_losses = []
            _test_measures = []
            num_samples = 0

            for i in range(num_batch[s]):

                if not recorded[s]:

                    data = next(test_iterator)
                    x = data[0].to(device)
                    y = data[1].to(device)
                    if odin_parameters:
                        x.requires_grad_(True)
                    with torch.no_grad():
                        _, logits, losses, testset_measures, mu, log_var, z = self.evaluate(x,
                                                                                            batch=i,
                                                                                            z_output=True)

                        if sample_recorders and s in sample_recorders:
                            batch_samples = dict()
                            if 'mu' in sample_recorders[s]:
                                batch_samples['mu'] = mu
                            if 'y' in sample_recorders[s]:
                                batch_samples['y'] = y
                            if 'y_nearest' in sample_recorders[s]:
                                batch_samples['y_nearest'] = losses['zdist'].argmin(0)
                            sample_recorders[s].append_batch(**batch_samples)

                    _test_measures.append({k: testset_measures[k] for k in testset_measures})
                    odin_softmax = {}
                    if odin_parameters:
                        for T in self.ODIN_TEMPS:
                            with torch.enable_grad():
                                _, no_temp_logits = self.forward(
                                    x, z_output=False)
                                softmax = (no_temp_logits[1:].mean(
                                    0) / T).softmax(-1).max(-1)[0]
                                X = softmax.sum()
                            # print('***', X.requires_grad, (X / batch_size).cpu().item())
                            X.backward()
                            dx = x.grad.sign()
                            for eps in self.ODIN_EPS:
                                _, odin_logits = self.forward(
                                    x + eps * dx, z_output=False)
                                out_probs = (odin_logits[1:].mean(
                                    0) / T).softmax(-1).max(-1)[0]
                                odin_softmax['odin-{:.0f}-{:.4f}'.format(
                                    T, eps)] = out_probs
                else:
                    components = [k for k in recorders[s].keys()
                                  if k in self.loss_components or k.startswith('odin')]
                    losses = recorders[s].get_batch(i, *components)
                    logits = recorders[s].get_batch(i, 'logits').T
                    odin_softmax = {}

                _test_losses.append({k: losses[k].float().mean().item() for k in losses})

                if recording[s]:
                    recorders[s].append_batch(
                        **losses, **odin_softmax, y_true=y, logits=logits.T)

                measures = self.batch_dist_measures(logits, dict(**losses, **odin_softmax),
                                                    ood_methods_per_set[s])

                for m in ood_methods_per_set[s]:
                    # w_str = '*** ood {} {} - {}'.format(m,
                    #                                     ' '.join(map(str, ind_measures[m].shape)),
                    #                                     ' '.join(map(str, measures[m].shape)))
                    # # print('*** ood', m, *ind_measures[m].shape, ',', *measures[m].shape)
                    # logging.error(w_str)
                    try:
                        ind_measures[m] = np.concatenate([ind_measures[m],
                                                          measures[m].cpu()])
                    except ValueError as e:
                        logging.error('For {} incompatible shapes {} {}'.format(
                            m, ind_measures[m].shape, measures[m].shape))
                        raise e

                    if update_self_ood:
                        # print('***', s)
                        if epoch not in self.ood_results:
                            self.ood_results[epoch] = {}
                        if s not in self.ood_results[epoch]:
                            self.ood_results[epoch][s] = {}

                        self.ood_results[epoch][s][m] = {'n': len(ind_measures[m]),
                                                         'epochs': epoch,
                                                         'mean': ind_measures[m].mean(),
                                                         'std:': ind_measures[m].std()}

                t_i = time.time() - t_0
                t_per_i = t_i / (i + 1)

                outputs.results(i, num_batch[s], 0, 1,
                                metrics={m: ind_measures[m].mean()
                                         for m in ood_methods_per_set[s]},
                                fpr={m: np.nan for m in ood_methods_per_set[s]},
                                time_per_i=t_per_i,
                                batch_size=batch_size[s],
                                preambule=testset.name)

            self.test_losses = {k: sum(_[k] for _ in _test_losses) / (i + 1)
                                for k in _test_losses[0]}

            if _test_measures:
                self.test_measures = {k: sum(_[k] for _ in _test_measures) / (i + 1)
                                      for k in _test_measures[0]}

            if recorders[s] is not None:
                recorders[s].restore_seed()

            if recording[s]:
                for d in sample_dirs:
                    f = os.path.join(d, f'record-{s}.pth')

                    recorders[s].save(f.format(s=s))

                recorded[s] = True
                recording[s] = False

        kept_tpr = [pc / 100 for pc in range(90, 100)]
        no_result = {'epochs': 0,
                     'n': 0,
                     'auc': 0,
                     'tpr': kept_tpr,
                     'fpr': [1 for _ in kept_tpr],
                     'thresholds': [None for _ in kept_tpr],
                     'mean': np.nan,
                     'std': np.nan
                     }

        for oodset in [_ for _ in oodsets if _] + [testset] if oodsets else []:

            s = oodset.name
            ood_n_batch = num_batch[s]

            ood_results[s] = {m: copy.deepcopy(
                no_result) for m in ood_methods_per_set[s]}
            ood_measures = {m: np.ndarray(0) for m in ood_methods_per_set[s]}

            fpr_ = {}
            tpr_ = {}
            thresholds_ = {}
            auc_ = {}
            r_ = {}

            if recorders[s] is not None:
                recorders[s].init_seed_for_dataloader()

            loader = torch.utils.data.DataLoader(oodset,
                                                 num_workers=0,
                                                 collate_fn=collate,
                                                 shuffle=shuffle[s],
                                                 batch_size=batch_size[s])

            _s = 'Computing for set {o} ({n}) with {b} batches of {k} images'
            logging.debug(_s.format(o=oodset.name, b=ood_n_batch,
                                    k=batch_size[s], n=len(oodset)))

            t_0 = time.time()
            test_iterator = iter(loader)

            n_samples = 0
            for i in range(ood_n_batch):

                if not recorded[s]:
                    data = next(test_iterator)
                    x = data[0].to(device)
                    y = data[1].to(device)
                    if odin_parameters:
                        x.requires_grad_(True)

                    with torch.no_grad():
                        _, logits, losses, testset_measures, mu, log_var, z = self.evaluate(x,
                                                                                            batch=i,
                                                                                            z_output=True)

                        if sample_recorders and s in sample_recorders:
                            if 'mu' in sample_recorders[s]:
                                sample_recorders[s].append_batch(mu=mu)

                    odin_softmax = {}
                    if odin_parameters:
                        for T in self.ODIN_TEMPS:
                            with torch.enable_grad():
                                _, no_temp_logits = self.forward(
                                    x, z_output=False)
                                softmax = (no_temp_logits[1:].mean(
                                    0) / T).softmax(-1).max(-1)[0]
                                X = softmax.sum()
                            # print('***', X.requires_grad, (X / batch_size).cpu().item())
                            X.backward()
                            dx = x.grad.sign()
                            for eps in self.ODIN_EPS:
                                _, odin_logits = self.forward(
                                    x + eps * dx, z_output=False)
                                out_probs = (odin_logits[1:].mean(
                                    0) / T).softmax(-1).max(-1)[0]
                                odin_softmax['odin-{:.0f}-{:.4f}'.format(
                                    T, eps)] = out_probs

                else:
                    components = [k for k in recorders[s].keys()
                                  if k in self.loss_components or k.startswith('odin')]
                    losses = recorders[s].get_batch(i, *components)
                    logits = recorders[s].get_batch(i, 'logits').T
                    odin_softmax = {}

                if recording[s]:
                    recorders[s].append_batch(
                        **losses, **odin_softmax, y_true=y, logits=logits.T)

                # logging.info('batch {}/{} losses for {} '.format(i, ood_n_batch, s) + ' - '.join(losses))
                measures = self.batch_dist_measures(logits, dict(**losses, **odin_softmax),
                                                    ood_methods_per_set[s])

                for m in ood_methods_per_set[s]:
                    ood_measures[m] = np.concatenate([ood_measures[m],
                                                      measures[m].cpu()])

                t_i = time.time() - t_0
                t_per_i = t_i / (i + 1)
                meaned_measures = {m: ood_measures[m].mean()
                                   for m in ood_methods_per_set[s]}

                t0 = time.time()

                if not (recorded[s] or i % 100) or i == ood_n_batch - 1:
                    for m in ood_methods_per_set[s]:
                        logging.debug(
                            f'Computing roc curves for with metrics {m}')
                        _debug = 'medium' if i == ood_n_batch - 1 else 'soft'
                        _debug = False
                        two_sided = False
                        if m.endswith('-2s'):
                            two_sided = 'around-mean'
                        if '-a-' in m:
                            two_sided = tuple(int(_)
                                              for _ in m.split('-')[-2:])

                        # print('***',s, m, two_sided)
                        auc_[m], fpr_[m], tpr_[m], thresholds_[m] = roc_curve(ind_measures[m], ood_measures[m],
                                                                              *kept_tpr,
                                                                              debug=_debug,
                                                                              two_sided=two_sided)
                        # print('*** fpr at tpr', s, m)
                        for _, __ in zip(fpr_[m], tpr_[m]):
                            pass
                            # print('{:.2%} : {:.2%}'.format(_, __))
                        r_[m] = fpr_at_tpr(fpr_[m],
                                           tpr_[m],
                                           0.95,
                                           thresholds_[m])

                    outputs.results(i, ood_n_batch, 0, 1,
                                    metrics=meaned_measures,
                                    fpr=r_,
                                    time_per_i=t_per_i,
                                    batch_size=batch_size[s],
                                    preambule=oodset.name)

            if recorders[s] is not None:
                recorders[s].restore_seed()

            if epoch not in self.ood_results:
                self.ood_results[epoch] = {}

            for m in ood_methods_per_set[s]:

                ood_results[s][m] = {'epochs': epoch,
                                     'n': len(ood_measures[m]),
                                     'mean': ood_measures[m].mean(),
                                     'std': ood_measures[m].std(),
                                     'auc': auc_[m],
                                     'tpr': kept_tpr,
                                     'fpr': list(fpr_[m]),
                                     'thresholds': list(thresholds_[m])}

                if update_self_ood:

                    if oodset.name not in self.ood_results[epoch]:
                        self.ood_results[epoch][oodset.name] = {}
                    self.ood_results[epoch][s][m] = ood_results[s][m]

            if recording[s]:
                for d in sample_dirs:
                    f = os.path.join(d, f'record-{s}.pth')

                    recorders[s].save(f.format(s=s))

        for s in sample_recorders:
            for sdir in sample_dirs:
                fp = os.path.join(sdir, 'samples-{}.pth'.format(s))
                sample_recorders[s].save(fp)

        return ood_results

    def misclassification_detection_rates(self,
                                          predict_methods='all',
                                          misclass_methods='all',
                                          epoch='last',
                                          shown_tpr=0.95,
                                          from_where=('json', 'recorders'),
                                          print_result=False,
                                          update_self_results=True,
                                          outputs=EpochOutput()):

        froms = available_results(self,
                                  where=from_where,
                                  wanted_epoch=epoch,
                                  oodsets=[],
                                  ood_methods=[],
                                  predict_methods=predict_methods,
                                  misclass_methods=misclass_methods)

        testset = self.training_parameters['set']
        f = f'record-{testset}.pth'

        epoch = next(iter(froms))
        available = froms[epoch][testset]

        _s = '\n'.join('{}: {}'.format(*_) for _ in available.items())
        logging.debug('epoch: {}, Avail:\n{}'.format(epoch, _s))

        if available['where']['recorders']:

            sample_dir = os.path.join(
                self.saved_dir, 'samples', '{:04d}'.format(epoch))
            recorder = LossRecorder.load(os.path.join(
                sample_dir, f), map_location=self.device)
        else:
            logging.debug('Nothing to do')
            return

        methods = {'predict': predict_methods, 'miss': misclass_methods}

        logging.debug('Computing mdr for methods'.format(methods))

        for which, all_methods in zip(('predict', 'miss'),
                                      (self.predict_methods, self.misclass_methods)):

            methods[which] = make_list(methods[which],
                                       develop_starred_methods(all_methods, self.methods_params))

            # print('*** methods for', which, ':', methods[which], '(', *all_methods, ')')

            for m in methods[which]:
                # print('*** |_', m)
                assert m in all_methods

        losses = recorder._tensors
        logits = losses.pop('logits').T.cpu()
        y = losses.pop('y_true').cpu()

        kept_tpr = [pc / 100 for pc in range(90, 100)]

        _p = 5.2
        _p_1 = 4.1

        for predict_method in methods['predict']:

            m_ = ['{}-{}'.format(predict_method, _) for _ in methods['miss']]

            available_m = [_ for _, __ in zip(
                methods['miss'], m_) if available['recorders'][__]]

            if not available_m:
                continue

            y_ = self.predict_after_evaluate(logits, losses, method=predict_method).cpu()
            missed = np.asarray(y_ != y)
            correct = np.asarray(y_ == y)

            acc = correct.sum() / (correct.sum() + missed.sum())
            test_measures = self.batch_dist_measures(
                logits, losses, available_m, to_cpu=True)

            fpr_, tpr_, precision_, recall_, thresholds_ = {}, {}, {}, {}, {}

            logging.debug(
                f'Acc. for method {predict_method}: ({100 * acc:{_p}f}) ****')

            max_P = 0

            for m in available_m:
                measures = np.asarray(test_measures[m])

                auc, fpr, tpr, thr = roc_curve(measures[correct],
                                               measures[missed], *kept_tpr, debug=False)

                thr = thr['low']
                tp, fp, tn, fn = [], [], [], []

                for t in thr:
                    pos = measures >= t
                    neg = ~pos
                    tp.append((pos * correct).sum())
                    fp.append((pos * missed).sum())
                    tn.append((neg * missed).sum())
                    fn.append((neg * correct).sum())

                    _print = True
                    _print = False
                    if _print:
                        print('**** Correct: {:5} + {:5}, Missed: {:5} + {:5}'.format(tp[-1],
                                                                                      fn[-1],
                                                                                      tn[-1],
                                                                                      fp[-1]),
                              end=' ')
                        print('**** P={:4.1f} TPR={:4.1f} FPR={:4.1f} t={}'.format(
                            100 * tp[-1] / (tp[-1] + fp[-1]),
                            100 * tp[-1] / (tp[-1] + fn[-1]),
                            100 * fp[-1] / (fp[-1] + tn[-1]),
                            t
                        ))

                # print('*** fpr: {:.1f} -> {:.1f}'.format(100 * fpr[0], 100 * fpr[-1]))
                # print('*** tpr: {:.1f} -> {:.1f}'.format(100 * tpr[0], 100 * tpr[-1]))
                t95 = fpr_at_tpr(fpr, tpr, shown_tpr, thr,
                                 return_threshold=True)[1]

                pos = measures >= t95
                neg = ~pos
                tp95 = (pos * correct).sum()
                fp95 = (pos * missed).sum()

                p95 = tp95 / (tp95 + fp95)

                dp95 = p95 - acc

                r95 = tp95 / correct.sum()
                fpr95 = fp95 / missed.sum()

                # tpr95 = p95

                precision_[m] = [(t / (t + f)) for t, f in zip(tp, fp)]
                recall_[m] = [t / correct.sum() for t in tp]

                if p95 > max_P:
                    best_m = m
                    max_P = p95
                logging.debug('{:16}: '.format(m) +
                              '\tP={:{p}f} '.format(100 * p95, p=_p) +
                              '({:+{p}f}) '.format(100 * dp95, p=_p_1) +
                              'R={:{p}f} FPR={:{p}f}'.format(100 * r95, 100 * fpr95, p=_p))

                n = len(y)
                try:
                    n_already = self.testing[epoch][predict_methods][m]['n']
                except KeyError:
                    n_already = 0
                if update_self_results and n >= n_already:
                    if epoch not in self.testing:
                        self.testing[epoch] = {}
                    if predict_method not in self.testing[epoch]:
                        r = {'n': n, 'epochs': epoch,
                             'sampling': self._latent_samplings['eval'], 'accuracy': acc}
                        self.testing[epoch][predict_method] = r
                    r = {'n': n, 'epochs': epoch,
                         'sampling': self._latent_samplings['eval']}
                    r.update(dict(tpr=list(tpr), fpr=list(fpr),
                                  auc=auc, precision=list(precision_[m])))
                    # print(epoch, predict_method, m)
                    self.testing[epoch][predict_method][m] = r

    def train_model(self,
                    trainset=None,
                    transformer=None,
                    data_augmentation=None,
                    optimizer=None,
                    epochs=50,
                    batch_size=100,
                    test_batch_size=100,
                    validation=4096,
                    device=None,
                    testset=None,
                    oodsets=None,
                    acc_methods=None,
                    fine_tuning=False,
                    warmup=[0, 0],
                    warmup_gamma=[0, 0],
                    latent_sampling=None,
                    validation_sample_size=1024,
                    full_test_every=10,
                    ood_detection_every=10,
                    train_accuracy=False,
                    save_dir=None,
                    outputs=EpochOutput(),
                    signal_handler=SIGHandler()):
        """

        """
        if epochs:
            self.training_parameters['epochs'] = epochs

        if trainset:

            try:
                set_name = trainset.name
            except (AttributeError):
                set_name = trainset.__str__().splitlines()[
                    0].split()[-1].lower()
            transformer = trainset.transformer

        if self.trained:
            logging.info(f'Network partially trained ({self.trained} epochs)')
            logging.debug('Ignoring parameters (except for epochs)')

        else:
            if trainset:
                self.training_parameters['set'] = set_name
                self.training_parameters['transformer'] = transformer
                self.training_parameters['validation'] = validation
                self.training_parameters['full_test_every'] = full_test_every
                ss = '?'
                if hasattr(trainset, 'data'):
                    ss = trainset.data[0].shape
                ns = self.input_shape
                logging.debug(f'Shapes : {ss} / {ns}')
                # assert ns == ss or ss == ns[1:]

            if batch_size:
                self.training_parameters['batch_size'] = batch_size

            if latent_sampling:
                self._latent_samplings['train'] = latent_sampling
                self.training_parameters['latent_sampling'] = latent_sampling

            if data_augmentation:
                self.training_parameters['data_augmentation'] = data_augmentation

        assert self.training_parameters['set']

        set_name = self.training_parameters['set']
        data_augmentation = self.training_parameters['data_augmentation']
        full_test_every = self.training_parameters.get('full_test_every', 10)

        logging.debug(f'Getting {set_name}')

        if self.training_parameters.get('validation_split_seed ') is None:
            np.random.seed()
            self.training_parameters['validation_split_seed'] = np.random.randint(
                0, 2 ** 12)

        trainset, testset = torchdl.get_dataset(set_name,
                                                transformer=transformer,
                                                data_augmentation=data_augmentation)

        seed = self.training_parameters['validation_split_seed']
        set_lengths = [validation, len(trainset) - validation]
        validationset, trainset = torch.utils.data.random_split(trainset, set_lengths,
                                                                generator=torch.Generator().manual_seed(seed))

        validationset.name = 'validation'

        validation_sample_size = min(validation, validation_sample_size)

        logging.debug('Choosing device')
        device = choose_device(device)
        logging.debug(f'done {device}')

        if optimizer is None:
            optimizer = self.optimizer

        max_batch_sizes = self.max_batch_sizes

        test_batch_size = min(max_batch_sizes['test'], test_batch_size)

        logging.info(
            'Test batch size wanted {} / max {}'.format(test_batch_size, max_batch_sizes['test']))

        if batch_size:
            train_batch_size = min(batch_size, max_batch_sizes['train'])
        else:
            train_batch_size = max_batch_sizes['train']
            logging.info(
                'Train batch size wanted {} / max {}'.format(train_batch_size, max_batch_sizes['train']))

        logging.info('Train batch size is {}'.format(train_batch_size))

        warmup_ = self.training_parameters.get('warmup', [0, 0])
        warmup_gamma_ = self.training_parameters.get('warmup_gamma', [0, 0])
        for _ in (0, 1):
            warmup[_] = max(warmup[_], warmup_[_])
            warmup_gamma[_] = max(warmup_gamma[_], warmup_gamma_[_])
        self.training_parameters['warmup'] = warmup
        self.training_parameters['warmup_gamma'] = warmup_gamma

        x_fake = torch.randn(
            test_batch_size, *self.input_shape, device=self.device)
        y_fake = torch.randint(0, 1, size=(
            test_batch_size,), device=self.device)

        _, logits, losses, measures = self.evaluate(x_fake)

        sets = [set_name]
        if validation:
            sets.append('validation')
        for s in oodsets:
            sets.append(s.name)

        develop_starred_methods(self.ood_methods, self.methods_params)

        odin_parameters = [_ for _ in self.ood_methods if _.startswith('odin')]

        fake_odin_softmax = {o: torch.zeros(
            test_batch_size) for o in odin_parameters}

        recorders = {s: LossRecorder(test_batch_size,
                                     **losses,
                                     **fake_odin_softmax,
                                     logits=logits.T,
                                     y_true=y_fake)
                     for s in sets}

        for s in recorders:
            logging.debug('Recorder created for %s %s', s, recorders[s])

        logging.debug('Creating dataloader for training with batch size %s'
                      ' and validation with batch size %s',
                      train_batch_size, test_batch_size)

        if validation:
            l_valid = len(validationset)
        else:
            l_valid = 0
        logging.debug('Length of datasets: train={}, valid={}'.format(
            len(trainset), l_valid))

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=train_batch_size,
                                                  # pin_memory=True,
                                                  shuffle=True,
                                                  num_workers=0)

        if validationset:
            validationloader = torch.utils.data.DataLoader(validationset,
                                                           batch_size=test_batch_size,
                                                           # pin_memory=True,
                                                           shuffle=True,
                                                           num_workers=0)

        logging.debug('...done')

        dataset_size = len(trainset)
        remainder = (dataset_size % train_batch_size) > 0
        per_epoch = dataset_size // train_batch_size + remainder

        done_epochs = self.train_history['epochs']
        if done_epochs == 0:
            self.train_history = {'epochs': 0}  # will not be returned

        if not acc_methods:
            acc_methods = self.predict_methods

        if oodsets:
            ood_methods = self.ood_methods

        if fine_tuning:
            for p in self.parameters():
                p.requires_grad_(True)
            # if self.features:
            #     for p in self.features.parameters():
            #         if not p.requires_grad:
            #             # print('**** turn on grad')
            #             p.requires_grad_(True)

            # if self.is_jvae or self.is_vae:
            #     for p in self.features.parameters():
            #         if not p.requires_grad:
            #             # print('**** turn on grad')
            #             p.requires_grad_(True)

        train_measures = {}
        logging.debug(f'Starting training loop with {signal_handler}')

        last_was_full_test = False
        for epoch in range(done_epochs, epochs + 1):

            self.train_history[epoch] = {}
            history_checkpoint = self.train_history[epoch]
            for s in recorders:
                recorders[s].reset()

            logging.debug(f'Starting epoch {epoch} / {epochs}')
            t_start_epoch = time.time()
            # test

            full_test = ((epoch - done_epochs) and
                         epoch % full_test_every == 0)

            full_test = full_test or epoch == epochs

            ood_detection = ((epoch - done_epochs) and
                             epoch % ood_detection_every == 0)

            ood_detection = ood_detection or epoch == epochs

            last_was_full_test = full_test
            if (full_test or not epoch or ood_detection) and save_dir:
                sample_dirs = [os.path.join(save_dir, 'samples', d)
                               for d in ('last', f'{epoch:04d}')]

                for d in sample_dirs:
                    if not os.path.exists(d):
                        os.makedirs(d)
            else:
                sample_dirs = []

            with torch.no_grad():
                self.test_losses = {}
                self.test_measures = {}
                if oodsets and ood_detection:

                    self.ood_detection_rates(oodsets=oodsets, testset=testset,
                                             batch_size=test_batch_size,
                                             num_batch='all',
                                             outputs=outputs,
                                             recorders=recorders,
                                             sample_dirs=sample_dirs,
                                             print_result='*')

                if full_test:
                    test_accuracy = self.accuracy(testset,
                                                  batch_size=test_batch_size,
                                                  num_batch='all',
                                                  # device=device,
                                                  method=acc_methods,
                                                  # log=False,
                                                  outputs=outputs,
                                                  sample_dirs=sample_dirs,
                                                  update_self_testing=full_test,
                                                  recorder=recorders[set_name],
                                                  print_result='TEST' if full_test else 'test')
                    test_loss = self.test_losses.copy()
                    test_measures = self.test_measures.copy()
                    # print('**** test_loss to checkpoint')
                    history_checkpoint['test_accuracy'] = test_accuracy
                    history_checkpoint['test_measures'] = test_measures
                    history_checkpoint['test_loss'] = test_loss

                if validation:
                    validation_accuracy = self.accuracy(validationset,
                                                        batch_size=test_batch_size,
                                                        num_batch='all',
                                                        # device=device,
                                                        method=acc_methods,
                                                        # log=False,
                                                        outputs=outputs,
                                                        sample_dirs=sample_dirs,
                                                        update_self_testing=False,
                                                        recorder=recorders['validation'],
                                                        print_result='VALID'
                                                        if full_test else
                                                        'valid')
                    validation_loss = self.test_losses.copy()
                    validation_measures = self.test_measures.copy()
                    for k, v in zip(('accuracy', 'measures', 'loss'),
                                    (validation_accuracy, validation_measures, validation_loss)):
                        history_checkpoint['validation_' + k] = v

                if signal_handler.sig > 3:
                    logging.warning(
                        f'Abruptly breaking training loop bc of {signal_handler}')
                    break
                if save_dir:
                    self.save(save_dir)
            # train

            if epoch == epochs:
                break

            if train_accuracy:
                with torch.no_grad():
                    train_accuracy = self.accuracy(trainset,
                                                   batch_size=test_batch_size,
                                                   num_batch='all',
                                                   device=device,
                                                   method=acc_methods,
                                                   update_self_testing=False,
                                                   log=False,
                                                   outputs=outputs,
                                                   print_result='acc')

            t_i = time.time()
            t_start_train = t_i
            train_mean_loss = {k: 0. for k in self.loss_components}
            train_total_loss = train_mean_loss.copy()

            if signal_handler.sig > 3:
                logging.warning(
                    f'Abruptly breaking training loop bc of {signal_handler}')
                break

            if save_dir:
                self.save(save_dir)

            current_measures = {}

            if signal_handler.sig > 2 or full_test and signal_handler.sig > 1:
                logging.warning(f'Breaking training loop bc of signal {signal_handler}'
                                f' after {epoch} epochs.')
                break

            self.encoder.prior.thaw_means(epoch)

            self.train()

            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                x, y = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                if self.training:
                    warmup_weighting = max(0., min(1., (epoch + 1 - warmup[0]) / (warmup[1] + 1)))
                    gamma_weighting = max(0., min(1., (epoch + 1 - warmup_gamma[0]) / (warmup_gamma[1] + 1)))
                else:
                    warmup_weighting = 1.
                    gamma_weightting = 1.

                # with autograd.detect_anomaly():
                # forward + backward + optimize
                (_, y_est,
                 batch_losses, measures) = self.evaluate(x, y,
                                                         batch=i,
                                                         with_beta=True,
                                                         kl_var_weighting=warmup_weighting,
                                                         gamma_weighting=gamma_weighting,
                                                         # mse_weighting=warmup_weighting,
                                                         current_measures=current_measures)

                current_measures = measures
                batch_loss = batch_losses['total'].mean()

                L = batch_loss

                for p in self.parameters():
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        print('GRAD NAN')
                        sys.exit(1)

                L.backward()
                optimizer.clip(self.parameters())
                optimizer.step()

                for k in batch_losses:
                    if k not in train_total_loss:
                        train_total_loss[k] = 0.0
                        train_mean_loss[k] = 0.0

                    train_total_loss[k] += batch_losses[k].mean().item()
                    train_mean_loss[k] = train_total_loss[k] / (i + 1)

                t_per_i = (time.time() - t_start_train) / (i + 1)
                outputs.results(i, per_epoch, epoch + 1, epochs,
                                preambule='train',
                                losses={_: train_mean_loss[_] for _ in self.loss_components},
                                metrics={_: measures[_] for _ in self.metrics},
                                accuracy={_: np.nan for _ in self.predict_methods},
                                time_per_i=t_per_i,
                                batch_size=train_batch_size,
                                end_of_epoch='\n')

            self.eval()
            train_measures = measures.copy()
            if train_accuracy:
                history_checkpoint['train_accuracy'] = train_accuracy
            history_checkpoint['train_loss'] = train_mean_loss
            history_checkpoint['train_measures'] = train_measures
            self.train_history['epochs'] += 1
            history_checkpoint['lr'] = self.optimizer.lr
            self.trained += 1
            if fine_tuning:
                self.training_parameters['fine_tuning'].append(epoch)

            optimizer.update_lr()

            if signal_handler.sig > 3:
                logging.warning(
                    f'Abruptly breaking training loop bc of {signal_handler}')
                break

            if save_dir:
                self.save(save_dir)

        for s in recorders:
            recorders[s].reset()
        if save_dir:
            sample_dirs = [os.path.join(save_dir, 'samples', d)
                           for d in ('last', f'{epoch + 1:04d}')]

            for d in sample_dirs:
                if not os.path.exists(d):
                    os.makedirs(d)

        if oodsets and not signal_handler.sig > 1:

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug('Clearing cuda cache')

            with torch.no_grad():
                self.ood_detection_rates(oodsets=oodsets, testset=testset,
                                         batch_size=test_batch_size,
                                         num_batch='all',
                                         outputs=outputs,
                                         recorders=recorders,
                                         sample_dirs=sample_dirs,
                                         print_result='*')

        if testset and not signal_handler.sig > 1:

            recorder = recorders[set_name]
            # print(num_batch, sample_size)
            with torch.no_grad():
                test_accuracy = self.accuracy(testset,
                                              batch_size=test_batch_size,
                                              method=acc_methods,
                                              recorder=recorder,
                                              sample_dirs=sample_dirs,
                                              # log=False,
                                              outputs=outputs,
                                              print_result='TEST')

        if signal_handler.sig > 3:
            logging.warning(f'Skipping saving because of {signal_handler}')
        elif save_dir:
            self.save(save_dir)

        logging.debug('Finished training')

    def summary(self):

        logging.warning('SUMMARY FUNCTION NOT IMPLEMENTED')

    @ property
    def device(self):
        return next(self.parameters()).device

    @ device.setter
    def device(self, d):
        self.to(d)

    def to(self, d):
        super().to(d)
        self.optimizer.to(d)

    @ property
    def nparams(self):
        return sum(p.nelement() for p in self.parameters())

    @ property
    def latent_sampling(self):
        return self._latent_sampling

    @ latent_sampling.setter
    def latent_sampling(self, v):
        self._latent_sampling = v
        self.encoder.sampling_size = v

    def plot_model(self, dir='.', suffix='.png', show_shapes=True,
                   show_layer_names=True):

        if dir is None:
            dir = '.'

        def _plot(net):
            logging.warning(f'PLOT HAS TO BE IMPLEMENTED WITH TB')
            # f_p = save_load.get_path(dir, net.name+suffix)
            # plot_model(net, to_file=f_p, show_shapes=show_shapes,
            #            show_layer_names=show_layer_names,
            #            expand_nested=True)

        _plot(self)
        _plot(self.encoder)
        _plot(self.decoder)
        _plot(self.classifier)

    def has_same_architecture(self, other_net, excludes=[]):

        a1 = self.print_architecture(sampling=True)
        a2 = other_net.print_architecture(sampling=True)
        # logging.debug(f'Comparing {a1}')
        # logging.debug(f'with      {a2}')

        if self.activation != other_net.activation:
            return False
        ls = self._sizes_of_layers
        ls_ = other_net._sizes_of_layers
        if len(ls) != len(ls_):
            return False
        for s, s_ in zip(ls, ls_):
            if np.any((np.array(s) != np.array(s_))):
                return False

        if self.latent_sampling != other_net.latent_sampling:
            return False

        # logging.debug('And they are the same')
        return True

    def has_same_training(self, other,
                          excludes=('epochs', 'batch_size', 'sampling')):

        t1 = self.training_parameters
        to = other.training_parameters

        for (t, t_) in ((t1, to), (to, t1)):
            for k in [i for i in t if i not in excludes]:
                # logging.debug(f'*** TBR {k}')
                if t[k] and t[k] != t_.get(k, None):
                    return False
                if not t[k] and t_.get(k, None):
                    return False

        return True

    def print_training(self, epochs=None, set=None):

        done_epochs = self.trained
        if not epochs:
            epochs = self.training_parameters['epochs']

        sampling = self.training_parameters['latent_sampling']
        if not set:
            set = self.training_parameters['set']
        s = f'{set}: {self.sigma} -- L={sampling} {done_epochs}/{epochs}'
        return s

    print_architecture = save_load.print_architecture
    option_vector = save_load.option_vector

    def save(self, dir_name=None, except_optimizer=False, except_state=False):
        """Save the params in params.json file in the directroy dir_name and, if
        trained, the weights inweights.h5.

        """
        job_dir = 'jobs'

        if dir_name is None:
            dir_name = os.path.join(job_dir, self.print_architecture,
                                    str(self.job_number))

        save_load.save_json(self.architecture, dir_name, 'params.json')
        save_load.save_json(self.training_parameters, dir_name, 'train_params.json')
        save_load.save_json(self.testing, dir_name, 'test.json')
        save_load.save_json(self.ood_results, dir_name, 'ood.json')
        save_load.save_json(self.train_history, dir_name, 'history.json')

        if self.trained and not except_state:
            w_p = save_load.get_path(dir_name, 'state.pth')
            torch.save(self.state_dict(), w_p)
            # print('**** state saved')
            if not except_optimizer:
                w_p = save_load.get_path(dir_name, 'optimizer.pth')
                torch.save(self.optimizer.state_dict(), w_p)

        return dir_name

    @ classmethod
    def load(cls, dir_name,
             build_module=True,
             load_state=True,
             load_train=True,
             load_test=True,
             strict=True,
             ):
        """dir_name : where params.json is (and weigths.h5 if applicable)

        """

        if not os.path.exists(os.path.join(dir_name, 'params.json')):
            raise NoModelError(dir_name)

        if os.path.exists(os.path.join(dir_name, 'deleted')):
            raise DeletedModelError(dir_name)

        if not build_module:
            load_state = False

        # default
        default_params = {'version': 1., 'output_distribution': 'gaussian'}

        train_params = {}

        loaded_params = save_load.load_json(dir_name, 'params.json')

        try:
            s = dir_name.split(os.sep)[-1]
            job_number_by_dir_name = int(s)
        except ValueError:
            job_number_by_dir_name = s

        loaded_params['job_number'] = loaded_params.get('job_number', job_number_by_dir_name)

        resumed_file = os.path.join(dir_name, 'RESUMED')
        is_resumed = os.path.exists(resumed_file)

        if is_resumed:
            with open(resumed_file, 'r') as resumed_f:
                is_resumed = resumed_f.read()
                try:
                    is_resumed = int(is_resumed)
                except ValueError:
                    is_resumed = False

        params = default_params.copy()
        params.update(loaded_params)

        loaded_train = False
        try:
            train_params.update(save_load.load_json(dir_name, 'train_params.json'))
            logging.debug('Training parameters loaded')
            loaded_train = True
        except (FileNotFoundError):
            pass

        loaded_test = False
        try:
            testing = save_load.load_json(dir_name, 'test.json',
                                          presumed_type=int)
            loaded_test = load_test

        except (FileNotFoundError):
            pass

        loaded_ood = False
        try:
            ood_results = save_load.load_json(dir_name, 'ood.json',
                                              presumed_type=int)

        except (FileNotFoundError):
            ood_results = {}
            pass

        try:
            train_history = save_load.load_json(dir_name, 'history.json', presumed_type=int)
        except (FileNotFoundError, IndexError):
            train_history = {'epochs': 0}

        resave_arch = False
        if not build_module:
            model = save_load.Shell()
            model.architecture = params.copy()
            model.type = model.architecture['type']
            model.loss_components = cls.loss_components_per_type[model.type]
            logging.debug('Ghost network loaded')
            model.job_number = params['job_number']
            model.ood_methods = cls.ood_methods_per_type[model.type]
            model.methods_params = cls.methods_params
            model.predict_methods = cls.predict_methods_per_type[model.type]
            model.misclass_methods = cls.misclass_methods_per_type[model.type]
            gamma = train_params['gamma']
            model.y_is_decoded = True
            if model.type in ('cvae', 'vae'):
                model.y_is_decoded = gamma

            if model.y_is_decoded and 'esty' not in model.predict_methods:
                model.predict_methods = model.predict_methods + ['esty']

            if model.y_is_decoded and 'cross_y' not in model.loss_components:
                model.loss_components += ('cross_y',)

            model.testing = {}
            if isinstance(train_params['sigma'], dict):
                model.sigma = Sigma(**train_params['sigma'])
            else:
                model.sigma = Sigma(train_params['sigma'])
        if build_module:
            logging.debug('Building the network')
            keys_out = ('set', 'epochs', 'data_augmentation',
                        'batch_size', 'fine_tuning', 'warmup',
                        'warmup_gamma',
                        'full_test_every', 'validation_split_seed',
                        'max_batch_sizes',
                        'pretrained_features',
                        'pretrained_upsampler',
                        'transformer', 'validation')
            train_params_for_const = train_params.copy()
            for _ in keys_out:
                train_params_for_const.pop(_, None)
            for _ in train_params:
                if _.startswith('early-'):
                    train_params_for_const.pop(_, None)
            model = cls(**params, **train_params_for_const)

            if train_params.get('pretrained_upsampler'):
                model.features.mame = train_params['pretrained_features']
                for p in model.features.parameters():
                    p.requires_grad_(False)

            if train_params.get('pretrained_upsampler'):
                model.imager.mame = train_params['pretrained_upsampler']
                for p in model.imager.parameters():
                    p.requires_grad_(False)

        model.saved_dir = dir_name
        model.trained = train_history['epochs']
        model.train_history = train_history
        model.is_resumed = is_resumed
        model.training_parameters = train_params
        if loaded_test:
            # logging.debug('Updating test_results ({}) with {}'.format('--'.join()))
            model.testing.update(testing)

        if load_test:
            model.ood_results = ood_results

        if load_state:  # and vae.trained:
            logging.debug('Loading state')
            w_p = save_load.get_path(dir_name, 'state.pth')
            try:
                state_dict = torch.load(w_p, map_location=model.device)
            except FileNotFoundError as e:
                raise StateFileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), e.filename)
            except RuntimeError as e:
                raise e
            try:
                keys = model.load_state_dict(state_dict, strict=strict).missing_keys

            except RuntimeError as e:
                if strict:
                    raise e

            w_p = save_load.get_path(dir_name, 'optimizer.pth')
            try:
                opt_state_dict = torch.load(w_p)
                model.optimizer.load_state_dict(opt_state_dict)
                logging.debug('optimizer loaded')

            except FileNotFoundError:
                logging.warning('Optimizer state file not found')
            model.optimizer.update_scheduler_from_epoch(model.trained)
            logging.debug('Loaded')

            if keys:
                raise MissingKeys(model, state_dict, keys)

        return model

    def copy(self, with_state=True):

        s = ''.join([random.choice('0123456789abcedf') for _ in range(30)])
        d = os.path.join(tempfile.gettempdir(), s)
        self.save(d)
        m = self.load(d, build_module=True, load_state=with_state)
        shutil.rmtree(d)
        return m


if __name__ == '__main__':

    pass
