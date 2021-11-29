import logging
import copy
import torch
import torch.utils.data
from torch import nn, autograd
from module.optimizers import Optimizer
from torch.nn import functional as F
from module.losses import x_loss, kl_loss, mse_loss

from utils.save_load import LossRecorder, last_samples, available_results, develop_starred_methods

from utils.misc import make_list

from module.vae_layers import VGGFeatures, ConvDecoder, Encoder, Decoder, Classifier, ConvFeatures, Sigma
from module.vae_layers import onehot_encoding

import utils.torch_load as torchdl
from utils.torch_load import choose_device
from utils import save_load
import numpy as np

from utils.roc_curves import roc_curve, fpr_at_tpr
# from sklearn.metrics import auc, roc_curve

from utils.print_log import EpochOutput

from utils.parameters import get_args

from utils.testing import testing_plan

from utils.signaling import SIGHandler

import os.path
import time

import re

DEFAULT_ACTIVATION = 'relu'
# DEFAULT_OUTPUT_ACTIVATION = 'sigmoid'
DEFAULT_OUTPUT_ACTIVATION = 'linear'
DEFAULT_LATENT_SAMPLING = 100


activation_layers = {'linear': nn.Identity,
                     'sigmoid': nn.Sigmoid, 
                     'relu': nn.ReLU} 


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
                                'cvae': ('cross_x', 'kl', 'total', 'zdist', 'var_kl', 'dzdist', 'iws'),
                                'xvae': ('cross_x', 'kl', 'total', 'zdist', 'iws'),
                                'vae': ('cross_x', 'kl', 'var_kl', 'total', 'iws'),
                                'vib': ('cross_y', 'kl', 'total')}
    
    predict_methods_per_type = {'jvae': ['loss', 'esty'],
                                # 'cvae': ('closest', 'iws'),
                                'cvae': ['iws', 'closest'],
                                'xvae': ['loss', 'closest'],
                                'vae': [],
                                'vib': ['esty']}

    metrics_per_type = {'jvae': ['std', 'snr', 'sigma'],
                        'cvae': ['std', 'snr',  'd-mind', 'ld-norm', 'sigma'],
                        'xvae': ['std', 'snr', 'zdist', 'd-mind', 'ld-norm', 'sigma'],
                        'vae': ['std', 'snr', 'sigma'],
                        'vib': ['sigma',]}

    ood_methods_per_type = {'cvae': ['iws-2s', 'iws', 'kl', 'mse', 'max', 'soft'],
                            'xvae': ['max', 'mean', 'std'],  # , 'mag', 'IYx'],
                            'jvae': ['max', 'sum',  'std'],  # 'mag'],
                            'vae': ['iws-2s', 'iws', 'logpx'],
                            'vib': ['odin*', 'baseline', 'logits']}

    misclass_methods_per_type = {'cvae': ['iws', 'kl', 'softkl*', 'softiws*'],
                           'xvae': [],
                           'jvae': [],
                           'vae': [],
                           'vib': ['odin*', 'baseline', 'logits']}
    
    ODIN_TEMPS = [_ * 10 ** i for _ in (1, 2, 5) for i in (0, 1, 2)] + [1000]
    ODIN_EPS = [_ / 20 * 0.004 for _ in range(21)]
    # ODIN_EPS = [_ / 40 * 0.008 for _ in range(41)]
    
    # ODIN_TEMPS = [500, 1000]
    # ODIN_TEMPS = [1, 1000]
    # ODIN_EPS = [0]  # , 0.002, 0.004]

    odin_params = []
    for T in ODIN_TEMPS:
        for eps in ODIN_EPS:
            odin_params.append('odin-{:.0f}-{:.4f}'.format(T, eps))
    methods_params = {'odin': odin_params,
                      'softiws': [f'softiws-{_:.0f}' for _ in ODIN_TEMPS],
                      'softkl': [f'softkl-{_:.0f}' for _ in ODIN_TEMPS]}
    

    def __init__(self,
                 input_shape,
                 num_labels,
                 type_of_net='jvae',  # or 'vib' or cvae or vae
                 y_is_coded=False,
                 job_number=0,
                 features=None,
                 pretrained_features=None,
                 features_channels=None,
                 conv_padding=1,
                 batch_norm=False,
                 encoder_layer_sizes=[36],
                 latent_dim=32,
                 latent_prior_variance=1,
                 beta=1.,
                 gamma=0.,
                 rho=0.,
                 rho_temp=np.inf,
                 dictionary_variance=1,
                 learned_coder=False,
                 dictionary_min_dist=None,
                 init_coder=True,
                 coder_capacity_regularization=False,
                 decoder_layer_sizes=[36],
                 upsampler_channels=None,
                 pretrained_upsampler=None,
                 classifier_layer_sizes=[36],
                 force_cross_y=0.,
                 name='joint-vae',
                 activation=DEFAULT_ACTIVATION,
                 latent_sampling=DEFAULT_LATENT_SAMPLING,
                 test_latent_sampling=None,  # if none will be the same as (train) latent_sampling
                 encoder_forced_variance=False,
                 output_activation=DEFAULT_OUTPUT_ACTIVATION,
                 sigma={'value': 0.5},
                 optimizer={},
                 shadow=False,
                 *args, **kw):

        super().__init__(*args, **kw)
        self.name = name

        self.job_number = job_number
        
        assert type_of_net in ('jvae', 'cvae', 'xvae', 'vib', 'vae')
        self.type = type_of_net

        self.loss_components = self.loss_components_per_type[self.type]
 
        self.metrics = self.metrics_per_type[self.type]
        self.predict_methods = self.predict_methods_per_type[self.type]
        self.ood_methods = self.ood_methods_per_type[self.type]
        self.misclass_methods = self.misclass_methods_per_type[self.type]
        
        self.is_jvae = type_of_net == 'jvae'
        self.is_vib = type_of_net == 'vib'
        self.is_vae = type_of_net == 'vae'
        self.is_cvae = type_of_net == 'cvae'
        self.is_xvae = type_of_net == 'xvae'

        assert not (y_is_coded and (self.is_vib or self.is_vae))
        self.y_is_coded = y_is_coded
        # self.y_is_decoded = self.is_vib or self.is_jvae
        self.y_is_decoded = True
        if self.is_cvae or self.is_vae:
            self.y_is_decoded = gamma

        self.coder_has_dict = self.is_cvae or self.is_xvae
        
        self.x_is_generated = not self.is_vib

        self.losses_might_be_computed_for_each_class = not self.is_vae
        
        logging.debug('y is%s coded', '' if self.y_is_coded else ' not')

        self._measures = {}
        
        self.force_cross_y = force_cross_y
        if not self.y_is_decoded and not force_cross_y:
            classifier_layer_sizes = []
        if not self.x_is_generated:
            decoder_layer_sizes = []
            upsampler_channels = None

        if self.y_is_decoded and 'esty' not in self.predict_methods:
            self.predict_methods =  self.predict_methods + ['esty']

        if self.y_is_decoded and 'cross_y' not in self.loss_components:
            self.loss_components += ('cross_y',)
            
        # no upsampler if no features
        assert (not upsampler_channels or features)

        if not features:
            batch_norm = False
        else:
            batch_norm_encoder = (batch_norm == 'encoder' or
                                  batch_norm == 'both')
            batch_norm_decoder = batch_norm == 'both'
        if features:
            logging.debug('Building features')
            if pretrained_features:
                feat_dict = torch.load(pretrained_features)
            else:
                feat_dict = None

            if features.startswith('vgg'):
                self.features = VGGFeatures(features, input_shape,
                                            channels=features_channels,
                                            batch_norm=batch_norm_encoder,
                                            pretrained=feat_dict)
                features_arch = self.features.architecture
                
            elif features == 'conv':
                self.features = ConvFeatures(input_shape,
                                             features_channels,
                                             batch_norm=batch_norm_encoder,
                                             padding=conv_padding,
                                             kernel=2*conv_padding+2)
                features_arch = {'features': features,
                                 'features_channels': features_channels,
                                 'conv_padding': conv_padding,}

            features_arch['name'] = self.features.name
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
            
        sampling = latent_sampling > 1 or self.sigma.learned or self.sigma.per_dim or bool(self.sigma.value > 0)
        if not sampling:
            logging.debug('Building a vanilla classifier')

        self.beta = beta

        if not rho and (rho_temp is not None) and np.isfinite(rho_temp):
            rho = 1
        self.rho = rho if self.coder_has_dict else None
        self.gamma = gamma if self.y_is_decoded else None
        self.rho_temp = rho_temp if self.coder_has_dict else None
        
        logging.debug(f'Gamma: {self.gamma}')
        
        self.latent_prior_variance = latent_prior_variance
        self.encoder = Encoder(encoder_input_shape, num_labels,
                               intermediate_dims=encoder_layer_sizes,
                               latent_dim=latent_dim,
                               y_is_coded = self.y_is_coded,
                               sigma_output_dim=self.sigma.output_dim if self.sigma.coded else 0,
                               forced_variance = encoder_forced_variance,
                               sampling_size=latent_sampling,
                               dictionary_variance=dictionary_variance,
                               learned_dictionary=learned_coder,
                               dictionary_min_dist=dictionary_min_dist,
                               activation=activation, sampling=sampling)

        if init_coder:
            self.encoder.init_dict()
        
        self.coder_capacity_regularization = coder_capacity_regularization
        activation_layer = activation_layers[activation]()

        if self.x_is_generated:
            decoder_layers = []
            input_dim = latent_dim
            for output_dim in decoder_layer_sizes:
                decoder_layers += [nn.Linear(input_dim, output_dim) ,
                                   activation_layer]
                input_dim = output_dim

            self.decoder = nn.Sequential(*decoder_layers)

            imager_input_dim = input_dim
            if upsampler_channels:
                upsampler_first_shape = self.features.output_shape
                if pretrained_upsampler:
                    upsampler_dict = torch.load(pretrained_upsampler)
                else:
                    upsampler_dict = None
                self.imager = ConvDecoder(imager_input_dim,
                                          upsampler_first_shape,
                                          upsampler_channels,
                                          upsampler_dict=upsampler_dict,
                                          batch_norm=batch_norm_decoder,
                                          output_activation=output_activation)

            else:
                upsampler_channels = None
                activation_layer = activation_layers[output_activation]()
                self.imager = nn.Sequential(nn.Linear(imager_input_dim,
                                                    np.prod(input_shape)),
                                            activation_layer)

        self.classifier = Classifier(latent_dim, num_labels,
                                     classifier_layer_sizes,
                                     activation=activation)

        self.input_shape = tuple(input_shape)
        self.num_labels = num_labels
        self.input_dim = len(input_shape)

        self.training_parameters = {}  # 
        
        self.batch_norm = batch_norm
        
        self._sizes_of_layers = [input_shape, num_labels,
                                 encoder_layer_sizes, latent_dim,
                                 decoder_layer_sizes,
                                 upsampler_channels,
                                 classifier_layer_sizes]

        self.architecture = {'input': input_shape,
                             'labels': num_labels,
                             'type': type_of_net,
                             # 'features': features_arch, 
                             'encoder': encoder_layer_sizes,
                             'batch_norm': batch_norm,
                             'activation': activation,
                             'encoder_forced_variance': self.encoder.forced_variance,
                             'latent_dim': latent_dim,
                             'test_latent_sampling': test_latent_sampling,
                             'latent_prior_variance': latent_prior_variance,
                             'decoder': decoder_layer_sizes,
                             'upsampler': upsampler_channels,
                             'classifier': classifier_layer_sizes,
                             'output': output_activation}

        self.depth = (len(encoder_layer_sizes)
                      + len(decoder_layer_sizes) 
                      + len(classifier_layer_sizes))
        
        self.width = (sum(encoder_layer_sizes)
                      + sum(decoder_layer_sizes)
                      + sum(classifier_layer_sizes)) 
        
        if features:
            self.architecture['features'] = features_arch

        self.training_parameters = {
            'sigma': self.sigma.params,
            'beta': self.beta,
            'gamma': self.gamma,
            'rho': self.rho,
            'rho_temp': self.rho_temp,
            'dictionary_variance': dictionary_variance,
            'learned_coder': learned_coder,
            'dictionary_min_dist': self.encoder.dictionary_dist_lb,
            'coder_capacity_regularization': coder_capacity_regularization,
            'force_cross_y': force_cross_y,
            'latent_sampling': latent_sampling,
            'set': None,
            'data_augmentation': [],
            'pretrained_features': pretrained_features,
            'pretrained_upsampler': pretrained_upsampler,
            'epochs': 0,
            'batch_size': None,
            'fine_tuning': [],}

        self.testing = {m: {'n':0, 'epochs':0, 'accuracy':0}
                        for m in self.predict_methods}
        # self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

        self.ood_results = {}

        self.optimizer = Optimizer(self.parameters(), **optimizer)
        self.training_parameters['optim'] = self.optimizer.params
            
        self.train_history = {'epochs': 0}

        self.latent_dim = latent_dim
        self.latent_sampling = latent_sampling
        self.latent_samplings = {'train': latent_sampling, 'eval': test_latent_sampling}
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.classifier_layer_sizes = classifier_layer_sizes
        self.upsampler_channels = upsampler_channels
        self.activation = activation
        self.output_activation = output_activation
            
        self.z_output = False

        self.eval()

    def train(self, *a, **k):
        state = 'train' if self.training else 'eval'
        super().train(*a, **k)
        new_state = 'train' if self.training else 'eval'
        logging.debug(f'Going from {state} to {new_state}')
        self.latent_sampling = self.latent_samplings[new_state]
        
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
                                          None if y is None else y.view(*batch_shape),
                                          x, **kw)

    def forward_from_features(self, x_features, y, x,
                              z_output=True, sampling_epsilon_norm_out=False, sigma_out=False):

        batch_shape = x_features.shape
        batch_size = batch_shape[:-len(self.encoder.input_shape)]  # N1 x...xNg
        reco_batch_shape = batch_size + self.input_shape
        
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
            x_output = self.imager(u)

        if self.is_cvae or self.is_vae:
            # y_output = self.classifier(z_mean.unsqueeze(0))  # for classification on the means
            y_output = self.classifier(z) # for classification on z
        else:
            y_output = self.classifier(z)
            
        # y_output of size LxN1x...xKgxC
        # print('**** y_out', y_output.shape)

        if self.is_vib:
            out = (x,)
        else:
            out = (x_output.reshape((self.latent_sampling + 1,) + reco_batch_shape),)

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
                 with_beta=False,
                 kl_var_weighting=1.,
                 mse_weighting=1.,
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

        cross_y_weight = False
        if self.y_is_decoded:
            if self.is_cvae or self.is_vae:
                cross_y_weight = self.gamma if self.training else False
            else: cross_y_weight = 1.
        
        if not batch:
            # print('*** training:', self.training)
            mode = 'training' if self.training else 'eval'
            logging.debug(f'Evaluating model in {mode} mode with batch size {x.shape[0]} '
                          'y {}in input'.format('' if y_in_input else '*not* '))
            pass

        C = self.num_labels
        
        if self.features:
            t = self.features(x)
        else:
            t = x

        t_shape = t.shape

        if len(t_shape) == len(self.input_shape):
            pass
            # t = t.unsqueeze(0)
            
        y_shape = x.shape[:-len(self.input_shape)]
        
        if x_repeated_along_classes:
            # build a C* N1* N2* Ng *D1 * Dt tensor of input x_features
            t = t.expand(C,  *t_shape)

        if y_is_built:
            # create a C * N1 * ... * Ng y tensor y[c,:,:,:...] = c
            y_shape_per_class = (1,) + y_shape
            y = torch.cat([c * torch.ones(y_shape_per_class,
                                          dtype=int,
                                          device=x.device)
                           for c in range(C)], dim=0)
            y_shape = y.shape

        y_in = y.view(y_shape) if self.y_is_coded else None
        
        if self.features:
            o = self.forward_from_features(t, y_in, x,
                                           sampling_epsilon_norm_out=True,
                                           sigma_out=True,
                                           **kw)
        else:
            o = self.forward(t, y_in, x,
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
            current_measures = {k: 0. for k in ('xpow', 'mse', 'snr',
                                                'imut-zy', 'd-mind',
                                                'ld-norm', 'var_kl',
                                                'zdist')}
            
        total_measures['sigma'] = self.sigma.value

        if self.x_is_generated:

            if self.sigma.coded:
                s_ = sigma_coded.view(-1, *self.sigma.output_dim)
                self.sigma.update(v=s_)
            else:
                s_ = self.sigma
            if self.sigma.is_rmse:
                sigma_ = 1.
                log_sigma = 0.
            else:
                sigma_ = s_.exp() if self.sigma.is_log else s_
                log_sigma = s_ if self.sigma.is_log else s_.log()

            # print('*** x', *x.shape, 'x_', *x_reco.shape, 's', *sigma_.shape) 
            weighted_mse_loss_sampling = 0.5 * mse_loss(x / sigma_,
                                                        x_reco[1:] / sigma_,
                                                        ndim=len(self.input_shape),
                                                        batch_mean=False)

            if self.sigma.is_rmse:
                sigma_ = weighted_mse_loss_sampling * 2
                log_sigma = sigma_.log()
                weighted_mse_loss_sampling = 0.5 * torch.ones_like(weighted_mse_loss_sampling)

            if not batch:
                pass
                # print('*** sigma', *self.sigma.shape, 'sigma_', sigma_.shape)  # 

            batch_quants['wmse'] = weighted_mse_loss_sampling.mean(0)
            batch_quants['mse'] = (batch_quants['wmse']).mean() * 2 * (sigma_ ** 2).mean()
            
            D = np.prod(self.input_shape)
            weighted_mse_remainder = D * weighted_mse_loss_sampling.min(0)[0]
            iws = (-D * weighted_mse_loss_sampling + weighted_mse_remainder).exp()
            if iws.isinf().sum(): logging.error('MSE INF')
            weighted_mse_remainder += D * (log_sigma.mean() + np.log(2 * np.pi) / 2)
            
            batch_quants['xpow'] = x.pow(2).mean().item()
            total_measures['xpow'] = (current_measures['xpow'] * batch 
                                      + batch_quants['xpow']) / (batch + 1)

            mse = batch_quants['mse'].mean().item()
            total_measures['mse'] = (current_measures['mse'] * batch
                                     + mse) / (batch + 1)

            total_measures['std'] = np.sqrt(total_measures['mse'])
            snr = total_measures['xpow'] / total_measures['mse']
            total_measures['snr'] = 10 * np.log10(snr)
            
        dictionary = self.encoder.latent_dictionary if self.coder_has_dict else None

        if not batch:
            logging.debug('warmup kl weight=%e', kl_var_weighting)

        kl_l, zdist, var_kl, sdist = kl_loss(mu, log_var,
                                             z=z[1:],
                                             y=y if self.coder_has_dict else None,
                                             prior_variance = self.latent_prior_variance,
                                             latent_dictionary=dictionary,
                                             var_weighting=kl_var_weighting,
                                             out=['kl', 'dist', 'var', 'sdist'],
                                             batch_mean=False)

        # print('*** wxjdjd ***', 'kl', *kl_l.shape, 'zd', *zdist.shape)
        
        total_measures['zdist'] = (current_measures['zdist'] * batch +
                                   zdist.mean().item()) / (batch + 1)

        total_measures['var_kl'] = (current_measures['var_kl'] * batch +
                                   zdist.mean().item()) / (batch + 1)
        
        batch_quants['latent_kl'] = kl_l

        batch_losses['zdist'] = zdist
        batch_losses['var_kl'] = var_kl

        if self.y_is_decoded:

            if y_is_built and not self.y_is_coded:
                y_in = None
            else:
                y_in = y
                
            batch_quants['cross_y'] = x_loss(y_in,
                                             y_est,
                                             batch_mean=False)


            # print('*** cvae:545 cross_y', *batch_quants['cross_y'].shape)

        batch_losses['total'] = torch.zeros_like(batch_quants['latent_kl'])

        if self.coder_has_dict:
            # batch_losses['zdist'] = 0
            dict_mean = dictionary.mean(0)
            zdist_to_mean = (mu - dict_mean).pow(2).sum(1)
            dict_norm_var = dictionary.pow(2).sum(1).mean(0) - dict_mean.pow(2).sum()
            batch_losses['dzdist'] = zdist_to_mean + dict_norm_var
            batch_quants['imut-zy'] = self.encoder.capacity()
            batch_quants['ld-norm'] = self.encoder.latent_dictionary.pow(2).mean()
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

            batch_logpx = -D * (log_sigma.mean() + np.log(2 * np.pi)
                                + batch_wmse)
                
            batch_losses['cross_x'] = - batch_logpx * mse_weighting

            batch_losses['total'] += batch_losses['cross_x'] 

            if sdist.dim() > iws.dim():
                if sdist.shape[1] == 1:
                    sdist = sdist.squeeze(1)
                else:
                    iws = iws.unsqueeze(1)

            sdist_remainder = sdist.min(0)[0] / 2
            p_z_y = (- sdist / 2 + sdist_remainder).exp()
            iws = iws * p_z_y
            if p_z_y.isinf().sum(): logging.error('P_Z_Y INF')
            
            log_inv_q_z_x = ((eps_norm + log_var.sum(-1)) / 2)
            
            if log_inv_q_z_x.dim() < iws.dim():
                log_inv_q_z_x = log_inv_q_z_x.unsqueeze(1)

            log_inv_q_remainder = log_inv_q_z_x.max(0)[0]
            inv_q = (log_inv_q_z_x - log_inv_q_remainder).exp()
            iws = iws * inv_q
            if inv_q.isinf().sum():
                logging.error('Q_Z_X INF')

            if log_inv_q_remainder.isinf().sum():
                logging.error('*** q_r is inf')
            if weighted_mse_remainder.isinf().sum():
                logging.error('*** mse_r is inf')
            if sdist_remainder.isinf().sum():
                logging.error('*** sd_r is inf')

            iws_ = (iws.mean(0) + 1e-40).log() + log_inv_q_remainder - sdist_remainder - weighted_mse_remainder

            if 'iws' in self.loss_components:
                batch_losses['iws'] = iws_
            # print('*** iws:', *iws.shape, 'eps', *eps_norm.shape)
            
        if self.y_is_decoded or self.force_cross_y:
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
                batch_losses['total'] = batch_losses['total'] + cross_y_weight * batch_losses['cross_y']
                
        batch_losses['kl'] = batch_quants['latent_kl']
        
        if self.is_vib:
            if not batch:
                logging.debug(f'KL coef={self.sigma}')
            # print('*** 612: T:', *batch_losses['total'].shape, 'kl', *batch_losses['kl'].shape)
            batch_losses['total'] += self.sigma * batch_losses['kl']
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
        
    
    def predict(self, x, method='mean', **kw):
        """x input of size (N1, .. ,Ng, D1, D2,..., Dt) 

        creates a x of size C * N1, ..., D1, ...., Dt)
        and a y of size C * N1 * ... * Ng

        - method: If 'mean'(default) output is of size N1 *...* and
        gives y predicted. If None output is C * N1 *... and gives
        p(y|x,y). If 'loss' returns the y which minimizes loss(x, y)

        """

        _, logits, batch_losses, measures = self.evaluate(x)

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
            if self.losses_might_be_computed_for_each_class:
                iws_max = iws.max(axis=0)[0]
                d_iws = iws - iws_max
            
        for m in methods:

            two_sided = m.endswith('-2s')
            if two_sided:
                m = m[:-3]
            if m == 'logpx':
                assert not self.losses_might_be_computed_for_each_class
                measures = logp

            elif m == 'iws':
                # print('*** iws:', 'iws' in losses, *losses.keys())
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
                measures = (losses['iws'] / T).softmax(0).max(axis=0)[0]
            elif m in ('soft', 'softkl'):
                # measures = logp.softmax(0).max(axis=0)[0]
                measures = (-losses['kl']).softmax(0).max(axis=0)[0]
            elif m.startswith('softkl-'):
                T = float(m[7:])
                measures = (- losses['kl'] / T).softmax(0).max(axis=0)[0]
            elif m == 'logits':
                measures = logits.max(axis=-1)[0]
            elif m == 'baseline':
                measures = logits.softmax(-1).max(axis=-1)[0]
            elif m == 't1000' and self.type in ('vib', 'jvae'):
                measures = (logits / 1000).softmax(-1).max(axis=-1)[0]
            elif m == 't1000' and self.type == 'cvae':
                # measures = (logp / 1000).softmax(0).max(axis=0)[0]
                measures = (-losses['kl'] / 1000).softmax(0).max(axis=0)[0]
            elif m == 'mag':
                measures = logp_max - logp.median(axis=0)[0]
            elif m == 'std':
                measures = logp.std(axis=0)
            elif m == 'mean':
                measures = d_logp.exp().mean(axis=0).log() + logp_max
            elif m == 'nstd':
                measures = (d_logp.exp().std(axis=0).log()
                            - d_logp.exp().mean(axis=0).log()).exp().pow(2)
            elif m == 'IYx':
                d_logp_x = d_logp.exp().mean(axis=0).log()
                
                measures =  ( (d_logp * (d_logp.exp())).sum(axis=0) / (C * d_logp_x.exp())
                            - d_logp_x )
            elif m == 'kl':
                measures = -losses['kl'].min(axis=0)[0]
            elif m == 'mse' and self.is_cvae:
                measures = -losses['cross_x']

            elif m.startswith('odin'):
                # print('odin losses:', *[_ for _ in losses if _.startswith('odin')])
                measures = losses[m]
            
            else:
                raise ValueError(f'{m} is an unknown ood method')

            dist_measures[m + ('-2s' if two_sided else '')] = measures.cpu() if to_cpu else measures

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

        x = torch.randn(batch_size, *self.input_shape, device=self.device)
        y = torch.ones(batch_size, dtype=int, device=self.device) if training else None

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
                logging.debug('Batch size of %s too much for %s.',
                              batch_size,
                              which)
                _s = str(e).split('\n')[0]
                logging.debug(_s)
                batch_size //= 2
                
    @property
    def max_batch_sizes(self):
        logging.debug('Calling max batch size')
        max_batch_sizes = self.training_parameters.get('max_batch_sizes', {})
        if max_batch_sizes:
            return max_batch_sizes
        self.compute_max_batch_size()
        return self.max_batch_sizes

    @max_batch_sizes.setter
    def max_batch_sizes(self, v):
        assert 'train' in v
        assert 'test' in v
        self.training_parameters['max_batch_sizes'] = v
    
    def accuracy(self, testset=None,
                 batch_size=100,
                 num_batch='all',
                 method='all',
                 print_result=False,
                 update_self_testing=True,
                 outputs=EpochOutput(),
                 sample_dirs=[],
                 recorder=None,
                 wygiwyu=False,
                 log=True):

        """return detection rate. 
        method can be a list of methods

        """

        MAX_SAMPLE_SAVE = 200
        
        device = next(self.parameters()).device
        
        if not testset:
            testset_name = self.training_parameters['set']
            _, testset = torchdl.get_dataset(testset_name)
        
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

        epoch = self.trained
        
        if num_batch == 'all':
            num_batch = len(testset) // batch_size
            shuffle = False

        if wygiwyu:
            from_r, _ = testing_plan(self, ood_sets=[], predict_methods=predict_methods, misclass_methods=[])
            if not from_r:
                acc = {m: self.testing[m]['accuracy'] for m in predict_methods}
                if only_one_method:
                    return acc[method]
                else:
                    return acc
            else:
                predict_methods = [m for m in from_r[testset_name] if from_r[testset_name][m]]
                rec_dir = os.path.join(self.saved_dir, 'samples', 'last')
                recorder = LossRecorder.loadall(rec_dir, testset_name)[testset_name]
                num_batch = len(recorder)
                batch_size = recorder.batch_size
                
        recorded = recorder is not None and len(recorder) >= num_batch 
        recording = recorder is not None and len(recorder) < num_batch 
        
        if recorded:
            logging.debug('Losses already recorded')
            num_batch = len(recorder)
            epoch = last_samples(self)

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
                                                 shuffle=shuffle)
        test_iterator = iter(testloader)
        start = time.time()

        total_loss = {k: 0. for k in self.loss_components}
        mean_loss = total_loss.copy()

        current_measures = {}
        measures = {}

        for i in range(num_batch):

            # save_batch_as_sample = sample_file and i < sample_save // batch_size            
            
            if not recorded:
                data = next(test_iterator)
                x_test, y_test = data[0].to(device), data[1].to(device)
                (x_, logits,
                 batch_losses, measures) = self.evaluate(x_test, batch=i,
                                                        current_measures=current_measures)
                
                # print('*** batch_losses:', *batch_losses)
                current_measures = measures
                self._measures = measures
            else:
                components = [k for k in recorder.keys() if k in self.loss_components]
                batch_losses = recorder.get_batch(i, *components)
                # logging.debug('TBD cvae:874: %s', ' '.join(self.loss_components))
                logits = recorder.get_batch(i, 'logits').T
                y_test = recorder.get_batch(i, 'y_true')

            y_pred = {}
            # print('*** predict methods:', *predict_methods)
            # logging.debug('TBD cvae:878: %s', ' '.join(batch_losses.keys()))
            for m in predict_methods:
                y_pred[m] = self.predict_after_evaluate(logits,
                                                        batch_losses,
                                                        method=m)

            if recording:
                recorder.append_batch(**batch_losses, y_true=y_test, logits=logits.T)
                
            # print('*** 842', y_test[0].item(), *y_test.shape)
            # print('*** 843', batch_losses['cross_y'].min(0)[0].mean())
            ind = y_test.unsqueeze(0)
            # print('*** 1156 y:', min(y_test), max(y_test)) 
            for k in batch_losses:
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
                    logging.warning(f'{k} shape has not been anticipated: {_s}')

                if not i:
                    logging.debug(f'Predicted shape for {k}: {shape}. Actual: {_s}') 
                try:
                    if shape == 'CxNxC':
                        batch_loss_y = batch_losses[k].max(-1)[0].gather(0, ind)
                    elif shape == 'CxN':
                        batch_loss_y = batch_losses[k].gather(0, ind)
                    else:
                        batch_loss_y = batch_losses[k]
                except RuntimeError:
                    logging.error(f'{k} shape has been wrongly anticipated: {shape} in lieu of {_s}')
                    total_loss[k] = 0.0
                    
                if k not in total_loss:
                    total_loss[k] = 0.0

                total_loss[k] += batch_loss_y.mean().item()
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
                                loss_components=self.loss_components,
                                losses=mean_loss,
                                acc_methods=predict_methods,
                                accuracies=acc,
                                metrics=self.metrics,
                                measures=self._measures,
                                time_per_i=time_per_i,
                                batch_size=batch_size,
                                preambule=print_result)
        self.test_loss = mean_loss

        if recording:
            logging.debug('Saving examples in' + ', '.join(sample_dirs))

            saved_dict = {
                'losses': {m: batch_losses[m][:MAX_SAMPLE_SAVE] for m in batch_losses},
                'measures': measures,
                'x': x_test[:MAX_SAMPLE_SAVE],
                'y': y_test[:MAX_SAMPLE_SAVE],
                'x_': x_[:MAX_SAMPLE_SAVE] if self.is_vib else x_.mean(0)[:MAX_SAMPLE_SAVE],
                'y_pred': {m: y_pred[m][:MAX_SAMPLE_SAVE] for m in y_pred},
                }
            if self.is_xvae or self.is_cvae:
                mu_y = self.encoder.latent_dictionary.index_select(0, y_test)
                saved_dict['mu_y'] = mu_y[:MAX_SAMPLE_SAVE]

            for d in sample_dirs:
                f = os.path.join(d, f'sample-{testset.name}.pth')
                torch.save(saved_dict, f)

                f = os.path.join(d, f'record-{testset.name}.pth')
                recorder.save(f)

        for m in predict_methods:

            update_self_testing_method = (update_self_testing and
                                          (epoch > self.testing[m]['epochs']
                                           or
                                           n > self.testing[m]['n']))
            if update_self_testing_method:
                if log:
                    logged = 'Updating accuracy %.3f%% for method %s (n=%s)'
                    logging.debug(logged,
                                  100 * acc[m],
                                  m, n)
                                  # self.testing[m]['n'],
                                  # self.testing[m]['epochs'],
                                  # n, self.trained)

                self.testing[m] = {'n': n,
                                   'epochs': epoch,
                                   'sampling': self.latent_samplings['eval'],
                                   'accuracy': acc[m]}

            elif log:

                logging.debug(f'Accuracies not updated')

        return acc[m] if only_one_method else acc

    def ood_detection_rates(self, oodsets=None,
                            testset=None,
                            batch_size=100,
                            num_batch='all',
                            method='all',
                            print_result=False,
                            update_self_ood=True,
                            updated_epoch=None,
                            outputs=EpochOutput(),
                            recorders=None,
                            wygiwyu=False,
                            sample_dirs=[],
                            log=True):

        if updated_epoch is None:
            updated_epoch = self.trained
        
        if not testset:
            testset_name = self.training_parameters['set']
            transformer = self.training_parameters['transformer']
            _, testset = torchdl.get_dataset(testset_name, transformer=transformer)

        if not method:
            return

        odin_parameters = [_ for _ in self.ood_methods if _.startswith('odin')]

        ood_methods = make_list(method, self.ood_methods)
        
        if oodsets is None:
            # print('*** 1291', *testset.same_size)
            oodsets = [torchdl.get_dataset(n, transformer=testset.transformer)[1]
                       for n in testset.same_size]
            logging.debug('Oodsets loaded: ' + ' ; '.join(s.name for s in oodsets))

        all_set_names = [testset.name] + [o.name for o in oodsets] 

        ood_methods_per_set = {s: ood_methods for s in all_set_names}
        all_ood_methods = ood_methods
        
        if not recorders:
            recorders = {n: None for n in all_set_names}

        max_num_batch = num_batch
        num_batch = {testset.name: max(len(testset) // batch_size, 1)}
        for o in oodsets:
            num_batch[o.name] = max(len(o) // batch_size, 1)

        shuffle = {s: False for s in all_set_names}
        recording = {}
        recorded = {}

        if wygiwyu:

            from_r, _ = testing_plan(self, ood_sets=[o.name for o in oodsets], ood_methods=ood_methods)
            if from_r:
                rec_dir = os.path.join(self.saved_dir, 'samples', 'last')
                recorders = LossRecorder.loadall(rec_dir, *all_set_names)
                num_batch = {s: len(recorders[s]) for s in recorders}
                batch_size = recorders[testset.name].batch_size
                ood_methods_per_set = {s: [m for m in ood_methods if from_r[s].get(m, {})] for s in from_r}
                for s in from_r:  #
                    if s in [o.name for o in oodsets]:
                        logging.debug('OOD methods for {}: '.format(s) + 
                                      '-'.join(ood_methods_per_set[s]))
                all_ood_methods = [m for m in ood_methods if any([m in from_r[s] for s in from_r])]
                # print('*** ood methods', *ood_methods)
                ood_methods_per_set[testset.name] = [m for m in ood_methods if m in all_ood_methods]
                
            oodsets = [o for o in oodsets if o.name in recorders]
            all_set_names = [s for s in all_set_names if s in recorders]
                
        for s in all_set_names:
            if type(max_num_batch) is int:
                num_batch[s] = min(num_batch[s], max_num_batch)
                shuffle[s] = True
            recording[s] = recorders[s] is not None and len(recorders[s]) < num_batch[s]
            recorded[s] = recorders[s] is not None and len(recorders[s]) >= num_batch[s]
            if recorded[s]:
                logging.debug('Losses already computed for %s %s', s, recorders[s])
            if recording[s]:
                recorders[s].reset()
                recorders[s].num_batch = num_batch[s]
                logging.debug('Recording session for %s %s', s, recorders[s])
                
        device = next(self.parameters()).device

        if oodsets:
            outputs.results(0, 0, -2, 0,
                            metrics=all_ood_methods,
                            acc_methods=all_ood_methods)
            outputs.results(0, 0, -1, 0, metrics=all_ood_methods,
                            acc_methods=all_ood_methods)

        if oodsets:

            logging.debug(f'Computing measures for set {testset.name}')
            ind_measures = {m: np.ndarray(0)
                            for m in ood_methods}

            s = testset.name
            if recorders[s] is not None:
                recorders[s].init_seed_for_dataloader()

            loader = torch.utils.data.DataLoader(testset,
                                                 shuffle=shuffle[s],
                                                 num_workers=0,
                                                 batch_size=batch_size)
            
            t_0 = time.time()

            test_iterator = iter(loader)
            for i in range(num_batch[s]):

                # print('*** 1378', i, num_batch[s])
                if not recorded[s]:

                    data = next(test_iterator)
                    x = data[0].to(device)
                    y = data[1].to(device)
                    if odin_parameters:
                        x.requires_grad_(True)
                    with torch.no_grad():
                        _, logits, losses, _ = self.evaluate(x, batch=i)

                    odin_softmax = {}
                    if odin_parameters:
                        for T in self.ODIN_TEMPS:
                            with torch.enable_grad():
                                _, no_temp_logits = self.forward(x, z_output=False)
                                softmax = (no_temp_logits[1:].mean(0) / T).softmax(-1).max(-1)[0]
                                X = softmax.sum()
                            # print('***', X.requires_grad, (X / batch_size).cpu().item())
                            X.backward()
                            dx = x.grad.sign()
                            for eps in self.ODIN_EPS:
                                _, odin_logits = self.forward(x + eps * dx, z_output=False)
                                out_probs = (odin_logits[1:].mean(0) / T).softmax(-1).max(-1)[0]
                                odin_softmax['odin-{:.0f}-{:.4f}'.format(T, eps)] = out_probs
                else:
                    components = [k for k in recorders[s].keys() if k in self.loss_components or k.startswith('odin')]
                    losses = recorders[s].get_batch(i, *components)
                    logits = recorders[s].get_batch(i, 'logits').T
                    odin_softmax = {}
                    
                if recording[s]:
                    recorders[s].append_batch(**losses, **odin_softmax, y_true=y, logits=logits.T)
                    
                measures = self.batch_dist_measures(logits, dict(**losses, **odin_softmax),
                                                    ood_methods_per_set[s])

                for m in ood_methods_per_set[s]:
                    # print('*** ood', m, *measures[m].shape)
                    
                    ind_measures[m] = np.concatenate([ind_measures[m],
                                                      measures[m].cpu()])
                t_i = time.time() - t_0
                t_per_i = t_i / (i + 1)
                                   
                outputs.results(i, num_batch[s], 0, 1,
                                metrics=ood_methods_per_set[s],
                                measures={m: ind_measures[m].mean()
                                          for m in ood_methods_per_set[s]},
                                acc_methods=ood_methods_per_set[s],
                                time_per_i=t_per_i,
                                batch_size=batch_size,
                                preambule=testset.name)

            if recording[s]:
                for d in sample_dirs:
                    f = os.path.join(d, f'record-{s}.pth')
        
                    recorders[s].save(f.format(s=s))
                
        kept_tpr = [pc / 100 for pc in range(90, 100)]
        no_result = {'epochs': 0,
                     'n': 0,
                     'auc': 0,
                     'tpr': kept_tpr,
                     'fpr': [1 for _ in kept_tpr],
                     'thresholds': [None for _ in kept_tpr]}
                     
        for oodset in oodsets:

            s = oodset.name
            ood_n_batch = num_batch[s]
            
            ood_results = {m: copy.deepcopy(no_result) for m in ood_methods_per_set[s]}
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
                                                 shuffle=shuffle[s],
                                                 batch_size=batch_size)

            logging.debug(f'Computing measures for set {oodset.name} with {ood_n_batch} batches')

            t_0 = time.time()
            test_iterator = iter(loader)
            
            for i in range(ood_n_batch):

                if not recorded[s]:
                    data = next(test_iterator)
                    x = data[0].to(device)
                    y = data[1].to(device)
                    if odin_parameters:
                        x.requires_grad_(True)

                    with torch.no_grad():
                        _, logits, losses, _ = self.evaluate(x, batch=i)

                    odin_softmax = {}
                    if odin_parameters:
                        for T in self.ODIN_TEMPS:
                            with torch.enable_grad():
                                _, no_temp_logits = self.forward(x, z_output=False)
                                softmax = (no_temp_logits[1:].mean(0) / T).softmax(-1).max(-1)[0]
                                X = softmax.sum()
                            # print('***', X.requires_grad, (X / batch_size).cpu().item())
                            X.backward()
                            dx = x.grad.sign()
                            for eps in self.ODIN_EPS:
                                _, odin_logits = self.forward(x + eps * dx, z_output=False)
                                out_probs = (odin_logits[1:].mean(0) / T).softmax(-1).max(-1)[0]
                                odin_softmax['odin-{:.0f}-{:.4f}'.format(T, eps)] = out_probs
                        
                else:
                    components = [k for k in recorders[s].keys() if k in self.loss_components or k.startswith('odin')]
                    losses = recorders[s].get_batch(i, *components)
                    logits = recorders[s].get_batch(i, 'logits').T
                    
                if recording[s]:
                    recorders[s].append_batch(**losses, **odin_softmax, y_true=y, logits=logits.T)

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
                for m in ood_methods_per_set[s]:
                    logging.debug(f'Computing roc curves for with metrics {m}')
                    _debug = 'medium' if i == ood_n_batch - 1 else 'soft'
                    auc_[m], fpr_[m], tpr_[m], thresholds_[m] = roc_curve(ind_measures[m], ood_measures[m],
                                                                          *kept_tpr,
                                                                          debug=_debug,
                                                                          two_sided=m.endswith('-2s'))
                    r_[m] = fpr_at_tpr(fpr_[m],
                                       tpr_[m],
                                       0.95,
                                       thresholds_[m])

                # print('*** cvae:1522 *** {} {:.0f}ms / method'.format(s, 1e3 * (time.time() - t0) / len(r_)))

                outputs.results(i, ood_n_batch, 0, 1,
                                metrics=ood_methods_per_set[oodset.name],
                                measures=meaned_measures,
                                acc_methods=ood_methods_per_set[oodset.name],
                                accuracies=r_,
                                time_per_i=t_per_i,
                                batch_size=batch_size,
                                preambule=oodset.name)

            for m in ood_methods_per_set[s]:

                ood_results[m] = {'epochs': updated_epoch,
                                  'n': ood_n_batch * batch_size,
                                  'auc': auc_[m],
                                  'tpr': kept_tpr,
                                  'fpr': list(fpr_[m]), 
                                  'thresholds': list(thresholds_[m])}
                
                if update_self_ood:
                    if oodset.name not in self.ood_results:
                        self.ood_results[oodset.name] = {}
                    self.ood_results[oodset.name][m] = ood_results[m]

            if recording[s]:
                for d in sample_dirs:
                    f = os.path.join(d, f'record-{s}.pth')
        
                    recorders[s].save(f.format(s=s))

    def misclassification_detection_rate(self,
                                         recorder=None,
                                         wygiswyu=True,
                                         sample_dir=None,
                                         predict_methods='all',
                                         misclass_methods='all',
                                         shown_tpr=0.95,
                                         outputs=EpochOutput):

        if not wygiswyu:
            logging.error('You have to compute accuracies '
                          'before computing misclassification rates')
            return

        from_r = testing_plan(self, ood_sets=[],
                              predict_methods=predict_methods,
                              misclass_methods=misclass_methods)
        
        if not sample_dir:
            sample_dir = os.path.join(self.saved_dir, 'samples', 'last')

        testset = self.training_parameters['set']
        
        if not recorder:
            f = f'record-{testset}.pth'
            recorder = LossRecorder.load(os.path.join(sample_dir, f))
        
        methods = {'predict': predict_methods, 'miss': misclass_methods}

        for which, all_methods in zip(('predict', 'miss'),
                                      (self.predict_methods, self.misclass_methods)):

            methods[which] = make_list(methods[which],
                                       develop_starred_methods(all_methods, self.methods_params))
            
            # print('*** methods for', which, ':', methods[which], '(', *all_methods, ')')

            for m in methods[which]:
                # print('*** |_', m)
                assert m in all_methods

        losses = recorder._tensors
        logits = losses.pop('logits').T
        y = losses.pop('y_true')

        kept_tpr = [pc / 100 for pc in range(90, 100)]

        _p = 5.2
        _p_1 = 4.1
        
        for predict_method in methods['predict']:
            
            y_ = self.predict_after_evaluate(logits, losses, method=predict_method)
            missed = y_ != y
            correct = y_ == y

            acc = correct.sum().item() / (correct.sum().item() + missed.sum().item())
                   
            test_measures = self.batch_dist_measures(logits, losses, methods['miss'], to_cpu=True)

            fpr_, tpr_, precision_, recall_, thresholds_ = {}, {}, {}, {}, {}

            logging.debug(f'*** {predict_method} ({100 * acc:{_p}f})')

            max_P = 0
            
            for m in methods['miss']:
                
                auc, fpr, tpr, thr = roc_curve(test_measures[m][correct],
                                               test_measures[m][missed], *kept_tpr)

                tp = [(test_measures[m][correct] > t).sum().item() for t in thr]
                tn =  [(test_measures[m][missed] <= t).sum().item() for t in thr]
                fp =  [(test_measures[m][missed] > t).sum().item() for t in thr]
                fn = [(test_measures[m][correct] <= t).sum().item() for t in thr]

                t95 = fpr_at_tpr(fpr, tpr, shown_tpr, thr, return_threshold=True)[1]
                
                tp95 = (test_measures[m][correct] > t95).sum().item()
                fp95 = (test_measures[m][missed] > t95).sum() .item()
                
                p95 = tp95 / (tp95 + fp95)

                dp95 = p95 - acc
                
                r95 = tp95 / correct.sum().item()
                fpr95 = fp95 / missed.sum().item()
                tpr95 = p95
                
                precision_[m] = [(t / (t + f)) for t, f in zip(tp, fp)]
                recall_[m] = [t / correct.sum().item() for t in tp]

                if p95 > max_P:
                    best_m = m
                    max_P = p95
                logging.debug('{:16}: '.format(m) +
                              '\tP={:{p}f} '.format(100 * p95, p=_p) +
                              '({:+{p}f}) '.format(100 * dp95, p = _p_1) +
                              'R={:{p}f} FPR={:{p}f}'.format(100 * r95, 100 * fpr95, p=_p))
            # print('*** {}: {:{p}f} ({:+{p1}f})'.format(best_m, 100 * max_P, 100 * (max_P -acc), p=_p, p1=_p-1.1))
                
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
                    warmup=0,
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
            except(AttributeError):
                set_name = trainset.__str__().splitlines()[0].split()[-1].lower()
            transformer=trainset.transformer
            
        if self.trained:
            logging.info(f'Network partially trained ({self.trained} epochs)')
            logging.debug('Ignoring parameters (except for epochs)')

        else:
            if trainset:
                self.training_parameters['set'] = set_name
                self.training_parameters['transformer'] = transformer
                self.training_parameters['validation'] = validation
                ss = trainset.data[0].shape
                ns = self.input_shape
                logging.debug(f'Shapes : {ss} / {ns}')
                # assert ns == ss or ss == ns[1:]
        
            if batch_size:
                self.training_parameters['batch_size'] = batch_size

            if latent_sampling:
                self.latent_samplings['train'] = latent_sampling
                self.training_parameters['latent_sampling'] = latent_sampling

            if data_augmentation:
                self.training_parameters['data_augmentation'] = data_augmentation
        
        assert self.training_parameters['set']

        set_name = self.training_parameters['set']
        data_augmentation = self.training_parameters['data_augmentation']
        
        logging.debug(f'Getting {set_name}')
        trainset, testset = torchdl.get_dataset(set_name,
                                                transformer=transformer,
                                                data_augmentation=data_augmentation)


        if self.training_parameters.get('validation_split_seed ') is None:
            np.random.seed()
            self.training_parameters['validation_split_seed'] = np.random.randint(0, 2 ** 12)

        torch.random.manual_seed(self.training_parameters['validation_split_seed'])
        validation = self.training_parameters.get('validation', 0)
        validationset, trainset = torch.utils.data.random_split(trainset,
                                                                (validation,
                                                                 len(trainset) - validation))
        torch.seed()
        
        logging.debug('Choosing device')
        device = choose_device(device)
        logging.debug(f'done {device}')

        if optimizer is None:
            optimizer = self.optimizer
        
        max_batch_sizes = self.max_batch_sizes

        test_batch_size = min(max_batch_sizes['test'], test_batch_size)
        
        if batch_size:
            train_batch_size = min(batch_size, max_batch_sizes['train'])
        else:
            train_batch_size = max_batch_sizes['train']

        warmup = max(warmup, self.training_parameters.get('warmup', 0))
        self.training_parameters['warmup'] = warmup
            
        x_fake = torch.randn(test_batch_size, *self.input_shape, device=self.device)
        y_fake = torch.randint(0, 1, size=(test_batch_size,), device=self.device)
        
        _, logits, losses, measures = self.evaluate(x_fake)
        
        sets = [set_name, 'validation']
        for s in oodsets:
            sets.append(s.name)

        develop_starred_methods(self.ood_methods, self.methods_params)
                
        odin_parameters = [_ for _ in self.ood_methods if _.startswith('odin')]

        fake_odin_softmax = {o: torch.zeros(test_batch_size) for o in odin_parameters}

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

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=train_batch_size,
                                                  # pin_memory=True,
                                                  shuffle=True,
                                                  num_workers=0)

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
            self.train_history['train_accuracy'] = []
            self.train_history['train_loss'] = []
            self.train_history['train_measures'] = []
            self.train_history['test_accuracy'] = []
            self.train_history['test_measures'] = []
            self.train_history['test_loss'] = []
            self.train_history['validation_accuracy'] = []
            self.train_history['validation_measures'] = []
            self.train_history['validation_loss'] = []
            self.train_history['lr'] = []
            
        if not acc_methods:
            acc_methods = self.predict_methods

        if oodsets:
            ood_methods = self.ood_methods
        
        outputs.results(0, 0, -2, epochs,
                        metrics=self.metrics,
                        loss_components=self.loss_components,
                        acc_methods=acc_methods)
        outputs.results(0, 0, -1, epochs,
                        metrics=self.metrics,
                        loss_components=self.loss_components,
                        acc_methods=acc_methods)

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
        for epoch in range(done_epochs, epochs):

            for s in recorders:
                recorders[s].reset()
                
            logging.debug(f'Starting epoch {epoch} / {epochs}')
            t_start_epoch = time.time()
            # test

            full_test = ((epoch - done_epochs) and
                         epoch % full_test_every == 0)
            ood_detection = ((epoch - done_epochs) and
                             epoch % ood_detection_every == 0)
            if (full_test or not epoch or ood_detection) and save_dir:
                sample_dirs = [os.path.join(save_dir, 'samples', d)
                               for d in ('last', f'{epoch:04d}')]

                for d in sample_dirs:
                    if not os.path.exists(d):
                        os.makedirs(d)
            else:
                sample_dirs = []

            with torch.no_grad():

                if oodsets and ood_detection:

                    self.ood_detection_rates(oodsets=oodsets, testset=testset,
                                             batch_size=test_batch_size,
                                             num_batch=len(testset) // test_batch_size,
                                             outputs=outputs,
                                             recorders=recorders,
                                             sample_dirs=sample_dirs,
                                             print_result='*')

                    outputs.results(0, 0, -2, epochs,
                                    metrics=self.metrics,
                                    loss_components=self.loss_components,
                                    acc_methods=acc_methods)
                    outputs.results(0, 0, -1, epochs,
                                    metrics=self.metrics,
                                    loss_components=self.loss_components,
                                    acc_methods=acc_methods)

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
                    test_loss = self.test_loss
                    test_measures = self._measures.copy()

                num_batch = 'all' if full_test else max(1, validation_sample_size // test_batch_size) 
                validation_accuracy =self.accuracy(validationset,
                                                   batch_size=test_batch_size,
                                                   num_batch=num_batch,
                                                   device=device,
                                                   method=acc_methods,
                                                   # log=False,
                                                   outputs=outputs,
                                                   sample_dirs=sample_dirs,
                                                   update_self_testing=False,
                                                   recorder=recorders['validation'],
                                                   print_result='VALID'
                                                   if full_test else
                                                   'test')
                validation_loss = self.test_loss
                validation_measures = self._measures.copy()
                
                if signal_handler.sig > 1:
                    logging.warning(f'Breaking training loop bc of {signal_handler}')
                    break
                if save_dir: self.save(save_dir)
            # train
            if train_accuracy:
                with torch.no_grad():
                    train_accuracy = self.accuracy(trainset,
                                                   batch_size=test_batch_size,
                                                   num_batch=valid_num_batch,
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
                               
            if signal_handler.sig > 1:
                logging.warning(f'Breaking training loop bc of {signal_handler}')
                break

            if save_dir:
                self.save(save_dir)

            current_measures = {}
            
            self.train()

            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                x, y = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                if self.training:
                    warmup_weighting = min(1., (epoch + 1) / (warmup + 1))
                else:
                    warmup_weighting = 1.

                # with autograd.detect_anomaly():
                # forward + backward + optimize
                (_, y_est,
                 batch_losses, measures) = self.evaluate(x, y,
                                                         batch=i,
                                                         with_beta=True,
                                                         kl_var_weighting=warmup_weighting ** 2,
                                                         # mse_weighting=warmup_weighting,
                                                         current_measures=current_measures)

                current_measures = measures
                batch_loss = batch_losses['total'].mean()

                L = batch_loss
                if self.coder_capacity_regularization and self.encoder.dictionary_dist_lb:
                    L += self.encoder.dist_barrier()

                if self.force_cross_y and not self.y_is_decoded:
                    L += self.force_cross_y * batch_losses['cross_y'].mean()

                if self.rho:
                    dict_var = self.encoder.latent_dictionary.pow(2).mean()
                    
                    log10 = np.log(10)
                    r_ = self.rho * torch.exp(-dict_var / self.rho_temp * log10)

                    L += r_ * (batch_losses['zdist'] - batch_losses['dzdist']).mean()
                    if not i:
                        logging.debug('rho_=%e', r_.item())
                    
                for p in self.parameters():
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        print('GRAD NAN')

                L.backward()
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
                                acc_methods=acc_methods,
                                loss_components=self.loss_components,
                                losses=train_mean_loss,
                                metrics=self.metrics,
                                measures=measures,
                                time_per_i=t_per_i,
                                batch_size=train_batch_size,
                                end_of_epoch='\n')
            
            self.eval()
            train_measures = measures.copy()
            if testset:
                self.train_history['test_accuracy'].append(test_accuracy)
                self.train_history['test_measures'].append(test_measures)
                self.train_history['test_loss'].append(test_loss)
            if train_accuracy:
                self.train_history['train_accuracy'].append(train_accuracy)
            self.train_history['train_loss'].append(train_mean_loss)
            self.train_history['train_measures'].append(train_measures)
            self.train_history['epochs'] += 1
            self.train_history['lr'].append(self.optimizer.lr)
            self.trained += 1
            if fine_tuning:
                self.training_parameters['fine_tuning'].append(epoch)

            optimizer.update_lr()

            
            if signal_handler.sig > 1:
                logging.warning(f'Breaking training loop bc of {signal_handler}')
                break

            if save_dir:
                self.save(save_dir)
    
            if full_test and signal_handler.sig:
                logging.warning(f'Gently stopping training loop bc of {signal_handler}'
                                'after {epoch} epochs')
                break
            
        for s in recorders:
            recorders[s].reset()
        if save_dir:
            sample_dirs = [os.path.join(save_dir, 'samples', d)
                           for d in ('last', f'{epoch + 1:04d}')]                

            for d in sample_dirs:
                if not os.path.exists(d):
                    os.makedirs(d)

        if oodsets and not signal_handler.sig > 1:
                            
            self.ood_detection_rates(oodsets=oodsets, testset=testset,
                                     batch_size=test_batch_size,
                                     num_batch=len(testset) // test_batch_size,
                                     outputs=outputs,
                                     recorders=recorders,
                                     sample_dirs=sample_dirs,
                                     print_result='*')

            outputs.results(0, 0, -2, epochs,
                            metrics=self.metrics,
                            loss_components=self.loss_components,
                            acc_methods=acc_methods)
            outputs.results(0, 0, -1, epochs,
                            metrics=self.metrics,
                            loss_components=self.loss_components,
                            acc_methods=acc_methods)

        if testset and not signal_handler.sig > 1:

            recorder = recorders[set_name]
            # print(num_batch, sample_size)
            with torch.no_grad():
                test_accuracy = self.accuracy(testset,
                                              batch_size=test_batch_size,
                                              # num_batch=num_batch,
                                              # device=device,
                                              method=acc_methods,
                                              recorder=recorder,
                                              sample_dirs=sample_dirs,
                                              # log=False,
                                              outputs=outputs,
                                              print_result='TEST')

        if signal_handler.sig > 1:
            logging.warning(f'Skipping saving because of {signal_handler}')
        elif save_dir:
            self.save(save_dir)

        logging.debug('Finished training')

    def summary(self):

        logging.warning('SUMMARY FUNCTION NOT IMPLEMENTED')

    @property
    def device(self):
        return next(self.parameters()).device

    @device.setter
    def device(self, d):
        self.to(d)

    def to(self, d):
        super().to(d)
        self.optimizer.to(d)
        

    @property
    def latent_sampling(self):
        return self._latent_sampling

    @latent_sampling.setter
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
        
    def save(self, dir_name=None):
        """Save the params in params.json file in the directroy dir_name and, if
        trained, the weights inweights.h5.

        """
        
        if dir_name is None:
            dir_name = os.path.join('jobs', self.print_architecture,
                                    str(self.job_number))

        save_load.save_json(self.architecture, dir_name, 'params.json')
        save_load.save_json(self.architecture, dir_name, 'architecture.json')
        save_load.save_json(self.training_parameters, dir_name, 'train.json')
        save_load.save_json(self.testing, dir_name, 'test.json')
        save_load.save_json(self.ood_results, dir_name, 'ood.json')
        save_load.save_json(self.train_history, dir_name, 'history.json')
        
        if self.trained:
            w_p = save_load.get_path(dir_name, 'state.pth')
            torch.save(self.state_dict(), w_p)
            w_p = save_load.get_path(dir_name, 'optimizer.pth')
            torch.save(self.optimizer.state_dict(), w_p)
                        
    @classmethod
    def load(cls, dir_name,
             load_net=True,
             load_state=True,
             load_train=True,
             load_test=True,):

        """dir_name : where params.json is (and weigths.h5 if applicable)

        """

        if not load_net:
            load_state = False
            
        # default
        default_params = {'type': 'jvae',
                          'batch_norm': False,
                          'encoder_forced_variance': False,
                          'test_latent_sampling': 0,
                          'latent_prior_variance': 1.,
        }
        
        train_params = {'pretrained_features': None,
                        'pretrained_upsampler': None,
                        'learned_coder': False,
                        'beta': 1.,
                        'warmup': 0,
                        'gamma': 0.,
                        'rho': 0.,
                        'rho_temp':np.inf,
                        'dictionary_min_dist': None,
                        'dictionary_variance': 1,
                        'data_augmentation': [],
                        'fine_tuning': [],
                        'optim': {},
                        'force_cross_y': 0.
        }

        loaded_params = save_load.load_json(dir_name, 'params.json')

        try:
            s = dir_name.split(os.sep)[-1]
            job_number_by_dir_name = int(s)
        except ValueError:
            job_number_by_dir_name = s
            
        job_number = loaded_params.get('job_number', job_number_by_dir_name)

        resumed_file = os.path.join(dir_name, 'RESUMED')
        is_resumed = os.path.exists(resumed_file)

        if is_resumed:
            with open(resumed_file, 'r') as resumed_f:
                is_resumed = resumed_f.read()
                try:
                    is_resumed = int(is_resumed)
                except ValueError:
                    pass
                
        logging.debug('Parameters loaded')
        if loaded_params.get('batch_norm', False) == True:
            loaded_params['batch_norm'] = 'encoder'

        params = default_params.copy()
        params.update(loaded_params)
        
        loaded_train = False
        try:
            train_params.update(save_load.load_json(dir_name, 'train.json'))
            logging.debug('Training parameters loaded')
            loaded_train = True
        except(FileNotFoundError):
            pass
        
        loaded_test = False
        try:
            testing = save_load.load_json(dir_name, 'test.json')
            loaded_test = load_test
            logging.debug('Results loaded')
        except(FileNotFoundError):
            pass

        loaded_ood = False
        try:
            ood_results = save_load.load_json(dir_name, 'ood.json')
            loaded_ood = True
        except(FileNotFoundError):
            ood_results = {}
            pass
        
        try:
            train_history = save_load.load_json(dir_name, 'history.json')
        except(FileNotFoundError, IndexError):
            train_history = {'epochs': 0}

        if not params.get('features', None):
            params['features'] = {}

        resave_arch = False
        if not load_net:
            vae = save_load.Shell()
            try:
                vae.architecture = default_params.copy()
                vae.architecture.update(save_load.load_json(dir_name, 'architecture.json'))
                vae.type = vae.architecture['type']
                vae.loss_components = cls.loss_components_per_type[vae.type]
                logging.debug('Ghost network loaded')
                vae.job_number = job_number
                vae.ood_methods = cls.ood_methods_per_type[vae.type]
                vae.methods_params = cls.methods_params
                vae.predict_methods = cls.predict_methods_per_type[vae.type]
                vae.misclass_methods = cls.misclass_methods_per_type[vae.type]
                classifier_layer_sizes = params['classifier']
                gamma = train_params['gamma']
                vae.y_is_decoded = True
                if vae.type in ('cvae', 'vae'):
                    vae.y_is_decoded = gamma

                if vae.y_is_decoded and 'esty' not in vae.predict_methods:
                    vae.predict_methods =  vae.predict_methods + ['esty']

                if vae.y_is_decoded and 'cross_y' not in vae.loss_components:
                    vae.loss_components += ('cross_y',)

                vae.testing = {}
                # print('*** sigma loaded from', train_params['sigma'], vae.job_number)
                if isinstance(train_params['sigma'], dict):
                    vae.sigma = Sigma(**train_params['sigma'])
                else:
                    vae.sigma = Sigma(train_params['sigma'])
            except FileNotFoundError as e:
                logging.debug(f'File {e.filename} not found, it will be created')
                resave_arch = True
                load_net = True 
        if load_net:
            logging.debug('Building the network')
            vae = cls(input_shape=params['input'],
                      num_labels=params['labels'],
                      type_of_net=params['type'],
                      job_number=job_number,
                      encoder_layer_sizes=params['encoder'],
                      latent_dim=params['latent_dim'],
                      decoder_layer_sizes=params['decoder'],
                      classifier_layer_sizes=params['classifier'],
                      latent_sampling=train_params['latent_sampling'],
                      test_latent_sampling=params['test_latent_sampling'],
                      batch_norm=params['batch_norm'],
                      activation=params['activation'],
                      sigma=train_params['sigma'],
                      beta=train_params['beta'],
                      gamma=train_params['gamma'],
                      rho=train_params['rho'],
                      rho_temp=train_params['rho_temp'],
                      dictionary_variance=train_params['dictionary_variance'],
                      learned_coder=train_params['learned_coder'],
                      dictionary_min_dist=train_params['dictionary_min_dist'],
                      init_coder=False,
                      optimizer=train_params['optim'],
                      upsampler_channels=params['upsampler'],
                      output_activation=params['output'],
                      pretrained_features=train_params['pretrained_features'],
                      pretrained_upsampler=train_params['pretrained_upsampler'],
                      shadow=not load_net,
                      **params['features'])

            logging.debug('Built')
            if resave_arch:
                save_load.save_json(vae.architecture, dir_name, 'architecture.json')
                logging.debug('Architecture file saved')
        
        vae.saved_dir = dir_name
        vae.trained = train_history['epochs']
        vae.train_history = train_history
        vae.is_resumed = is_resumed
        vae.training_parameters = train_params
        if loaded_test:
            vae.testing.update(testing)

        if load_test:
            vae.ood_results = ood_results
            
        if load_state and vae.trained:
            w_p = save_load.get_path(dir_name, 'state.pth')
            try:
                state_dict = torch.load(w_p)
                _sigma = state_dict.pop('_sigma', None)
                if _sigma:
                    # print(f'cvae:1735: sigma: {vae.sigma.shape}, _sigma:{_sigma.shape}')
                    state_dict['sigma'] = _sigma.reshape(1)
            except RuntimeError as e:
                raise e
            try:
                vae.load_state_dict(state_dict)
            except RuntimeError as e:
                state_dict_vae = vae.state_dict()
                s = ''
                for (state, other) in [(state_dict, state_dict_vae),
                                       (state_dict_vae, state_dict)]:
                    for k, t in state.items():
                        s_ = f'{k}: {tuple(t.shape)}'
                        t_ = other.get(k, torch.Tensor([]))
                        s += f'{s_:40} === {tuple(t_.shape)}'
                        s+='\n'
                        s+='\n'*4    
                        logging.debug(f'DUMPED\n{dir_name}\n{e}\n\n{s}\n{vae}')
                raise e
            w_p = save_load.get_path(dir_name, 'optimizer.pth')
            try:
                state_dict = torch.load(w_p)
                vae.optimizer.load_state_dict(state_dict)

            except FileNotFoundError:
                logging.warning('Optimizer state file not found') 
            vae.optimizer.update_scheduler_from_epoch(vae.trained)
                
            logging.debug('Loaded')
        return vae

    def log_pxy(self, x, normalize=True, batch_losses=None, **kw):

        if batch_losses is None:
            _, _, batch_losses, measures = self.evaluate(x, **kw)

        normalized_log_pxy = - batch_losses / (2 * self.sigma)

        if normalize:
            return normalized_log_pxy

        D = np.prod(self.input_shape)
        a = np.log(self.sigma * 2 * np.pi)
        return normalized_log_pxy - D / 2 * a

    def log_px(self, x, normalize=True, method='max', batch_losses=None, **kw):
        """Computes a lower bound on log(p(x)) with the loss which is an upper
        bound on -log(p(x, y)).  - normalize = True forgets a constant
        (2pi sigma^2)^(-d/2) - method ='sum' computes p(x) as the sum
        of p(x, y), method='max' computes p(x) as the max_y of p(x, y)

        """
        if batch_losses is None:
            _, _, batch_losses_dict, _ = self.evaluate(x, **kw)
        batch_losses = batch_losses_dict['total']
            
        log_pxy = - batch_losses - np.log(self.num_labels) * self.is_cvae


        m_log_pxy = log_pxy.max(0)[0]

        if method == 'max':
            return m_log_pxy
        
        d_log_pxy = log_pxy - m_log_pxy

        # p_xy = d_p_xy * m_pxy
        d_pxy = d_log_pxy.exp()
        if method == 'sum':
            d_px = d_pxy.sum(0)
            return d_px.log + m_log_pxy
 
    def log_py_x(self, x, batch_losses=None, **kw):

        if batch_losses is None:
            _, _, batch_losses, _ = self.evaluate(x, **kw)

        # batch_losses is C * N1 * ... *
        log_pxy = - batch_losses / (2 * self.sigma)

        m_log_pxy = log_pxy.max(0)[0]
        d_log_pxy = log_pxy - m_log_pxy

        d_log_px = d_log_pxy.exp().sum(0).log()

        log_py_x = d_log_pxy - d_log_px

        return log_py_x

    def HY_x(self, x, method='pred', y_pred=None, losses=None, **kw):

        if method == 'pred':
            if y_pred is None:
                y_pred = self.blind_predict(x)
                return -(np.log(y_pred) * y_pred).sum(axis=-1)

        log_p = self.log_py_x(x, losses=losses, **kw)
        return -(np.exp(log_p) * log_p).sum(axis=-1)


if __name__ == '__main__':

    dir = save_load.get_path_by_input()
    argv = ['--debug',
            # '--force_cpu',
            # '-c', 'cifar-ola',
            # '-c', 'fashion-conv-code-linear-decode', 
            # '-c', 'svhn',
            # '-c', '',
            # '-c', 'cifar10-vgg16',
            # '-K', '128',
            # '-L', '50',
            # '-m', '50',
            '-b', '1e-4',
            '-j', 'test-jobs']
    argv = None
    args = get_args(argv)

    debug = args.debug
    verbose = args.verbose

    force_cpu = args.force_cpu

    epochs = args.epochs
    batch_size = args.batch_size
    test_sample_size = args.test_sample_size
    sigma = args.sigma

    latent_sampling = args.latent_sampling
    latent_dim = args.latent_dim

    features = args.features

    encoder = args.encoder
    decoder = args.decoder
    upsampler = args.upsampler
    conv_padding = args.conv_padding
    features_channels = args.features_channels

    output_activation = args.output_activation

    classifier = args.classifier

    dataset = args.dataset
    transformer = args.transformer

    refit = args.refit
    load_dir = args.load_dir
    print(f'**** LOAD {load_dir} ****')
    save_dir = load_dir if not refit else None
    job_dir = args.job_dir
    
    # load_dir = None
    save_dir = load_dir
    rebuild = load_dir is None

    if debug:
        for k in vars(args).items():
            print(*k)

    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Used device:', device)

    trainset, testset = torchdl.get_dataset(dataset, transformer=transformer)
    _, oodset = torchdl.get_svhn()

    testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                             shuffle=True, num_workers=0)

    test_batch = next(iter(testloader))
    x, y = test_batch[0].to(device), test_batch[1].to(device)

    input_shape = x.shape[1:]
    num_labels = len(torch.unique(y))

    if not rebuild:
        print('Loading...', end=' ')
        try:
            jvae = ClassificationVariationalNetwork.load(load_dir)
            print(f'done', end=' ')
            done_epochs = jvae.train_history['epochs']
            verb = 'resuming' if done_epochs else 'starting'
            print(f'{verb} training since epoch {done_epochs}')
            print(jvae.print_training())
        except(FileNotFoundError, NameError) as err:
            print(f'*** NETWORK NOT LOADED -- REBUILDING bc of {err} ***')
            rebuild = True

    if rebuild:
        print('Building network...', end=' ')
        jvae = ClassificationVariationalNetwork(input_shape, num_labels,
                                                features=features,
                                                features_channels=features_channels,
                                                conv_padding=conv_padding,
                                                # pretrained_features='vgg11.pth',
                                                encoder_layer_sizes=encoder,
                                                latent_dim=latent_dim,
                                                latent_sampling=latent_sampling,
                                                decoder_layer_sizes=decoder,
                                                upsampler_channels=upsampler,
                                                classifier_layer_sizes=classifier,
                                                sigma=sigma,
                                                output_activation=output_activation)

    
        if not save_dir:
            bs = f'sigma={sigma:.2e}--sampling={latent_sampling}--pretrained=both'
            save_dir_root = os.path.join(job_dir, dataset,
                                         jvae.print_architecture(),
                                         bs)
            i = 0
            save_dir = os.path.join(save_dir_root, f'{i:02d}')
            while os.path.exists(save_dir):
                i += 1
                save_dir = os.path.join(save_dir_root, f'{i:02d}')
                
        print('done.', 'Will be saved in\n' + save_dir)
            
    arch = jvae.print_architecture()
    print(arch)
    print(jvae.print_architecture(True, True))

    pattern='classifier=([0-9]+\-)*[0-9]+'
    vae_arch = re.sub(pattern, 'classifier=.', arch)
    
    jvae.to(device)

    ae_dir = os.path.join('.', 'jobs',
                          testset.name, vae_arch,
                          f'vae-sigma={sigma:.2e}--sampling={latent_sampling}',
                          '00')

    trained_ae_exists = os.path.exists(ae_dir)
    print(ae_dir, 'does' + ('' if trained_ae_exists else ' not'), 'exist')

    if trained_ae_exists:
        autoencoder = ClassificationVariationalNetwork.load(ae_dir)
        feat_dict = autoencoder.features.state_dict()
        up_dict = autoencoder.imager.upsampler.state_dict()
        ae_dict = autoencoder.state_dict()

        # jvae.load_state_dict(ae_dict)
        jvae.features.load_state_dict(feat_dict)
        jvae.imager.upsampler.load_state_dict(up_dict)
        
        for p in jvae.features.parameters():
            p.requires_grad_(False)
            
        for p in jvae.imager.upsampler.parameters():
            p.requires_grad_(False)    
    else:
        pass
        logging.warning('Trained autoencoder does not exist')
    """
    """
    
    out = jvae(x, y)

    for o in out:
        print(o.shape)


    # jvae2 = ClassificationVariationalNetwork.load('/tmp')
    
    # print('\nTraining\n')
    # print(refit)

    def train_the_net(epochs=3, **kw):
        jvae.train(trainset, epochs=epochs,
                   batch_size=batch_size,
                   device=device,
                   testset=testset,
                   sample_size=test_sample_size,  # 10000,
                   save_dir=save_dir,
                   **kw)
    
    # train_the_net(500, latent_sampling=latent_sampling, sigma=sigma)
    # jvae3 = ClassificationVariationalNetwork.load('/tmp')
    
    """
    for net in (jvae, jvae2, jvae3):
        print(net.print_architecture())
    for net in (jvae, jvae2, jvae3):
        print(net.print_architecture(True, True))

    print(jvae.training_parameters)
    train_the_net(2, latent_sampling=3, sigma=2e-5)
    if save_dir is not None:
        jvae.save(save_dir)

    x_, y_, mu, lv, z = jvae(x, y)
    x_reco, y_out, batch_losses = jvae.evaluate(x)
    
    y_est_by_losses = batch_losses.argmin(0)
    y_est_by_mean = y_out.mean(0).argmax(-1)
    """

