import logging
import copy
import torch
import torch.utils.data
from torch import nn, autograd
from utils.optimizers import Optimizer
from torch.nn import functional as F
from utils.losses import x_loss, kl_loss, mse_loss

from utils.save_load import LossRecorder

from vae_layers import VGGFeatures, ConvDecoder, Encoder, Decoder, Classifier, ConvFeatures, Sigma
from vae_layers import onehot_encoding

import data.torch_load as torchdl
from data.torch_load import choose_device
from utils import save_load
import numpy as np

from utils.roc_curves import fpr_at_tpr
from sklearn.metrics import auc, roc_curve

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
                                'cvae': ('cross_x', 'kl', 'total', 'zdist', 'dzdist', 'cross_y'),
                                'xvae': ('cross_x', 'kl', 'total'),
                                'vae': ('cross_x', 'kl', 'total'),
                                'vib': ('cross_y', 'kl', 'total')}
    
    predict_methods_per_type = {'jvae': ('loss', 'mean'),
                                'cvae': ('loss', 'closest'),
                                'xvae': ('loss', 'closest'),
                                'vae': (),
                                'vib': ('esty',)}

    metrics_per_type = {'jvae': ('std', 'snr', 'sigma'),
                        'cvae': ('std', 'snr',  'd-mind', 'ld-norm', 'sigma'),
                        'xvae': ('std', 'snr', 'zdist', 'd-mind', 'ld-norm', 'sigma'),
                        'vae': ('std', 'snr', 'sigma'),
                        'vib': ('sigma',)}

    ood_methods_per_type ={'cvae': ('max', 'kl', 'mse', 'std', 'mag'), # , 'mag', 'IYx'),
                           'xvae': ('max', 'mean', 'std'), # , 'mag', 'IYx'),
                           'jvae': ('max', 'sum',  'std'), # 'mag'), 
                           'vae': ('logpx',),
                           'vib': ()}

    def __init__(self,
                 input_shape,
                 num_labels,
                 type_of_net = 'jvae', # or 'vib' or cvae or vae
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
                 gamma_temp=np.inf,
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
        
        self.is_jvae = type_of_net == 'jvae'
        self.is_vib = type_of_net == 'vib'
        self.is_vae = type_of_net == 'vae'
        self.is_cvae = type_of_net == 'cvae'
        self.is_xvae = type_of_net == 'xvae'
        
        self.y_is_coded = self.is_jvae or self.is_xvae
        self.y_is_decoded = self.is_vib or self.is_jvae

        self.coder_has_dict = self.is_cvae or self.is_xvae
        
        self.x_is_generated = not self.is_vib

        self.losses_might_be_computed_for_each_class = not (self.is_vae or self.is_vib)
        
        logging.debug('y is%s coded', '' if self.y_is_coded else ' not')

        self._measures = {}
        
        self.force_cross_y = force_cross_y
        if not self.y_is_decoded and not force_cross_y:
            classifier_layer_sizes = []
        if not self.x_is_generated:
            decoder_layer_sizes = []
            upsampler_channels = None
            
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
        if type(sigma) == Sigma:
            self.sigma = sigma
        elif type(sigma) == dict:
            self.sigma = Sigma(**sigma)
        else:
            self.sigma = Sigma(value=sigma)
            
        sampling = latent_sampling > 1 or bool(self.sigma > 0)
        if not sampling:
            logging.debug('Building a vanilla classifier')

        self.beta = beta
        if np.isfinite(gamma_temp) and not gamma:
            gamma = 1
        self.gamma = gamma if self.coder_has_dict else None
        self.gamma_temp = gamma_temp if self.coder_has_dict else None
        
        logging.debug(f'Gamma: {self.gamma}')
        
        self.latent_prior_variance = latent_prior_variance
        self.encoder = Encoder(encoder_input_shape, num_labels,
                               intermediate_dims=encoder_layer_sizes,
                               latent_dim=latent_dim,
                               y_is_coded = self.y_is_coded,
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

        self.training = {} # 
        
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

        self.training = {
            'sigma': self.sigma.params,
            'beta': self.beta,
            'gamma': self.gamma,
            'gamma_temp': self.gamma_temp,
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
        self.training['optim'] = self.optimizer.params
            
        self.train_history = {'epochs': 0}

        self.latent_dim = latent_dim
        self.latent_sampling = latent_sampling
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.classifier_layer_sizes = classifier_layer_sizes
        self.upsampler_channels = upsampler_channels
        self.activation = activation
        self.output_activation = output_activation
            
        self.z_output = False
        
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

    def forward_from_features(self, x_features, y, x, z_output=True):

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
            z_mean, z_log_var, z = self.encoder(x_, y_onehot)
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

        return out

    def evaluate(self, x,
                 y=None,
                 batch=0,
                 current_measures=None,
                 with_beta=False,
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

        C = self.num_labels
        
        if self.features:
            t = self.features(x)
            # print('*** cvae:444 x:', *x.shape,
            #      't:', *t.shape)
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
            # create a C * N1 * ... * Ng y tensor y[c,:,:,:...] = c


        if losses_computed_for_each_class:
            y_shape_per_class = (1,) + y_shape
            y = torch.cat([c * torch.ones(y_shape_per_class,
                                          dtype=int,
                                          device=x.device)
                           for c in range(C)], dim=0)
            y_shape = y.shape
        
        # _ = ('*',) if y is None else y.shape
        # print('***', 'cvae:470',
        #       'y:', *_,
        #       't:', *t.shape)
                
        y_in = y.view(y_shape) if self.y_is_coded else None
        
        if self.features:
            x_reco, y_est, mu, log_var, z = self.forward_from_features(t, y_in, x, **kw)
        else:
            x_reco, y_est, mu, log_var, z = self.forward(t, y_in, x, **kw)

        # print('*** cvae:472 logits:', 't:', *t.shape, 'x_:', *x_reco.shape)
            
        batch_quants = {}
        batch_losses = {}
        total_measures = {}

        if not current_measures:
            current_measures =  {k: 0.
                                for k in ('xpow',
                                          'mse', 
                                          'snr',
                                          'imut-zy',
                                          'd-mind',
                                          'ld-norm',
                                          'zdist')}
            
        total_measures['sigma'] = self.sigma.value

        if self.x_is_generated:
            batch_quants['mse'] = mse_loss(x, x_reco[1:],
                                           ndim=len(self.input_shape),
                                           batch_mean=False)

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

        kl_l, zdist = kl_loss(mu, log_var,
                              y=y if self.coder_has_dict else None,
                              prior_variance = self.latent_prior_variance,
                              latent_dictionary=dictionary,
                              out_zdist=True,
                              batch_mean=False)

        # print('*** wxjdjd ***', 'kl', *kl_l.shape, 'zd', *zdist.shape)
        
        total_measures['zdist'] = (current_measures['zdist'] * batch +
                                   zdist.mean().item()) / (batch + 1)
        
        batch_quants['latent_kl'] = kl_l

        batch_losses['zdist'] = zdist

        if not self.is_vae: # self.y_is_decoded or self.force_cross_y:
            if y_in_input or (self.y_is_decoded and losses_computed_for_each_class):
                y_target = y
            else:
                y_target = None

            batch_quants['cross_y'] = x_loss(y_target,
                                             y_est if self.classifier else zdist.T,
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
            batch_mse = batch_quants['mse']
            D = np.prod(self.input_shape)

            if self.sigma.is_rmse:
                batch_logpx = (- D / 2 * torch.log(batch_mse * np.pi)
                               - D / 2)

            else:
                batch_logpx = (- D / 2 * torch.log(self.sigma**2 * np.pi)
                               - D / (2 * self.sigma**2) * batch_mse)

            batch_losses['cross_x'] = - batch_logpx

            batch_losses['total'] += batch_losses['cross_x'] 
            
        if not self.is_vae:  # self.y_is_decoded or self.force_cross_y:
            batch_losses['cross_y'] = batch_quants['cross_y']
            """ print('*** cvae:528', 'losses:',
                  'y', *batch_losses['cross_y'].shape,
                  'T', *batch_losses['total'].shape)
            """
        
            # print('*** cvae:602', 'T:', *batch_losses['total'].shape,
            #      'Xy:', *batch_losses['cross_y'].shape)
            
            if self.y_is_decoded:
                batch_losses['total'] = batch_losses['total'] + batch_losses['cross_y']
                
        batch_losses['kl'] = batch_quants['latent_kl']
        
        if self.is_vib:
            # print('*** cvae 612: T:', *batch_losses['total'].shape, 'kl', *batch_losses['kl'].shape)
            batch_losses['total'] += self.sigma * batch_losses['kl']
        else:
            beta = self.beta if with_beta else 1.
            batch_losses['total'] += beta * batch_losses['kl']
            
        if not self.is_vib:
            pass
            # print('******* x_', x_reco.shape)
            # x_reco = x_reco.mean(0)

        # logging.debug('Losses computed')
        out = (x_reco, y_est.mean(0), batch_losses, total_measures)
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

        if method == 'closest':
            return losses['zdist'].argmin(0)

        raise ValueError(f'Unknown method {method}')

    def batch_dist_measures(self, logits, losses, methods):

        dist_measures = {m: None for m in methods}
        for m in methods:
            assert m in self.ood_methods
        
        loss = losses['total']

        logp = - loss
        # ref is max of logp
        logp_max = logp.max(axis=0)[0]
        d_logp = logp - logp_max
        
        for m in methods:

            if m == 'logpx':
                assert not self.losses_might_be_computed_for_each_class
                measures = logp
            elif m == 'sum':
                measures = d_logp.exp().sum(axis=0).log() + logp_max 
            elif m == 'max':
                measures = logp_max
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
                C = self.num_labels
                
                measures =  ( (d_logp * (d_logp.exp())).sum(axis=0) / (C * d_logp_x.exp())
                            - d_logp_x )
            elif m == 'kl':
                measures = -losses['kl'].min(axis=0)[0]
            elif m == 'mse' and self.is_cvae:
                measures = -losses['cross_x']
                    
            else:
                raise ValueError(f'{m} is an unknown ood method')

            dist_measures[m] = measures

        return dist_measures

    def compute_max_batch_size(self, batch_size=1024, which='all'):
        if which == 'all':
            self.compute_max_batch_size(batch_size, which='train')
            self.compute_max_batch_size(batch_size, which='test')
            return

        logging.debug('Computing max batch size for %s', which)
        if 'max_batch_sizes' not in self.training:
            self.training['max_batch_sizes'] = {}
            
        training = which == 'train'

        x = torch.randn(batch_size, *self.input_shape, device=self.device)
        y = torch.ones(batch_size, dtype=int, device=self.device) if training else None

        while batch_size > 2:
            x = x[:batch_size]
            if y is not None:
                y = y[:batch_size]
            try:
                logging.debug('Trying batch size of %s for %s of %s.',
                              batch_size,
                              which,
                              self.job_number)
                if training:
                    logging.debug('Evaling net')
                    _, _, batch_losses, _ = self.evaluate(x, y=y)
                    logging.debug('Net evaled')
                    L = batch_losses['total'].mean()
                    logging.debug('Backwarding net')
                    L.backward()
                else:
                    with torch.no_grad():
                        self.evaluate(x, y=y)
                self.training['max_batch_sizes'][which] = batch_size // 2
                logging.debug('Found max batch size for %s : %s',
                              which, batch_size)
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
        max_batch_sizes = self.training.get('max_batch_sizes', {})
        if max_batch_sizes:
            return max_batch_sizes
        self.compute_max_batch_size()
        return self.max_batch_sizes

    @max_batch_sizes.setter
    def max_batch_sizes(self, v):
        assert 'train' in v
        assert 'test' in v
        self.training['max_batch_sizes'] = v
    
    def accuracy(self, testset=None,
                 batch_size=100,
                 num_batch='all',
                 method='all',
                 print_result=False,
                 update_self_testing=True,
                 outputs=EpochOutput(),
                 sample_dirs=[],
                 recorder=None,
                 log=True):

        """return detection rate. 
        method can be a list of methods

        """

        MAX_SAMPLE_SAVE = 200
        
        device = next(self.parameters()).device
        
        if not testset:
            testset_name = self.training['set']
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
        
        if num_batch == 'all':
            num_batch = len(testset) // batch_size
            shuffle = False
            
        recorded = recorder is not None and len(recorder) >= num_batch
        recording = recorder is not None and len(recorder) < num_batch
        
        if recorded:
            logging.debug('Losses already recorded')

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

                current_measures = measures
                self._measures = measures
            else:
                batch_losses = recorder.get_batch(i, *self.loss_components)
                logging.debug('TBD cvae:874: %s', ' '.join(self.loss_components))
                logits = recorder.get_batch(i, 'logits').T
                y_test = recorder.get_batch(i, 'y_true')

            y_pred = {}
            logging.debug('TBD cvae:878: %s', ' '.join(batch_losses.keys()))
            for m in predict_methods:
                y_pred[m] = self.predict_after_evaluate(logits,
                                                        batch_losses,
                                                        method=m)

            if recording:
                recorder.append_batch(**batch_losses, y_true=y_test, logits=logits.T)
                
            # print('*** 842', y_test[0].item(), *y_test.shape)
            # print('*** 843', batch_losses['cross_y'].min(0)[0].mean())
            ind = y_test.unsqueeze(0)
            for k in batch_losses:
                if k == 'cross_y' and self.is_xvae:
                    shape = 'CxNxC'
                elif k == 'cross_y' or self.is_jvae:
                    shape = 'CxN'
                elif self.is_cvae and k != 'cross_x' and k != 'dzdist':
                    shape = 'CxN'
                elif self.is_vib and k == 'total':
                    shape = 'CxN'
                elif self.is_xvae:
                    shape = 'CxN'
                else:
                    shape = 'N'
                # print('*** cvae 859', k, *batch_losses[k].shape, shape)

                if shape == 'CxNxC':
                    batch_loss_y = batch_losses[k].max(-1)[0].gather(0, ind)
                elif shape == 'CxN':
                    batch_loss_y = batch_losses[k].gather(0, ind)
                else:
                    batch_loss_y = batch_losses[k]
                # print('*** cvae:866 bl_y:', *batch_loss_y.shape)
                
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
                                          (self.trained > self.testing[m]['epochs']
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
                                   'epochs': self.trained,
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
                            outputs=EpochOutput(),
                            recorders=None,
                            sample_dirs=[],
                            log=True):

        if not testset:
            testset_name = self.training['set']
            transformer = self.training['transformer']
            _, testset = torchdl.get_dataset(testset_name, transformer=transformer)
        
        if method=='all':
            ood_methods = self.ood_methods

        elif type(method) is str:
            assert method in self.ood_methods
            ood_methods = [method]
            
        else:
            try:
                method.__iter__()
                ood_methods = method
            except AttributeError:
                raise ValueError(f'{method} is not a '
                                 'valid method / list of method')

        if not method:
            return
        
        if oodsets is None:
            oodsets = {n: torchdl.get_dataset(n, transformer=testset.transformer)[1]
                       for n in testset.same_size}
            logging.debug('Oodsets loaded: ' + ' ; '.join(s.name for s in oodsets.values()))

        all_set_names = [testset.name] + [o.name for o in oodsets] 

        if not recorders:
            recorders = {n: None for n in all_set_names}

        max_num_batch = num_batch
        num_batch = {testset.name: max(len(testset) // batch_size, 1)}
        for o in oodsets:
            num_batch[o.name] = max(len(o) // batch_size, 1)

        shuffle = {s: False for s in all_set_names}
        recording = {}
        recorded = {}
        
        if type(max_num_batch) is int:
            for s in all_set_names:
                num_batch[s] = min(num_batch[s], max_num_batch)
                recording[s] = recorders[s] is not None and len(recorders[s]) < num_batch[s]
                recorded[s] = recorders[s] is not None and len(recorders[s]) >= num_batch[s]
                shuffle[s] = True
                if recorded[s]:
                    logging.debug('Losses already computed for %s %s', s, recorders[s])
                if recording[s]:
                    recorders[s].reset()
                    recorders[s].num_batch = num_batch[s]
                    logging.debug('Recording session for %s %s', s, recorders[s])
                
        device = next(self.parameters()).device

        if oodsets:
            outputs.results(0, 0, -2, 0,
                            metrics=ood_methods,
                            acc_methods=ood_methods)
            outputs.results(0, 0, -1, 0, metrics=ood_methods,
                            acc_methods=ood_methods)

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

                if not recorded[s]:
                    data = next(test_iterator)
                    x = data[0].to(device)
                    y = data[1].to(device)
                    with torch.no_grad():
                        _, logits, losses, _  = self.evaluate(x)
                else:
                    losses = recorders[s].get_batch(i, *self.loss_components)
                    logits = recorders[s].get_batch(i, 'logits').T
                    
                if recording[s]:
                    recorders[s].append_batch(**losses, y_true=y, logits=logits.T)
                    
                measures = self.batch_dist_measures(logits, losses, ood_methods)
                for m in ood_methods:
                    ind_measures[m] = np.concatenate([ind_measures[m],
                                                      measures[m].cpu()])
                t_i = time.time() - t_0
                t_per_i = t_i / (i + 1)
                outputs.results(i, num_batch[s], 0, 1, metrics=ood_methods,
                                measures = {m: ind_measures[m].mean()
                                            for m in ood_methods},
                                acc_methods = ood_methods,
                                time_per_i = t_per_i,
                                batch_size=batch_size,
                                preambule = testset.name)

            if recording[s]:
                for d in sample_dirs:
                    f = os.path.join(d, f'record-{s}.pth')
        
                    recorders[s].save(f.format(s=s))
        

                
        keeped_tpr = [pc / 100 for pc in range(90, 100)]
        no_result = {'epochs': 0,
                     'n': 0,
                     'auc': 0,
                     'tpr': keeped_tpr,
                     'fpr': [1 for _ in keeped_tpr],
                     'thresholds':[None for _ in keeped_tpr]}
                     
        for oodset in oodsets:

            s = oodset.name
            ood_n_batch = num_batch[s]
            
            ood_results = {m: copy.deepcopy(no_result) for m in ood_methods}
            i_ood_measures = {m: ind_measures[m] for m in ood_methods}
            ood_labels = np.ones(batch_size * num_batch[testset.name])
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
                    with torch.no_grad():
                        _, logits, losses, _ = self.evaluate(x)
                else:
                    losses = recorders[s].get_batch(i, *self.loss_components)
                    logits = recorders[s].get_batch(i, 'logits').T
                    
                if recording[s]:
                    recorders[s].append_batch(**losses, y_true=y, logits=logits.T)

                measures = self.batch_dist_measures(None, losses, ood_methods)
                for m in ood_methods:
                    i_ood_measures[m] = np.concatenate([i_ood_measures[m],
                                                      measures[m].cpu()])
                ood_labels = np.concatenate([ood_labels, np.zeros(batch_size)])
                t_i = time.time() - t_0
                t_per_i = t_i / (i + 1)
                meaned_measures = {m: i_ood_measures[m][len(ind_measures):].mean()
                                   for m in ood_methods}
                for m in ood_methods:
                    # logging.debug(f'Computing roc curves for with metrics {m}')
                    fpr_[m], tpr_[m], thresholds_[m] =  roc_curve(ood_labels,
                                                                  i_ood_measures[m])
                    auc_[m] = auc(fpr_[m], tpr_[m])

                    r_[m] = fpr_at_tpr(fpr_[m],
                                       tpr_[m],
                                       0.95,
                                       thresholds_[m])

                outputs.results(i, ood_n_batch, 0, 1, metrics=ood_methods,
                                measures=meaned_measures,
                                acc_methods=ood_methods,
                                accuracies=r_,
                                time_per_i = t_per_i,
                                batch_size=batch_size,
                                preambule = oodset.name)

            for m in ood_methods:
                fpr_and_thresholds = [fpr_at_tpr(fpr_[m], tpr_[m], a,
                                                 thresholds=thresholds_[m],
                                                 return_threshold=True) for a in keeped_tpr] 
                fpr_m = [f[0] for f in fpr_and_thresholds]
                t_m = [f[1] for f in fpr_and_thresholds]
                ood_results[m] = {'epochs': self.trained,
                                  'n': ood_n_batch * batch_size,
                                  'auc': auc_[m],
                                  'tpr': keeped_tpr,
                                  'fpr': fpr_m, 
                                  'thresholds': t_m }
                
            if update_self_ood:
                a = self.ood_results
                self.ood_results[oodset.name] = ood_results

            if recording[s]:
                for d in sample_dirs:
                    f = os.path.join(d, f'record-{s}.pth')
        
                    recorders[s].save(f.format(s=s))
        
    def train(self,
              trainset=None,
              transformer=None,
              data_augmentation=None,
              optimizer=None,
              epochs=50,
              batch_size=100, device=None,
              testset=None,
              oodsets=None,
              acc_methods=None,
              fine_tuning=False,
              latent_sampling=None,
              sample_size=1000,
              full_test_every=10,
              ood_detection_every=10,
              train_accuracy=False,
              save_dir=None,
              outputs=EpochOutput(),
              signal_handler=SIGHandler()):
        """

        """
        if epochs:
            self.training['epochs'] = epochs
            
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
                self.training['set'] = set_name
                self.training['transformer'] = transformer
                ss = trainset.data[0].shape
                ns = self.input_shape
                logging.debug(f'Shapes : {ss} / {ns}')
                # assert ns == ss or ss == ns[1:]
        
            if batch_size:
                self.training['batch_size'] = batch_size

            if latent_sampling:
                self.latent_sampling = latent_sampling
            self.training['latent_sampling'] = self.latent_sampling

            if data_augmentation:
                self.training['data_augmentation'] = data_augmentation
        
        assert self.training['set']

        set_name = self.training['set']
        data_augmentation = self.training['data_augmentation']
        
        logging.debug(f'Getting {set_name}')
        trainset, testset = torchdl.get_dataset(set_name,
                                                transformer=transformer,
                                                data_augmentation=data_augmentation)

        logging.debug('Choosing device')
        device = choose_device(device)
        logging.debug(f'done {device}')

        if optimizer is None:
            optimizer = self.optimizer
        
        max_batch_sizes = self.max_batch_sizes

        test_batch_size = max_batch_sizes['test']
        
        if batch_size:
            train_batch_size = min(batch_size, max_batch_sizes['train'])
        else:
            train_batch_size = max_batch_sizes['train']

        x_fake = torch.randn(test_batch_size, *self.input_shape, device=self.device)
        y_fake = torch.randint(0, 1, size=(test_batch_size,), device=self.device)
        
        _, logits, losses, measures = self.evaluate(x_fake)
        
        sets = [set_name]
        for s in oodsets:
            sets.append(s.name)
            
        recorders = {s: LossRecorder(test_batch_size, **losses,
                                     logits=logits.T,
                                     y_true=y_fake)
                     for s in sets}

        for s in recorders:
            logging.debug('Recorder created for %s %s', s, recorders[s])
            
        logging.debug('Creating dataloader for training with batch size %s',
                      train_batch_size)

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=train_batch_size,
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
            self.train_history['train_loss'] = []
            self.train_history['test_accuracy'] = []
            self.train_history['train_accuracy'] = []
            self.train_history['train_measures'] = []
            self.train_history['test_measures'] = []
            self.train_history['test_loss'] = []
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

            num_batch = max(sample_size // test_batch_size, 1)
            if testset:
                # print(num_batch, sample_size)
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
                        recorder = recorders[set_name]
                        # recorder.reset()
                    else:
                        recorder = None

                    test_accuracy = self.accuracy(testset,
                                                  batch_size=test_batch_size,
                                                  num_batch='all' if full_test else num_batch,
                                                  # device=device,
                                                  method=acc_methods,
                                                  # log=False,
                                                  outputs=outputs,
                                                  sample_dirs=sample_dirs,
                                                  recorder=recorder,
                                                  print_result='TEST' if full_test else 'test')
                    test_loss = self.test_loss
                if signal_handler.sig > 1:
                    logging.warning(f'Breaking training loop bc of {signal_handler}')
                    break
                if save_dir: self.save(save_dir)
                test_measures = self._measures.copy()
            # train
            if train_accuracy:
                with torch.no_grad():
                    train_accuracy = self.accuracy(trainset,
                                                   batch_size=test_batch_size,
                                                   num_batch=num_batch,
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

            if 'std' in train_measures:
                self.sigma.decay_to(train_measures['std'])
                               
            if signal_handler.sig > 1:
                logging.warning(f'Breaking training loop bc of {signal_handler}')
                break

            if save_dir:
                self.save(save_dir)

            current_measures = {}
            
            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                x, y = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # with autograd.detect_anomaly():
                # forward + backward + optimize
                (_, y_est,
                 batch_losses, measures) = self.evaluate(x, y,
                                                         batch=i,
                                                         with_beta=True,
                                                         current_measures=current_measures)

                current_measures = measures
                batch_loss = batch_losses['total'].mean()

                L = batch_loss
                if self.coder_capacity_regularization and self.encoder.dictionary_dist_lb:
                        L += self.encoder.dist_barrier()

                if self.force_cross_y and not self.y_is_decoded:
                    L += self.force_cross_y * batch_losses['cross_y'].mean()

                if self.gamma:
                    dict_var = self.encoder.latent_dictionary.pow(2).mean()
                    log2 = np.log(2)
                    
                    g_ = self.gamma * torch.exp(-dict_var / self.gamma_temp * log2)
                    L += g_ * (batch_losses['zdist'] - batch_losses['dzdist']).mean()
                    # logging.debug('adding gamma loss')
                    
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
                self.training['fine_tuning'].append(epoch)

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
            
        if testset and not signal_handler.sig > 1:
            # print(num_batch, sample_size)
            with torch.no_grad():
                test_accuracy = self.accuracy(testset,
                                              batch_size=test_batch_size,
                                              # num_batch=num_batch,
                                              # device=device,
                                              method=acc_methods,
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

        t1 = self.training
        to = other.training

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
            epochs = self.training['epochs']

        sampling = self.training['latent_sampling']
        if not set:
            set = self.training['set']
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
        save_load.save_json(self.training, dir_name, 'train.json')
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
                          'latent_prior_variance': 1.,
        }
        
        train_params = {'pretrained_features': None,
                        'pretrained_upsampler': None,
                        'learned_coder': False,
                        'beta': 1.,
                        'gamma': 0.,
                        'gamma_temp':np.inf,
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

        logging.debug('Parameters loaded')
        if loaded_params.get('batch_norm', False) == True:
            loaded_params['batch_norm'] = 'encoder'

        params = default_params.copy()
        params.update(loaded_params)
        
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
        
        loaded_train = False
        try:
            train_params.update(save_load.load_json(dir_name, 'train.json'))
            logging.debug('Training parameters loaded')
            loaded_train = True
        except(FileNotFoundError):
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
                logging.debug('Ghost network loaded')
                vae.job_number = job_number
                vae.ood_methods = cls.ood_methods_per_type[vae.architecture['type']]
                vae.predict_methods = cls.predict_methods_per_type[vae.architecture['type']]
                vae.testing = {}
                vae.sigma = Sigma(**train_params['sigma'])

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
                      batch_norm=params['batch_norm'],
                      activation=params['activation'],
                      sigma=train_params['sigma'],
                      beta=train_params['beta'],
                      gamma=train_params['gamma'],
                      gamma_temp=train_params['gamma_temp'],
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
                
        vae.trained = train_history['epochs']
        vae.train_history = train_history

        vae.training = train_params
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

    print(jvae.training)
    train_the_net(2, latent_sampling=3, sigma=2e-5)
    if save_dir is not None:
        jvae.save(save_dir)

    x_, y_, mu, lv, z = jvae(x, y)
    x_reco, y_out, batch_losses = jvae.evaluate(x)
    
    y_est_by_losses = batch_losses.argmin(0)
    y_est_by_mean = y_out.mean(0).argmax(-1)
    """

