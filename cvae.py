import logging
import copy
import torch
import torch.utils.data
from torch import nn, autograd
from utils.optimizers import Optimizer
from torch.nn import functional as F
from utils.losses import x_loss, kl_loss, mse_loss

from vae_layers import VGGFeatures, ConvDecoder, Encoder, Decoder, Classifier, ConvFeatures
from vae_layers import onehot_encoding

import data.torch_load as torchdl
from data.torch_load import choose_device
from utils import save_load
import numpy as np

from roc_curves import ood_roc, fpr_at_tpr
from sklearn.metrics import auc, roc_curve

from utils.print_log import Outputs, debug_nan

from utils.parameters import get_args

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
                                'cvae': ('cross_x', 'kl', 'total'),
                                'xvae': ('cross_x', 'kl', 'total'),
                                'vae': ('cross_x', 'kl', 'total'),
                                'vib': ('cross_y', 'kl', 'total')}
    
    predict_methods_per_type = {'jvae': ('loss', 'mean'),
                                'cvae': ('loss', 'closest'),
                                'xvae': ('loss', 'closest'),
                                'vae': (),
                                'vib': ('esty',)}

    metrics_per_type = {'jvae': ('std', 'snr', 'sigma'),
                        'cvae': ('std', 'snr', 'zdist', 'd-mind', 'ld-norm', 'sigma'),
                        'xvae': ('std', 'snr', 'zdist', 'd-mind', 'ld-norm', 'sigma'),
                        'vae': ('std', 'snr', 'sigma'),
                        'vib': ('sigma',)}

    ood_methods_per_type ={'cvae': ('max', 'mean', 'std'), # , 'mag', 'IYx'),
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
                 learned_coder=False,
                 dictionary_min_dist=None,
                 init_coder=True,
                 coder_capacity_regularization=True,
                 decoder_layer_sizes=[36],
                 upsampler_channels=None,
                 pretrained_upsampler=None,
                 classifier_layer_sizes=[36],
                 name='joint-vae',
                 activation=DEFAULT_ACTIVATION,
                 latent_sampling=DEFAULT_LATENT_SAMPLING,
                 output_activation=DEFAULT_OUTPUT_ACTIVATION,
                 sigma=0.5,
                 sigma_reach=0,
                 sigma_decay=0,
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
        
        if not self.y_is_decoded:
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

            encoder_input_shape = self.features.output_shape
            logging.debug('Features built')
            
        else:
            encoder_input_shape = input_shape
            self.features = None
            
        sampling = latent_sampling > 1 or sigma > 0
        if not sampling:
            logging.debug('Building a vanilla classifier')
            
        self.encoder = Encoder(encoder_input_shape, num_labels,
                               intermediate_dims=encoder_layer_sizes,
                               latent_dim=latent_dim,
                               y_is_coded = self.y_is_coded,
                               sampling_size=latent_sampling,
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
        self._sigma = torch.nn.Parameter(requires_grad=False)
        self.sigma = sigma
        self.sigma_reach = sigma_reach

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
                             'latent_dim': latent_dim,
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

        self.trained = 0
        self.training = {'sigma': sigma,
                         'sigma_reach': sigma_reach,
                         'learned_coder': learned_coder,
                         'dictionary_min_dist': self.encoder.dictionary_dist_lb,
                         'coder_capacity_regularization':coder_capacity_regularization,
                         'latent_sampling': latent_sampling,
                         'set': None,
                         'data_augmentation': [],
                         'pretrained_features': pretrained_features,
                         'pretrained_upsampler': pretrained_upsampler,
                         'epochs': 0,
                         'kl_loss_weight': None,
                         'mse_loss_weight': None,
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
            out = (x_output.reshape((self.latent_sampling,) + reco_batch_shape),)

        out += (y_output,)

        if z_output:
            out += (z_mean, z_log_var, z)

        return out

    def evaluate(self, x,
                 y=None,
                 batch=0,
                 current_measures=None,
                 z_output=False,
                 **kw):
        """x input of size (N1, .. ,Ng, D1, D2,..., Dt) 

        creates a x of size C * N1, ..., D1, ...., Dt)
        and a y of size C * N1 * ... * Ng

        ----- Returns 

        x_ (C,N1,..., D1...) tensor,

        y_est (C,N1,...,C) tensor, 

        batch_losses (C, N1,...N) tensor
        total_losses
        batch_metrics
        total_metrics

        """
        y_in_input = y is not None
        x_repeated_along_classes = self.y_is_coded and not y_in_input
        losses_computed_for_each_class = (self.losses_might_be_computed_for_each_class
                                          and not y_in_input)

        x_is_decoded = not self.is_vib
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
            x_reco, y_est, mu, log_var, z = self.forward_from_features(t, y_in, x)
        else:
            x_reco, y_est, mu, log_var, z = self.forward(t, y_in, x)

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
            
        total_measures['sigma'] = self.sigma

        if not self.is_vib:
            batch_quants['mse'] = mse_loss(x, x_reco,
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
            
        if self.y_is_decoded:
            y_target = y if y_in_input or losses_computed_for_each_class else None
            batch_quants['cross_y'] = x_loss(y_target,
                                             y_est,
                                             batch_mean=False)

            # print('*** cvae:485 cross_y', *batch_quants['cross_y'].shape)
            
        dictionary = self.encoder.latent_dictionary if self.coder_has_dict else None

        kl_l, zdist = kl_loss(mu, log_var,
                              y=y if self.coder_has_dict else None,
                              latent_dictionary=dictionary,
                              out_zdist=True,
                              batch_mean=False)

        # print('*** wxjdjd ***', 'kl', *kl_l.shape, 'zd', *zdist.shape)
        
        total_measures['zdist'] = (current_measures['zdist'] * batch +
                                   zdist.mean().item()) / (batch + 1)
        
        batch_quants['latent_kl'] = kl_l

        batch_losses['zdist'] = zdist
        batch_losses['total'] = torch.zeros_like(batch_quants['latent_kl'])

        if self.coder_has_dict:
            # batch_losses['zdist'] = 0
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
            sigma = self._sigma
            batch_logpx = (- D / 2 * torch.log(sigma**2 * np.pi)
                           - D / (2 * sigma**2) * batch_mse)
            batch_losses['cross_x'] = - batch_logpx

            batch_losses['total'] += batch_losses['cross_x'] 
            
        if self.y_is_decoded:
            batch_losses['cross_y'] = batch_quants['cross_y']
            """ print('*** cvae:528', 'losses:',
                  'y', *batch_losses['cross_y'].shape,
                  'T', *batch_losses['total'].shape)
            """
            batch_losses['total'] += batch_losses['cross_y']
                
        batch_losses['kl'] = batch_quants['latent_kl']
        
        if self.is_vib:
            batch_losses['total'] += self.sigma * batch_losses['kl']
        else:
            batch_losses['total'] += batch_losses['kl']
            
        # print('***fcdf***', 'T:', *batch_losses['total'].shape,
        #       'Xy:', *batch_losses['cross_y'].shape,
        #       'kl:', *batch_losses['kl'].shape)
        

        if not self.is_vib:
            pass
            # print('******* x_', x_reco.shape)
            # x_reco = x_reco.mean(0)

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
        ref = -loss.min(axis=0)[0]
        d_logp = -loss - ref
        for m in methods:

            if m == 'logpx':
                assert not self.losses_might_be_computed_for_each_class
                measures = -loss
            elif m == 'sum':
                measures = d_logp.exp().sum(axis=0).log() + ref 
            elif m == 'max':
                measures = -loss.min(axis=0)[0]
            elif m == 'mag':
                measures = d_logp.max(axis=0)[0] - d_logp.mean(axis=0)
            elif m == 'std':
                measures = d_logp.exp().std(axis=0).log() + ref
            elif m == 'mean':
                measures = d_logp.exp().mean(axis=0).log() + ref
            elif m == 'nstd':
                measures = (d_logp.exp().std(axis=0).log()
                            - d_logp.exp().mean(axis=0).log()).exp().pow(2)
            elif m == 'IYx':
                d_logp_x = d_logp.exp().mean(axis=0).log()
                C = self.num_labels
                
                measures =  ( (d_logp * (d_logp.exp())).sum(axis=0) / (C * d_logp_x.exp())
                            - d_logp_x )
            else:
                raise ValueError(f'{m} is an unknown ood method')

            dist_measures[m] = measures

        return dist_measures

    def compute_max_batch_size(self, batch_size=1024, which='all'):
        if which=='all':
            self.compute_max_batch_size(batch_size, which='train')
            self.compute_max_batch_size(batch_size, which='test')
            return

        if 'max_batch_sizes' not in self.training:
            self.training['max_batch_sizes'] = {}
            
        training = which=='train'  

        x = torch.randn(batch_size, *self.input_shape, device=self.device)
        y = torch.ones(batch_size, dtype=int, device=self.device) if training else None
        while True:
            x = x[:batch_size]
            if y is not None:
                y = y[:batch_size]
            try:
                logging.debug('Trying batch size of %s for %s.',
                              batch_size,
                              which)
                if training:
                    _, _, batch_losses, _ = self.evaluate(x, y=y)
                    L = batch_losses['total'].mean()
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
                batch_size//=2
                
    @property
    def max_batch_sizes(self):
        # logging.debug('Calling max bathc size')
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
                 return_mismatched=False,
                 print_result=False,
                 update_self_testing=True,
                 outputs=Outputs(),
                 log=True):

        """return detection rate. If return_mismatched is True, indices of
        mismatched are also retuned.
        method can be a list of methods

        """

        device = next(self.parameters()).device
        
        if not testset:
            testset_name=self.training['set']
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

        n_err = dict()
        mismatched = dict()
        acc = dict()
        for m in predict_methods:
            n_err[m] = 0
            mismatched[m] = []
        n = 0

        # logging.debug('Creating dataloader for accuracy with batch size %s',
        #              batch_size)

        # print('**** cvae.py:698', testset.transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        iter_ = iter(testloader)
        start = time.time()

        total_loss = {k: 0. for k in self.loss_components}
        mean_loss = total_loss.copy()

        current_measures = {}

        for i in range(num_batch):
            data = next(iter_)
            x_test, y_test = data[0].to(device), data[1].to(device)

            (_, y_est,
             batch_losses, measures) = self.evaluate(x_test, batch=i,
                                                    current_measures=current_measures)
            current_measures = measures

            if self.is_jvae or self.is_cvae:
                ind = y_test.unsqueeze(0)
            
            for k in batch_losses:
                # print('*****', ind.shape)
                # print('*****', k, batch_losses[k].shape)
                if self.is_jvae or (self.is_cvae and k != 'cross_x'):
                    batch_loss_y = batch_losses[k].gather(0, ind)
                else:
                    batch_loss_y = batch_losses[k].mean(0)
                if k not in total_loss:
                    total_loss[k] = 0.0
                total_loss[k] += batch_loss_y.mean().item()
                mean_loss[k] = total_loss[k] / (i + 1)

            for m in predict_methods:
                y_pred = self.predict_after_evaluate(y_est,
                                                     batch_losses,
                                                     method=m)

                n_err[m] += (y_pred != y_test).sum().item()
                mismatched[m] += [torch.where(y_test != y_pred)[0]]
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
                                measures=measures,
                                time_per_i=time_per_i,
                                batch_size=batch_size,
                                preambule=print_result)

        self._measures = measures

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
                _reason = ''
                _reason += ' does not update' if not update_self_testing else ''
                _reason += ' {} <= {}'.format(self.trained, self.testing[m]['epochs']) if self.trained <= self.testing[m]['epochs'] else ''
                _reason += ' {} <= {}'.format(n, self.testing[m]['n']) if n <= self.testing[m]['n'] else '' 
                logging.debug(f'Accuracies not updated bc{_reason}')

        if return_mismatched:
            if only_one_method:
                return acc[m], mismatched[m]
            return acc, mismatched

        return acc[m] if only_one_method else acc


    def ood_detection_rates(self, oodsets=None,
                            testset=None,
                            ind_measures=None,
                            batch_size=100,
                            num_batch='all',
                            method='all',
                            print_result=False,
                            update_self_ood=True,
                            outputs=Outputs(),
                            log=True):

        if not testset and not ind_measures:
            testset_name=self.training['set']
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
            oodsets = [torchdl.get_dataset(n, transformer=testset.transformer)[1]
                       for n in testset.same_size]
            logging.debug('Oodsets loaded: ' + ' ; '.join(s.name for s in oodsets))

        if ind_measures:
            try:
                for m in ood_methods:
                    assert m in ind_measures
            except AssertionError:
                ind_measures=None
                logging.warning('Not all in distribution measures' 
                                ' were provided')

        device = next(self.parameters()).device

        shuffle = False
        test_n_batch = len(testset) // batch_size
        ood_n_batchs = [min(len(oodset), len(testset)) // batch_size for oodset in oodsets]
        if type(num_batch) is int:
            shuffle = True
            test_n_batch = min(num_batch, test_n_batch)
            ood_n_batchs = [min(num_batch, n) for n in ood_n_batchs]

        if oodsets:
            outputs.results(0, 0, -2, 0,
                            metrics=ood_methods,
                            acc_methods=ood_methods)
            outputs.results(0, 0, -1, 0, metrics=ood_methods,
                            acc_methods=ood_methods)

        if not ind_measures and oodsets:

            logging.debug(f'Computing measures for set {testset.name}')
            ind_measures = {m: np.ndarray(0)
                            for m in ood_methods}

            loader = torch.utils.data.DataLoader(testset,
                                                 shuffle=shuffle,
                                                 batch_size=batch_size)
            t_0 = time.time()

            iter_ = iter(loader)
            for i in range(test_n_batch):

                data = next(iter_)
                x = data[0].to(device)
                with torch.no_grad():
                    _, _, losses, _  = self.evaluate(x)
                measures = self.batch_dist_measures(None, losses, ood_methods)
                for m in ood_methods:
                    ind_measures[m] = np.concatenate([ind_measures[m],
                                                      measures[m].cpu()])
                t_i = time.time() - t_0
                t_per_i = t_i / (i + 1)
                outputs.results(i, test_n_batch, 0, 1, metrics=ood_methods,
                                measures = {m: ind_measures[m].mean()
                                            for m in ood_methods},
                                acc_methods = ood_methods,
                                time_per_i = t_per_i,
                                batch_size=batch_size,
                                preambule = testset.name)
                
        keeped_tpr = [pc / 100 for pc in range(90, 100)]
        no_result = {'epochs': 0,
                     'n': 0,
                     'auc': 0,
                     'tpr': keeped_tpr,
                     'fpr': [1 for _ in keeped_tpr],
                     'thresholds':[None for _ in keeped_tpr]}
                     
        for oodset, ood_n_batch in zip(oodsets, ood_n_batchs):
                        
            ood_results = {m: copy.deepcopy(no_result) for m in ood_methods}
            i_ood_measures = {m: ind_measures[m] for m in ood_methods}
            ood_labels = np.ones(batch_size * test_n_batch)
            fpr_ = {}
            tpr_ = {}
            thresholds_ = {}
            auc_ = {}
            r_ = {}

            loader = torch.utils.data.DataLoader(oodset,
                                                 shuffle=shuffle,
                                                 batch_size=batch_size)

            logging.debug(f'Computing measures for set {oodset.name} with {ood_n_batch} batches')

            t_0 = time.time()
            iter_ = iter(loader)
            for i in range(ood_n_batch):

                data = next(iter_)
                x = data[0].to(device)
                with torch.no_grad():
                    _, _, losses, _  = self.evaluate(x)
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


        # for m in ood_methods:

        #     if print_result:
        #         print(m, ': ', end='')
        #     result = ood_results[m]

        #     with torch.no_grad():
        # fpr, tpr, thresholds = ood_roc(self, testset, oodset,
        #                                        method=m, batch_size=batch_size,
        #                                        num_batch=num_batch,
        #                                        print_result=print_result,
        #                                        device=device)

        #     auc_ = auc(fpr, tpr)
        #     result['auc'] = auc_
        #     n = min(len(oodset), len(testset))
        #     if type(num_batch) is int:
        #         n = min(n, batch_size * num_batch)
                
        #     result.update({'epochs': self.trained,
        #                     'n': n})

        #     str_res = []
        #     for i, t  in enumerate(keeped_tpr):

        #         r_ = fpr_at_tpr(fpr, tpr, t, thresholds, True)

        #         result['fpr'][i], result['thresholds'][i] = r_
        #         str_res.append(f'{t:.0%}:{r_[0]:.2%}')

        #     if print_result:
        #         print('--'.join(str_res + [f'auc:{auc_:.2%}']))

        # if update_self_ood:

    def loss(self, x, y,
             x_reconstructed, y_estimate,
             mu_z, log_var_z,
             mse_loss_weight=None,
             x_loss_weight=None,
             kl_loss_weight=None,
             return_all_losses=False,
             **kw):

        if mse_loss_weight is None:
            mse_loss_weight = self.mse_loss_weight
        if x_loss_weight is None:
            x_loss_weight = self.x_entropy_loss_weight
        if kl_loss_weight is None:
            kl_loss_weight = self.kl_loss_weight

        batch_mse_loss = mse_loss(x, x_reconstructed,
                                  ndim=len(self.input_shape), **kw)
        
        batch_kl_loss = kl_loss(mu_z, log_var_z, **kw)

        batch_x_loss = x_loss(y, y_estimate, **kw)
                        
        batch_loss = torch.zeros_like(batch_x_loss)
        
        if mse_loss_weight > 0:
            batch_loss = batch_loss + mse_loss_weight * batch_mse_loss
        if x_loss_weight > 0:
            batch_loss = batch_loss + x_loss_weight * batch_x_loss
        if kl_loss_weight > 0:
            batch_loss += kl_loss_weight * batch_kl_loss

        """
        print('****')
        print('all', *batch_loss.shape)
        print('kl', *batch_kl_loss.shape, kl_loss_weight)
        print('mse', *batch_mse_loss.shape, mse_loss_weight)
        print('x', *batch_x_loss.shape, x_loss_weight)
        batch_loss = (mse_loss_weight * batch_mse_loss +
                      x_loss_weight * batch_x_loss +
                      kl_loss_weight * batch_kl_loss)
        """
        if return_all_losses:
            D = np.prod(self.input_shape)
            sigma = self.sigma
            batch_logpx = (- D / 2 * np.log(D * sigma * np.pi)
                           - 1 / sigma * batch_mse_loss)
            batch_elbo = batch_logpx - batch_x_loss - batch_kl_loss  
            # print('*** -logpx', -batch_logpx.mean().item() )
            return {'mse': batch_mse_loss,
                    '-logpx': -batch_logpx,
                    'x': batch_x_loss,
                    'kl': batch_kl_loss,
                    'total': batch_loss,
                    '-elbo': -batch_elbo}

        return batch_loss

    
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
              sigma=None,
              fine_tuning=False,
              latent_sampling=None,
              mse_loss_weight=None,
              x_loss_weight=None,
              kl_loss_weight=None,
              sample_size=1000,
              full_test_every=10,
              ood_detection_every=10,
              train_accuracy=False,
              save_dir=None,
              outputs=Outputs()):
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
            mse_loss_weight = self.training['mse_loss_weight']
            x_loss_weight = self.training['x_loss_weight']
            kl_loss_weight = self.training['kl_loss_weight']
            
        else:

            if trainset:
                self.training['set'] = set_name
                self.training['transformer'] = transformer
                ss = trainset.data[0].shape
                ns = self.input_shape
                logging.debug(f'Shapes : {ss} / {ns}')
                # assert ns == ss or ss == ns[1:]
        
            if sigma:
                self.sigma = sigma

            if batch_size:
                self.training['batch_size'] = batch_size

            if latent_sampling:
                self.latent_sampling = latent_sampling
            self.training['latent_sampling'] = self.latent_sampling

            self.training['x_loss_weight'] = x_loss_weight
            self.training['kl_loss_weight'] = kl_loss_weight
            self.training['mse_loss_weight'] = mse_loss_weight

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
            
        logging.debug('Creating dataloader for training with batch size %s',
                      train_batch_size)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=train_batch_size,
                                                  shuffle=True, num_workers=0)

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
        for epoch in range(done_epochs, epochs):

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

                with torch.no_grad():

                    if oodsets and ood_detection:


                        self.ood_detection_rates(oodsets=oodsets, testset=testset,
                                                 batch_size=test_batch_size,
                                                 num_batch=len(testset) // batch_size,
                                                 outputs=outputs,
                                                 print_result='*')

                        outputs.results(0, 0, -2, epochs,
                                        metrics=self.metrics,
                                        loss_components=self.loss_components,
                                        acc_methods=acc_methods)
                        outputs.results(0, 0, -1, epochs,
                                        metrics=self.metrics,
                                        loss_components=self.loss_components,
                                        acc_methods=acc_methods)
                    
                    test_accuracy = self.accuracy(testset,
                                                  batch_size=test_batch_size,
                                                  num_batch='all' if full_test else num_batch,
                                                  # device=device,
                                                  method=acc_methods,
                                                  # log=False,
                                                  outputs=outputs,
                                                  print_result='TEST' if full_test else 'test')
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

            if 'std' in train_measures and self.sigma_reach:
                sigma_n = (0.9 * self.sigma +
                           0.1 * train_measures['std'] * self.sigma_reach )
                # print('*** sigma_n ***', type(sigma_n), sigma_n)
                self.sigma = sigma_n
                # print('*** sigma_ ***', type(self._sigma), self._sigma.device, type(self.sigma))   

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
                                                         current_measures=current_measures)

                current_measures = measures
                batch_loss = batch_losses['total'].mean()

                L = batch_loss
                if self.coder_capacity_regularization:
                        L += self.encoder.dist_barrier()

                for p in self.parameters():
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        print('GRAD NAN')

                L.backward()
                # self._sigma.grad *= 1e-10
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
            if train_accuracy:
                self.train_history['train_accuracy'].append(train_accuracy)
            self.train_history['train_loss'].append(train_mean_loss)
            self.train_history['train_measures'].append(train_measures)
            self.train_history['epochs'] += 1
            self.train_history['lr'].append(self.optimizer.lr)
            self.trained += 1
            if fine_tuning:
                self.training['fine_tuning'].append(epoch)


            if save_dir:
                self.save(save_dir)

            optimizer.update_lr()

        if testset:
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
                
        logging.debug('Finished training')

    def summary(self):

        logging.warning('SUMMARY FUNCTION NOT IMPLEMENTED')

    @property
    def sigma(self):
        return self._sigma.detach().item()

    # decorator to change sigma in the decoder if changed in the vae.
    @sigma.setter
    def sigma(self, value):
        device = None
        try:
            device = self._sigma.device
            # print('*** device ***', device)
        except(AttributeError):
            # print('*** device error')
            pass
        self._sigma.data = torch.tensor(value, device=device)
        self.training['sigma'] = self.sigma
        # self._sigma.to(device)
        # print('*** device_ ***', self._sigma.device)
        
        if self.is_jvae:
            self.x_entropy_loss_weight = 2 * value
            self.kl_loss_weight = 2 * value
        elif self.is_vib:
            self.kl_loss_weight = value
        elif self.is_vae:
            self.kl_loss_weight = 2 * value

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
    def kl_loss_weight(self):
        return self._kl_loss_weight

    @kl_loss_weight.setter
    def kl_loss_weight(self, value):
        self._kl_loss_weight = value

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
        sigma = self.training['sigma']
        sampling = self.training['latent_sampling']
        if not set:
            set = self.training['set']
        s = f'{set}: {sigma:.1e} -- L={sampling} {done_epochs}/{epochs}'
        return s

    def print_architecture(self, sigma=False, sampling=False, excludes=[], short=False):

        def _l2s(l, c='-', empty='.'):
            if l:
                return c.join(str(_) for _ in l)
            return empty

        def s_(s):
            return s[0] if short else s
        
        features = None
        if self.features:
            features = self.features.name
        s = ''
        if 'type' not in excludes:
            s += s_('type') + f'={self.type}--'
        if 'activation' not in excludes:
            if not self.is_vib:
                s += s_('output') + f'={self.output_activation}--'
            s += s_('activation') + f'={self.activation}--'
        if 'latent_dim' not in excludes: 
            s += s_('latent-dim') + f'={self.latent_dim}--'
        # if sampling:
        #    s += f'sampling={self.latent_sampling}--'
        if features:
            s += s_('features') + f'={features}--'
        if 'batch_norm' not in excludes:
            w = '-both' if self.batch_norm == 'both' else ''
            s += f'batch-norm{w}--' if self. batch_norm else ''
            
        s += s_('encoder') + f'={_l2s(self.encoder_layer_sizes)}--'
        if 'decoder' not in excludes:
            s += s_('decoder') + f'={_l2s(self.decoder_layer_sizes)}--'
            if self.upsampler_channels:
                s += s_('upsampler') + f'={_l2s(self.upsampler_channels)}--'
        s += s_('classifier') + f'={_l2s(self.classifier_layer_sizes)}'

        if sigma:
            s += '--'
            s += s_('sigma')
            s += f'={self.sigma:1.2e}'

        if sampling:
            s += '--'
            s += s_('sampling')
            s += f'={self.latent_sampling}'

        return s

    def option_vector(self):

        v_ = []
        if self.features:
            w = ''
            w += 'p:'
            if self.training['pretrained_features']:
                w+= 'f'
            else:
                w+= ' '

            if self.upsampler_channels:
                if self.training['pretrained_upsampler']:
                    w += 'u'
                else:
                    w += ' '
            v_.append(w)
            
        w = 't:' + self.training.get('transformer', 'd')[0]
        v_.append(w)
        
        w = 'bn:'
        if not self.batch_norm:
            c = ' '
        else:
            # print('****', self.batch_norm)
            c = self.batch_norm[0]
        w += c
        v_.append(w)
        
        w = 'a:'
        for m in ('flip', 'crop'):
            if m in self.training['data_augmentation']:
                w += m[0]
            else: w += ' '
        v_.append(w)

        w = 'c:'
        if self.training['learned_coder']:
                w += 'l'
        else:
            w += 'r'
        _md = self.training['dictionary_min_dist']
        if _md:
            w += f'{_md:.1f}'
        
        v_.append(w)
            
        return ' '.join(v_)
    
    def save(self, dir_name=None):
        """Save the params in params.json file in the directroy dir_name and, if
        trained, the weights inweights.h5.

        """
        
        if dir_name is None:
            dir_name = os.path.join('jobs', self.print_architecture,
                                    str(self.job_number))

        save_load.save_json(self.architecture, dir_name, 'params.json')
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
        params = {'type': 'jvae',
                  'batch_norm': False
        }
        
        train_params = {'pretrained_features': None,
                        'pretrained_upsampler': None,
                        'learned_coder': False,
                        'dictionary_min_dist': None,
                        'sigma_reach': 0,
                        'data_augmentation': [],
                        'fine_tuning': [],
                        'optim': {}}

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
            pass
        
        loaded_train = False
        try:
            train_params.update(save_load.load_json(dir_name, 'train.json'))
            loaded_train = load_train
            logging.debug('Training parameters loaded')
        except(FileNotFoundError):
            pass
        
        try:
            train_history = save_load.load_json(dir_name, 'history.json')
        except(FileNotFoundError):
            train_history = {'epochs': 0}

        if not params.get('features', None):
            params['features'] = {}

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
                  sigma_reach=train_params['sigma_reach'],
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
        vae.trained = train_history['epochs']
        vae.train_history = train_history
        vae.training = train_params
        if loaded_test:
            vae.testing.update(testing)

        if load_test and loaded_ood:
            vae.ood_results = ood_results
        
        if load_state and vae.trained:
            w_p = save_load.get_path(dir_name, 'state.pth')
            try:
                state_dict = torch.load(w_p)
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

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
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
                   mse_loss_weight=None,
                   x_loss_weight=None,
                   kl_loss_weight=None,
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

