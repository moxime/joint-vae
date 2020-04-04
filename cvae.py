import torch
import torch.utils.data
from torch import nn, optim
# from torch.nn import functional as F
from utils.losses import x_loss, kl_loss, mse_loss
import argparse, configparser
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from vae_layers import VGGFeatures, ConvDecoder, Encoder, Decoder, Classifier, ConvFeatures
from vae_layers import onehot_encoding

import data.torch_load as torchdl
from data.torch_load import choose_device
from utils import save_load
import numpy as np

from utils.print_log import print_results

import os.path
import time


DEFAULT_ACTIVATION = 'relu'
# DEFAULT_OUTPUT_ACTIVATION = 'sigmoid'
DEFAULT_OUTPUT_ACTIVATION = 'linear'
DEFAULT_LATENT_SAMPLING = 100


activation_layers = {'linear': nn.Identity,
                     'sigmoid': nn.Sigmoid, 
                     'relu': nn.ReLU} 


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


    predict_methods = ['mean', 'loss']

    def __init__(self,
                 input_shape,
                 num_labels,
                 features=None,
                 pretrained_features=None,
                 features_channels=None,
                 encoder_layer_sizes=[36],
                 latent_dim=32,
                 decoder_layer_sizes=[36],
                 upsampler_channels=None,
                 classifier_layer_sizes=[36],
                 name='joint-vae',
                 activation=DEFAULT_ACTIVATION,
                 latent_sampling=DEFAULT_LATENT_SAMPLING,
                 output_activation=DEFAULT_OUTPUT_ACTIVATION,
                 beta=1e-6,
                 verbose=1,
                 *args, **kw):

        super().__init__(*args, **kw)
        self.name = name

        # no upsampler if no features
        assert (not upsampler_channels or features)

        features_arch = {'features': features,
                         'features_channels': features_channels,
                         'pretrained_features': pretrained_features}

        if features:
            if pretrained_features:
                feat_dict = torch.load(pretrained_features)
            else:
                feat_dict = None

            if features.startswith('vgg'):
                self.features = VGGFeatures(features, input_shape,
                                            pretrained=feat_dict)
            elif features == 'conv':
                self.features = ConvFeatures(input_shape, features_channels)
            encoder_input_shape = self.features.output_shape
        else:
            encoder_input_shape = input_shape
            self.features = None

        self.encoder = Encoder(encoder_input_shape, num_labels, latent_dim,
                               encoder_layer_sizes,
                               beta=beta, sampling_size=latent_sampling,
                               activation=activation)

        activation_layer = activation_layers[activation]()
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
            self.imager = ConvDecoder(imager_input_dim,
                                      upsampler_first_shape,
                                      upsampler_channels,
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
        self.input_dims = (input_shape, num_labels)

        self.beta = beta

        self._sizes_of_layers = [input_shape, num_labels,
                                 encoder_layer_sizes, latent_dim,
                                 decoder_layer_sizes,
                                 upsampler_channels,
                                 classifier_layer_sizes]

        self.architecture = {'input': input_shape,
                             'labels': num_labels,
                             'features': features_arch, 
                             'encoder': encoder_layer_sizes,
                             'activation': activation,
                             'latent_dim': latent_dim,
                             'decoder': decoder_layer_sizes,
                             'upsampler': upsampler_channels,
                             'classifier': classifier_layer_sizes,
                             'output': output_activation}

        self.trained = 0
        self.training = None # {'beta': beta, 'sampling': latent_sampling}
        
        # self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.train_history = {'epochs': 0}

        self.latent_dim = latent_dim
        self.latent_sampling = latent_sampling
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.classifier_layer_sizes = classifier_layer_sizes
        self.upsampler_channels = upsampler_channels
        self.activation = activation
        self.output_activation = output_activation

        self.mse_loss_weight = 1

        self.z_output = False

    def forward(self, x, y, x_features=None, **kw):
        """inputs: x, y where x, and y are tensors sharing first dims.

        - x is of size N1x...xNgxD1x..xDt
        - y is of size N1x....xNg(x1)

        """

        if not self.features:
            x_features = x
        if x_features is None:
            x_features = self.features(x.view(-1, *self.input_shape))

        return self.forward_features(x_features, y, **kw)
            
    def forward_features(self, x_features, y, z_output=True):

        batch_shape = x_features.shape
        batch_size = batch_shape[:-len(self.encoder.input_shape)]  # N1 x...xNg
        reco_batch_shape = batch_size + self.input_shape
        
        x_ = x_features.view(*batch_size, -1)  # x_ of size N1x...xNgxD

        y_onehot = onehot_encoding(y, self.num_labels).float()

        z_mean, z_log_var, z = self.encoder(x_, y_onehot)
        # z of size LxN1x...xNgxK

        u = self.decoder(z)
        # x_output of size LxN1x...xKgxD
        x_output = self.imager(u)
        
        y_output = self.classifier(z)
        # y_output of size LxN1x...xKgxC

        out = (x_output.reshape((self.latent_sampling,) + reco_batch_shape),
               y_output)
        if z_output:
            out += (z_mean, z_log_var, z)

        return out

    def evaluate(self, x, **kw):
        """x input of size (N1, .. ,Ng, D1, D2,..., Dt) 

        creates a x of size C * N1, ..., D1, ...., Dt)
        and a y of size C * N1 * ... * Ng

        ----- Returns 

        x_ (C,N1,..., D1...) tensor,

        y_est (C,N1,...,C) tensor, 

        batch_losses (C, N1,...N) tensor

        """

        if self.features:
            x_features = self.features(x)
        else:
            x_features = x
            
        # build a C* N1* N2* Ng *D1 * Dt tensor of input x_features

        C = self.num_labels
        s_f = x_features.shape
        
        s_f = (1, ) + s_f
        rep_dims = (C, ) + tuple([1 for _ in s_f[1:]])

        f_repeated = x_features.reshape(s_f).repeat(rep_dims)

        # create a C * N1 * ... * Ng y tensor y[c,:,:,:...] = c

        s_y = f_repeated.shape[:-len(self.input_shape)]

        y = torch.zeros(s_y, dtype=int, device=x.device)

        for c in range(C):
            y[c] = c  # maybe a way to accelerate this ?

        if self.features:
            x_reco, y_est, mu, log_var, z = self.forward_features(f_repeated, y)
        else:
            x_reco, y_est, mu, log_var, z = self.forward(f_repeated, y)
            
        batch_losses = self.loss(x, y,
                                 x_reco, y_est,
                                 mu, log_var,
                                 batch_mean=False, **kw)

        return x_reco.mean(0), y_est.mean(0), batch_losses

    def predict(self, x, method='mean', **kw):
        """x input of size (N1, .. ,Ng, D1, D2,..., Dt) 

        creates a x of size C * N1, ..., D1, ...., Dt)
        and a y of size C * N1 * ... * Ng

        - method: If 'mean'(default) output is of size N1 *...* and
        gives y predicted. If None output is C * N1 *... and gives
        p(y|x,y). If 'loss' returns the y which minimizes loss(x, y)

        """

        _, y_est, batch_losses = self.evaluate(x)

        # print('cvae l. 192', x.device, batch_losses.device)
        return self.predict_after_evaluate(y_est, batch_losses, method=method)

    def predict_after_evaluate(self, y_est, losses, method='mean'):

        if method is None:
            return y_est

        if method == 'mean':
            return y_est.mean(0).argmax(-1)

        if method == 'loss':
            return losses.argmin(0)

    def accuracy(self, testset, batch_size=100, num_batch='all', method='mean',
                 device=None, return_mismatched=False, print_result=False):
        """return detection rate. If return_mismatched is True, indices of
        mismatched are also retuned.
        method can be a list of methods

        """
        if method == 'all':
            method = self.predict_methods
        if type(method) is str:
            methods = [method]
            only_one_method = True
        else:
            methods = method
            only_one_method = False

        shuffle = True
        if num_batch == 'all':
            num_batch = len(testset) // batch_size
            shuffle = False

        if device is None:
            has_cuda = torch.cuda.is_available
            device = torch.device('cuda' if has_cuda else 'cpu')

        self.to(device)
        testloader = torch.utils.data.DataLoader(testset,
                                                 shuffle=shuffle,
                                                 batch_size=batch_size)
        n_err = dict()
        mismatched = dict()
        acc = dict()
        for m in methods:
            n_err[m] = 0
            mismatched[m] = []
        n = 0.0
        iter_ = iter(testloader)
        start = time.time()

        mean_loss = {'mse': 0, 'x': 0., 'kl': 0., 'total': 0.} 
        total_loss = mean_loss.copy()
        for i in range(num_batch):
            data = next(iter_)
            x_test, y_test = data[0].to(device), data[1].to(device)
            _, y_est, batch_losses = self.evaluate(x_test,
                                                   return_all_losses=True)
            for k in batch_losses:
                total_loss[k] += batch_losses[k].mean().item()
                mean_loss[k] = total_loss[k] / (i + 1)
            for m in methods:
                y_pred = self.predict_after_evaluate(y_est,
                                                     batch_losses['total'],
                                                     method=m)
                n_err[m] += (y_pred != y_test).sum().item()
                mismatched[m] += [torch.where(y_test != y_pred)[0]]
            n += len(y_test)
            time_per_i = (time.time() - start) / (i + 1)
            for m in methods:
                acc[m] = 1 - n_err[m] / n
            if print_result:
                print_results(i, num_batch, 0, 0,
                              losses=mean_loss, accuracies=acc,
                              time_per_i=time_per_i,
                              batch_size=batch_size,
                              preambule=print_result)

        if return_mismatched:
            if only_one_method:
                return acc[m], mismatched[m]
            return acc, mismatched

        return acc[m] if only_one_method else acc

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
        batch_x_loss = x_loss(y, y_estimate, **kw)
        batch_kl_loss = kl_loss(mu_z, log_var_z, **kw)

        batch_loss = (mse_loss_weight * batch_mse_loss +
                      x_loss_weight * batch_x_loss +
                      kl_loss_weight * batch_kl_loss)

        if return_all_losses:
            return {'mse': batch_mse_loss, 'x': batch_x_loss,
                    'kl': batch_kl_loss, 'total': batch_loss}

        return batch_loss

    def train(self, trainset, optimizer=None, epochs=50,
              batch_size=100, device=None,
              testset=None,
              beta=None,
              latent_sampling=None,
              mse_loss_weight=None,
              x_loss_weight=None,
              kl_loss_weight=None,
              sample_size=1000,
              train_accuracy=False,
              save_dir=None,
              verbose=1):
        """

        """
        try:
            set_name = trainset.name
        except(AttributeError):
            set_name = trainset.__str__().splitlines()[0].split()[-1].lower()

        if beta:
            self.beta = beta
        
        if latent_sampling:
            self.latent_sampling = latent_sampling
        
        device = choose_device(device)

        if optimizer is None:
            optimizer = self.optimizer
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=0)

        dataset_size = len(trainset)
        remainder = (dataset_size % batch_size) > 0 
        per_epoch = dataset_size // batch_size + remainder

        done_epochs = self.train_history['epochs']
        if done_epochs == 0:
            self.train_history = {'epochs': 0}  # will not be returned
            self.train_history['train_loss'] = []
            self.train_history['test_accuracy'] = []
            self.train_history['train_accuracy'] = [] 
            self.training = {'beta': self.beta,
                             'sampling': self.latent_sampling,
                             'set': set_name,
                             'epochs': epochs}
        else:
            assert self.beta == self.training['beta']
            assert self.latent_sampling == self.training['sampling']
            
        print_results(0, 0, -2, epochs)
        print_results(0, 0, -1, epochs)

        for epoch in range(done_epochs, epochs):

            t_start_epoch = time.time()
            # test

            num_batch = sample_size // batch_size
            if testset:
                # print(num_batch, sample_size)
                with torch.no_grad():
                    test_accuracy = self.accuracy(testset,
                                                  batch_size=batch_size,
                                                  num_batch=num_batch,
                                                  device=device,
                                                  method='all',
                                                  print_result='test')
            # train
            if train_accuracy:
                with torch.no_grad():
                    train_accuracy = self.accuracy(trainset,
                                                   batch_size=batch_size,
                                                   num_batch=num_batch,
                                                   device=device,
                                                   method='all',
                                                   print_result='acc')

                
            t_i = time.time()
            t_start_train = t_i
            train_mean_loss = {'mse': 0.0, 'x': 0.0, 'kl': 0.0, 'total': 0.0}
            train_total_loss = train_mean_loss.copy()
            
            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                x, y = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                x_reco, y_est, mu_z, log_var_z, z = self.forward(x, y)

                batch_loss = self.loss(x, y, x_reco, y_est, mu_z,
                                       log_var_z,
                                       mse_loss_weight=mse_loss_weight,
                                       x_loss_weight=x_loss_weight,
                                       kl_loss_weight=kl_loss_weight,
                                       return_all_losses=True)
                batch_loss['total'].backward()
                optimizer.step()

                for k in batch_loss:
                    train_total_loss[k] += batch_loss[k].item()
                    train_mean_loss[k] = train_total_loss[k] / (i + 1)

                t_per_i = (time.time() - t_start_train) / (i + 1)
                print_results(i, per_epoch, epoch + 1, epochs,
                              preambule='train',
                              losses=train_mean_loss,
                              time_per_i=t_per_i,
                              batch_size=batch_size,
                              end_of_epoch='\n')

            if testset:
                self.train_history['test_accuracy'].append(test_accuracy)
            if train_accuracy:
                self.train_history['train_accuracy'].append(train_accuracy)
            self.train_history['train_loss'].append(train_mean_loss)
            self.train_history['epochs'] += 1
            self.trained += 1
            if save_dir:
                self.save(save_dir)

        print('\nFinished Training')

    def summary(self):

        print('SUMMARY FUNCTION NOT IMPLEMENTED')

    @property
    def beta(self):
        return self._beta

    # decorator to change beta in the decoder if changed in the vae.
    @beta.setter
    def beta(self, value):
        self._beta = value
        self.encoder.beta = value
        self.kl_loss_weight = 2 * value
        self.x_entropy_loss_weight = 2 * value

    @property
    def kl_loss_weight(self):
        return self._kl_loss_weight

    @kl_loss_weight.setter
    def kl_loss_weight(self, value):
        self._kl_loss_weight = value
        self.encoder.kl_loss_weight = value

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
            print(f'PLOT HAS TO BE IMPLEMENTED WITH TB')
            # f_p = save_load.get_path(dir, net.name+suffix)
            # plot_model(net, to_file=f_p, show_shapes=show_shapes,
            #            show_layer_names=show_layer_names,
            #            expand_nested=True)

        _plot(self)
        _plot(self.encoder)
        _plot(self.decoder)
        _plot(self.classifier)

    def has_same_architecture(self, other_net):

        out = True

        out = out and self.activation == other_net.activation
        # print(out)
        out = out and self._sizes_of_layers == other_net._sizes_of_layers
        # print(out)
        out = out and self.latent_sampling == other_net.latent_sampling
        # print(out)
        return out

    def print_architecture(self, beta=False, sampling=False):

        def _l2s(l, c='-', empty='.'):
            if l:
                return c.join(str(_) for _ in l)
            return empty

        features = None
        if self.features:
            features = self.features.name

        s = f'output={self.output_activation}--'
        s += f'activation={self.activation}--'
        s += f'latent-dim={self.latent_dim}--'
        if sampling:
            s += f'sampling={self.latent_sampling}--'
        if features:
            s += f'features={features}--'
        s += f'encoder={_l2s(self.encoder_layer_sizes)}--'
        s += f'decoder={_l2s(self.decoder_layer_sizes)}--'
        if self.upsampler_channels:
            s += f'upsampler={_l2s(self.upsampler_channels)}--'
        s += f'classifier={_l2s(self.classifier_layer_sizes)}'

        if beta:
            s += f'--beta={self.beta:1.2e}'

        return s

    def save(self, dir_name=None):
        """Save the params in params.json file in the directroy dir_name and, if
        trained, the weights inweights.h5.

        """
        
        if dir_name is None:
            dir_name = './jobs/' + self.print_architecture()

        save_load.save_json(self.architecture, dir_name, 'params.json')
        save_load.save_json(self.training, dir_name, 'train.json')
        save_load.save_json(self.train_history, dir_name, 'history.json')
        
        if self.trained:
            w_p = save_load.get_path(dir_name, 'state.pth')
            torch.save(self.state_dict(), w_p)

    @classmethod
    def load(cls, dir_name,
             load_state = True,
             verbose=1,
             default_output_activation=DEFAULT_OUTPUT_ACTIVATION):
        """dir_name : where params.json is (and weigths.h5 if applicable)

        """
        params = save_load.load_json(dir_name, 'params.json')
        train_params = save_load.load_json(dir_name, 'train.json')

        if load_state:
            try:
                train_history = save_load.load_json(dir_name, 'history.json')
            except(FileNotFoundError):
                train_history = {'epochs': 0}
        else:
            train_history = {'epochs': 0}

        if not train_params:
            train_params = {'beta': 1e-4,
                            'sampling': 64}

        if not params['features']:
            params['features'] = {}
            
        vae = cls(input_shape=params['input'],
                  num_labels=params['labels'],
                  encoder_layer_sizes=params['encoder'],
                  latent_dim=params['latent_dim'],
                  decoder_layer_sizes=params['decoder'],
                  classifier_layer_sizes=params['classifier'],
                  latent_sampling=train_params['sampling'],
                  activation=params['activation'],
                  beta=train_params['beta'],
                  upsampler_channels=params['upsampler'],
                  output_activation=params['output'],
                  verbose=verbose,
                  **params['features'])

        vae.trained = train_history['epochs']
        vae.train_history = train_history
        vae.training = train_params

        if load_state and vae.trained:
            w_p = save_load.get_path(dir_name, 'state.pth')
            vae.load_state_dict(torch.load(w_p))

        return vae

    def log_pxy(self, x, normalize=True, batch_losses=None, **kw):

        if batch_losses is None:
            _, _, batch_losses = self.evaluate(x, **kw)

        normalized_log_pxy = - batch_losses / (2 * self.beta)

        if normalize:
            return normalized_log_pxy

        D = np.prod(self.input_shape)
        a = np.log(self.sigma * 2 * np.pi)
        return normalized_log_pxy - D / 2 * a

    def log_px(self, x, normalize=True, method='sum', batch_losses=None, **kw):
        """Computes a lower bound on log(p(x)) with the loss which is an upper
        bound on -log(p(x, y)).  - normalize = True forgets a constant
        (2pi sigma^2)^(-d/2) - method ='sum' computes p(x) as the sum
        of p(x, y), method='max' computes p(x) as the max_y of p(x, y)

        """
        if batch_losses is None:
            _, _, batch_losses = self.evaluate(x, **kw)

        log_pxy = - batch_losses / (2 * self.beta)

        m_log_pxy = log_pxy.max(0)[0]
        d_log_pxy = log_pxy - m_log_pxy

        # p_xy = d_p_xy * m_pxy
        d_pxy = d_log_pxy.exp()
        if method == 'sum':
            d_px = d_pxy.sum(0)
        elif method == 'max':
            d_px = d_pxy.max(0)

        normalized_log_px = d_px.log() + m_log_pxy
        if normalize:
            return normalized_log_px
        else:
            D = np.prod(self.input_shape)
            a = np.log(self.sigma * 2 * np.pi)
            return normalized_log_px - D / 2 * a

    def log_py_x(self, x, batch_losses=None, **kw):

        if batch_losses is None:
            _, _, batch_losses = self.evaluate(x, **kw)

        # batch_losses is C * N1 * ... *
        log_pxy = - batch_losses / (2 * self.beta)

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

    default_latent_dim = 40
    default_latent_sampling = 50
    default_batch_size = 50
    default_test_sample_size = 2000
    default_dataset = 'cifar10'
    default_epochs = 50
    default_beta = 1e-4
    default_job_dir = './jobs'

    config = configparser.ConfigParser()
    config.read('config.ini')

    used_config = config['DEFAULT']
    # used_config = config['svhn-vgg16']
    used_config = config['fashion-conv']
    # used_config = config['dense']
    # used_config = config['test']
    # used_config = config['autoencoder']
    
    for k in used_config:
        print(k, used_config[k])
    
    dataset = used_config['dataset']
    transformer = used_config['transformer']
    
    epochs = int(used_config['epochs'])
    batch_size = int(used_config['batch_size'])
    test_sample_size = int(used_config['test_sample_size'])
    latent_dim = int(used_config['latent_dim'])
    latent_sampling = int(used_config['latent_sampling'])
    output_activation = used_config['output_activation']
    beta = used_config.getfloat('beta')

    features = used_config.get('features', None)
    if features.lower() == 'none':
        features = None

    ls = used_config.get('features_channels', '')
    features_channels = [int(l) for l in ls.split()]        

    upsampler = used_config.get('upsampler', None)
    if upsampler.lower() == 'none':
        upsampler = ''

    encoder = [int(l) for l in used_config['encoder'].split()]
    decoder = [int(l) for l in used_config['decoder'].split()]
    upsampler = [int(l) for l in upsampler.split()]
    classifier = [int(l) for l in used_config['classifier'].split()]
    
    job_dir = default_job_dir

    load_dir = None
    save_dir = load_dir
    rebuild = load_dir is None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Used device:', device)
    trainset, testset = torchdl.get_dataset(dataset, transformer=transformer)
    _, oodset = torchdl.get_svhn()

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)

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
        except(FileNotFoundError, NameError) as err:
            print(f'*** NETWORK NOT LOADED -- REBUILDING bc of {err} ***')
            rebuild = True

    if rebuild:
        print('Building network...', end=' ')
        jvae = ClassificationVariationalNetwork(input_shape, num_labels,
                                                features=features,
                                                features_channels=features_channels,
                                                # pretrained_features='vgg11.pth',
                                                encoder_layer_sizes=encoder,
                                                latent_dim=latent_dim,
                                                latent_sampling=latent_sampling,
                                                decoder_layer_sizes=decoder,
                                                upsampler_channels=upsampler,
                                                classifier_layer_sizes=classifier,
                                                beta=beta,
                                                output_activation=output_activation)

        """
        if not save_dir:
            save_dir_root = os.path.join(job_dir, dataset,
                                         jvae.print_architecture(),
                                         f'{beta:.2e}')
            i = 0
            save_dir = os.path.join(save_dir_root, f'{i:02d}')
            while os.path.exists(save_dir):
                i += 1
                save_dir = os.path.join(save_dir_root, f'{i:02d}')
                
        print('done.', 'Will be saved in\n' + save_dir)
        """
        
    print(jvae.print_architecture())
    print(jvae.print_architecture(True, True))

    jvae.to(device)

    out = jvae(x, y)

    for o in out:
        print(o.shape)

    jvae.save('/tmp')

    jvae2 = ClassificationVariationalNetwork.load('/tmp')
    
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
                   save_dir='/tmp', **kw)

    # train_the_net(100, latent_sampling=128, beta=1e-4)
    # jvae3 = ClassificationVariationalNetwork.load('/tmp')
    
    """
    for net in (jvae, jvae2, jvae3):
        print(net.print_architecture())
    for net in (jvae, jvae2, jvae3):
        print(net.print_architecture(True, True))

    print(jvae.training)
    train_the_net(2, latent_sampling=3, beta=2e-5)
    if save_dir is not None:
        jvae.save(save_dir)

    x_, y_, mu, lv, z = jvae(x, y)
    x_reco, y_out, batch_losses = jvae.evaluate(x)
    
    y_est_by_losses = batch_losses.argmin(0)
    y_est_by_mean = y_out.mean(0).argmax(-1)
    """

