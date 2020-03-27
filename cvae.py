from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
# from torch.nn import functional as F
from utils.losses import x_loss, kl_loss, mse_loss

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from vae_layers import VGGFeatures, ConvDecoder, Encoder, Decoder, Classifier
from vae_layers import onehot_encoding

import data.torch_load as torchdl
from data.torch_load import choose_device
from utils import save_load
import numpy as np

from utils.print_log import print_results

import time


DEFAULT_ACTIVATION = 'relu'
# DEFAULT_OUTPUT_ACTIVATION = 'sigmoid'
DEFAULT_OUTPUT_ACTIVATION = 'linear'
DEFAULT_LATENT_SAMPLING = 100


class ClassificationVariationalNetwork(nn.Module):

    predict_methods = ['mean', 'loss']

    def __init__(self,
                 input_shape,
                 num_labels,
                 features=None,
                 pretrained_features=None,
                 encoder_layer_sizes=[36],
                 latent_dim=32,
                 decoder_layer_sizes=[36],
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

        if features:
            if pretrained_features:
                vgg_dict = torch.load(pretrained_features)
            else:
                vgg_dict = None

            self.features = VGGFeatures(features, input_shape,
                                        pretrained=vgg_dict)

            dense_input_shape = self.features.output_shape
        else:
            dense_input_shape = input_shape
            self.features = None

        self.encoder = Encoder(dense_input_shape, num_labels, latent_dim,
                               encoder_layer_sizes,
                               beta=beta, sampling_size=latent_sampling,
                               activation=activation)
        if features:
            self.decoder = ConvDecoder(latent_dim, dense_input_shape,
                                       decoder_layer_sizes,
                                       output_activation=output_activation)
        else:
            self.decoder = Decoder(latent_dim, input_shape,
                                   decoder_layer_sizes,
                                   activation=activation,
                                   output_activation=output_activation)

        self.classifier = Classifier(latent_dim, num_labels,
                                     classifier_layer_sizes,
                                     activation=activation)

        self.input_shape = tuple(input_shape)
        self.num_labels = num_labels
        self.input_dims = (input_shape, num_labels)

        self.beta = beta

        self._sizes_of_layers = [input_shape, num_labels,
                                 encoder_layer_sizes, latent_dim,
                                 decoder_layer_sizes, classifier_layer_sizes]

        self.trained = False
        # self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.train_history = dict()

        self.latent_dim = latent_dim
        self.latent_sampling = latent_sampling
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.classifier_layer_sizes = classifier_layer_sizes
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
        if not x_features:
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

        x_output = self.decoder(z)
        # x_output of size LxN1x...xKgxD

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
              mse_loss_weight=None,
              x_loss_weight=None,
              kl_loss_weight=None,
              resume_training=False,
              sample_size=1000,
              train_accuracy=False,
              save_dir=None,
              verbose=1):
        """

        """
        methods = self.predict_methods
        device = choose_device(device)

        if optimizer is None:
            optimizer = self.optimizer
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=0)

        dataset_size = len(trainset)
        remainder = 1 if (dataset_size % batch_size) > 0 else 0
        per_epoch = dataset_size // batch_size + remainder

        if not resume_training or not self.trained:
            self.train_history = dict()  # will not be returned
            self.train_history['train_loss'] = []
            self.train_history['train_x_loss'] = []
            self.train_history['train_mse_loss'] = []
            self.train_history['train_kl_loss'] = []
            self.train_history['test_accuracy'] = [] if testset else None
            self.train_history['train_accuracy'] = []
            self.train_history['train_loss'] = []
        done_epochs = len(self.train_history['train_loss'])

        print_results(0, 0, -2, epochs)
        print_results(0, 0, -1, epochs)        

        for epoch in range(done_epochs, epochs):

            t_start_epoch = time.time()
            # test

            if testset:
                num_batch = sample_size // batch_size
                with torch.no_grad():
                    test_accuracy = self.accuracy(testset,
                                                  batch_size=batch_size,
                                                  num_batch='all',
                                                  device=device,
                                                  method='all',
                                                  print_result='test')
                self.train_history['test_accuracy'].append(test_accuracy)

            # train
            if train_accuracy:
                with torch.no_grad():
                    train_accuracy = self.accuracy(trainset,
                                                   batch_size=batch_size,
                                                   num_batch='all',
                                                   device=device,
                                                   method='all',
                                                   print_result='acc')

                self.train_history['train_accuracy'].append(train_accuracy)
                
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

            self.train_history['train_loss'].append(train_mean_loss)
            if save_dir:
                self.save(save_dir)

        print('\nFinished Training')
        self.trained = trainset.name

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

    def print_architecture(self, beta=False):

        def _l2s(l, c='-', empty='.'):
            if len(l) == 0:
                return empty
            return c.join(str(_) for _ in l)

        s = f'output-activation={self.output_activation}--'
        s += f'activation={self.activation}--'
        s += f'latent-dim={self.latent_dim}--'
        s += f'sampling={self.latent_sampling}--'
        s += f'encoder-layers={_l2s(self.encoder_layer_sizes)}--'
        s += f'decoder-layers={_l2s(self.decoder_layer_sizes)}--'
        s += f'classifier-layers={_l2s(self.classifier_layer_sizes)}'

        if beta:
            s += f'--beta={beta:1.3e}'

        return s

    def save(self, dir_name=None):
        """Save the params in params.json file in the directroy dir_name and, if
        trained, the weights inweights.h5.

        """
        ls = self._sizes_of_layers

        if dir_name is None:
            dir_name = './jobs/' + self.print_architecture()

        param_dict = {'layer_sizes': self._sizes_of_layers, 'trained':
                      self.trained, 'beta': self.beta,
                      'latent_sampling': self.latent_sampling,
                      'activation': self.activation,
                      'output_activation': self.output_activation}

        save_load.save_json(param_dict, dir_name, 'params.json')

        if self.trained:
            save_load.save_json(self.train_history, dir_name, 'training.json')
            w_p = save_load.get_path(dir_name, 'state.pth')
            torch.save(self.state_dict(), w_p)

    @classmethod
    def load(cls, dir_name, verbose=1,
             default_output_activation=DEFAULT_OUTPUT_ACTIVATION):
        """dir_name : where params.json is (and weigths.h5 if applicable)

        """
        p_dict = save_load.load_json(dir_name, 'params.json')

        try:
            train_history = save_load.load_json(dir_name, 'training.json')
        except(FileNotFoundError):
            train_history = dict()

        latent_sampling = p_dict.get('latent_sampling', 1)
        output_activation = p_dict.get('output_activation',
                                       default_output_activation)

        ls = p_dict['layer_sizes']
        # print(ls)

        vae = cls(ls[0], ls[1], ls[2], ls[3], ls[4], ls[5],
                  latent_sampling=latent_sampling,
                  activation=p_dict['activation'],
                  beta=p_dict['beta'],
                  output_activation=output_activation,
                  verbose=verbose)

        vae.trained = p_dict['trained']
        vae.train_history = train_history

        if vae.trained:
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
    default_dataset = 'cifar10'
    default_epochs = 50
    
    parser = argparse.ArgumentParser(description="train a network")

    parser.add_argument('--dataset', default=default_dataset,
                        choices=['fashion', 'mnist', 'cifar10'])

    parser.add_argument('--epochs', type=int, default=default_epochs)

    parser.add_argument('-b', '--batch_size', type=int,
                        default=default_batch_size)

    parser.add_argument('-K', '--latent_dim', type=int,
                        default=default_latent_dim)
    parser.add_argument('-L', '--latent_sampling', type=int,
                        default=default_latent_sampling)
    
    args = parser.parse_args()
    
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    latent_dim = args.latent_dim
    latent_sampling = args.latent_sampling
    
    save_dir = './jobs/features/cifar10/job-2'
    load_dir = None
    # load_dir = save_dir

    resume_training = True
    resume_training = False
    refit = True
    refit = False

    rebuild = load_dir is None
    # rebuild = True

    e_ = [512, 256]
    # e_ = []
    d_ = e_.copy()
    d_ = [32, 32, 32, 32, 3]
    c_ = [20, 10]

    beta = 1e-9

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('*** USED DEVICE', device, '***')
    trainset, testset = torchdl.get_cifar10()
    _, oodset = torchdl.get_svhn()

    # output_activation = 'linear'  # 'sigmoid'
    output_activation = 'sigmoid'
    data_loaded = True

    if not rebuild:
        print('*** LOADING... ***')
        try:
            jvae = ClassificationVariationalNetwork.load(load_dir)
            print(f'*** NETWORK LOADED',
                  f'{"AND" if jvae.trained else "BUT NOT"}',
                  'TRAINED ***')
        except(FileNotFoundError, NameError) as err:
            print(f'*** NETWORK NOT LOADED -- REBUILDING bc of {err} ***')
            rebuild = True

    if rebuild:
        t = time.time()
        print('*'*4+' BUILDING '+'*'*4)
        jvae = ClassificationVariationalNetwork((3, 32, 32), 10,
                                                features='vgg11',
                                                pretrained_features='vgg11.pth',
                                                encoder_layer_sizes=e_,
                                                latent_dim=latent_dim,
                                                latent_sampling=latent_sampling,
                                                decoder_layer_sizes=d_,
                                                classifier_layer_sizes=c_,
                                                beta=beta,
                                                output_activation=output_activation)
        print('*'*4 + f' BUILT' + '*'*4)

    print(jvae.print_architecture())
    epochs = 50
    # batch_size = 7

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    test_batch = next(iter(testloader))
    x, y = test_batch[0].to(device), test_batch[1].to(device)

    jvae.to(device)

    print('TRAINING\n\n')
    if not jvae.trained or refit or resume_training:

        jvae.train(trainset, epochs=epochs,
                   batch_size=batch_size,
                   device=device,
                   testset=testset,
                   resume_training=resume_training,
                   sample_size=200,  # 10000,
                   mse_loss_weight=None,
                   x_loss_weight=None,
                   kl_loss_weight=None,
                   save_dir=save_dir)

    if save_dir is not None:
        jvae.save(save_dir)

    """
    x_, y_, mu, lv, z = jvae(x, y)
    x_reco, y_out, batch_losses = jvae.evaluate(x)
    
    y_est_by_losses = batch_losses.argmin(0)
    y_est_by_mean = y_out.mean(0).argmax(-1)
    """

    def timing(vae, x, N=1):
        from utils.print_log import Time
    
        t0 = time.time()

        for n in range(N):
            _ = vae.evaluate(x)

        t1 = time.time()
        print('Evaluate:', (Time(t1) - t0) / N / len(x))

        t0 = t1

        for n in range(N):

            u = vae.features(x)
              
        t1 = time.time()
        print('Features:', (Time(t1) - t0) / N / len(x))
