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
from vae_layers import Encoder, Decoder, Classifier, onehot_encoding

import data.generate as dg
from utils import save_load 
import numpy as np

from utils.print_log import print_epoch

import time

DEFAULT_ACTIVATION = 'relu'
# DEFAULT_OUTPUT_ACTIVATION = 'sigmoid'
DEFAULT_OUTPUT_ACTIVATION = 'linear'
DEFAULT_LATENT_SAMPLING = 100


class ClassificationVariationalNetwork(nn.Module):

    def __init__(self,
                 input_shape,
                 num_labels,
                 encoder_layer_sizes=[36],
                 latent_dim=32,
                 decoder_layer_sizes=[36],
                 classifier_layer_sizes=[36],
                 name = 'joint-vae',
                 activation=DEFAULT_ACTIVATION,
                 latent_sampling=DEFAULT_LATENT_SAMPLING,
                 output_activation=DEFAULT_OUTPUT_ACTIVATION,
                 beta=1e-6,
                 verbose=1,
                 *args, **kw):

        super().__init__(*args, **kw)
        self.name = name

        # if beta=0 in Encoder(...) loss is not computed by layer
        self.encoder = Encoder(input_shape, num_labels, latent_dim,
                               encoder_layer_sizes,
                               beta=beta, sampling_size=latent_sampling,
                               activation=activation)

        self.decoder = Decoder(latent_dim, input_shape,
                               decoder_layer_sizes,
                               activation=activation,
                               output_activation=output_activation)
    
        self.classifier = Classifier(latent_dim, num_labels,
                                     classifier_layer_sizes,
                                     activation=activation)

        self.input_shape = input_shape
        self.num_labels = num_labels
        self.input_dims = (input_shape, num_labels)

        self.beta = beta
            
        self._sizes_of_layers = [input_shape, num_labels,
                                 encoder_layer_sizes, latent_dim,
                                 decoder_layer_sizes, classifier_layer_sizes]

        self.trained = False
        # self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        
        self.latent_dim = latent_dim
        self.latent_sampling = latent_sampling
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.classifier_layer_sizes = classifier_layer_sizes
        self.activation = activation
        self.output_activation = output_activation

        self.mse_loss_weight = 1

        self.z_output = False

    def forward(self, x, y=None, z_output=True):
        """inputs: x, y where x, and y are tensors sharing first dims.

        - x is of size N1x...xNgxD1x..xDt
        - y is of size N1x....xNg(x1)

        """
        shape = x.size()
        shape_ = shape[:-len(self.input_shape)] + (-1,)
        x_ = x.reshape(*shape_) # x_ of size N1x...xNgxD with D=D1*...Dt
        
        y_onehot = onehot_encoding(y, self.num_labels).float()

        # y_onehot of size N1x...xNgxC with
        
        z_mean, z_log_var, z = self.encoder(x_, y_onehot)
        # z_mean and z_log_var of size N1...NgxK
        # z of size LxN1x...xNgxK
        
        x_output = self.decoder(z)
        # x_output of size LxN1x...xKgxD
        
        y_output = self.classifier(z)
        # y_output of size LxN1x...xKgxC
        
        out = (x_output.reshape((self.latent_sampling,)+shape), y_output)
        if z_output:
            out += (z_mean, z_log_var, z)
 
        return out
    
    def loss(self, x, y,
             x_reconstructed, y_estimate,
             mu_z, log_var_z,
             mse_loss_weight=None,
             x_loss_weight=None,
             kl_loss_weight=None, **kw):

        if mse_loss_weight is None: mse_loss_weight = self.mse_loss_weight
        if x_loss_weight is None: x_loss_weight = self.x_entropy_loss_weight
        if kl_loss_weight is None: kl_loss_weight = self.kl_loss_weight

        return (mse_loss_weight * mse_loss(x, x_reconstructed, **kw) +
                x_loss_weight * x_loss(y, y_estimate, **kw) +
                kl_loss_weight * kl_loss(mu_z, log_var_z, **kw))        

    def train(self, trainset, optimizer=None, epochs=50,
              batch_size=64, device=None, x_loss_weight=None,
              kl_loss_weight=None , verbose=1):
        """

        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        if optimizer is None: optimizer = self.optimizer
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        dataset_size = trainset.data.shape[0]
        per_epoch = dataset_size // batch_size
        for epoch in range(epochs):
            t_start_epoch = time.time()
            epoch_total_loss = 0.0            
            data = next(iter(trainloader))
            x, y = data[0].to(device), data[1].to(device)
            x_reco, y_est, mu_z, log_var_z, z = self.forward(x, y)
            batch_kl_loss = kl_loss(mu_z, log_var_z).item()
            batch_x_loss = x_loss(y, y_est).item()
            batch_mse_loss = mse_loss(x, x_reco).item()
            first_batch_loss = self.loss(x, y, x_reco, y_est, mu_z,
                                         log_var_z,
                                         kl_loss_weight=kl_loss_weight,
                                         x_loss_weight=x_loss_weight).item()
            print(f'epoch {epoch + 1:2d}/{epochs} 1st batch ' + 
                  f'mse: {batch_mse_loss:.2e} kl: {batch_kl_loss:.2e} ' + 
                  f'x: {batch_x_loss:.2e} L: {first_batch_loss:.2e}')
            t_i = time.time()
            for i, data in enumerate(trainloader, 0):
                tick = time.time()
                t_per_i = tick - t_i
                t_epoch = tick - t_start_epoch
                info = f'{1e6 * t_epoch/(i * batch_size + 1e-100):.0f} us per sample'
                t_i = tick
                print_epoch(i, per_epoch, epoch, epochs,
                            epoch_total_loss / (i + 1e-100), info=info)
                # get the inputs; data is a list of [inputs, labels]
                x, y = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                x_reco, y_est, mu_z, log_var_z, z = self.forward(x, y)

                batch_loss = self.loss(x, y, x_reco, y_est, mu_z,
                                       log_var_z,
                                       x_loss_weight=x_loss_weight,
                                       kl_loss_weight=kl_loss_weight) 
                batch_loss.backward()
                optimizer.step()
                epoch_total_loss += batch_loss.item()

                # print statistics
                # running_loss += loss.item()
                # if i % 2000 == 1999:    # print every 2000 mini-batches
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, i + 1, running_loss / 2000))
                #     running_loss = 0.0

        print('Finished Training')
        self.trained = True

                
    def summary(self):

        print('SUMMARY FUNCTION NOT IMPLEMENTED')
                
    @property
    def beta(self):
        return self._beta

    @beta.setter # decorator to change beta in the decoder if changed
                 # in the vae.
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
        out = out and self._sizes_of_layers == other_net._sizes_of_layers
        out = out and self.latent_sampling == other_net.latent_sampling
        
        return out

    def print_architecture(self, beta=False):

        def _l2s(l, c='-', empty='.'):
            if len(l) == 0:
                return empty
            return c.join(str(_) for _ in l)

        s  = f'output-activation={self.output_activation}--' 
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
        """Save the params in params.json file in the directroy dir_name and,
        if trained, the weights inweights.h5.

        """

        ls = self._sizes_of_layers
        
        if dir_name is None:
            dir_name = './jobs/' + self.print_architecture()
            
        param_dict = {'layer_sizes': self._sizes_of_layers,
                      'trained': self.trained,
                      'beta': self.beta,
                      'latent_sampling': self.latent_sampling,
                      'activation': self.activation,
                      'output_activation': self.output_activation
                      }

        save_load.save_json(param_dict, dir_name, 'params.json')

        if self.trained:
            w_p = save_load.get_path(dir_name, 'state.pth')
            torch.save(self.state_dict(), w_p)


    @classmethod        
    def load(cls,
             dir_name,
             verbose=1,
             default_output_activation=DEFAULT_OUTPUT_ACTIVATION):
        """dir_name : where params.json is (and weigths.h5 if applicable)

        """
        
        p_dict = save_load.load_json(dir_name, 'params.json')

        latent_sampling = p_dict.get('latent_sampling', 1)
        output_activation = p_dict.get('output_activation',
                                       default_output_activation)
        
        vae = cls(ls[0], ls[1], ls[2], ls[3], ls[4], ls[5],
                  latent_sampling=latent_sampling,
                  activation=p_dict['activation'],
                  beta=p_dict['beta'],
                  output_activation=output_activation,
                  verbose=verbose)

        vae.trained = p_dict['trained']

        _input = np.ndarray((1, ls[0]))

        # call the network once to instantiate it
        if ls[1] is not None:
            _input = [_input, np.ndarray((1, ls[1]))]
        _ = vae(_input)
        vae.summary()

        if vae.trained:
            w_p = save_load.get_path(dir_name, 'weights.h5')
            vae.load_state_dict(torch.load(w_p))

        return vae

    def naive_predict(self, x,  verbose=1):
        """for x a single input find y for which log P(x, y) is maximum
        (actually the ELBO < log P(x,y) is maximum)

        """
        
        x = np.atleast_2d(x)
        assert x.shape[0] == 1
        assert len(self.input_dims) > 1
        num_labels = self.input_dims[-1]

        y_ = np.eye(num_labels)

        loss_ = np.inf

        for i in range(num_labels):

            y = np.atleast_2d(y_[:,i])
            loss = super.evaluate([x, y], verbose=verbose)
            if loss < loss_:
                i_ = i
                loss_ = loss

        return i_, loss_

    def naive_evaluate(self, x, verbose=0):
        """for x an input or a tensor or an array of inputs computes the
        losses (and returns as a list) for each possible y.

        """
        num_labels = self.input_dims[-1]
        y_ = np.eye(num_labels)

        losses = []

        x2d = np.atleast_2d(x)
        
        for y in y_:
            y2d = np.atleast_2d(y)
            loss = super().evaluate([x2d, y2d], verbose=verbose)
            losses.append(loss)

        return losses

    def evaluate(self, x, **kw):
        """
        x input of size (N1, .. ,Ng, D1, D2,..., Dt) 
        x_output of size (L, N1, ..., Ng, D1, D2,..., Dt) where L is sampling size, 
        """

        # build a C* N1* N2* Ng *D1 * Dt tensor of input X

        C = self.num_labels
        s_x = x.shape
        if s_x[0] != 1:
            s_x = (1, ) + s_x
        s_x_ = (C, )  + s_x[1:] 

        x_ = x.reshape(s_x).repeat(s_x_)

        # create a C * N1 * ... * Ng y tensor y[c,:,:,:...] = c

        s_y = s_x_[:-len(self.input_shape)]
        
        y = torch.zeros(s_y)
        for c in range(C):
            y[c] = c # maybe a way to accelrate this ?

        return self.forward(x_, y_)

    def log_pxy(self, x, normalize=True, losses=None, **kw):

        beta2pi = self.beta * 2 * np.pi
        d = x.shape[-1]

        if losses is None:
            losses = self.evaluate(x, **kw)

        c = losses.shape[-1]
        log_pxy = - losses  / (2 * self.beta)

        if normalize:
            return log_pxy

        return log_pxy - d / 2 * np.log(d * beta2pi)

    def elbo_xy_pred(self, x, normalize=True, pred_method='blind', **kw):

        """computes the elbo for x, y0 with y0 the predicted

        """

        elbo_xy = self.log_pxy(np.atleast_2d(x), normalize=normalize, **kw)

        if pred_method=='max':
            return elbo_xy.max(axis=-1)

        if pred_method=='blind':
            y = self.blind_predict(x).argmax(axis=-1)
            return np.hstack([elbo_xy[i, y0] for (i, y0) in enumerate(y)])
            
    
    def log_px(self, x, normalize=True, method='sum', losses=None, **kw):
        """computes a lower bound on log(p(x)) with the loss which is an
        upper bound on -log(p(x, y)).  - normalize = True forgets a
        constant (2pi sigma^2)^(-d/2) - method ='sum' computes p(x) as
        the sum of p(x, y), method='max' computes p(x) as the max_y of
        p(x, y)

        """
        
        beta2pi = self.beta * 2 * np.pi
        d = x.shape[-1]
        if losses is None:
            losses = self.evaluate(x, **kw)

        c = losses.shape[-1]
        log_pxy = - losses  / (2 * self.beta)

        m_log_pxy = log_pxy.max(axis=-1)
        mat_log_pxy = np.hstack([np.atleast_1d(m_log_pxy)[:, None]] * c)
        d_log_pxy = log_pxy - mat_log_pxy

        # p_xy = d_p_xy * m_pxy 
        d_pxy = np.exp(d_log_pxy)
        if method == 'sum':
            d_px = d_pxy.sum(axis=-1)
        elif method == 'max':
            d_px = d_pxy.max(axis=-1)
        else: # mean
            p_y = self.blind_predict(x)
            d_px = (d_pxy * p_y).sum(axis=-1)
            
        if normalize:
            return np.squeeze(np.log(d_px) + m_log_pxy)
        else:
            return np.squeeze(np.log(d_px) + m_log_pxy
                              - d / 2 * np.log(d * beta2pi))
        
    def log_py_x(self, x, losses=None, **kw):

        d = x.shape[-1]
        if losses is None:
            losses = self.evaluate(x, **kw)

        c = losses.shape[-1]
        log_pxy = - losses  / (2 * self.beta)

        m_log_pxy = log_pxy.max(axis=-1)
        mat_log_pxy = np.hstack([np.atleast_1d(m_log_pxy)[:, None]] * c)
        d_log_pxy = log_pxy - mat_log_pxy

        d_pxy = np.exp(d_log_pxy)
        d_log_px = np.log(d_pxy.sum(axis=-1))

        log_py_x = d_log_pxy - np.hstack([np.atleast_1d(d_log_px)[:, None]]
                                         * c)

        return log_py_x

    def HY_x(self, x, method='pred', y_pred=None, losses=None, **kw):

        if method=='pred':
            if y_pred is None:
                y_pred = self.blind_predict(x)
            return -(np.log(y_pred) * y_pred).sum(axis=-1)
        
        log_p = self.log_py_x(x, losses=losses, **kw)
        return -(np.exp(log_p) * log_p).sum(axis=-1)
        
        
    def naive_call(xy_vae, x):
        """for a single input x returns [x_, y_] estimated by the network ofr
        every possible input [x, y]

        """

        x = np.atleast_2d(x)
        assert len(xy_vae.input_dims) > 1
        assert x.shape[0] == 1
        num_labels = xy_vae.input_dims[-1]

        y = np.eye(num_labels)
        x = np.vstack([x]*num_labels)

        x_, y_ = xy_vae([x, y])

        return x_, y_

    def blind_predict(self, x):

        x_n = np.atleast_2d(x)
        num_labels = self.input_dims[-1]
        y = np.ones((x_n.shape[0], num_labels)) / num_labels
        # print('Y SHAPE=', y.shape)
        [x_, y_] = super().predict([x_n, y])

        return y_

    def accuracy(self, x_test, y_test, return_mismatched=False):
        """return detection rate. If return_mismatched is True, indices of
        mismatched are also retuned.

        """

        y_pred = self.blind_predict(x_test).argmax(axis=-1)
        y_test_ = y_test.argmax(axis=-1)

        n = len(y_test_)

        mismatched = np.argwhere(y_test_ != y_pred)
        acc = 1 - len(mismatched)/n

        if return_mismatched:
            return acc, mismatched

        return acc
        

if __name__ == '__main__':

    # load_dir = './jobs/mnist/job5'
    # load_dir = './jobs/fashion-mnist/latent-dim=20-sampling=100-encoder-layers=3/beta=5.00000e-06-0'
    load_dir = ('./jobs/output-activation=sigmoid' +
                '/activation=relu--latent-dim=50' +
                '--sampling=100' +
                '--encoder-layers=1024-1024-512-512-256-256' +
                '--decoder-layers=256-512' +
                '--classifier-layers=10' +
                '/beta=1.00000e-06-1')
    
    load_dir = None
    # save_dir = './jobs/mnist/job5'
    save_dir = None
                  
    rebuild = load_dir is None
    # rebuild = True
    
    e_ = [512, 512, 256]
    # e_ = []
    d_ = e_.copy()
    d_.reverse()
    c_ = [20, 10]

    beta = 1e-4
    latent_dim = 40
    latent_sampling = 100

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # try:
    #     data_loaded
    # except(NameError):
    #     data_loaded = False

    
    data_loaded = False
    if not data_loaded:
        mnist_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        #
        # mnist_transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Lambda(lambda x: x / 255.0)])

        
        
        # transform = transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(root='./data',
                                              train=True,
                                              download=True,
                                              transform=mnist_transform)
        data_loaded = True
        
    if not rebuild:
        try:
            vae = ClassificationVariationalNetwork.load(load_dir)
            print('Is loaded. Is trained:', vae.trained)
        except(FileNotFoundError, NameError):
            print('Not loaded, rebuilding')
            rebuild = True

    if rebuild:
        t = time.time()
        print('*'*20+' BUILDING '+'*'*20+'\n')
        jvae = ClassificationVariationalNetwork((1, 28, 28), 10, e_,
                                               latent_dim, d_, c_,
                                               latent_sampling=latent_sampling,
                                               beta=beta) 
        # vae.plot_model(dir=load_dir)

        
    
    print('*'*20 + f' BUILT in {(time.time() -t) * 1e3:.0f} ms  ' + '*'*20)
    
    epochs = 1
    batch_size = 500

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                      shuffle=True, num_workers=0)

    batch = next(iter(trainloader))
    x, y = batch[0].to(device), batch[1].to(device)
    
    refit = False
    # refit = True

    jvae.to(device)
    
    if not jvae.trained or refit:
        jvae.train(trainset,
                   epochs=epochs,
                   batch_size=batch_size,
                   device=device,
                   x_loss_weight=None,
                   kl_loss_weight=None)
    
    if save_dir is not None:
        jvae.save(save_dir)
