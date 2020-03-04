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
import data.torch_load as torchdl
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

    def forward(self, x, y=None, z_output=True):
        """inputs: x, y where x, and y are tensors sharing first dims.

        - x is of size N1x...xNgxD1x..xDt
        - y is of size N1x....xNg(x1)

        """
        shape = x.shape
        shape_ = shape[:-len(self.input_shape)] + (-1,)
        x_ = x.reshape(*shape_) # x_ of size N1x...xNgxD with D=D1*...Dt

        # print('cvae l. 100 y', y.shape)
        y_onehot = onehot_encoding(y, self.num_labels).float()
        # print('cvae l. 102 y', y.shape)
        
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

    def evaluate(self, x, **kw):
        """
        x input of size (N1, .. ,Ng, D1, D2,..., Dt) 

        creates a x of size C * N1, ..., D1, ...., Dt)
        and a y of size C * N1 * ... * Ng

        """

        # build a C* N1* N2* Ng *D1 * Dt tensor of input X

        C = self.num_labels
        s_x = x.shape
        if s_x[0] != 1:
            s_x = (1, ) + s_x
        s_x_ = (C, )  + tuple([1 for _ in s_x[1:]]) 

        x_ = x.reshape(s_x).repeat(s_x_)

        # create a C * N1 * ... * Ng y tensor y[c,:,:,:...] = c

        s_y = x_.shape[:-len(self.input_shape)]

        # print('cvae l. 428', 's_y', s_y)
        # y = torch.zeros(s_y, requires_grad=False, dtype=int, device=x.device)
        y = torch.zeros(s_y, dtype=int, device=x.device)
        # print('cvae l. 430', 'y', y.shape)
        for c in range(C):
            y[c] = c # maybe a way to accelerate this ?

        # print('cva l. 433', x_.shape, x_.dtype, y.shape, s_y, y.dtype)
        x_reco, y_est, mu, log_var, z = self.forward(x_, y)

        # print('cvae l. 436 x_', x_.shape, 'y', y.shape, '\nx_r', x_reco.shape,
        #       'y_est', y_est.shape, 'mu', mu.shape, 'log_var',
        #       log_var.shape, 'z', z.shape) 
        batch_loss = self.loss(x_, y, x_reco, y_est, mu, log_var, batch_mean=False)

        return x_reco.mean(0), y_est.mean(0), batch_loss

    def predict(self, x, method='mean', **kw):

        """x input of size (N1, .. ,Ng, D1, D2,..., Dt) 

        creates a x of size C * N1, ..., D1, ...., Dt)
        and a y of size C * N1 * ... * Ng

        - method: If 'mean'(default) output is of size N1 *...* and
        gives y predicted. If None output is C * N1 *... and gives
        p(y|x,y). If 'loss' returns the y which minimizes loss(x, y)

        """
        
        _, y_est, loss = self.evaluate(x)

        if method is None:
            return y_est

        if method == 'mean':
            return y_est.mean(0).argmax(-1)

        if method == 'loss':
            return loss.argmin(0)
            
    def accuracy(self, testset, batch_size=100, num_batch='all',
                 device=None, return_mismatched=False):
        """return detection rate. If return_mismatched is True, indices of
        mismatched are also retuned.

        """
        shuffle = True
        if num_batch == 'all':
            num_batch = len(testset) // batch_size
            shuffle = False

        if device is None:
            has_cuda = torch.cuda.is_available
            device = torch.device('cuda' if has_cuda else 'cpu')

        testloader = torch.utils.data.DataLoader(testset,
                                                 shuffle=shuffle,
                                                 batch_size=batch_size)
        n_err = 0
        n = 0.0
        mismatched = []
        iter_ = iter(testloader)
        for i in range(num_batch):
            data = next(iter_)
            x_test, y_test = data[0].to(device), data[1].to(device)
            y_pred = self.predict(x_test)
            n_err += (y_pred != y_test).sum().item()
            n += len(y_test)

            mismatched += [torch.where(y_test != y_pred)[0]]

        acc = 1 - n_err / n
        if return_mismatched:
            return acc, mismatched

        return acc
        
    def loss(self, x, y,
             x_reconstructed, y_estimate,
             mu_z, log_var_z,
             mse_loss_weight=None,
             x_loss_weight=None,
             kl_loss_weight=None, **kw):

        # print('cvae l. 137 | y', y.shape)
        if mse_loss_weight is None:
            mse_loss_weight = self.mse_loss_weight
        if x_loss_weight is None:
            x_loss_weight = self.x_entropy_loss_weight
        if kl_loss_weight is None:
            kl_loss_weight = self.kl_loss_weight

        # print('cvae l. 134 | kl_loss_weight', kl_loss_weight)
        # w_kl = kl_loss_weight * kl_loss(mu_z, log_var_z, **kw)
        # print('cvae l. 136 | x', x.shape)
        # print('cvae l. 138 | w*kl', w_kl.shape)
        # print('cvae l. 139 | x_loss', x_loss(y, y_estimate, **kw).shape)
        # print('cvae l. 140 | mse_loss',
        # mse_loss(x, x_reconstructed,
        #         ndim=len(self.input_shape), **kw).shape)

        return (mse_loss_weight *
                mse_loss(x, x_reconstructed,
                         ndim=len(self.input_shape), **kw) +
                x_loss_weight *
                x_loss(y, y_estimate, **kw) +
                kl_loss_weight *
                kl_loss(mu_z, log_var_z, **kw))        

    def train(self, trainset, optimizer=None, epochs=50,
              batch_size=100, device=None,
              testset=None,
              mse_loss_weight=None,
              x_loss_weight=None,
              kl_loss_weight=None,
              verbose=1):
        """

        """
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        if optimizer is None: optimizer = self.optimizer
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        # if testset:
        #     testloader = torch.utils.data.DataLoader(testset,
        #                                              batch_size=batch_size,
        #                                              shuffle=False,
        #                                              num_workers=0)
  
        dataset_size = len(trainset)
        per_epoch = dataset_size // batch_size
        self.train_history = dict() # will not be returned
        self.train_history['train batch loss'] = []
        self.train_history['train batch xent loss'] = []
        self.train_history['train batch mse loss'] = []
        self.train_history['train batch kl loss'] = []
        self.train_history['train loss'] = []
        self.train_history['test accuracy'] = [] if testset else None
        self.train_history['train accuracy'] = [] 
            
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
                                         mse_loss_weight=mse_loss_weight,
                                         kl_loss_weight=kl_loss_weight,
                                         x_loss_weight=x_loss_weight).item()
            self.train_history['train batch loss'].append(first_batch_loss)
            self.train_history['train batch xent loss'].append(batch_x_loss)
            self.train_history['train batch mse loss'].append(batch_mse_loss)
            self.train_history['train batch kl loss'].append(batch_kl_loss)
            train_accuracy = self.accuracy(trainset,
                                           batch_size=batch_size,
                                           num_batch=1,
                                           device=device)
            self.train_history['train accuracy'].append(train_accuracy)

            if testset:
                test_accuracy = self.accuracy(testset,
                                              batch_size=batch_size,
                                              num_batch=1,
                                              device=device)
                self.train_history['test accuracy'].append(test_accuracy)
            acc = test_accuracy if testset else train_accuracy
            print(f'epoch {epoch + 1:2d}/{epochs} 1st batch ',
                  f'mse: {batch_mse_loss:.2e} kl: {batch_kl_loss:.2e} ',
                  f'x: {batch_x_loss:.2e} L: {first_batch_loss:.2e}',
                  f'acc: {100*acc:.2f} %',
                  f'({"test" if testset else "train"})')

            t_i = time.time()
            for i, data in enumerate(trainloader, 0):
                tick = time.time()
                t_per_i = tick - t_i
                t_epoch = tick - t_start_epoch
                mus = 1e6 * t_epoch/(i * batch_size) if i>0 else 0
                info = f'{mus:.0f} us per sample'
                t_i = tick
                mean_loss = epoch_total_loss / i if i > 0 else 0
                print_epoch(i, per_epoch, epoch, epochs, mean_loss,
                            info=info)
                # get the inputs; data is a list of [inputs, labels]
                x, y = data[0].to(device),data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                x_reco, y_est, mu_z, log_var_z, z = self.forward(x, y)
                              
                batch_loss = self.loss(x, y, x_reco, y_est, mu_z,
                                       log_var_z,
                                       mse_loss_weight=mse_loss_weight,
                                       x_loss_weight=x_loss_weight,
                                       kl_loss_weight=kl_loss_weight) 
                batch_loss.backward()
                optimizer.step()
                epoch_total_loss += batch_loss.item()
                self.train_history['train loss'].append(epoch_total_loss)
                
        print('Finished Training')
        self.trained = str(trainset)
                
    def summary(self):
                              
        print('SUMMARY FUNCTION NOT IMPLEMENTED')
                              
    @property
    def beta(self):
        return self._beta
                              
    @beta.setter # decorator to change beta in the decoder if changed in the vae.
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
                              
    def log_pxy(self, x, normalize=True, losses=None, **kw):
                              
        if losses is None:
            _, _, losses = self.evaluate(x, **kw)
                              
        log_pxy = - losses  / self.beta
                              
        if normalize:
            return log_pxy
                              
        return log_pxy - np.prod(self.input_shape) / 2 * np.log(beta2pi)
                              
    def elbo_xy_pred(self, x, normalize=True, pred_method='blind', **kw):
        """computes the elbo for x, y0 with y0 the predicted

        """
        print('TO BE DONE... OR NOT')
        elbo_xy = self.log_pxy(x, normalize=normalize, **kw)
        
        if pred_method=='max':
            return elbo_xy.max(axis=-1)
        
        if pred_method=='blind':
            y = self.blind_predict(x).argmax(axis=-1)
            return np.hstack([elbo_xy[i, y0] for (i, y0) in enumerate(y)])
                              
    def log_px(self, x, normalize=True, method='sum', losses=None, **kw):
        """TO BE DONE computes a lower bound on log(p(x)) with the loss which is an upper
        bound on -log(p(x, y)).  - normalize = True forgets a constant
        (2pi sigma^2)^(-d/2) - method ='sum' computes p(x) as the sum
        of p(x, y), method='max' computes p(x) as the max_y of p(x, y)

        """
        if losses is None:
            losses = self.evaluate(x, **kw)
                              
            log_pxy = - losses  / self.beta
                              
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

        print('NOT IMPLEMENTED')
        if True:
            return None
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
                              
                              
if __name__ == '__main__':
                              
    # load_dir = './jobs/mnist/job5'
    # load_dir = './jobs/fashion-mnist/latent-dim=20-sampling=100-encoder-layers=3/beta=5.00000e-06-0'
    # load_dir = ('./jobs/output-activation=sigmoid' +
    #             '/activation=relu--latent-dim=50' +
    #             '--sampling=100' +
    #             '--encoder-layers=1024-1024-512-512-256-256' +
    #             '--decoder-layers=256-512' +
    #             '--classifier-layers=10' +
    #             '/beta=1.00000e-06-1')
    
    save_dir = './jobs/fashion/job-1'
    load_dir = save_dir
    # load_dir = None
    # save_dir = None
                              
    rebuild = load_dir is None
    rebuild = True
                              
    e_ = [1024, 512, 512]
    # e_ = []
    d_ = e_.copy()
    d_.reverse()
    c_ = [20, 10]
                              
    beta = 1e-5
    latent_dim = 20
    latent_sampling = 50
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('*** USED DEVICE', device, '***')
    # try:
    #     data_loaded
    # except(NameError):
    #     data_loaded = False
    
    data_loaded = False
    if not data_loaded:
        # mnist_transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.1307,), (0.3081,))])
        #
                              
        trainset, testset = torchdl.get_fashion_mnist()        
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
        jvae = ClassificationVariationalNetwork((1, 28, 28), 10,
                                                e_, latent_dim,
                                                d_, c_,
                                                latent_sampling=latent_sampling,
                                                beta=beta,
                                                output_activation=output_activation) 
        print('*'*4 + f' BUILT in {(time.time() -t) * 1e3:.0f} ms  ' + '*'*4)

    print(jvae.print_architecture())
    epochs = 50
    batch_size = 200
            
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    test_batch = next(iter(testloader))
    x, y = test_batch[0].to(device), test_batch[1].to(device)

    refit = False
    # refit = True

    jvae.to(device)

    if not jvae.trained or refit:
        jvae.train(trainset, epochs=epochs,
                   batch_size=batch_size,
                   device=device,
                   testset=testset,
                   mse_loss_weight=None,
                   x_loss_weight=None,
                   kl_loss_weight=None)

    if save_dir is not None:
        jvae.save(save_dir)
        
    x_, y_, mu, lv, z = jvae(x, y)

    # jvae.latent_sampling = 20
    print(f'latent_sampling: {jvae.latent_sampling}')
    x_reco, y_out, batch_losses = jvae.evaluate(x)

    y_est_by_losses = batch_losses.argmin(0)
    y_est_by_mean = y_out.mean(0).argmax(-1)
                              
