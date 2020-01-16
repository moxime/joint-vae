import tensorflow as tf
from tensorflow.keras.models import Model
# from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import mse
from tensorflow.keras.losses import categorical_crossentropy as x_entropy
# from tensorflow.keras.utils import to_categorical
import data.generate as dg
from utils import save_load 
import numpy as np
from vae_layers import Encoder, Decoder, Classifier


def __make_iter__(a):
    try:
        _ = [e for e in a]
    except TypeError:
        return [a]
    return a


DEFAULT_ACTIVATION = 'relu'
DEFAULT_OUTPUT_ACTIVATION = 'sigmoid'
DEFAULT_LATENT_SAMPLING = 1000


class ClassificationVariationalNetwork(Model):

    def __init__(self,
                 input_shape,
                 num_labels,
                 encoder_layer_sizes=[36],
                 latent_dim=4,
                 decoder_layer_sizes=[36],
                 classifier_layer_sizes=[36],
                 name = 'xy-vae',
                 activation=DEFAULT_ACTIVATION,
                 latent_sampling=DEFAULT_LATENT_SAMPLING,
                 output_activation=DEFAULT_OUTPUT_ACTIVATION,
                 beta=1e-3,
                 verbose=1,
                 *args, **kw):

        super().__init__(name=name, *args, **kw)

        # if beta=0 in Encoder(...) loss is not computed by layer
        self.encoder = Encoder(latent_dim, encoder_layer_sizes,
                               beta=beta, sampling_size=latent_sampling,
                               activation=activation)
        self.decoder = Decoder(input_shape, decoder_layer_sizes,
                               activation=activation,
                               output_activation=output_activation)
    
        self.joint = Concatenate()
        self.classifier = Classifier(num_labels,
                                     classifier_layer_sizes,
                                     activation=activation)
        self.input_dims = [input_shape]
        self.input_dims.append(num_labels)
        self.beta = beta
            
        self._sizes_of_layers = [input_shape, num_labels,
                                 encoder_layer_sizes, latent_dim,
                                 decoder_layer_sizes, classifier_layer_sizes]

        self.trained = False

        self.latent_dim = latent_dim
        self.latent_sampling = latent_sampling
        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.classifier_layer_sizes = classifier_layer_sizes
        self.activation = activation
        self.output_activation = output_activation

        self.mse_loss_weight = 1

        self.z_output = False

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

    def call(self, inputs):
        """
        inputs: [x, y] where x, and y are tensors sharing first dim.
        the loss is computed during call (no loss function is defined) 
        """

        x_input, y_input = inputs
        joint_input = self.joint(inputs)
        
        z_mean, z_log_var, z = self.encoder(joint_input)

        x_output = self.decoder(z)

        y_output = self.classifier(z)

        """The loss is computed with a mean with respect to the sampled Z.

        """
        if self.mse_loss_weight > 0:
            self.add_loss(self.mse_loss_weight *
                          tf.reduce_mean(mse(x_input, x_output), axis=0))
        if not self.z_output:
            x_output = tf.reduce_mean(x_output, 0)

        if self.x_entropy_loss_weight > 0:
            self.add_loss(self.x_entropy_loss_weight *
                          tf.reduce_mean(x_entropy(y_input, y_output), axis=0))
        if not self.z_output:
            y_output = tf.reduce_mean(y_output, 0)

        out = [x_output, y_output]
        if self.z_output:
            out += [z]
 
        return out

    def fit(self, *args, **kwargs):
        """ Just the super().fit() 
        """

        h = super().fit(*args, **kwargs)
        self.trained = True

        return h
    
    def plot_model(self, dir='.', suffix='.png', show_shapes=True,
                   show_layer_names=True):

        if dir is None:
            dir = '.'
        
        def _plot(net):
            f_p = save_load.get_path(dir, net.name+suffix)
            plot_model(net, to_file=f_p, show_shapes=show_shapes,
                       show_layer_names=show_layer_names,
                       expand_nested=True)

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
            w_p = save_load.get_path(dir_name, 'weights.h5')
            self.save_weights(w_p)

    @classmethod        
    def load(cls,
             dir_name,
             verbose=1,
             default_output_activation=DEFAULT_OUTPUT_ACTIVATION):
        """dir_name : where params.json is (and weigths.h5 if applicable)

        """
        
        p_dict = save_load.load_json(dir_name, 'params.json')

        ls = p_dict['layer_sizes']
        print(ls)

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
            vae.load_weights(w_p)

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

    def evaluate(self, x, batch_size=None, **kw):
        """for x an array of n inputs (first dim of the array) returns a n*num_labels
        array of losses

        """
        # print(f'x.shape={x.shape}')
        
        c = self.input_dims[-1] # num of classes
        n, d = np.atleast_2d(x).shape # num of inputs, dim of input
        x_ = np.vstack([x[None]] * c).reshape(n * c, d)

        i_c = np.eye(c)
        y_ = np.concatenate([np.expand_dims(i_c, axis=1)] * n,
                            axis=1).reshape( n * c, c)
        
        # print(f'n: {n} x_:{x_.shape}, y_:{y_.shape}\n')
        if batch_size is None:
            new_batch_size = c * n
        else:
            new_batch_size = c * batch_size
        losses = super().evaluate([x_, y_], batch_size=new_batch_size, **kw)
        return losses.reshape(-1, c, order='F').squeeze()

    def log_pxy(self, x, normalize=True, **kw):

        beta2pi = self.beta * 2 * np.pi
        d = x.shape[-1]
        losses = self.evaluate(x, **kw)

        c = losses.shape[-1]
        log_pxy = - losses  / (2 * self.beta)

        if normalize:
            return log_pxy

        return log_pxy - d / 2 * np.log(d * beta2pi)

    def elbo_xy_pred(self, x, normalize=True, pred_method='blind', **kw):

        """computes the elbo for x, y0 with y0 the predicted

        """

        elbo_xy = self.log_pxy(x, normalize=normalize, **kw)

        if pred_method=='max':
            return elbo_xy.max(axis=-1)

        if pred_method=='blind':
            y = self.blind_predict(x).argmax(axis=-1)
            return np.hstack([elbo_xy[i, y0] for (i, y0) in enumerate(y)])
            
    
    def log_px(self, x, normalize=True, method='sum', **kw):
        """computes a lower bound on log(p(x)) with the loss which is an
        upper bound on -log(p(x, y)).  - normalize = True forgets a
        constant (2pi sigma^2)^(-d/2) - method ='sum' computes p(x) as
        the sum of p(x, y), method='max' computes p(x) as the max_y of
        p(x, y)

        """
        
        beta2pi = self.beta * 2 * np.pi
        d = x.shape[-1]
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
    load_dir = None
    load_dir = ('./jobs/output-activation=sigmoid' +
                '/activation=relu--latent-dim=50' +
                '--sampling=100' +
                '--encoder-layers=1024-1024-512-512-256-256' +
                '--decoder-layers=256-512' +
                '--classifier-layers=10' +
                '/beta=1.00000e-06-1')
    
    # save_dir = './jobs/mnist/job5'
    save_dir = None
                  
    rebuild = load_dir is None
    # rebuild = True
    
    e_ = [1024, 1024, 512, 256]
    d_ = e_.copy()
    d_.reverse()
    c_ = [2]

    beta = 0.001
    latent_dim = 20
    latent_sampling = int(20)

    try:
        data_loaded
    except(NameError):
        data_loaded = False
    
    if not data_loaded:
        (x_train, y_train, x_test, y_test, x_ood, y_ood) = \
            dg.get_fashion_mnist(ood='mnist')
        data_loaded = True

    N = x_test.shape[0]
    C = y_test.shape[1]
    
    if not rebuild:
        try:
            vae = ClassificationVariationalNetwork.load(load_dir)
            print('Is loaded. Is trained:', vae.trained)
        except(FileNotFoundError, NameError):
            print('Not loaded, rebuilding')
            rebuild = True

    if rebuild:
        print('\n'*2+'*'*20+' BUILDING '+'*'*20+'\n'*2)
        vae = ClassificationVariationalNetwork(28**2, 10, e_,
                                               latent_dim, d_, c_,
                                               latent_sampling=latent_sampling,
                                               beta=beta) 
        # vae.plot_model(dir=load_dir)

    [x_n, y_n] = vae([x_train[0:3], y_train[0:3]])
        
    vae.compile(
        # loss = [mse, x_entropy],
        # loss_weights = [1, 2*vae.beta],
        optimizer='Adam')

    vae.summary()
    vae.encoder.summary()
    vae.decoder.summary()
    vae.classifier.summary()
    
    print('\n'*2+'*'*20+' BUILT   '+'*'*20+'\n'*2)
    
    epochs = 2
    
    refit = False
    # refit = True

    if not vae.trained or refit:
        history = vae.fit(x=[x_train, y_train],
                epochs=epochs,
                batch_size=10)
    
    x0 = np.atleast_2d(x_test[0:3])
    y0 = np.atleast_2d(y_test[0:3])

    """
    x1 = np.atleast_2d(x_test[1])
    y1 = np.atleast_2d(y_test[1])

   
    t0_ = vae.encoder(vae.joint([x0, y0]))
    mu0 = t0_[0]
    logsig0 = t0_[1]
    sig0 = np.exp(logsig0)
    t0 = t0_[2]
    # print(' -- '.join(str(i) for i in [sig0.min(), sig0.mean(), sig0.max()]))

    x_pred, y_pred = vae.predict([x_test, y_test])

    x_y_test = np.concatenate([x_test, y_test], axis=-1)
    t_enc_ = vae.encoder(x_y_test)

    t_enc = t_enc_[2]
    logsig_enc = t_enc_[1]
    sig_enc = np.exp(logsig_enc)   
    mu_enc = t_enc_[0]

    [x_dec, y_dec] = vae.decoder(t_enc)
    """

    """
    acc = vae.accuracy(x_test, y_test)
    print(f'test accuracy: {acc}\n')
    """
    
    if save_dir is not None:
        vae.save(save_dir)

    beta = vae.beta
    
    loss = vae.evaluate(x0)
    elbo_xy0 = vae.elbo_xy_pred(x0)
    elbo_xy = vae.log_pxy(x0)
    log_px = vae.log_px(x0)

    vae.kl_loss_weight = 1e-60
    vae.mse_loss_weight = 0
    vae.x_entropy_loss_weight = 1
    vae.compile()
    x_loss = vae.evaluate(x0)

    # vae = ClassificationVariationalNetwork.load(tmp_dir)
    vae.kl_loss_weight = 1
    vae.mse_loss_weight = 0
    vae.x_entropy_loss_weight = 1e-60

    vae.compile()
    kl_loss = vae.evaluate(x0) 
    vae.kl_loss_weight = 1e-60
    vae.mse_loss_weight = 1
    vae.x_entropy_loss_weight = 0

    vae.compile()
    mse_loss = vae.evaluate(x0)
    

    loss_ = mse_loss + 2*beta*kl_loss + 2*beta*x_loss

    # input()
    print(f'mse_loss = {mse_loss}')
    print(f'x_loss = {x_loss}')
    print(f'kl_loss = {kl_loss}')
    print(f'loss_ = {loss_}')

    b = vae.beta
    vae.kl_loss_weight = 2 * b
    vae.mse_loss_weight = 1
    vae.x_entropy_loss_weight = 2 * b

    vae.z_output = True
    vae.compile()
    
    [x_, y_, z] = vae([x0, y0])
        
    from tensorflow.keras.losses import mse, categorical_crossentropy as x_entropy

    mse_ = mse(x0, x_)
    xent_ = x_entropy(y0, y_)

    mse_mean = mse_.numpy().mean(axis=0)
    xent_mean = xent_.numpy().mean(axis=0)

    """

    """
    print('elbo_xy:')
    print(elbo_xy)
    
    print('elbo_xy0:\n', elbo_xy0)
    print('log_px:\n',log_px)

    
