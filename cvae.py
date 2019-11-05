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
from vae_layers import Encoder, Decoder


def __make_iter__(a):
    try:
        _ = [e for e in a]
    except TypeError:
        return [a]
    return a


DEFAULT_ACTIVATION = 'relu'
DEFAULT_LATENT_SAMPLING = 1000

class ClassificationVariationalNetwork(Model):

    def __init__(self,
                 input_shape,
                 num_labels=None,
                 encoder_layer_sizes=[36],
                 latent_dim=4,
                 decoder_layer_sizes=[36],
                 name = 'xy-vae',
                 activation=DEFAULT_ACTIVATION,
                 latent_sampling=DEFAULT_LATENT_SAMPLING,
                 beta=1e-3,
                 verbose=1,
                 *args, **kw):

        super().__init__(name=name, *args, **kw)

        # if beta=0 in Encoder(...) loss is not computed by layer
        self.encoder = Encoder(latent_dim, encoder_layer_sizes,
                               beta=beta, sampling_size=latent_sampling,
                               activation=activation)
        self.decoder = Decoder(input_shape, num_labels,
                               decoder_layer_sizes, activation=activation)

        self.x_y = num_labels is not None
        if self.x_y:
            self.joint = Concatenate()
        self.input_dims = [input_shape]
        self.input_dims.append(num_labels if self.x_y else 0)
        self.beta = beta
            
        self._sizes_of_layers = [input_shape, num_labels,
                                 encoder_layer_sizes, latent_dim,
                                 decoder_layer_sizes]

        self.trained = False

        self.latent_sampling = latent_sampling
        self.activation = activation

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        self.encoder.beta = value

    def call(self, inputs):
        
        if self.x_y:
            num_labels = self.input_dims[-1]
            x_input = inputs[0]
            y_input = inputs[1]

            """ A REVOIR 
            if isinstance(y_input, int):
                y_input = tf.keras.utils.to_categorical(y_input, num_labels)
            if y_input.ndim==1:
                if len(y_input) == x_input.shape[0]:
                    y_input = tf.keras.utils.to_categorical(y_input, num_labels)
                else:
                    assert(len(y_input) == num_labels and
                           ((y_input == 0) + (y_input == 1)).all())
            """
            joint_input = self.joint([x_input, y_input])
        else:
            joint_input = inputs
            x_input = np.atleast_2d(inputs)

        z_mean, z_log_var, z = self.encoder(joint_input)

        reconstructed = self.decoder(z)

        if self.x_y:
            x_output = reconstructed[0]
            y_output = reconstructed[1]
        else:
            x_output = reconstructed

        self.add_loss(tf.reduce_mean(mse(x_input, x_output)))

        if self.x_y and self.beta>0:
            self.add_loss(2*self.beta*tf.reduce_mean(x_entropy(y_input, y_output)))
        
        return reconstructed

    def fit(self, *args, **kwargs):

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
        
    def save(self, dir_name=None):

        ls = self._sizes_of_layers
        
        if dir_name is None:
            dir_name = ('./jobs/' + str(ls[0]) + '-' + str(ls[1])
                        + '-[' + ','.join([str(_) for _ in ls[2]])
                        + ']-' + str(ls[3]) + '-['
                        + ','.join([str(_) for _ in ls[4]])
                        + ']')

        param_dict = {'layer_sizes': self._sizes_of_layers,
                      'trained': self.trained,
                      'beta': self.beta,
                      'latent_sampling': self.latent_sampling,
                      'activation': self.activation
                      }

        save_load.save_json(param_dict, dir_name, 'params.json')

        if self.trained:
            w_p = save_load.get_path(dir_name, 'weights.h5')
            self.save_weights(w_p)

    @classmethod        
    def load(cls, dir_name, verbose=1):

        p_dict = save_load.load_json(dir_name, 'params.json')

        ls = p_dict['layer_sizes']
        print(ls)

        if 'latent_sampling' in p_dict.keys():
            latent_sampling = p_dict['latent_sampling']
        else:
            sapling_size = 1
        
        vae = cls(ls[0], ls[1], ls[2], ls[3], ls[4],
                  latent_sampling=latent_sampling,
                  activation=p_dict['activation'],
                  beta=p_dict['beta'], verbose=verbose)

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

        x = np.atleast_2d(x)
        assert x.shape[0] == 1
        assert len(self.input_dims) > 1
        num_labels = self.input_dims[-1]

        y_ = np.eye(num_labels)

        loss_ = np.inf

        for i in range(num_labels):

            y = np.atleast_2d(y_[:,i])
            loss = self.evaluate([x, y], verbose=verbose)
            if loss < loss_:
                i_ = i
                loss_ = loss

        return i_, loss_


    def naive_evaluate(self, x):

        num_labels = self.input_dims[-1]
        y_ = np.eye(num_labels)

        losses = []

        for y in y_:

            losses.append(self.evaluate([np.atleast_2d(x), np.atleast_2d(y)]))

        return losses
        
    
    def naive_call(xy_vae, x):

        x = np.atleast_2d(x)
        assert len(xy_vae.input_dims) > 1
        assert x.shape[0] == 1
        num_labels = xy_vae.input_dims[-1]

        y = np.eye(num_labels)
        x = np.vstack([x]*num_labels)

        x_, y_ = xy_vae([x, y])

        return x_, y_

    def blind_predict(self, x):

        num_labels = self.input_dims[-1]
        y = np.ones((x.shape[0], num_labels)) / num_labels
        [x_, y_] = super().predict([x, y])

        return y_

    def accuracy(self, x_test, y_test):

        y_pred = self.blind_predict(x_test).argmax(axis=-1)
        y_test_ = y_test.argmax(axis=-1)

        n = len(y_test_)

        return sum(y_test_ == y_pred)/n
        
if __name__ == '__main__':

    load_dir = None
    # load_dir = './jobs/mnist/job5'

    # save_dir = './jobs/mnist/job5'
    save_dir = None
                  
    rebuild = load_dir is None
    # rebuild = True
    
    e_ = [1024, 1024]
    d_ = e_.copy()
    d_.reverse()

    beta = 0.001
    latent_dim = 100
    latent_sampling = int(1e3)

    try:
        data_loaded
    except(NameError):
        data_loaded = False
    
    if not data_loaded:
        (x_train, y_train, x_test, y_test) = dg.get_fashion_mnist()
        data_loaded = True
        
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
                                               latent_dim, # d_,
                                               latent_sampling=latent_sampling,
                                               beta=beta) 
        # vae.plot_model(dir=load_dir)

    [x_, y_] = vae([x_train[0:3], y_train[0:3]])
        
    vae.compile(
        # loss = [mse, x_entropy],
        # loss_weights = [1, 2*vae.beta],
        optimizer='Adam')

    vae.summary()
    vae.encoder.summary()
    vae.decoder.summary()
    print('\n'*2+'*'*20+' BUILT   '+'*'*20+'\n'*2)
    
    epochs = 20
    
    refit = False
    # refit = True

    if not vae.trained or refit:
        history = vae.fit(x=[x_train, y_train],
                epochs=epochs,
                batch_size=10)

    x0 = np.atleast_2d(x_test[0])
    y0 = np.atleast_2d(y_test[0])

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

    acc = vae.accuracy(x_test, y_test)
    print(f'test accuracy: {acc}\n')
    
    if save_dir is not None:
        vae.save(save_dir)
