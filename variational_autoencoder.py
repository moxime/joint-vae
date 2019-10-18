import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, Concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import mse
from tensorflow.keras.losses import categorical_crossentropy as x_entropy
# from tensorflow.keras.utils import to_categorical
import data.generate as dg
from utils import save_load 
import utils.mutual_information as mi
import numpy as np
import tensorflow.keras.backend as K
from vae_layers import *

def __make_iter__(a):
    try:
        _ = [e for e in a]
    except TypeError:
        return [a]
    return a


DEFAULT_ACTIVATION = 'relu'




class ClassificationVariationalNetwork(Model):

    def __init__(self,
                 input_shape,
                 num_labels=None,
                 encoder_layer_sizes=[36],
                 latent_dim=4,
                 decoder_layer_sizes=[36],
                 name = 'xy-vae',
                 activation=DEFAULT_ACTIVATION,
                 beta=1,
                 *args, **kw):

        super().__init__(name=name, *args, **kw)

        self.joint_layer = JointLayer(input_shape, num_labels)
        self.encoder = Encoder(latent_dim, encoder_layer_sizes)
        self.decoder = Decoder(input_shape, num_labels, decoder_layer_sizes)

        self.x_y = num_labels is not None
        self.input_dims = [input_shape]
        self.input_dims.append(num_labels if self.x_y else 0)
        self.beta = beta
            
        self._sizes_of_layers = [input_shape, num_labels,
                                 encoder_layer_sizes, latent_dim,
                                 decoder_layer_sizes]

        self.compile(optimizer='Adam')
                    # loss = mse,
                    # loss='categorical_crossentropy',
                    # loss = vae.loss_function(),
                    # metrics=['mae']
        
    def call(self, inputs):

        joint_input = self.joint_layer(inputs)
        print('joint_input shape', joint_input.shape)
        z_mean, z_log_var, z = self.encoder(joint_input)
        for l in [z_mean, z_log_var, z]:
            print ('z:', l.shape)
        reconstructed = self.decoder(z)

        if self.x_y:
            [x_input, y_input] = inputs
            [x_output, y_output] = reconstructed

        else:
            x_input = inputs
            x_output = reconstructed

        mse_loss = mse(x_input, x_output)
        self.add_loss(mse_loss)
                
        # Add KL divergence regularization loss.
        if not beta == 0:
            kl_loss = - 0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
            self.add_loss(beta*kl_loss)

            if self.x_y:
                x_loss = x_entropy(y_input, y_output)
                self.add_loss(beta*x_loss)
            
        return reconstructed

    def plot_model(self, dir='.', suffix='.png', show_shapes=True, show_layer_names=True):

        if dir is None:
            dir = '.'
        
        def _plot(net):
            f_p = save_load.get_path(dir, net.name+suffix)
            plot_model(net, to_file=f_p, show_shapes=show_shapes,
                       show_layer_names=show_layer_names)

        _plot(self)
        _plot(self.encoder)
        _plot(self.decoder)

    def save(self, dir_name):

        param_dict = {'layer_sizes': self._sizes_of_layers,
                      'trained': self.trained,
                      'beta': self.beta,
                      'activation': self.activation
                      }

        save_load.save_json(param_dict, dir_name, 'params.json')

        w_p = save_load.get_path(dir_name, 'weights.h5')
        if self.trained:
            self.save_weights(w_p)

    @classmethod        
    def load(cls, dir_name):

        p_dict = save_load.load_json(dir_name, 'params.json')

        ls = p_dict['layer_sizes']

        vae = cls(ls[0], ls[1], ls[2], ls[3], ls[4],
                  activation=p_dict['activation'],
                  beta=p_dict['beta'])

        vae.trained = p_dict['trained']

        if vae.trained:
            vae.load_weights(save_load.get_path(dir_name, 'weights.h5'))
        
        return vae

    
if __name__ == '__main__':

    load_dir = None
    # load_dir = './jobs/vae-mnist/191016'
                  
    # rebuild = load_dir is None
    rebuild = True
    
    e_ = [36]
    d_ = e_.copy()
    d_.reverse()
    
    (x_train, y_train, x_test, y_test) = dg.get_mnist() 

    if not rebuild:
        try:
            vae = ClassificationVariationalNetwork.load(load_dir)
            print('Is loaded. Is trained:', vae.trained)
        except(FileNotFoundError, NameError):
            print('Not loaded, rebuilding')
            rebuild = True
    if rebuild:
        print('\n'*2+'*'*20+' BUILDING '+'*'*20+'\n'*2)
        beta = 1
        vae = ClassificationVariationalNetwork(28**2, 10, e_, 2,  # 
                                               d_, beta=beta) 
        # vae.plot_model(dir=load_dir)
        
        vae.call([x_train, y_train])
        vae.summary()
        vae.encoder.summary()
        vae.decoder.summary()
        print('\n'*2+'*'*20+' BUILT   '+'*'*20+'\n'*2)


    epochs = 2
    
    refit = False
    # refit = True

    vae.fit(x=[x_train, y_train],
            y=[x_train, y_train],
            epochs=epochs,
            # validation_data=(x_test, x_test)
            )

    x0 = np.atleast_2d(x_test[0])
    y0 = np.atleast_2d(y_test[0])
    
    t0_ = vae.encoder.predict([x0, y0])
    mu0 = t0_[0]
    logsig0 = t0_[1]
    sig0 = np.exp(logsig0)
    t0 = t0_[2]
    # print(' -- '.join(str(i) for i in [sig0.min(), sig0.mean(), sig0.max()]))

    y_pred = vae.predict([x_test, y_test])
    t_enc_ = vae.encoder.predict([x_test, y_test])

    t_enc = t_enc_[2]
    ls_enc = t_enc_[1]

    mu0_enc = t_enc_[0]

    y_dec = vae.decoder.predict(t_enc)
    
