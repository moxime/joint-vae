import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import mse, binary_crossentropy, categorical_crossentropy
# from tensorflow.keras.utils import to_categorical
import data.generate as dg
from utils import save_load 
import utils.mutual_information as mi
import numpy as np
import tensorflow.keras.backend as K


def __make_iter__(a):
    try:
        _ = [e for e in a]
    except TypeError:
        return [a]
    return a


DEFAULT_ACTIVATION = 'relu'


class ClassificationVariationalNetwork(Model):

    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)

    @classmethod
    def build_model(cls, input_shape, encoder_layer_sizes, latent_dim,
                    decoder_layer_sizes, num_labels=None,
                    activation=DEFAULT_ACTIVATION, beta=0.01):

        if np.ndim(input_shape) == 0:
            input_shape = (input_shape,)

        # build encoder
        x_inputs = Input(shape=input_shape, name='encoder_input')
        if num_labels is not None:
            y_inputs = Input(shape=(num_labels,), name='y_true')
            inputs = [x_inputs, y_inputs]
        else: 
            xy_inputs = [x_inputs]
        x = x_inputs
        for i, l in enumerate(encoder_layer_sizes):
            x = Dense(l, activation=activation, name='enc_layer_' + str(i))(x)

        t_mean = Dense(latent_dim, name = 'latent_mean')(x)
        t_log_var = Dense(latent_dim, name = 'latent_log_var')(x)

        t = Lambda(dg.sampling,
                   output_shape=(latent_dim,),
                   name='latent')([t_mean, t_log_var])

        # instantiate encoder
        encoder = Model(x_inputs, [t_mean, t_log_var, t], name='encoder')

        # build decoder
        latent_inputs = Input(shape=(latent_dim,), name='t_sampling')

        x = latent_inputs
        if len(decoder_layer_sizes) > 0:
            for i, l in enumerate(decoder_layer_sizes):
                x = Dense(l, activation=activation, name='dec_layer_'+str(i))(x)

        x_outputs = Dense(np.prod(input_shape), activation='linear')(x)

                          
        # instantiate decoder model
        decoder = Model(latent_inputs, x_outputs, name='decoder')

        # instantiate VAE model
        connected_outputs = decoder(encoder(x_inputs)[2])
        vae = cls(x_inputs, connected_outputs, name='vae_mlp')

        vae.dim = np.prod(input_shape)
        vae.encoder = encoder
        vae.decoder = decoder
        vae.x_inputs = x_inputs
        # vae.y_inputs = y_inputs
        # vae.y_outputs = outputs
        vae.t_mean = t_mean
        vae.t_log_var = t_log_var

        the_optimizer = optimizers.RMSprop(lr=0.001)

        vae.trained = False
        vae.beta = beta
        vae.activation = activation
        vae._sizes_of_layers = [input_shape, encoder_layer_sizes,
                                latent_dim, decoder_layer_sizes, num_labels]

        print('*'*10+' beta=', beta,'*'*10)
        vae.compile(optimizer=the_optimizer,
                    loss = mse,
                    # loss='categorical_crossentropy',
                    # loss = vae.loss_function(),
                    metrics=['mae'])

        vae.built_model = True

        return vae


    
    def loss_function(self, beta=None):

        if beta is None:
            beta = self.beta
            
        def loss(x, x_):
      
            reconstruction_loss = mse(x, x_)
            
            kl_loss = 1 + self.t_log_var - K.square(self.t_mean) - K.exp(self.t_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + beta*kl_loss)

            """
            vae_loss = K.mean(reconstruction_loss)
            """
            
            return vae_loss
        
        if beta==0:
            return mse

        return loss
    
    def fit(self, force=False, *args, **kw):

        print('*'*100+'\n'+' '*20+'LETS FIT\n'+'*'*100)
        if not self.trained or force:
            super().fit(*args, **kw)

        self.trained = True            

    def plot_model(self, suffix='.png', show_shapes=True, show_layer_names=True):

        def _plot(net):
            plot_model(net, to_file=net.name+suffix, show_shapes=show_shapes,
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

        vae = cls.build_model(ls[0], ls[1], ls[2], ls[3], ls[4],
                              activation=p_dict['activation'],
                              beta=p_dict['beta'])

        vae.trained = p_dict['trained']

        if vae.trained:
            vae.load_weights(save_load.get_path(dir_name, 'weights.h5'))
        
        return vae
        
if __name__ == '__main__':

    load_dir = None
    # load_dir = './jobs/vae-mnist'
                  
    rebuild = load_dir is None
    
    if not rebuild:
        try:
            vae = ClassificationVariationalNetwork.load(load_dir)
            print('Is loaded. Is trained:', vae.trained)
        except(FileNotFoundError, NameError):
            print('Not loaded, rebuilding')
            rebuild = True
    if rebuild:
        beta = 1
        vae = ClassificationVariationalNetwork.build_model(28**2, [512, 200], 20,
                                                           [200, 512], 10, beta=beta) 
        # vae.plot_model()

    (x_train, y_train, x_test, y_test) = dg.get_mnist() 

    epochs = 1
    
    refit = False
    # refit = True
    
    vae.fit(force=refit, x=x_train, y=x_train, epochs=epochs,
            validation_data=(x_test, x_test))

    
    x0 = np.atleast_2d(x_test[0])
    
    t0_ = vae.encoder.predict(x0)
    mu0 = t0_[0]
    logsig0 = t0_[1]
    sig0 = np.exp(logsig0)
    t0 = t0_[2]
    # print(' -- '.join(str(i) for i in [sig0.min(), sig0.mean(), sig0.max()]))

    y_pred = vae.predict(x_test)
    t_enc_ = vae.encoder.predict(x_test)

    t_enc = t_enc_[2]
    ls_enc = t_enc_[1]

    

    mu0_enc = t_enc_[0]

    y_dec = vae.decoder.predict(t_enc)
