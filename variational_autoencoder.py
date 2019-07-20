from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras import optimizers
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
# from tensorflow.keras.utils import to_categorical
import data.generate as dg
import utils.save_load as disk
import utils.mutual_information as mi
import numpy as np
import keras.backend as K

def __make_iter__(a):
    try:
        _ = [e for e in a]
    except TypeError:
        return [a]
    return a

DEFAULT_ACTIVATION = 'relu'

class VariationalNetwork(Model):

    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)

    @classmethod
    def build_model(cls, input_shape, encoder_layers, latent_dim,
                    decoder_layers, num_labels,
                    activation=DEFAULT_ACTIVATION, beta=0.01):

        if np.ndim(input_shape) == 0:
            input_shape = (input_shape,)

        # build encoder
        x_inputs = Input(shape=input_shape, name='encoder_input')
        # y_inputs = Input(shape=(num_labels,), name='y_true')
        # xy_inputs = [x_inputs, y_inputs]
        
        x = Dense(encoder_layers[0], activation=activation)(x_inputs)
        for l in encoder_layers[1:]:
            x = Dense(l, activation=activation)(x)

        t_mean = Dense(latent_dim, name = 't_mean')(x)
        t_log_var = Dense(latent_dim, name = 't_log_var')(x)

        t = Lambda(dg.sampling,
                   output_shape=(latent_dim,),
                   name='t')([t_mean, t_log_var])

        # instantiate encoder
        encoder = Model(x_inputs, [t_mean, t_log_var, t])

        # build decoder
        latent_inputs = Input(shape=(latent_dim,), name='t_sampling')

        if len(decoder_layers)>0:
            x = Dense(decoder_layers[0], activation=activation)(latent_inputs)
            for l in decoder_layers[1:]:
                x = Dense(l, activation=activation)(x)
        else:
            x = latent_inputs
                
        outputs = Dense(num_labels, activation='softmax')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = decoder(encoder(x_inputs)[2])
        vae = cls(x_inputs, outputs, name='vae_mlp')

        vae.dim = np.prod(input_shape)
        vae.encoder = encoder
        vae.decoder = decoder
        vae.x_inputs = x_inputs
        # vae.y_inputs = y_inputs
        vae.y_outputs = outputs
        vae.t_mean = t_mean
        vae.t_log_var = t_log_var

        vae.beta = beta
        
        vae.trained = False

        the_optimizer = optimizers.RMSprop(lr=0.001)
        
        vae.compile(optimizer=the_optimizer,
                    #loss='categorical_crossentropy',
                    loss = vae.loss_function(),
                    metrics=['accuracy'])

        vae.built_model = True

        return vae

    def loss_function(self, beta=None):

        if beta is None:
            beta = self.beta
            
        def loss(y_true, y_pred):
      
            reconstruction_loss = categorical_crossentropy(y_true, y_pred)
            kl_loss = 1 + self.t_log_var - K.square(self.t_mean) - K.exp(self.t_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + beta*kl_loss)
            vae_loss = K.mean(reconstruction_loss)

            return vae_loss
        if beta==0:
            return categorical_crossentropy
        return loss
    
    def fit(self, force=False, *args, **kw):

        print('*'*100+'\n'+' '*20+'LETS FIT\n'+'*'*100)
        if not self.trained or force:
            super().fit(*args, **kw)

        self.trained = True
            


    def trust_d(self, x, M=100):

        d = 8

        y_pred = self.predict(x)
        for m in range(M):
            t_enc = self.encoder.predict(x)[2]
            y_dec = self.decoder.predict(t_enc)
            d += mi.d_kl(y_pred, y_dec, axis=0)

        return d/M
            
if __name__ == '__main__':

    rebuild = False or True
    if not rebuild:
        try:
            vae.trained
        except(NameError):
            rebuild = True
    if rebuild:
        vae = VariationalNetwork.build_model(28**2, [200, 100], 2, [], 10, beta=0) 

    (x_train, y_train, x_test, y_test) = dg.get_mnist() 

    epochs = 40
    
    refit = False
    # refit = True
    
    vae.fit(force=refit, x=x_train, y=y_train, epochs=epochs, validation_data=(x_test, y_test))
    
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

    

    m0_enc = t_enc_[0]

    y_dec = vae.decoder.predict(t_enc)

    
    N_false = 100

    x_false = np.ndarray((N_false, x_train.shape[1]))

    dim = x_test.shape[1]
    mu = np.ones(dim)
    sig = 0.1 * np.eye(dim)
        
    for i in range(N_false):
        x_false[i, :] = np.random.multivariate_normal(mu, sig)

    x_false = np.clip(x_false, 0, 1)

    
