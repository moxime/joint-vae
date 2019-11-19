import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Concatenate, Input
from tensorflow.keras.models import Model

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def __init__(self, *args, sampling_size=1, **kwargs):

        self.sampling_size = sampling_size
        super().__init__(*args, **kwargs)
    
    def call(self, inputs):
        sampling_size = self.sampling_size
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(
            shape=(sampling_size, batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class JointLayer(Layer):

    def __init__(self, input_dim, num_labels=None, name='joint_input', **kw):

        if num_labels is None:
            num_labels=0
            self.x_y = False
        else:
            self.x_y = True

        self.input_dims = [input_dim, num_labels]
        
        super(JointLayer, self).__init__(name=name, **kw)

    def call(self, inputs):
        # print(self.name+' is called')
        if self.x_y:
            
            return tf.concat(inputs, axis=-1)
        
        return inputs

class Encoder(Model):

    def __init__(self,
                 latent_dim=32,
                 intermediate_dims=[64],
                 name='encoder',
                 beta=0,
                 activation='relu',
                 sampling_size=10,
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_projs = [Dense(u, activation=activation) for u in intermediate_dims]
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling(sampling_size=sampling_size)
        self.beta=beta
        
    def call(self, inputs):
        x = inputs 
        for l in self.dense_projs:
            x = l(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        if not self.beta == 0:
            kl_loss = - 0.5 * tf.reduce_sum(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, -1)
            self.add_loss(2 * self.beta * kl_loss)

        
        return z_mean, z_log_var, z

          
class Decoder(Model):

    def __init__(self,
                 original_dim,
                 num_labels=None,
                 intermediate_dims=[64],
                 name='decoder',
                 activation='relu',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.dense_projs = [Dense(u, activation=activation) for u in intermediate_dims]

        self.x_output = Dense(original_dim)

        self.x_y = num_labels is not None
        if self.x_y:
            self.y_output = Dense(num_labels, activation='softmax')
      
    def call(self, inputs):
        x = inputs
        # print('decoder inputs', inputs.shape)
        for l in self.dense_projs:
            # print('l:', l)
            x = l(x)
        if self.x_y:
            return [self.x_output(x), self.y_output(x)]

        return self.x_output(x)
            
