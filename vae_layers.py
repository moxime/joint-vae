import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Concatenate, Input


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class JointLayer(Layer):

    def __init__(self, input_dim, num_labels=None, name='joint_input', **kw):

        if num_labels is None:
            num_labels=0
            self.x_y = False
        else:
            self.x_y = True

        self.input_dims = [input_dim, num_labels]
        
        super(Layer, self).__init__(name=name, **kw)

    def call(self, inputs):
        # print(self.name+' is called')
        if self.x_y:
            
            return tf.concat(inputs, axis=-1)
        
        return inputs

class Encoder(Layer):

    def __init__(self,
                 latent_dim=32,
                 intermediate_dims=[64],
                 name='encoder',
                 **kwargs):
        super(Layer, self).__init__(name=name, **kwargs)
        self.dense_projs = [Dense(u, activation='relu') for u in intermediate_dims]
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = inputs 
        for l in self.dense_projs:
            x = l(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

          
class Decoder(Layer):

    def __init__(self,
                 original_dim,
                 num_labels=None,
                 intermediate_dims=[64],
                 name='decoder',
                 activation='relu',
                 **kwargs):
        super(Layer, self).__init__(name=name, **kwargs)

        self.dense_projs = [Dense(u, activation=activation) for u in intermediate_dims]

        x_output = Dense(original_dim)

        x_y = num_labels is not None
        if x_y:
            y_output = Dense(num_labels, activation='softmax')
            self.dense_output = [x_output, y_output]
        else:
            self.dense_output = x_output 
      
    def call(self, inputs):
        x = inputs
        print('decoder inputs', inputs.shape)
        for l in self.dense_projs:
            x = l(x)
        return self.dense_output(x)
