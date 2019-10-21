import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, categorical_crossentropy as xent


class Sampling(Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Pipe(Layer):
    def __init__(self, **kw):
        super(Pipe, self).__init__(**kw)
        self.mean_layer = Dense(36)
        self.log_var_layer = Dense(36)
        self.sampling = Sampling()

    def call(self, input):
        m_output = self.mean_layer(input)
        lv_output = self.log_var_layer(input)
        self.add_loss(tf.reduce_mean(tf.square(m_output), -1))
        return self.sampling([m_output, lv_output])


class Out(Layer):
    def __init__(self, out_dims, **kw):
        super(Out, self).__init__(**kw)
        self.layers = []
        for d in out_dims:
            l = Dense(d)
            self.layers.append(l)

    def call(self, input):
        outputs = []
        for l in self.layers:
            outputs.append(l(input))
        return outputs


class Khi(Model):
    def __init__(self, dims, **kw):
        super(Khi, self).__init__(**kw)
        self.joint = Concatenate()
        self.pipe = Pipe()
        self.out = Out(dims)

    def call(self, inputs):
        # print('\n'*5 + '*'*80, 'khi called', '\n'*5)
        joint_input = self.joint(inputs)
        latent = self.pipe(joint_input)
        outs = self.out(latent)
        for [i, o] in zip(inputs, outs):
            # print('='*80, mse(i, o),'\n'*5)
            self.add_loss(mse(i, o))
        return outs
    
        
M_ = [100, 200, 50, 300]
N = int(1e5)

x_ = []

for M in M_:
    x_.append(np.random.randn(N, M))   

print('\n'*40)
net = Khi(M_)
net.compile(optimizer='Adam')
net.fit(x_, epochs=30, batch_size=100)

for loss in net.losses:
    print(loss)
