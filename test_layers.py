from vae_layers import Encoder, Decoder
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Concatenate, Input



class JointLayer(Layer):

    def __init__(self, **kw):
        super(Layer, self).__init__(**kw)

    def call(self, x):
        return tf.concat(x, axis=-1)

        
M = 100
C = 2
T = 10
N = 1000

x = np.random.randn(N, M)   
y = to_categorical(np.random.randint(0, C, size=N))

joint_input = JointLayer()
encoder = Encoder(C)
decoder = Decoder(M, C)

j_out = joint_input([x, y])
e_out = encoder(j_out)
d_out = decoder(e_out[2])
