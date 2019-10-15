import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense, Activation, Concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.losses import mse, binary_crossentropy, categorical_crossentropy
# from tensorflow.keras.utils import to_categorical
import data.generate as dg
from utils import save_load 
import utils.mutual_information as mi
import numpy as np
import tensorflow.keras.backend as K




N = 10000
C = 2
M = 20
T = 2


x_input = Input(shape=M)
y_input = Input(shape=C)

log_y = Activation('softmax')(y_input)

x_y = Concatenate()([x_input, log_y])

z = Dense(T)(x_y)

x_output = Dense(M)(z)
y_output = Dense(C)(z)

net = Model([x_input, y_input], [x_output, y_output])
net.summary()



def my_loss(net, type='mix'):

    def loss(in_, out_):
        x = in_[0]
        y = in_[1]

        x_ = out_[0]
        y_ = out_[1]

        if type is 'mix':
            return K.mean(mse(x, x_) + binary_crossentropy(y, y_))

        elif type is 'bin':
            return K.mean(binary_crossentropy(y, y_))

        else:
            return K.mean(mse(x, x_))
        


    return loss
    

if False:
    u = np.random.rand(M)


x = np.random.randn(N, M)
y = to_categorical(np.dot(x, u) > 0)


x_train = x[N//5:]
y_train = y[N//5:]
x_test = x[:N//5]
y_test = y[:N//5]





net.compile(optimizer='Adam', loss=my_loss(net, 'bin'))
net.fit([x_train, y_train], [x_train, y_train], epochs=30)


