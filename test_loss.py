import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, categorical_crossentropy as xent
from data import generate as dg



class MixedModel(Model):

    def __init__(self, **kw):

        super().__init__(**kw)
        self.dense_1 = Dense(64, activation='relu', name='dense_1')
        self.dense_2 = Dense(64, activation='relu', name='latent')
        self.predict =  Dense(10, activation='softmax', name='predictions')
        self.reconstruct = Dense(784, name='reconstructed')

    def call(self, inputs):

        x_input = inputs[0]
        y_input = inputs[1]
        o1 = self.dense_1(x_input)
        o2 = self.dense_2(o1)
        y_ = self.predict(o2)
        x_ = self.reconstruct(o2)
        self.add_loss(mse(x_input, x_))
        self.add_loss(xent(y_, y_input))
        
        return [x_, y_]

model = MixedModel()


(x_train, y_train, x_test, y_test) = dg.get_mnist() 

# model.compile(loss=[None, categorical_crossentropy], optimizer='Adam')
model.compile(optimizer='Adam')

model.fit([x_train, y_train], epochs=10)
