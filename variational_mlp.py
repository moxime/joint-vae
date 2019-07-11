from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

def __make_iter__(a):
    try:
        _ = [e for e in a]
    except TypeError:
        return [a]
    return a


DEFAULT_ACTIVATION='relu'


class VariationalNetwork(object):


    def __init__(self):

        self.net = models.Sequential()
        self.built_layers = False
        self.layer_sizes = []
        
        
        self.var_decoders = []
        
        
        def build_layers(self, input_dim, num_labels, layer_sizes,
                         activation=DEFAULT_ACTIVATION):

            self.layer_sizes = layer_sizes

            self.net.add(layers.Dense(layer_sizes[0],
                                      activation=activation,
                                      input_shape=(input_dim,)))

            for layer_size in layer_sizes[1:]:
                model.add(layers.Dense(layer_size,
                                       activation=activation))
            model.add(layers.Dense(num_labels, activation='softmax'))

            self.built_layers = True
            self.num_labels = num_labels
            
        def build_var_dec(self, decoded_layers=None, hidden_vlayers=[10], activation=''):
            """Build adjacent networks to create the variational decoders q(y|t) for
            the layers t specified in argument (default is None for
            all hidden layers).
            """

            if decoded_layers is None:
                decoded_layers = [i+1 for i in range(len(self.layer_sizes))]

            decoded_layers = __make_iter__(decoded_layers)

            for t in decoded_layers:
                var_dec = model.Sequential()
                var_dec.add(layers.Dense(hidden_vlayers[0],
                                         activation=activation,
                                         input_shape = (self.t,)))

                for layer_size in hidden_vlayers[1:]:
                    var_dec.add(layers.Dense(layer_size,
                                             activation=activation))

                var_dec.add(layers.Dense(self.num_labels, activation='softmax'))

                self.var_decoders.append(var_dec)



if __name__ == '__main__':

    var_net = VariationalNetwork()
    num_labels = 2
    
