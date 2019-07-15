from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import data.generate as dg


def __make_iter__(a):
    try:
        _ = [e for e in a]
    except TypeError:
        return [a]
    return a


DEFAULT_ACTIVATION='relu'


class VariationalNetwork:

    def __init__(self):


        self.nets = {'main': Sequential()}

        self.params = {'built_model': False}
        self.params['layer_sizes'] = []
        self.params['trained'] = False

        self.data = {'x_train': None,
                     'y_train': None,
                     'x_test': None,
                     'y_test': None,
                     'name': ''}

        self.params['num_labels'] = 0
        self.params['dim'] = 0
        
        self.nets['var_decoders'] = []

        self.params['decoded_layers'] = []
        self.params['var_trained'] = False
        self.data['intermediate_outputs'] = None
        
    def build_model(self, layer_sizes, input_dim=None,
                     num_labels=None, activation=DEFAULT_ACTIVATION):

        if num_labels is None:
            num_labels = self.params['num_labels']
        if input_dim is None:
            input_dim = self.params['dim']
        self.params['layer_sizes'] = layer_sizes        

        print('net:', input_dim, '-', *layer_sizes, '-',  num_labels)

        self.nets['main'].add(layers.Dense(layer_sizes[0], activation=activation,
                                  input_shape=(input_dim,)))

        for layer_size in layer_sizes[1:]:
            # print(layer_size)
            self.nets['main'].add(layers.Dense(layer_size, activation=activation))

        self.nets['main'].add(layers.Dense(num_labels, activation='softmax'))

        the_optimizer = optimizers.RMSprop(lr=0.001)
        
        self.nets['main'].compile(optimizer=the_optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

        self.params['built_model'] = True
        self.params['num_labels'] = num_labels

    def build_var_dec(self, decoded_layers=None, hidden_vlayers=[10],
                      activation='linear'):
        """Build adjacent networks to create the variational decoders q(y|t)
            for the layers t specified in argument (default is None
            for all hidden layers).

        """
        sizes = self.params['layer_sizes']

        if decoded_layers is None:
            decoded_layers = [i for i in range(len(sizes))]
            
        decoded_layers = __make_iter__(decoded_layers)

        self.params['decoded_layers'] = decoded_layers
            
        for t in decoded_layers:
            t_dim = sizes[t]
            print('var_dec:', t_dim, hidden_vlayers)
            var_dec = models.Sequential()
            var_dec.add(layers.Dense(hidden_vlayers[0],
                                     activation=activation,
                                     input_shape = (t_dim,)))

            for layer_size in hidden_vlayers[1:]:
                var_dec.add(layers.Dense(layer_size,
                                         activation=activation))

            var_dec.add(layers.Dense(self.params['num_labels'],
                                     activation='softmax'))

            the_optimizer = optimizers.RMSprop(lr=0.001)

            var_dec.compile(optimizer=the_optimizer,
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

            self.nets['var_decoders'].append(var_dec)

    def get_data(self, source='mnist'):

        if source is 'mnist':
            self.data['name'] = 'mnist'
            train_data, train_labels, test_data, test_labels = dg.get_mnist()
            self.data['x_train'] = train_data
            self.data['y_train'] = train_labels

            self.data['x_test'] = test_data
            self.data['y_test'] = test_labels

            self.params['dim'] = 28**2
            self.params['num_labels'] = 10

        return test_data, test_labels

    def fit(self, x=None, y=None, force=False, *args, **kw):

        if x is None:
            x = self.data['x_train']
        if y is None:
            y = self.data['y_train']
        if not self.params['trained'] or force:
            self.nets['main'].fit(x=x, y=y, *args, **kw)
        self.params['trained'] = True
        
        self.data['intermediate_outputs'] = None

    def calculated_intermediate_outputs(self):
        """ return wether the intermediate output have been calculated """
        return self.data['intermediate_outputs'] is not None
        
    def get_intermediate_outputs(self, force=False):
        """ calculate intermeidate outputs if they are not caclulated yet or if force is True"""

        self.fit()
        main_net = self.nets['main']
        ts = self.params['decoded_layers']
        
        if not self.calculated_intermediate_outputs() or force:
            self.data['intermediate_outputs'] = []
            for t in ts:
                intermediate_layer_model = Model(inputs=main_net.input,
                                                 outputs=main_net.layers[t].output)
                output = intermediate_layer_model.predict(self.data['x_train'])
                self.data['intermediate_outputs'].append(output)

            self.params['var_trained'] = False

    def var_fit(self, y=None, var_epochs=50, *arg, **kw):
        """ trains auxiliary networks """
        
        self.get_intermediate_outputs()

        var_decs = self.nets['var_decoders']
        sizes = self.params['layer_sizes']
        inputs = self.data['intermediate_outputs']
        ts = self.params['decoded_layers']
        
        if y is None:
            y = self.data['y_train']
        
        for i, t in enumerate(ts):
            vad = var_decs[i]
            print('Training variational decoder number {}, input of dim {}'.format(i, sizes[t]))

            vad.fit(x=inputs[i], y=y,
                    epochs=var_epochs, *arg, **kw)

        self.params['var_trained'] = True

    def save_data(dir_name):
	"""Save the data to the file """
	directory = '{0}/{1}{2}/'.format(os.getcwd(), parent_dir, self.params['directory'])

		data = {'information': self.information,
		        'test_error': self.test_error, 'train_error': self.train_error, 'var_grad_val': self.grads,
		        'loss_test': self.loss_test, 'loss_train': self.loss_train, 'params': self.params
			, 'l1_norms': self.l1_norms, 'weights': self.weights, 'ws': self.ws}

		if not os.path.exists(directory):
			os.makedirs(directory)
		self.dir_saved = directory
		with open(self.dir_saved + file_to_save, 'wb') as f:
			cPickle.dump(data, f, protocol=2)

        

        
if __name__ == '__main__':

    var_net = VariationalNetwork()
    
    layer_sizes = [64, 16]
    
    test_data, test_labels = var_net.get_data(source='mnist')

    var_net.build_model(layer_sizes)
    var_net.build_var_dec()    

    epochs = 50
    var_net.fit(epochs=epochs, batch_size=512,
                validation_data=(test_data, test_labels)) 

    var_net.get_intermediate_outputs()

    var_net.var_fit(var_epochs=50)
