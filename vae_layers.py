import torch
from torch import nn, optim
import numpy as np

class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def __init__(self, latent_dim, sampling_size=1, **kwargs):

        self.sampling_size = sampling_size
        super().__init__(*args, **kwargs)

    def forward(self, z_mean, z_log_var):
        
        sampling_size = self.sampling_size
        size = z_log_var.size() + (sampling_size,)
        epsilon = torch.randn(size)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

 
class Encoder(nn.Module):

    def __init__(self, input_shape, num_labels,
                 latent_dim=32,
                 intermediate_dims=[64],
                 name='encoder',
                 beta=0,
                 activation='relu',
                 sampling_size=10,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.name = name
        self.beta = beta
        self.kl_loss_weight = 2 * beta

        self.input_shape = input_shape

        self.dense_projs = nn.ModuleList()
        input_dim = np.prod(input_shape)
        for d in intermediate_dims:
            l_ = nn.Linear(input_dim, d)
            self.dense_projs.append(l_)
            input_dim = d

        self.dense_mean = nn.Linear(input_dim, latent_dim)
        self.dense_log_var = nn.Linear(input_dim, latent_dim)

        self.sampling = Sampling(latent_dim, sampling_size)
        
    def call(self, inputs):
        x = inputs 
        for l in self.dense_projs:
            x = l(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        # if self.kl_loss_weight > -1:
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_log_var) - 1 -
                                      z_log_var +
                                      tf.square(z_mean), -1)
        self.add_loss(self.kl_loss_weight * kl_loss)
        
        return z_mean, z_log_var, z

          
class Decoder(nn.Module):           # 

    def __init__(self, 
                 latent_dim, reconstructed_dim,
                 intermediate_dims=[64],
                 name='decoder',
                 activation='relu',
                 output_activation='sigmoid',
                 **kwargs):

        super(Decoder, self).__init__(**kwargs)
        self.name = name
        
        self.dense_layers = [Dense(u, activation=activation) for u in
                             intermediate_dims]

        self.x_output = Dense(reconstructed_dim, activation=output_activation)
      
    def call(self, inputs):
        x = inputs
        # print('decoder inputs', inputs.shape)
        for l in self.dense_layers:
            # print('l:', l)
            x = l(x)
        return self.x_output(x)
            

class Classifier(nn.Module):

    def __init__(self, latent_dim,
                 num_labels,
                 intermediate_dims=[],
                 name='classifier',
                 activation='relu',
                 **kwargs):

        super().__init__(**kwargs)
        self.name = name
        
        self.dense_layers = [Dense(u, activation=activation) for u in
                             intermediate_dims]
        self.y_output = Dense(num_labels, activation='softmax')

    def call(self, inputs):
        x = inputs
        for l in self.dense_layers:
            x = l(x)
        return self.y_output(x)
