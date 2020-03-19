import torch
from torch import nn, optim
import numpy as np
from torch.nn import functional as F


def onehot_encoding(y, C):

    s_y = y.squeeze().size()
    s_ = s_y + (1,)
    s = s_y + (C,)

    # print('vae_l l. 16 y:', y.shape, 's_', s_)
    y_onehot = torch.zeros(s, device=y.device)
    # y_onehot = torch.LongTensor(s)
    # y_onehot.zero_()
    y_onehot.scatter_(-1, y.reshape(s_), 1)

    return y_onehot


def _no_activation(a):
    return a


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the latent vector.
    - z_mean and a_log_var have the same dimensions N1xN2x...xNgxK
    - the output z has dimensions LxN1x...xNgxK where L is the samoling size. 
    """

    def __init__(self, latent_dim, sampling_size=1, **kwargs):

        self.sampling_size = sampling_size
        super().__init__(**kwargs)

    def forward(self, z_mean, z_log_var):
        
        sampling_size = self.sampling_size
        size = (sampling_size,) + z_log_var.size()  
        epsilon = torch.randn(size, device=z_mean.device)
        # print((f'***** z_log_var: {z_log_var.size()} '+
        #        f'z_mean: {z_mean.size()} ' +
        #        f'epsilon: {epsilon.size()}'))
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class Convolutional(nn.Module):

    def __init__(self, input_shape,
                 kernel_size=5,
                 padding=0, max_pool=2,
                 outputs_per_channel=3,
                 activation='relu', **kw):

        super(Convolutional, self).__init__(**kw)

        assert len(input_shape) == 3
        in_channels, height, width = input_shape
        
        out_channels = in_channels * outputs_per_channel
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=5, padding=0)

        h = (2 * padding + height - kernel_size + 1) // max_pool
        w = (2 * padding + width - kernel_size + 1) // max_pool

        self.input_shape = input_shape
        self.max_pool = max_pool
        self.output_shape = (out_channels, h, w)

        if activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f'{activation} is not implemented in {self.__class__})')

    def forward(self, x):

        batch_shape = x.shape[:-len(self.input_shape)]
        x_ = x.view(-1, *self.input_shape)
        x_ = self.activation(self.conv(x_))
        if self.max_pool > 1:
            return F.max_pool2d(x_, kernel_size=self.max_pool, stride=self.max_pool)
        else: return x_.view(batch_shape + self.output_shape)

        
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

        if activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f'{activation} is not implemented in {self.__class__})')

        self.input_shape = input_shape
        self.num_labels = num_labels

        self._sampling_size = sampling_size
        
        self.dense_projs = nn.ModuleList()
        input_dim = np.prod(input_shape) + num_labels
        for d in intermediate_dims:
            l_ = nn.Linear(input_dim, d)
            self.dense_projs.append(l_)
            input_dim = d

        self.dense_mean = nn.Linear(input_dim, latent_dim)
        self.dense_log_var = nn.Linear(input_dim, latent_dim)

        self.sampling = Sampling(latent_dim, sampling_size)

    @property
    def sampling_size(self):
        return self._sampling_size
        
    @sampling_size.setter
    def sampling_size(self, v):
        self._sampling_size = v
        self.sampling.sampling_size = v

    def forward(self, x, y):
        """ 
        - x input of size N1xN2x...xNgxD 
        - y of size N1xN2x...xNgxC
        - output of size (N1x...xNgxK, N1x...NgxK, LxN1x...xNgxK)
        """
        # print('*****', 'x:', x.shape, 'y:', y.shape)
        # u = torch.cat((x, y), dim=-1)
        # cat not working
        D = x.shape[-1]
        C = y.shape[-1]
        s = x.shape[:-1] + (D + C, )
        N = int(np.prod(s[:-1]))
        # print(N)
        u = torch.zeros(N, D + C, device=x.device)
        u[:, :D] = x.reshape(N, D)
        u[:, D:] = y.reshape(N, C)
        u = u.reshape(s)

        for l in self.dense_projs:
            u = self.activation(l(u))
        z_mean = self.dense_mean(u)
        z_log_var = self.dense_log_var(u)
        z = self.sampling(z_mean, z_log_var)
        
        return z_mean, z_log_var, z

          
class Decoder(nn.Module):           # 
    """
    - input: N1 x N2 x ... Ng x K
    - output : N1 x... x Ng x D where D is product of dims of reconstructed dims)
    """
    def __init__(self, 
                 latent_dim, reconstructed_dim,
                 intermediate_dims=[64],
                 name='decoder',
                 activation='relu',
                 output_activation='sigmoid',
                 **kwargs):

        super(Decoder, self).__init__(**kwargs)
        self.name = name

        if activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f'{activation} is not implemented in {self.__class__})')

        if output_activation == 'sigmoid':
            self.output_activation = torch.sigmoid
        elif output_activation == 'linear':
            self.output_activation = _no_activation
        else:
            raise ValueError(f'{output_activation} is not implemented in {self.__class__})')

        self.dense_layers = nn.ModuleList()
        input_dim = latent_dim
        for d in intermediate_dims:
            l_ = nn.Linear(input_dim, d)
            self.dense_layers.append(l_)
            input_dim = d

        self.output_layer = nn.Linear(input_dim, np.prod(reconstructed_dim))
      
    def forward(self, z):
        h = z
        # print('decoder inputs', inputs.shape)
        for l in self.dense_layers:
            # print('decoder h:', h.shape)
            h = self.activation(l(h))
        return self.output_activation(self.output_layer(h))
            

class Classifier(nn.Module):
    """Classifer

    - input: N1 x N2 x ... Ng x K

    - output : N1 x... x Ng x C where C is the number of class

    """
 
    def __init__(self, latent_dim,
                 num_labels,
                 intermediate_dims=[],
                 name='classifier',
                 activation='relu',
                 **kwargs):

        super().__init__(**kwargs)
        self.name = name
        
        if activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f'{output_activation} is not implemented in {self.__class__})')

        self.dense_layers = nn.ModuleList()
        input_dim = latent_dim
        for d in intermediate_dims:
            l_ = nn.Linear(input_dim, d)
            self.dense_layers.append(l_)
            input_dim = d

        self.output_layer = nn.Linear(input_dim, num_labels)
      
    def forward(self, z):
        u = z
        # print('decoder inputs', inputs.shape)
        for l in self.dense_layers:
            # print('l:', l)
            u = self.activation(l(u))
        return self.output_layer(u).softmax(dim=-1)

    
if __name__ == '__main__':


    

    def test_sampling(z_dims, latent_size, z_mean=None, z_log_var= None):

        if z_mean is None:
            z_mean = torch.randn(z_dims)
        if z_log_var is None:
            z_log_var = torch.randn(z_dims)

        sampling_layer = Sampling(1, latent_size)

        z = sampling_layer(z_mean, z_log_var)
        print(f'z size: {z.size()}')
        return z


    input_dims = (4, 3)
    num_labels = 10
    latent_dim = 7
    sampling = 11
    N_ = (13, 3)
    
    encoder = Encoder(input_dims, num_labels, latent_dim=latent_dim,
                      sampling_size=sampling)
    
    x = torch.randn(*N_, *input_dims)
    s_ = x.shape[:-len(input_dims)] + (-1,)
    x_ = x.reshape(*s_)
    y = torch.randint(0, num_labels, N_)

    y_onehot = onehot_encoding(y, num_labels).type(torch.Tensor)
    
    mu, ls, z = encoder(x_, y_onehot)

    print('x: ', x.shape)
    print('x_: ', x_.shape)
    print('y: ', y.shape)
    print('y_1: ', y_onehot.shape)
      
    print('mu: ', mu.shape)
    print('var: ', ls.shape)
    print('z: ', z.shape)
    
    decoder = Decoder(latent_dim, input_dims)

    x_reco = decoder(z)
            
    print('x_reco: ', x_reco.shape)
    
    classifier = Classifier(latent_dim, num_labels)

    y_est = classifier(z)
    
    print('y_est: ', y_est.shape)
