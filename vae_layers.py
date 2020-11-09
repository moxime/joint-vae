import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from utils.print_log import debug_nan

def onehot_encoding(y, C):

    s_y = y.squeeze().size()
    s_y = y.size() 
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


activation_layers = {'linear': nn.Identity,
                     'sigmoid': nn.Sigmoid, 
                     'relu': nn.ReLU} 


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the latent vector.
    - z_mean and a_log_var have the same dimensions N1xN2x...xNgxK
    - the output z has dimensions LxN1x...xNgxK where L is the samoling size.
    """

    def __init__(self, latent_dim, sampling_size=1, sampling=True, **kwargs):

        self.sampling_size = sampling_size
        self.is_sampled = sampling
        super().__init__(**kwargs)

    def forward(self, z_mean, z_log_var):

        sampling_size = self.sampling_size
        size = (sampling_size,) + z_log_var.size()
        epsilon = torch.randn(size, device=z_mean.device)
        # print((f'***** z_log_var: {z_log_var.size()} '+
        #        f'z_mean: {z_mean.size()} ' +
        #        f'epsilon: {epsilon.size()}'))
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon * self.is_sampled


vgg_cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M',
              512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256,
              'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
              'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
              512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGFeatures(nn.Sequential):
    
    def __init__(self, vgg_name, input_shape, batch_norm=False, channels=None, pretrained=None):

        cfg = vgg_cfg.get(vgg_name, channels)
            
        layers = self._make_layers(cfg, input_shape, batch_norm)
        super(VGGFeatures, self).__init__(*layers)
        self.architecture = {'features': vgg_name}

        self.pretrained = pretrained
        if pretrained:
            self.load_state_dict(pretrained)
            for p in self.parameters():
                p.requires_grad_(False)
        if vgg_name not in vgg_cfg:
            self.name = 'vgg-' + '-'.join(str(c) for c in channels)
            self.architecture['features_channels'] = channels
        else:
            self.name = vgg_name 

    def _make_layers(self, cfg, input_shape, batch_norm):
        layers = []
        in_channels, h, w = input_shape
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                h = h // 2
                w = w // 2
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                h = h // 2
                w = w // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                if batch_norm:
                    layers += [nn.BatchNorm2d(x),]
                layers+= [nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        self.output_shape = (in_channels, h, w)
        return layers


class ConvFeatures(nn.Sequential):

    def __init__(self, input_shape, channels,
                 padding=1, kernel=4,
                 batch_norm=False,
                 pretrained=None,
                 activation='relu'):

        assert (kernel == 2 * padding + 2)

        layers = self._make_layers(channels, padding, kernel,
                                   input_shape,
                                   activation, batch_norm)
        super(ConvFeatures, self).__init__(*layers)

        
        self.name = 'conv-' + f'p{padding}-'
        self.name += '-'.join([str(c) for c in channels])

        self.pretrained = pretrained
        if pretrained:
            self.load_state_dict(pretrained)
            for p in self.parameters():
                p.requires_grad_(False)

        self.input_shape = input_shape
        self.channels = channels
        self.padding = padding
        self.kernel = kernel
        self.activation = activation

        self.params_dict = {'input_channels': self.input_shape,
                            'channels': self.channels,
                            'padding': self.padding,
                            'kernel':self.kernel,
                            'activation':self.activation}
        
    def _make_layers(self, channels, padding, kernel, input_shape,
                     activation, batch_norm):
        layers = []
        in_channels, h, w  = input_shape
        activation_layer = activation_layers[activation]()
        for channel in channels:
            layers.append(nn.Conv2d(in_channels, channel, kernel,
                                    stride=2, padding=padding))
            if batch_norm:
                layers.append(nn.BatchNorm2d(channel))
            # layers.append(activation_layer)
            layers.append(nn.ReLU(inplace=True))
            h = h//2
            w = w//2
            in_channels = channel
                          
        self.output_shape = (in_channels, h, w)
        return layers

                                   
class Encoder(nn.Module):

    def __init__(self, input_shape, num_labels,
                 latent_dim=32,
                 intermediate_dims=[64],
                 name='encoder',
                 activation='relu',
                 sampling_size=10,
                 sampling=True,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.name = name

        if activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(
                f'{activation} is not implemented in {self.__class__})')

        self.input_shape = input_shape
        self.num_labels = num_labels

        self._sampling_size = sampling_size

        dense_layers = []

        input_dim = np.prod(input_shape) + num_labels
        for d in intermediate_dims:
            dense_layers += [nn.Linear(input_dim, d),
                             activation_layers[activation]()]
            input_dim = d
        self.dense_projs = nn.Sequential(*dense_layers)
        
        self.dense_mean = nn.Linear(input_dim, latent_dim)
        self.dense_log_var = nn.Linear(input_dim, latent_dim)

        self.sampling = Sampling(latent_dim, sampling_size, sampling)

        centroids = torch.randn(num_labels, latent_dim)
        
        self.latent_dictionary = torch.nn.Parameter(centroids, requires_grad=False)

    @property
    def sampling_size(self):
        return self._sampling_size

    @sampling_size.setter
    def sampling_size(self, v):
        self._sampling_size = v
        self.sampling.sampling_size = v

    def capacity(self):
        """ 
        Approximation (upper-bound) of I(Z ; Y)
        """
        m = self.latent_dictionary
        # K = self.latent_dim
        C = self.num_labels
        # E = np.exp(1)
    
        cdm = torch.cdist(m, m)
        I = np.log(C) - 1 / C * torch.exp(-cdm.pow(2)/4).sum(0).log().sum()
        # + K/2 * np.log(2 / E) 

        return I
        
    def forward(self, x, y):
        """ 
        - x input of size N1xN2x...xNgxD 
        - y of size N1xN2x...xNgxC
        - output of size (N1x...xNgxK, N1x...NgxK, LxN1x...xNgxK)
        """
        # print('**** v_l l. 239', 'x:', x.shape, 'y:', y.shape)
        u = torch.cat((x, y), dim=-1)

        """ At first cat was not working, so...
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
        """

        # debug_nan(u, u, 'u1')
        u = self.dense_projs(u)
        if torch.isnan(u).any():
             for p in self.dense_projs.parameters():
                 print(torch.isnan(p).sum().item(), 'nans in',
                       'parameters of size',
                       *p.shape)  
             raise ValueError('ERROR')
        # debug_nan(u, self.dense_projs.parameters(), 'dpp')
        z_mean = self.dense_mean(u)
        # debug_nan(z_mean, u, 'µz')
        z_log_var = self.dense_log_var(u)
        # debug_nan(z_log_var, z_mean, 'sigz')
        z = self.sampling(z_mean, z_log_var)

        return z_mean, z_log_var, z


class ConvDecoder(nn.Module):
    """
    -- One refeactoring linear layer for input_dim to first_shape
    
    -- Successive upsampling layers

    -- input: N1x...xNqxK tensors
    -- output: N1x...xNqx C x H x W

    """

    
    def __init__(self, input_dim, first_shape, channels,
                 output_activation='linear',
                 upsampler_dict=None,
                 activation='relu',
                 batch_norm=False,
                 **kwargs):

        super(ConvDecoder, self).__init__(**kwargs)
           
        layers = self._makelayer(first_shape, channels, output_activation, batch_norm)
        activation_layer = activation_layers[activation]()
        
        self.first_shape = first_shape
        self.refactor = nn.Sequential(nn.Linear(input_dim, np.prod(first_shape)),
                                      activation_layer)
        self.upsampler = nn.Sequential(*layers)
        if upsampler_dict:
            self.upsampler.load_state_dict(upsampler_dict)
            for p in self.upsampler.parameters():
                p.requires_grad_(False)


    def forward(self, z):

        t = self.refactor(z)

        batch_shape = t.shape[:-1]

        out =  self.upsampler(t.view(-1, *self.first_shape))

        output_dim = out.shape[1:]
        return out.view(*batch_shape, *output_dim)

    def _makelayer(self, first_shape, channels, output_activation, batch_norm ):

        layers = []

        input_channels = first_shape[0]
        for output_channels in channels:
            layers += [nn.ConvTranspose2d(input_channels, output_channels,
                                          4, stride=2, padding=1)]
            if batch_norm:
                       layers.append(nn.BatchNorm2d(output_channels)),
            layers.append(nn.ReLU(inplace=True))
            input_channels = output_channels

        layers[-1] = activation_layers.get(output_activation, nn.Identity)()
        
        return layers


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
            raise ValueError(
                f'{activation} is not implemented in {self.__class__})')

        if output_activation == 'sigmoid':
            self.output_activation = torch.sigmoid
        elif output_activation == 'linear':
            self.output_activation = _no_activation
        else:
            raise ValueError(
                f'{output_activation} is not implemented in {self.__class__})')

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

    
class Classifier(nn.Sequential):
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
        
        activation_layer = activation_layers[activation]()
        layers = []
        input_dim = latent_dim
        for d in intermediate_dims:
            layers.append(nn.Linear(input_dim, d))
            layers.append(activation_layer)
            input_dim = d
            
        layers.append(nn.Linear(input_dim, num_labels))
        # layers.append(nn.Softmax(dim=-1))
        super().__init__(*layers, **kwargs)
        self.name = name
        

if __name__ == '__main__':

    def test_sampling(z_dims, latent_size, z_mean=None, z_log_var=None):

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
