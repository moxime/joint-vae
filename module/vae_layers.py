import torch
from torch import nn
import numpy as np
from torch.nn import functional as F, Parameter
from utils.print_log import texify_str
import logging
from torchvision import models


def onehot_encoding(y, C):

    s_y = y.shape
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


class Rgb2hsv(nn.Module):

    def __init__(self, input_dims, epsilon=1e-10, hmax=1.):
        super().__init__()
        self.dims = input_dims
        self.epsilon = epsilon
        assert len(input_dims) == 3
        assert input_dims[0] == 3
        self.hmax = hmax
        
    def forward(self, x):
        sixth = self.hmax / 6

        r, g, b = tuple(torch.index_select(x, -3, torch.LongTensor([i]).to(x.device)).squeeze(-3)
                        for i in range(3))
        max_rgb, argmax_rgb = x.max(-3)
        min_rgb, argmin_rgb = x.min(-3)

        max_min = max_rgb - min_rgb + self.epsilon

        h1 = sixth * (g - r) / max_min + sixth
        h2 = sixth * (b - g) / max_min + 3 * sixth
        h3 = sixth * (r - b) / max_min + 5 * sixth
        h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
        s = max_min / (max_rgb + self.epsilon)
        v = max_rgb

        return torch.stack((h, s, v), dim=-3)


class Hsv2rgb(Rgb2hsv):

    def forward(self, x):
        sixth = self.hmax / 6

        h, s, v = tuple(torch.index_select(x, -3, torch.LongTensor([i]).to(x.device)).squeeze(-3)
                        for i in range(3))

        h_ = (h - torch.floor(h / self.hmax) * self.hmax) / sixth
        
        c = s * v
        x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))
        
        zero = torch.zeros_like(c)
        y = torch.stack((
            torch.stack((c, x, zero), dim=-3),
            torch.stack((x, c, zero), dim=-3),
            torch.stack((zero, c, x), dim=-3),
            torch.stack((zero, x, c), dim=-3),
            torch.stack((x, zero, c), dim=-3),
            torch.stack((c, zero, x), dim=-3),
        ), dim=0)

        index = torch.repeat_interleave(torch.floor(h_).unsqueeze(-3), 3, dim=-3).unsqueeze(0).to(torch.long)

        rgb = y.gather(dim=0, index=index).squeeze(0) + (v - c).unsqueeze(-3)

        # print('***', *rgb.shape, *v.shape, *c.shape)
        return rgb

    
class Sigma(Parameter):

    @staticmethod
    def __new__(cls, value=None, sdim=1, input_dim=False, learned=False, is_rmse=False, is_log=False, **kw):

        assert value is not None or is_rmse or input_dim
        if is_rmse or input_dim and value is None:
            value = 0
        if input_dim:
            learned = True
        if learned:
            is_log = True
        if is_log:   
            value = np.log(value)

        return super().__new__(cls, torch.zeros(sdim).fill_(value), requires_grad=learned)
   
    def __init__(self, value=None, learned=False,  is_rmse=False,
                 sdim=1,
                 input_dim=False,
                 reach=1, decay=0, max_step=None, sigma0=None, is_log=False):

        assert not learned or not is_rmse
        assert not decay or not learned

        self._rmse = np.nan
        self.is_rmse = is_rmse
        self.sigma0 = value if (sigma0 is None and not is_rmse) else sigma0
        self.learned = learned
        self.input_dim = input_dim

        self.is_log = learned or is_log or input_dim

        self.decay = decay if not is_rmse else 1
        self.reach = reach if decay or is_rmse else None
        self.max_step = max_step
        self.sdim = sdim
        if self.coded:
            self._output_dim = input_dim if self.per_dim else (1,) * len(input_dim)
        else:
            self._output_dim = None

    @property
    def value(self):

        with torch.no_grad():
            if self.is_log:
                return (self.data * 2).exp().mean().sqrt().item()
            else:
                return self.data.pow(2).mean().sqrt().item()

    @property
    def coded(self):
        return bool(self.input_dim)

    @property
    def per_dim(self):
        return self.sdim != 1

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def params(self):

        d = self.__dict__.copy()
        for k in [_ for _ in d if _.startswith('_')]:
            d.pop(k)
        d['value'] = self.value
        return d

    def update(self, rmse=None, v=None):

        assert (rmse is None) or (v is None)

        if v is not None:
            mean_dims = tuple(range(v.dim() - self.dim()))

            v_ = v.mean(mean_dims) if mean_dims else v

            assert v_.dim() == self.dim()
            self.data = v_
            return

        if rmse is None:
            return

        self._rmse = rmse
        if self.learned or not self.decay:
            return
        delta = self.decay * (self.reach * rmse - self.data)
        if self.max_step and abs(delta) > self.max_step:
            delta = self.max_step if delta > 0 else -self.max_step
        self.data += delta

    def __format__(self, spec):

        if spec.endswith(('f', 'g', 'e')):
            return self.value.__format__(spec)
        if spec.endswith('x'):
            return texify_str(str(self), num=True)
        if spec.endswith('i'):
            if self.is_rmse:
                return 'e'
            if self.coded:
                return 'C' if self.per_dim else 'c'
            if self.learned:
                return 'l'
            return str(self)
                
        return str(self)
        
    def __str__(self):

        if self.is_rmse:
            if self._rmse is np.nan:
                return 'rmse'
            return f'rmse ({self._rmse:g})'

        if self.coded:
            return 'coded {}'.format('mask' if self.per_dim else 'scalar')
        if self.learned:
            return f'{self.sigma0:g}->rmse[l] ({self.value:g})'
            return f'learned from {self.sigma0:g}'
        if not self.decay:
            with torch.no_grad():
                return f'{self.data.item():g}'
        _mult = '' if self.reach == 1 else f'{self.reach:g}*'
        _max = f'<{self.max_step:g}' if self.max_step else ''
        return f'{self.sigma0:g}->{_mult}rmse[-{self.decay:g}*{_max}]'

    def __repr__(self):

        if self.is_rmse:
            return 'Sigma will be RMSE'
        s = super().__repr__()
        if self.decay:
            return s[:-1] + f', decaying to {self.reach}*mse with rate {self.decay})'
        return s


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
        size = (sampling_size + 1,) + z_log_var.size()
        epsilon = torch.randn(size, device=z_mean.device)
        epsilon[0] = 0
        # print((f'***** z_log_var: {z_log_var.size()} '+
        #        f'z_mean: {z_mean.size()} ' +
        #        f'epsilon: {epsilon.size()}'))
        # print('vl:136', self.is_sampled)
        return (z_mean + torch.exp(0.5 * z_log_var) * epsilon * self.is_sampled,
                epsilon[1:])


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


class ResOrDenseNetFeatures(nn.Sequential):

    def __init__(self, model_name='resnet152', input_shape=(3, 32, 32), pretrained=True):

        assert input_shape[0] == 3

        model = getattr(models, model_name)(pretrained=pretrained)
        modules = list(model.children())

        super(ResOrDenseNetFeatures, self).__init__(*modules[:-1])

        self.architecture = {'features': model_name}

        self.pretrained = pretrained

        self.name = model_name

        _, w, h = input_shape

        if model_name.startswith('resnet'):
            w, h = 1, 1
        elif model_name.startswith('densenet'):
            w //= 32
            h //= 32

        self.output_shape = (modules[-1].in_features, w, h) 


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

    capacity_log_barrier = 0.001

    def __init__(self, input_shape, num_labels,
                 representation='rgb',
                 y_is_coded=False,
                 latent_dim=32,
                 intermediate_dims=[64],
                 name='encoder',
                 activation='relu',
                 sampling_size=10,
                 sampling=True,
                 sigma_output_dim=0,
                 forced_variance=False,
                 dictionary_variance=1,
                 coder_means=None,
                 dictionary_min_dist=None,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.name = name
        self.y_is_coded = y_is_coded

        if activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(
                f'{activation} is not implemented in {self.__class__})')

        self.input_shape = input_shape
        self.num_labels = num_labels

        self.forced_variance = forced_variance

        self._sampling_size = sampling_size

        dense_layers = []

        input_dim = np.prod(input_shape) + num_labels * y_is_coded
        for d in intermediate_dims:
            dense_layers += [nn.Linear(input_dim, d),
                             activation_layers[activation]()]
            input_dim = d
        self.dense_projs = nn.Sequential(*dense_layers)

        self.dense_mean = nn.Linear(input_dim, latent_dim)
        self.dense_log_var = nn.Linear(input_dim, latent_dim)

        self.sigma_output_dim = sigma_output_dim
        if sigma_output_dim:
            self.sigma = nn.Linear(input_dim, np.prod(sigma_output_dim))

        self.sampling = Sampling(latent_dim, sampling_size, sampling)

        if coder_means in (None, 'random', 'learned'):
            centroids = np.sqrt(dictionary_variance) * torch.randn(num_labels, latent_dim)

        elif coder_means == 'onehot':
            centroids = torch.zeros(num_labels, latent_dim)
            for k in range(num_labels):
                centroids[k, k] = 1.

        else:
            logging.error('%s unknown', coder_means)    
            
        learned_dictionary = coder_means == 'learned'
            
        self.latent_dictionary = torch.nn.Parameter(centroids, requires_grad=learned_dictionary)
            
        self.dictionary_is_learned = learned_dictionary
        
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

    def dict_min_distance(self):

        C = self.num_labels
        dictionary = self.latent_dictionary

        max_norm = dictionary.norm(dim=1).max()
        diag = 2 * max_norm * torch.eye(C, device=dictionary.device)
        
        dist = torch.cdist(dictionary, dictionary) + diag

        return dist.min()

    def dict_distances(self):

        return torch.cdist(self.latent_dictionary,
                           self.latent_dictionary)

    def forward(self, x, y=None):
        """
        - x input of size N1xN2x...xNgxD 
        - y of size N1xN2x...xNgxC
        - output of size (N1x...xNgxK, N1x...NgxK, LxN1x...xNgxK)
        """
        if y is not None:
            pass
            # print('*** v_l:311', 'x:', *x.shape, 'y:', *y.shape)
        else:
            pass
            # print('*** v_l:319', 'x:', *x.shape)
            
        u = x if y is None else torch.cat((x, y), dim=-1) 

        # print('**** vl l 242', 'y mean', y.mean().item())

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

        u = self.dense_projs(u)
        if torch.isnan(u).any():
            for p in self.dense_projs.parameters():
                print(torch.isnan(p).sum().item(), 'nans in',
                      'parameters of size',
                       *p.shape)  
            raise ValueError('ERROR')

        z_mean = self.dense_mean(u)

        if self.forced_variance:
            z_log_var = np.log(self.forced_variance) * torch.ones_like(z_mean)
            # logging.debug(f'Variance forced {z_log_var.mean()} +- {z_log_var.std()}')
        else:
            z_log_var = self.dense_log_var(u)

        z, e = self.sampling(z_mean, z_log_var)

        if self.sigma_output_dim:
            sigma = self.sigma(u)
        else:
            sigma = None

        return z_mean, z_log_var, z, e, sigma


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

        out = self.upsampler(t.view(-1, *self.first_shape))

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
