import numpy as np
from torch import nn
from .misc import activation_layers

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


ivgg_cfg = {}

# ivgg_cfg = {'ivgg19_': ['U', 512, 512, 512, 'U', 512, 512, 512, 512,
#                         'U', 256, 256, 256, 256, 'U', 128, 128, 'U',
#                         64, 64, -1], }

for c in vgg_cfg:
    ivgg_cfg['i' + c] = ['U' if _ == 'M' else _ for _ in vgg_cfg[c][-1::-1]]
    ivgg_cfg['i' + c].append(-1)


vgg_cfg_a = {}
for cfg in vgg_cfg:
    vgg_cfg_a[cfg + '-a'] = ['A' if _ == 'M' else _ for _ in vgg_cfg[cfg]]
vgg_cfg.update(vgg_cfg_a)


class VGGFeatures(nn.Sequential):

    def __init__(self, vgg_name, input_shape, batch_norm=False, channels=None, pretrained=None):

        cfg = vgg_cfg.get(vgg_name, channels)

        layers = self._make_layers(cfg, input_shape, batch_norm)
        super(VGGFeatures, self).__init__(*layers)
        self.architecture = {'features': vgg_name}

        self.pretrained = pretrained
        if pretrained:
            model_name = vgg_name.split('-')[0] + ('_bn' if batch_norm else '')
            if hasattr(models, model_name):
                pretrained_vgg = getattr(models, model_name)(pretrained=True)
                feat_to_inject = pretrained_vgg.features.state_dict()
                self.load_state_dict(feat_to_inject)
                logging.debug('% state injection successful')
            else:
                logging.error('Model %s not found in zoo', model_name)
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
                    layers += [nn.BatchNorm2d(x), ]
                layers += [nn.ReLU(inplace=True)]
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
                            'kernel': self.kernel,
                            'activation': self.activation}

    def _make_layers(self, channels, padding, kernel, input_shape,
                     activation, batch_norm):
        layers = []
        in_channels, h, w = input_shape
        activation_layer = activation_layers[activation]()
        for channel in channels:
            layers.append(nn.Conv2d(in_channels, channel, kernel,
                                    stride=2, padding=padding))
            if batch_norm:
                layers.append(nn.BatchNorm2d(channel))
            # layers.append(activation_layer)
            layers.append(nn.ReLU(inplace=True))
            h = h // 2
            w = w // 2
            in_channels = channel

        self.output_shape = (in_channels, h, w)
        return layers


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

    def _makelayer(self, first_shape, channels, output_activation, batch_norm):

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


class VGGDecoder(ConvDecoder):

    def __init__(self, ivgg_name, input_dim, first_shape, image_channel=None, channels=None, **kw):

        channels = ivgg_cfg.get(ivgg_name, channels)
        assert channels is not None
        assert -1 not in channels or image_channel is not None
        channels = [image_channel if _ == -1 else _ for _ in channels]

        super(VGGDecoder, self).__init__(input_dim, first_shape, channels,
                                         **kw)

    def _makelayer(self, first_shape, channels, output_activation, batch_norm):

        layers = []

        input_channels = first_shape[0]
        for x in channels:

            if x == 'U':
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))

            else:
                layers.append(nn.ConvTranspose2d(input_channels, x,
                                                 3, stride=1, padding=1))
                input_channels = x

                if batch_norm:
                    layers.append(nn.BatchNorm2d(x)),
                layers.append(nn.ReLU(inplace=True))

        if x == 'U':
            layers.append(None)
        layers[-1] = activation_layers.get(output_activation, nn.Identity)()

        return layers


