import os
import configparser
import numpy as np
from torch import nn
from .misc import activation_layers, Reshape
import re
import logging

conv_config = configparser.ConfigParser()

this_dir = os.path.dirname(__file__)
conv_config.read(os.path.join(this_dir, 'conv-models.ini'))

features_dict = dict(conv_config['features'])
upsampler_dict = dict(conv_config['upsampler'])


def _parse_conv_layer_name(s, ltype='conv', out_channels=32, kernel_size=5,
                           padding='*', stride=None, output_padding=0,
                           activation='relu',
                           output_activation='linear',
                           where='input'):
    r"""Parse conv layer name s

    -- padding: '*' is 'same' if conv else 0 (for pooling)

    -- stride: if None will be one for conv layers, kernel_size for pooling layers

    """

    delimiters = {'out_channels': '^', 'kernel_size': 'x', 'padding': '\+', 'stride': ':'}

    if where == 'output':
        delimiters['output_padding'] = '\+\+'
        delimiters['conv_in_deconv'] = '\!'
        ltype = 'deconv'

    if s[0].lower() in 'am':
        ltype = s[0].lower() + 'pooling'
        s = s[1:]

    params = dict(ltype=ltype, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    if ltype == 'deconv':
        params['output_padding'] = output_padding

    if ltype.endswith('pooling'):
        params.pop('out_channels')
        delimiters.pop('out_channels')

    for k, c in delimiters.items():
        pattern = '{}(?P<{}>[0-9|\*]*)'.format(c, k)
        res = re.search(pattern, s)
        if res:
            try:
                params[k] = int(res.groupdict()[k])
            except ValueError:
                params[k] = params.get(k)

    if 'conv_in_deconv' in params:
        params['ltype'] = 'conv'
        params['out_channels'] = params.pop('conv_in_deconv')
        params.pop('output_padding')

    if params.get('padding') == '*':
        params['padding'] = params['kernel_size'] // 2 if ltype == 'conv' else 0

    if params['stride'] is None and ltype.endswith('conv'):
        params['stride'] = 1

    return params


def _conv_layer_name(conv_layer):
    if isinstance(conv_layer, (nn.Conv2d, nn.ConvTranspose2d)):
        _s = '{}x{}'.format(conv_layer.out_channels, conv_layer.kernel_size[0])
        if conv_layer.padding[0] != conv_layer.kernel_size[0] // 2:
            _s += '+{}'.format(conv_layer.padding[0])
        if conv_layer.stride[0] != 1:
            _s += ':{}'.format(conv_layer.stride[0])
        return _s
    elif isinstance(conv_layer, (nn.MaxPool2d, nn.AvgPool2d)):
        _s = '{}x{}'.format(str(conv_layer)[0], conv_layer.kernel_size)
        if conv_layer.stride != conv_layer.kernel_size:
            _s += ':{}'.format(conv_layer.stride)

        return _s


def build_de_conv_layers(input_shape, layers_name, batch_norm=False,
                         append_un_flatten=False, where='input',
                         activation='relu', output_activation='linear', output_distribution='gaussian'):
    """ make (de)conv features

    -- if where is input, conv, else (output) deconv
    """
    name = None
    if where == 'input' and layers_name in features_dict:
        name = layers_name
        layers_name = features_dict[layers_name]

    if where == 'output' and layers_name in upsampler_dict:
        name = layers_name
        layers_name = upsampler_dict[layers_name]

    if isinstance(input_shape, int):
        input_shape = (input_shape, 1, 1)

    default_params = {}
    if layers_name[0] == '[':
        end_default = layers_name.find(']')

        default_layer_params = layers_name[1:end_default].split('-')
        for s in default_layer_params:
            layer_params = _parse_conv_layer_name(s, where=where)
            ltype = layer_params.pop('ltype')
            default_params[ltype] = layer_params
        layers_name = layers_name[end_default + 1:]
    layer_names = layers_name.split('-')
    layers = []

    in_channels, h, w = input_shape
    layer_names_ = []

    for i, layer_name in enumerate(layer_names):
        last_layer = i == len(layer_names) - 1
        layer_params = _parse_conv_layer_name(layer_name, where=where)
        ltype = layer_params.pop('ltype')
        layer_params = _parse_conv_layer_name(layer_name, **default_params.get(ltype, {}), where=where)
        ltype = layer_params.pop('ltype')

        if where == 'output' and last_layer and output_distribution == 'categorical':
            out_channels = layer_params['out_channels']
            logging.debug('Output channel {} -> {} for categorical output'.format(out_channels, 256 * out_channels))
            out_channels = 256 * out_channels
            layer_params['out_channels'] = out_channels

        out_channels, kernel_size, padding, stride = (layer_params.get(_) for _ in
                                                      ('out_channels', 'kernel_size', 'padding', 'stride'))
        if ltype == 'conv':
            conv_layer = nn.Conv2d(in_channels, **layer_params)
            in_channels = out_channels
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1

        elif ltype == 'deconv':
            conv_layer = nn.ConvTranspose2d(in_channels, **layer_params)
            in_channels = out_channels
            h = (h - 1) * stride - 2 * padding + kernel_size + layer_params['output_padding']
            w = (w - 1) * stride - 2 * padding + kernel_size + layer_params['output_padding']
        else:
            Layer = {'m': nn.MaxPool2d, 'a': nn.AvgPool2d}[ltype[0]]
            conv_layer = Layer(**layer_params)
            out_channels = in_channels
            h = (h + 2 * conv_layer.padding - kernel_size) // conv_layer.stride + 1
            w = (w + 2 * conv_layer.padding - kernel_size) // conv_layer.stride + 1

        layers.append(conv_layer)
        if ltype.endswith('conv'):
            if batch_norm:
                layers.append(nn.BatchNorm2d(in_channels))
            kw = {'relu': {'inplace': True}}
            layers.append(activation_layers[activation](**kw.get(activation, {})))
            last_activation_i = len(layers) - 1
        if append_un_flatten and where == 'input':
            layers.append(nn.Flatten(-3, -1))

        layer_names_.append(_conv_layer_name(conv_layer))

    out_channels = (out_channels,)
    if where == 'output':
        layers[last_activation_i] = activation_layers[output_activation]()
        if output_distribution == 'categorical':
            layers.append(Reshape((256, out_channels[0] // 256, h, w)))
            out_channels = (256, out_channels[0] // 256)
    conv = nn.Sequential(*layers)
    conv.name = name or '-'.join(layer_names_)
    conv.output_shape = (*out_channels, h, w)
    conv.input_shape = input_shape

    return conv


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
                 stride=1,
                 batch_norm=False,
                 pretrained=None,
                 activation='relu'):

        if isinstance(padding, int):
            padding = [padding if isinstance(_, int) else None for _ in channels]
        if isinstance(kernel, int):
            kernel = [kernel if isinstance(_, int) else None for _ in channels]
        if isinstance(stride, int):
            stride = [stride if isinstance(_, int) else None for _ in channels]

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
