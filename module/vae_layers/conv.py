import os
import configparser
import numpy as np
from torch import nn
from torchvision import models
from .misc import activation_layers, Reshape
import re
import logging


conv_config = configparser.ConfigParser()

this_dir = os.path.dirname(__file__)
conv_config.read(os.path.join(this_dir, 'conv-models.ini'))

features_dict = dict(conv_config['features'])
upsampler_dict = dict(conv_config['upsampler'])


def parse_conv_layer_name(s, ltype='conv', out_channels=32, kernel_size=5,
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

    elif s[0].lower() == 'u':
        ltype = 'upsampler'
        s = s[1:]

    elif s[0].lower() == 'r':
        ltype = 'rehsape'
        s = s[1:]

    params = dict(ltype=ltype, out_channels=out_channels, kernel_size=kernel_size,
                  padding=padding, stride=stride)

    if ltype == 'deconv':
        params['output_padding'] = output_padding

    if ltype.endswith('pooling'):
        params.pop('out_channels')
        delimiters.pop('out_channels')

    if ltype == 'upsampler':
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


def conv_layer_name(conv_layer):
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

    elif isinstance(conv_layer, nn.UpsamplingNearest2d):
        _s = 'u:{}'.format(conv_layer.scale_factor)

    return _s


def find_input_shape(layers_name, wanted_output_shape, input_shape=(1, 1)):

    deconv = build_de_conv_layers((1, *input_shape), layers_name, where='output')

    output_shape = deconv.output_shape[1:]

    if tuple(output_shape) == tuple(wanted_output_shape):
        logging.debug('Found input_shape for {}: {}, {}'.format(layers_name, *input_shape))
        return input_shape

    if output_shape[0] > wanted_output_shape[0] or output_shape[1] > wanted_output_shape[1]:
        e = 'Did not find an input shape yielding output size ({}, {}) for {}'
        raise ValueError(e.format(*wanted_output_shape, layers_name))

    i0 = input_shape[0] + int(output_shape[0] < wanted_output_shape[0])
    i1 = input_shape[1] + int(output_shape[1] < wanted_output_shape[1])

    return find_input_shape(layers_name, wanted_output_shape, input_shape=(i0, i1))


def build_de_conv_layers(input_shape, layers_name, batch_norm=False,
                         where='input', activation='relu',
                         output_activation='linear',
                         output_distribution='gaussian',
                         pretrained_dict=None, ):
    """ make (de)conv features

    -- if where is input, conv, else (output) deconv
    """

    if where == 'input' and layers_name.startswith('resnet'):
        conv = ResOrDenseNetFeatures(model_name=layers_name, input_shape=input_shape)
        return conv

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
            layer_params = parse_conv_layer_name(s, where=where)
            ltype = layer_params.pop('ltype')
            default_params[ltype] = layer_params

        layers_name = layers_name[end_default + 1:]
    layer_names = layers_name.split('-')
    layers = []

    in_channels, h, w = input_shape
    layer_names_ = []

    shapes = [input_shape]

    for i, layer_name in enumerate(layer_names):
        last_layer = i == len(layer_names) - 1
        layer_params = parse_conv_layer_name(layer_name, where=where)
        ltype = layer_params.pop('ltype')
        layer_params = parse_conv_layer_name(layer_name, **default_params.get(ltype, {}), where=where)
        ltype = layer_params.pop('ltype')

        if where == 'output' and last_layer and output_distribution == 'categorical':
            out_channels = layer_params['out_channels']
            logging.debug('Output channel {} -> {} for categorical output'.format(out_channels,
                                                                                  256 * out_channels))
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

        elif ltype.endswith('pooling'):
            Layer = {'m': nn.MaxPool2d, 'a': nn.AvgPool2d}[ltype[0]]
            conv_layer = Layer(**layer_params)
            out_channels = in_channels
            h = (h + 2 * conv_layer.padding - kernel_size) // conv_layer.stride + 1
            w = (w + 2 * conv_layer.padding - kernel_size) // conv_layer.stride + 1

        elif ltype == 'upsampler':
            scale_factor = layer_params['stride']
            conv_layer = nn.UpsamplingNearest2d(scale_factor=scale_factor)
            h = int(h * scale_factor)
            w = int(w * scale_factor)

        layers.append(conv_layer)
        if ltype.endswith('conv'):
            if batch_norm:
                layers.append(nn.BatchNorm2d(in_channels))
            kw = {'relu': {'inplace': True}}
            layers.append(activation_layers[activation](**kw.get(activation, {})))
            last_activation_i = len(layers) - 1

        layer_names_.append(conv_layer_name(conv_layer))
        shapes.append((out_channels, h, w))

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
    conv.shapes = shapes

    if pretrained_dict:

        conv.load_state_dict(pretrained_dict)
        logging.debug('Pretrained conv layers for {}'.format(where))
        for p in conv.parameters():
            p.requires_grad_(False)

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
