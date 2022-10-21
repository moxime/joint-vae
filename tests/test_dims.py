from cvae import ClassificationVariationalNetwork as Net
import torch
from module.losses import kl_loss, x_loss
import logging

# logging.getLogger().setLevel(logging.DEBUG)

D = (3, 32, 32)
C = 10

nets = {}
out = {}
out_y = {}

N = (20,)
L = 12
K = 4

d = 'cuda'

x = torch.randn(*N, *D, device=d)
y = torch.randint(0, C, N, device=d)

types = ('cvae',)
types = ('vib', 'vae', 'jvae', 'cvae', 'xvae')
types = ('cvae', 'jvae', 'vae', 'vae', 'cvae')

cls_cvae = []
gamma = 0
y_coded = True
y_coded = False

if y_coded:
    types = ('jvae', 'xvae')

for ntype in types:

    print('TYPE:', ntype)
    n = Net(D, C,
            type_of_net=ntype,
            y_is_coded=y_coded and ntype not in ('vib', 'vae'),
            batch_norm='encoder',
            features='vgg16',
            encoder_layer_sizes=[],
            decoder_layer_sizes=[],
            classifier_layer_sizes=cls_cvae if ntype == 'cvae' else [20, 10],
            sigma=0,
            gamma=gamma,
            force_cross_y=0,
            latent_sampling=L,
            latent_dim=K)
    n.to(d)
    nets[ntype] = n
    # n.compute_max_batch_size(batch_size=1024)
    # print(n.max_batch_sizes)

    if n.y_is_coded:
        pass

    if ntype != 'vae':
        print('y in input')
        out_y[ntype] = n.evaluate(x, y)
    print('y is none')
    out[ntype] = n.evaluate(x)


for o, _y in zip((out, out_y), ('*', 'y')):
    for t in o:

        for k in o[t][2]:
            ll = o[t][2][k]
            print(t, _y, k, 'loss  :', *ll.shape)
        logits = o[t][1]
        print(t, _y, 'logits:', *logits.shape)
        x_ = o[t][0]
        print(t, _y, 'x_:', *x_.shape)

        print('=' * 30)
    print('=' * 30)


if 'vib' in nets:
    x_, logits, mu, log_var, z = nets['vib'](x)

y_ = torch.cat([c * torch.ones((1,) + N, dtype=int) for c in range(C)])
x_ent = x_loss(None, logits, batch_mean=False)
dictionary = nets['cvae'].encoder.latent_dictionary
# kl = kl_loss(mu, log_var, y=y_, latent_dictionary=dictionary, batch_mean=False)
