import sys
import logging
import warnings
import argparse
import torch
from torch import nn
import torchvision
import PIL
import time
import matplotlib.pyplot as plt

HALF_PLANE = True
HALF_PLANE = False


def print_time(t):
    if t < 1e-6:
        return '{:.0f} ns'.format(t * 1e9)
    elif t < 1e-3:
        return '{:.0f} us'.format(t * 1e6)
    elif t < 1:
        return '{:.0f} ms'.format(t * 1e3)
    elif t < 1e2:
        return '{:.0f} s'.format(t)
    elif t < 3600:
        return '{:.0f} m'.format(t / 60)


class Mask(nn.Module):

    def __init__(self, H, W, T=180, weighing_harmonics=2, init_lp='rand', store_tensors=False, **kw):

        super().__init__(**kw)
        self._masks = {}
        self._shape = (H, W)
        self.T = T

        self.store_tensors = store_tensors

        self.thetas = nn.Parameter(torch.linspace(0, torch.pi * (1 - 1 / self.T), self.T), requires_grad=False)

        self.lp = nn.Parameter(torch.randn(weighing_harmonics + 1))

        if not init_lp:
            self.lp.data = torch.zeros_like(self.lp)
            self.lp.data[0] = 1.
            self.lp.requires_grad_(False)

        self._reset_masks()

    def __repr__(self):
        return 'Mask()'

    def __str__(self):
        return 'Mask()'

    @property
    def device(self):
        return self.lp.device

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, v):

        if self.shape[0] != v[0] or self.shape[1] != v[1]:
            self._shape = v
            self._reset_masks()

    def _reset_masks(self, i=None):

        if i is not None:
            self._masks[i] = None
        else:
            for i in range(self.T):
                self._reset_masks(i)

    def _compute_masks(self, *i):
        i = list(i)
        if 0 in i:
            logging.debug('*** Computing masks for {}'.format(', '.join(str(_) for _ in i)))
        H, W = self.shape

        assert not H % 2 and not W % 2

        if HALF_PLANE:
            Hmin = 0
            nH = H // 2
        else:
            Hmin = -H // 2
            nH = H

        Wmin = - W // 2
        nW = W

        Hmax = H // 2 - 1
        Wmax = W // 2 - 1

        dim_names = ('P', 'theta', 'm', 'n')
        device = self.device
        m_ = ((torch.linspace(0, nH - 1, nH, device=device)[None, None, :, None] - Hmin) % nH) + Hmin
        m_.rename_(*dim_names) 

        n_ = ((torch.linspace(0, nW - 1, nW, device=device)[None, None, None, :] - Wmin) % nW) + Wmin
        n_.rename_(*dim_names) 

        theta = self.thetas[i]
        cost = torch.cos(theta)[None, :, None, None].rename(*dim_names)
        sint = torch.sin(theta)[None, :, None, None].rename(*dim_names)

        """
        print('m  ', *m_.names, *m_.shape)
        print('cos', *cost.names, *cost.shape)
        print('n  ', *n_.names, *n_.shape)
        print('sin', *sint.names, *sint.shape)
        """

        p = torch.LongTensor([_ for _, l in enumerate(self.lp) if l])
        lp = self.lp[p]

        mt = (m_ * cost + n_ * sint).rename(None).expand(len(p), len(theta), nH, nW).rename(*dim_names)

        # print(' '.join('{}'.format(_) for _ in p), '...', ' '.join('{:g}'.format(_) for _ in lp))

        p_ = p[:, None, None, None].rename(*dim_names).to(device)

        sinc = torch.sinc((mt - p_).rename(None)) + torch.sinc((mt + p_).rename(None))

        lp_ = lp[:, None, None, None].rename(*dim_names)

        masks = (sinc.rename(*dim_names) * lp_).sum('P')

        if self.store_tensors:

            self._masks.update({idx: masks[_] for _, idx in enumerate(i)})

        return masks

    def __getitem__(self, i):

        if self._masks[i] is None:
            return self._compute_masks(i)[0]

        return self._masks[i]


class ExtractModule(nn.Module):

    def __init__(self, padding, shape=[1024, 1024], T=180, norm=1,
                 weighing_harmonics=2, init_lp='rand', estimate_mean=False,
                 store_masks_tensors=False,
                 **kw):
        assert norm == 2 and padding == 2 or not estimate_mean, 'Estimate mean only for norm 2 ({})'.format(norm)

        super().__init__(**kw)
        self.padding = padding

        self.masks = Mask(1, 1, T=T, weighing_harmonics=weighing_harmonics,
                          init_lp=init_lp, store_tensors=store_masks_tensors)

        self.T = T
        self.P = len(self.masks.lp)

        self.norm = norm

        self.shape = shape

        self.estimate_mean = estimate_mean

    def __str__(self):

        s = 'ExtractModule(padding={pad}, T={T}, estimate_mean={mean}\n(masks): {mask})'
        return s.format(pad=self.padding, T=self.T, mean=self.estimate_mean, mask=self.masks)

    def __repr__(self):
        return self.__str__()
        
    def train(self, v=True):
        super().train(v)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, v):
        self.masks.shape = (int(self.padding * v[0]), int(self.padding * v[1]))
        self._shape = v

    def forward(self, batch, squeeze=False):

        if batch.ndim > 3:
            assert batch.shape[1] == 1
            return self.forward(batch.squeeze(1))

        if batch.ndim == 2:
            return self.forward(batch.unsqueeze(0), squeeze=True)

        assert batch.ndim == 3, 'batch shape: {}'.format(batch.shape)

        self.shape = batch.shape[-2:]
        H, W = self.masks.shape

        if HALF_PLANE:
            H_, W_ = H // 2, W
        else:
            H_, W_ = H, W

        image_fft = torch.fft.fft2(batch.rename(None), s=self.masks.shape)
        pseudo_image = torch.fft.ifft2(image_fft.abs().pow(self.norm)).real[:, -H_:, -W_:]

        if self.estimate_mean:
            K, L = H_ // 2, W // 2
            g0 = pseudo_image[:, 0, 0][:, None, None]
            alpha_l = (pseudo_image[:, 0, 1] * L / (L - 1))[:, None, None] / g0
            alpha_k = (pseudo_image[:, 1, 0] * K / (K - 1))[:, None, None] / g0

            k_ = ((torch.linspace(0, 2 * K - 1, 2 * K)[None, :, None] + K) % (2 * K) - K).to(batch.device)
            l_ = ((torch.linspace(0, 2 * L - 1, 2 * L)[None, None, :] + L) % (2 * L) - L).to(batch.device)
            am = (L - abs(l_)) * (K - abs(k_)) / K / L

            # print('im', *pseudo_image.shape)
            # print('g0', *g0.shape)
            # print('alpha', *alpha_k.shape)
            # print('k_', *k_.shape)
            # print('l_', *l_.shape)
            # print('am', *am.shape)

            pseudo_image -= g0 * alpha_k ** (k_.abs()) * alpha_l ** (l_.abs()) * am

        return torch.stack([(self.masks[_].rename(None) * pseudo_image).sum((-2, -1))
                            for _ in range(self.T)], dim=1) / H / W


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.DEBUG)
    warnings.filterwarnings('ignore', category=UserWarning)

    K, L = 512, 512
    default_shape = [K, L]
    default_batch_size = 1
    default_T = 180

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', nargs='?', dest='device', default='cuda', const='cpu')
    parser.add_argument('--shape', '-s', default=default_shape, type=int, nargs=2)
    parser.add_argument('--batch-size', '-N', default=default_batch_size, type=int)
    parser.add_argument('-T', default=default_T, type=int)
    parser.add_argument('--no-pow', action='store_false', dest='pow_of_two')
    parser.add_argument('--store_masks', action='store_true')
    parser.add_argument('-p', action='count', default=0)
    parser.add_argument('-I', nargs='+')

    args_from_py = '--cpu'.split()
    args_from_py = ''.split()

    args = parser.parse_args(args_from_py if len(sys.argv) < 2 else None)

    K, L = args.shape
    batch_size = args.batch_size

    if args.I:
        to_tensor = torchvision.transforms.ToTensor()
        image_list = []
        for image_path in args.I:
            image = PIL.Image.open(image_path)
            image_tensor = to_tensor(image).mean(0).to(args.device)
            print('Image {} shape:'.format(image_path), *image_tensor.shape)
            image_list.append(image_tensor)

        K = min([_.shape[0] for _ in image_list])
        L = min([_.shape[1] for _ in image_list])

        image_cropper = torchvision.transforms.CenterCrop((K, L))
        batch = torch.stack([image_cropper(_) for _ in image_list])
        print('*** {} images of shape {}x{}'.format(*batch.shape))

    else:
        batch = torch.rand(batch_size, K, L, device=args.device)

    plt.close('all')

    plot = ['']
    plot_thetas = []

    if args.p:
        plot.append('signature')
    if args.p > 1:
        plot_thetas = [0, 45, 90, 120]
    if args.p > 2:
        plot.append('pseudo_image')

    norm_ = [2, 2, 2]
    padding_ = [2, 4, 8]
    norm_ = [2]
    padding_ = [2]

    for norm, padding in zip(norm_, padding_):
        print('extraction with norm {} and padding {}'.format(norm, padding))

        e = ExtractModule(padding, norm=norm, T=args.T, init_lp=0, store_masks_tensors=args.store_masks)
        e.to(args.device)

        #    with torch.no_grad():
        s = e(batch)

        if 'pseudo_image' in plot:

            image_fft = torch.fft.fft2(batch[0].rename(None), s=[padding * K, padding * L], norm='ortho')
            pseudo_image = torch.fft.ifft2(image_fft.abs().pow(norm), norm='ortho').real
            plt.imshow(pseudo_image.cpu())
            plt.title('Pseudo Image')
            plt.show(block=False)

        logging.getLogger().setLevel(logging.ERROR)

        for t in plot_thetas:
            plt.figure()
            mask = e.masks[t]
            m, M = mask.min().item(), mask.max().item()
            mask = (mask - m) / (M - m)
            plt.imshow(mask.cpu())
            plt.title(t)
            plt.show(block=False)

        if 'signature' in plot:
            plt.figure()
            s_ = s[:, 1:] - s[:, :-1]
            plt.plot(s_.T.cpu())
            plt.title('s with norm {} and padding {}'.format(norm, padding))
            plt.show(block=False)

    if len(sys.argv) > 1 and args.p:
        input()
