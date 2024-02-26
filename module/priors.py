import torch
from torch import nn
import numpy as np
from utils.print_log import texify_str
import logging
from torch.nn import functional as F, Parameter


def adapt_batch_function(arg=1, last_shapes=1):

    def adapter(func):
        return adapt_batch_dim(func, arg=arg, last_shapes=last_shapes)
    return adapter


def adapt_batch_dim(func, arg=1, last_shapes=1):

    def func_(*a, **kw):
        x = a[arg]
        batch_shape = x.shape[:-last_shapes]
        working_shape = x.shape[-last_shapes:]
        a = list(a)
        try:
            a[arg] = x.contiguous().view(-1, *working_shape)
        except RuntimeError as e:
            print('***  ERROR', 'x', *x.shape, 'w:', *working_shape)
            raise e
        res = func(*a, **kw)
        if isinstance(res, float):
            logging.error('\n'.join(str(_) for _ in a))
        return res.view(batch_shape)
    return func_


def build_prior(dim, distribution='gaussian', **kw):

    if kw.get('num_priors', 1) == 1:
        kw.pop('learned_means', False)
    valid_dist = ('gaussian', 'tilted', 'uniform')
    assert distribution in valid_dist, '{} unknown (try one of: {})'.format(distribution, ', '.join(valid_dist))
    if distribution == 'gaussian':
        if kw.pop('tau', None) is not None:
            logging.debug('discarded value of tau for gaussian prior')
        return GaussianPrior(dim, **kw)
    elif distribution == 'tilted':
        if kw.pop('var_dim', 'scalar') != 'scalar':
            logging.info('discarded variance type for tilted gaussian prior')
        return TiltedGaussianPrior(dim, **kw)
    elif distribution == 'uniform':
        if kw.pop('var_dim', 'scalar') != 'scalar':
            logging.info('discarded variance type for tilted gaussian prior')
        return UniformWithGaussianTailPrior(dim, **kw)


class GaussianPrior(nn.Module):

    def __init__(self, dim, var_dim='scalar', num_priors=1,
                 init_mean=0, mean_shift=0, learned_means=False, freeze_means=0):
        """
        var_dim : scalar, diag or full
        num_priors: 1 for non conditional prior, else number of classes

        """

        assert not learned_means or num_priors > 1

        super().__init__()

        self.num_priors = num_priors
        self.var_dim = var_dim
        self.learned_var = var_dim != 'scalar'
        self.learned_means = learned_means
        self.freeze_means = freeze_means
        self.dim = dim

        if num_priors == 1:
            self.conditional = False
            mean_tensor = init_mean * torch.tensor(1.) + mean_shift

        else:
            self.conditional = True
            if init_mean == 'onehot':
                assert dim >= num_priors, 'K={}<C={}'.format(dim, num_priors)
                mean_tensor = torch.zeros(dim, num_priors)
                for i in range(num_priors):
                    mean_tensor[i, i] = 1

            else:
                unit_mean = torch.randn(num_priors, dim).squeeze()

                try:
                    float(init_mean)
                    mean_tensor = init_mean * unit_mean + mean_shift
                except ValueError:
                    mean_tensor = init_mean.squeeze()

        self._frozen_means = not learned_means or freeze_means > 0
        self.mean = Parameter(mean_tensor, requires_grad=not self._frozen_means)

        if var_dim == 'scalar':
            var_param_per_class = torch.tensor(1.)
        elif var_dim == 'diag':
            var_param_per_class = torch.ones(dim)
        elif var_dim == 'full':
            var_param_per_class = torch.eye(dim)
        else:
            raise ValueError('var_dim {} unknown'.format(var_dim))
        if self.conditional:
            var_tensor = torch.stack([var_param_per_class for _ in range(num_priors)])
        else:
            var_tensor = var_param_per_class
        self._var_parameter = Parameter(var_tensor, requires_grad=self.learned_var)

        self._inv_var_is_computed = False
        self._inv_var = None

        self.params = {'distribution': 'gaussian', 'dim': dim,
                       'init_mean': init_mean,
                       'var_dim': self.var_dim, 'num_priors': self.num_priors}
        if self.conditional:
            self.params.update({'learned_means': self.learned_means,
                                'freeze_means': freeze_means})

    def thaw_means(self, epoch=None):
        if not self.learned_means or not self._frozen_means:
            return
        if epoch is None or epoch >= self.freeze_means:
            logging.debug('Defreezing prior means')
            self.mean.requires_grad_()
            self._frozen_means = True

    @ property
    def inv_trans(self):

        if self.var_dim == 'full':
            return self._var_parameter.tril()
        else:
            return self._var_parameter

    @ property
    def inv_var(self):

        if self._inv_var_is_computed:
            return self._inv_var

        self._compute_inv_var()
        return self._inv_var

    def _compute_inv_var(self):

        if self.var_dim == 'scalar':
            self._inv_var = self._var_parameter ** 2
            self._inv_var_is_computed = not self.training

        elif self.var_dim == 'diag':
            self._inv_var = self._var_parameter ** 2
            self._inv_var_is_computed = not self.training

        elif self.var_dim == 'full':
            self._inv_var = torch.matmul(self.inv_trans.transpose(-1, -2), self.inv_trans)
            self._inv_var_is_computed = not self.training

    def log_det_per_class(self):
        """ return log det of Sigma """

        if self.var_dim == 'full':
            # return -self.inv_trans.abs().logdet() * 2
            if self.conditional:
                return -2 * torch.stack([M_.diag().abs().log().sum() for M_ in self.inv_trans])
            else:
                return -2 * self.inv_trans.diag().abs().log().sum()

        elif self.var_dim == 'diag':
            return -self.inv_trans.abs().log().sum(-1) * 2
        else:
            return -self.dim * self.inv_trans.log() * 2

    def whiten(self, x, y=None):

        # logging.debug('TBR in whiten')
        # logging.debug('x : %s y:%s', x.shape, y.shape if y is not None else y)
        assert self.conditional ^ (y is None)

        if self.conditional:
            transform = self.inv_trans.index_select(0, y.view(-1))
        else:
            transform = self.inv_trans

        # print('**** transf shape', *transform.shape, 'x', *x.shape)
        if self.var_dim == 'full':
            transformed = torch.matmul(transform, x.unsqueeze(-1)).squeeze(-1)

        elif self.var_dim == 'diag':
            transformed = x * transform

        else:
            transformed = x * transform.unsqueeze(-1)

        return transformed

    @ adapt_batch_function()
    def mahala(self, x, y=None):

        # logging.debug('TBR in mahala')
        # logging.debug('x : %s y:%s', x.shape, y.shape if y is not None else y)

        assert self.conditional ^ (y is None)

        if self.conditional:
            means = self.mean.index_select(0, y.view(-1))
        else:
            means = self.mean.unsqueeze(-1)

        return self.whiten(x - means, y).pow(2).sum(-1)

    @ adapt_batch_function()
    def trace_prod_by_var(self, var, y=None):
        """ Compute tr(LS^-1) """

        assert self.conditional ^ (y is None)

        if self.var_dim == 'full':
            prior_inv_var_diag = self.inv_trans.pow(2).sum(-2)

        else:
            prior_inv_var_diag = self.inv_trans.pow(2)

        if self.conditional:
            prior_inv_var_diag = prior_inv_var_diag.index_select(0, y.view(-1))

        if self.var_dim == 'scalar':
            prior_inv_var_diag.unsqueeze_(-1)

        try:
            return (var * prior_inv_var_diag).sum(-1)
        except RuntimeError as e:
            error_msg = '*** var: {} diag: {}'.format(' '.join(str(_) for _ in var.shape),
                                                      ' '.join(str(_) for _ in prior_inv_var_diag.shape))
            logging.error(error_msg + ' error msg: {}'.format(str(e)))
            return 0.

    def kl(self, mu, log_var, y=None, output_dict=True, var_weighting=1.):
        """Params:

        -- mu: NxK means

        -- log_var: NxK diag log_vars

        -- y: None or tensor of size N

        """

        if y is not None and y.ndim == mu.ndim:
            expand_shape = list(mu.unsqueeze(0).shape)
            expand_shape[0] = y.shape[0]
            return self.kl(mu.expand(*expand_shape), log_var.expand(*expand_shape),
                           y=y, output_dict=output_dict, var_weighting=var_weighting)

        debug_msg = 'TBR in kl '
        debug_msg += 'mu: ' + ' '.join(str(_) for _ in mu.shape)
        debug_msg += 'var: ' + ' '.join(str(_) for _ in log_var.shape)
        debug_msg += 'y: ' + ('None' if y is None else ' '.join(str(_) for _ in y.shape))
        # logging.debug(debug_msg)

        var = log_var.exp()

        prior_trans = self.inv_trans

        if prior_trans.isnan().any():

            print('*** STOPPIN')
            torch.save(self.state_dict(), '~/prior.pth')
            return

        loss_components = {}

        loss_components['trace'] = self.trace_prod_by_var(var, y)

        """ log |Sigma| """
        loss_components['log_det_prior'] = self.log_det_per_class()

        if loss_components['log_det_prior'].isnan().any():
            print('*** Log det nan')
            inv_var = self.inv_var
            log_det_prior = inv_var.logdet()
            is_nan_after = log_det_prior.isnan().any()
            print('*** real log det is nan:', is_nan_after)

        if self.conditional:
            log_detp = loss_components['log_det_prior']
            loss_components['log_det_prior'] = log_detp.index_select(0, y.view(-1)).view(y.shape)

        loss_components['log_det'] = log_var.sum(-1)

        loss_components['distance'] = self.mahala(mu, y)

        stop = False
        for k in loss_components:
            if loss_components[k].isnan().any():
                print('***', k, 'is nan')
                logging.error('Will stop bc {} is nan'.format(k))
                stop = True
        if stop:
            return

        # for k in loss_components:
        #    logging.debug('TBR in KL %s %s', k, loss_components[k].shape)
        loss_components['var_kl'] = (loss_components['trace'] -
                                     loss_components['log_det'] +
                                     loss_components['log_det_prior'] -
                                     self.dim)

        loss_components['kl'] = 0.5 * (loss_components['distance'] +
                                       var_weighting * loss_components['var_kl'])

        return loss_components if output_dict else loss_components['kl']

    def log_density(self, z, y=None):

        # logging.debug('TBR in log_density')
        # logging.debug('z : %s y:%s', z.shape, y.shape if y is not None else y)

        assert self.conditional ^ (y is None)

        u = self.mahala(z, y)

        log_det = self.log_det_per_class()
        if self.conditional:
            log_det = log_det.index_select(0, y.view(-1)).view(u.shape)

        # print('**** log_det', *log_det.shape, 'u', *u.shape)
        return -np.log(2 * np.pi) * self.dim / 2 - u / 2 - log_det / 2

    def __repr__(self):

        pre = 'conditional ' if self.conditional else ''
        var = ('learned ' if self.learned_var else '') + self.var_dim + ' variance'
        mean = ''
        if self.conditional:
            mean = '{} {}means and '.format(self.num_priors, 'learned ' if self.learned_means else '')
        else:
            mean = 'mean centered on {} '.format(self.params['init_mean']) if self.params['init_mean'] else ''
        return 'gaussian {p}prior of dim {K} with {m}{v}'.format(p=pre, m=mean, v=var, K=self.dim)


class TiltedGaussianPrior(GaussianPrior):

    def __init__(self, dim, num_priors=1, init_mean=0, learned_means=False,
                 tau=25, **kw):

        super().__init__(dim,
                         num_priors=num_priors,
                         init_mean=init_mean,
                         learned_means=learned_means,
                         var_dim='scalar', **kw)

        self.tau = tau
        self._mu_star = tau

        self.params['distribution'] = 'tilted'
        self.params['tau'] = tau

    def __repr__(self):
        if self.num_priors > 1:
            _m = ' with {} {}means'.format(self.num_priors, 'learned ' if self.learned_means else '')
        else:
            _m = ''
        return 'tilted gaussian {c}prior{m}, tau={tau}'.format(c='conditional ' if self.conditional else '',
                                                               m=_m, tau=self.tau)

    def log_density(self, z, y=None):

        return super().log_density(z, y) - z.norm(dim=-1)

    @ property
    def mu_star(self):
        return self._mu_star

    def kl(self, mu, log_var, y=None, output_dict=True, var_weighting=1.):

        if var_weighting != 1.:
            logging.debug('var weighting != 1 but tilted gaussian does not care')

        if y is not None and y.ndim == mu.ndim:
            expand_shape = list(mu.unsqueeze(0).shape)
            expand_shape[0] = y.shape[0]
            return self.kl(mu.expand(*expand_shape), log_var.expand(*expand_shape),
                           y=y, output_dict=output_dict)
        loss_components = {}
        distance = self.mahala(mu, y)
        loss_components['distance'] = distance
        mu_norm = distance.sqrt()
        kl = 0.5 * (mu_norm - self.mu_star) ** 2

        loss_components['mu_norm'] = mu_norm
        loss_components['var_kl'] = torch.zeros_like(mu_norm)
        loss_components['kl'] = kl
        return loss_components if output_dict else loss_components['kl']


class UniformWithGaussianTailPrior(GaussianPrior):

    def __init__(self, dim, num_priors=1, init_mean=0,
                 learned_means=False, tau=5, **kw):

        super().__init__(dim, num_priors=num_priors,
                         init_mean=init_mean,
                         learned_means=learned_means,
                         var_dim='scalar')

        self.tau = tau

        phi_tau = torch.distributions.Normal(0, 1).cdf(torch.tensor(tau)).item()
        self._alpha = np.log(2 * tau) - np.log(2 * phi_tau - 1)  # log rho(z) between -tau an tau

        self.params['distribution'] = 'uniform'
        self.params['tau'] = tau

    def kl(self, mu, log_var, y=None, output_dict=True, var_weighting=1.0):

        if y is not None and y.ndim == mu.ndim:
            expand_shape = list(mu.unsqueeze(0).shape)
            expand_shape[0] = y.shape[0]
            return self.kl(mu.expand(*expand_shape), log_var.expand(*expand_shape),
                           y=y, output_dict=output_dict, var_weighting=var_weighting)

        tau = self.tau
        alpha = self._alpha
        c = np.log(2 * np.pi)

        assert self.conditional ^ (y is None)

        if self.conditional:
            means = self.mean.index_select(0, y.view(-1)).view(*mu.shape)
        else:
            means = self.mean.unsqueeze(-1)

        loss_components = {}

        span = 2 * np.sqrt(3) * (0.5 * log_var).exp()

        mu = mu - means
        distance = mu.square()
        loss_components['distance'] = distance.sum(-1)

        a = mu - 0.5 * span
        b = mu + 0.5 * span

        a_ = tau * F.hardtanh(a / tau)
        b_ = tau * F.hardtanh(b / tau)
        Elogq = -0.5 * log_var - 0.5 * np.log(12)  # -log(span)

        negElogrho = (c + distance + span.square() / 12) / 2
        negElogrho += (alpha - c / 2) * (b_ - a_) / span
        negElogrho -= (b_.pow(3) - a_.pow(3)) / span / 6

        var_kl = (Elogq + alpha).sum(-1)
        kl = torch.max(Elogq.sum(-1) + negElogrho.sum(-1), var_kl)
        loss_components['var_kl'] = 2 * var_kl
        loss_components['kl'] = kl

        if var_weighting != 1.0:
            kl = kl + (var_weighting - 1) * var_kl
            loss_components['kl'] = kl

        return loss_components if output_dict else loss_components['kl']

    def log_density(self, z, y=None):

        assert self.conditional ^ (y is None)

        if self.conditional:
            means = self.mean.index_select(0, y.view(-1)).view(*y.shape, -1)
            z = z - means

        c = np.log(2 * np.pi)
        logp = - self._alpha * torch.ones_like(z)
        i = z.abs() > self.tau
        logp[i] = - c / 2 - z[i].square() / 2

        return logp.sum(-1)

    def __repr__(self):
        if self.num_priors > 1:
            _m = ' with {} {}means'.format(self.num_priors, 'learned ' if self.learned_means else '')
        else:
            _m = ''
        return 'uniform {c}prior{m}, tau={tau}'.format(c='conditional ' if self.conditional else '',
                                                       m=_m, tau=self.tau)
