from torch import optim
from torch import nn
import torch
import logging

default_lr = {'sgd': 0.01,
              'adam': 0.001}

params_by_type = {'sgd': ('momentum', 'nesterov', 'weight_decay'),
                  'adam': ('betas', 'weight_decay', 'amsgrad')}

class Optimizer:
        
    def __init__(self, parameters, optim_type='adam', lr=0, lr_decay=0, epoch=0, **kw):

        self.kind = optim_type

        if not lr:
            lr = default_lr[optim_type]

        self.params = {'optim_type': optim_type,
                       'lr': lr,
                       'lr_decay': lr_decay}
        self.params.update(kw)
        
        self.init_lr = lr

        _ = optim_type
        if _ == 'sgd':
            constructor = optim.SGD
        elif _ == 'adam':
            constructor = optim.Adam
            
        self._opt = constructor(parameters, lr=lr, **kw)

        self.lr_decay = lr_decay
        if lr_decay:
            self._lr_scheduler = optim.lr_scheduler.ExponentialLR(self._opt,
                                                                  gamma=1-lr_decay,
                                                                  last_epoch=-1)

    @property
    def lr(self):

        return self._opt.param_groups[0]['lr']

    @classmethod
    def load(cls, params_dict, state_dict, epoch):

        o = cls.__init__(**params_dict)
        o._opt.load(**state_dict)
        o.update_scheduler(epoch)

        return o

    def __str__(self):

        return self.__format__('10')
        
    def __format__(self, format_spec):

        try:
            level = int(format_spec)
        except ValueError:
            level = 0
            
        if not level:
            return self.__str__()
        
        s_ = [self.kind]
        s_ += [f'-lr={self.init_lr}']
        if self.lr_decay: s_ += [f'-decay={self.lr_decay}']
        else: s_ += ['']

        d_ = self._opt.param_groups[0]

        s = ''
        for k in params_by_type[self.kind]:
            v = d_[k]
            if v:
                if type(v) is bool:
                    s+= f'{v.lower()}'
                else:
                    s += f'-{k}={d_[k]}'
        s_.append(s)
                    
        return ''.join(s_[:level])

    def zero_grad(self, *a, **kw):
        self._opt.zero_grad(*a, **kw)
    
    def step(self):

        self._opt.step()

    def update_lr(self):
        if self.lr_decay:
            old_lr = self.lr
            self._lr_scheduler.step()
            logging.debug(f'lr updated from {old_lr:.4e} to {self.lr:.4e}')

    def update_scheduler_from_epoch(self, n):

        for i in range(n):
            self._lr_scheduler.step()


if __name__ == '__main__':

    p = nn.Parameter(torch.randn(1))

    sgd = Optimizer([p], optim_type='sgd', lr_decay=0.01)
    adam = Optimizer([p], lr_decay=0, optim_type='adam')
