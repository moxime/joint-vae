import os
from cvae import ClassificationVariationalNetwork as M
import torch
from utils import save_load
import logging


class IteratedModels(M):

    def __new__(cls, *models):

        m = models[-1].copy()
        m.__class__ = cls
        
        return m
    
    def __init__(self, *models):

        assert len(models) > 1
        self._modules = {str(_): m for _, m in enumerate(models)}
        self._models = models
        self.predict_methods = ['iter']
        self.ood_results = {}
        self.testing = {}
        # self.training_parameters = self._models.training_parameters
        
    def __len__(self):
        return len(self._models)

    def to(self, device):
        for m in self._models:
            m.to(device)
    
    def save(self, dir_name=None):
        if dir_name is None:
            dir_name = os.path.join('iterated-jobs', '-'.join(str(_.job_number) for _ in self._models))
        architecture = {_: m.saved_dir for _, m in enumerate(self._models)}
            
        save_load.save_json(architecture, dir_name, 'params.json')
        save_load.save_json(self.testing, dir_name, 'test.json')
        save_load.save_json(self.ood_results, dir_name, 'ood.json')
        self.saved_dir = dir_name
        
    @classmethod
    def load(cls, dir_name, *a, **kw):

        architecture = save_load.load_json(dir_name, 'params.json')
        models = [architecture[str(_)] for _ in range(len(architecture))]

        m = cls(*[M.load(_, *a, **kw) for _ in models])

        try:
            m.testing = save_load.load_json(dir_name, 'test.json', presumed_type=int)
        except(FileNotFoundError):
            pass

        try:
            m.ood_results = save_load.load_json(dir_name, 'ood.json', presumed_type=int)
        except(FileNotFoundError):
            pass

        m.saved_dir = dir_name
        
        return m

    def evaluate(self, x,
                 y=None,
                 z_output=False,
                 **kw):

        input = {'x': x, 'y': y, 'z_output': z_output}

        x_ = []
        y_ = []
        losses_ = []
        measures_ = []

        mse_ = []

        for m in self._models:

            out = m.evaluate(**input, **kw)
            input['x'] = out[0][1]
            input['y'] = out[1].argmax(-1) if y else None

            """
            for k in 'xy':
                print('***', k, ':', *input[k].shape if input[k] != None else 'None')

            for k in out[2]:
                print('*** losses', k, ':', *out[2][k].shape)

            for k in out[3]:
                print('*** meas', k, ':', type(out[3][k]))
            """

            x_.append(out[0])
            y_.append(out[1])
            losses_.append(out[2])
            measures_.append(out[3])

        x_ = torch.stack(x_)
        y_ = torch.stack(y_)

        input_dims = tuple([0] + [_ - self.input_dim for _ in range(self.input_dim)])
        
        for i in range(len(x_) + 1):
            for j in range(i):

                x_i = x_[i - 1][1:]
                x_j = x.unsqueeze(0) if not j else x_[j - 1][1:]

                i_ = 'I'  if not i else f'O{i}'
                j_ = 'I'  if not j else f'O{j}'
                print('***', i_, ':',  *x_i.shape, '--', j_, ':',  *x_j.shape)

                mse_.append((x_i - x_j).pow(2).mean(input_dims))

                print('*** mse:', *mse_[-1].shape)

        output_losses = {}
        output_measures = {}

        for k in losses_[0]:
            output_losses[k] = torch.stack([_[k] for _ in losses_])

        for k in measures_[0]:
            output_measures[k] = torch.tensor([_[k] for _ in measures_])

        output_losses['mse'] = torch.stack(mse_)
            
        return x_, y_, output_losses, output_measures

    def predict_after_evaluate(self, logits, losses, method='iter'):

        if method == 'iter':
            return logits[-1].max(axis=0)
        
        return self._models[-1].predict_after_evaluate(logits[-1], losses[-1], **kw)

    def batch_dist_measures(self):
        pass


def iterate_with_prior(logp_x_y):

    """Args:

    -- p_x_y is a tensor of dim MxCxN with M the number of models, C
    the number of labels and N the number of sample

    """

    M, C, N = logp_x_y.shape
    prior = torch.ones(C, N, device=logp_x_y.device) / C
    posterior = torch.zeros_like(logp_x_y)

    for i in range(M):

        joint = logp_x_y[i] * prior
        p_x = joint.sum(0, keepdim=True)
        # print('***', prior.shape, joint.shape, p_x.shape)
        posterior[i] = joint / p_x
        prior = posterior[i]

    return posterior
        
