import os
from cvae import ClassificationVariationalNetwork as M
import torch
from utils import save_load


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

        return m

    def evaluate(self, x,
                 y=None,
                 z_output=False,
                 **kw):

        out = [(x, y)]
        
        models = self._models
        
        for m in models[1:]:

            out.append(m.evaluate(*out[-1][:2], z_output=z_output, **kw))

        return (torch.stack(_) for _ in out[1:])

    def predict_after_evaluate(self, logits, losses, method='iter'):

        if method == 'iter':
            return logits[-1].max(axis=0)
        
        return self._models[-1].predict_after_evaluate(logits[-1], losses[-1], **kw)

    def batch_dist_measures(self):
        pass


