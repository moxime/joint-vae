import os
from cvae import ClassificationVariationalNetwork as M
import torch
from utils import save_load
import copy


class IteratedModels(M):

    def __new__(cls, *models):

        m = copy.deepcopy(models[-1])
        m.__class__ = cls
        
        return m
    
    def __init__(self, *models):

        assert len(models) > 1
        self._modules = {str(_): m for _, m in enumerate(models)}
        self._models = models
        self.training_parameters = self._models.training_parameters
        
    def __len__(self):
        return len(self._models)

    def save(self, dir_name=None):
        if dir_name is None:
            dir_name = os.path.join('iterated-jobs', '-'.join(str(_.job_number) for _ in self._models))
        architecture = {_: m.saved_dir for _, m in enumerate(self._models)}
            
        save_load.save_json(architecture, dir_name, 'params.json')

    @classmethod
    def load(cls, d, *a, **kw):

        architecture = save_load.load_json(d, 'params.json')
        models = [architecture[str(_)] for _ in range(len(architecture))]

        return cls(*[M.load(_, *a, **kw) for _ in models])
            
    def evaluate(self, x,
                 y=None,
                 z_output=False,
                 **kw):

        out = [(x, y)]
        
        models = self._models
        
        for i in models[1:]:

            out.append(models[i].evaluate(*out[-1][:2], z_output=z_output, **kw))

        return (torch.stack(_) for _ in out[1:])

    def predict_after_evaluate(self, logits, losses, **kw):

        return self._models[-1].predict_after_evaluate(logits[-1], losses[-1], **kw)

    def batch_dist_measures(self):
        pass


