import torch
from cvae import ClassificationVariationalNetwork as M
import utils.torch_load as torchdl
from module.priors import build_prior


class WIMVariationalNetwork(M):

    def __init__(self, *a, alternate_prior={}, **kw):

        super().__init__(*a, **kw)
        self._original_prior = self.encoder.prior,
        self._alternate_prior = build_prior(**alternate_prior)

    def alternate_prior_finetune(self, *sets, epochs=5, alpha=0.1, optimizer=None):

        if optimizer is None:
            optimizer = self.optimizer

        self.train()

        set_name = self.training_parameters['set']
        transformer = self.training_parameters['transoformer']
        data_augmentation = self.training_parameters['data_augmentation']
        batch_size = self.training_parameters['batch_size']

        trainset, testset = torchdl.get_dataset(set_name,
                                                transformer=transformer,
                                                data_augmentation=data_augmentation)

        moving_sets = {_: torchdl.get_dataset(_, transformer=transformer, splits=['test'])[1] for _ in sets}
        moving_sets['test'] = testset

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  # pin_memory=True,
                                                  shuffle=True,
                                                  num_workers=0)

        moving_loaders = {_: torch.utils.data.DataLoader(moving_sets[_],
                                                         batch_size=batch_size,
                                                         # pin_memory=True,
                                                         shuffle=True,
                                                         num_workers=0)
                          for _ in moving_sets}

        current_measures = {}
        moving_current_measures = {}

        device = next(self.parameters()).device

        for epoch in range(epochs):

            moving_iters = {_: iter(moving_loaders[_]) for _ in moving_loaders}
            moving_batches = {_: next(moving_iters[_]) for _ in moving_iters}

            for i, (x, y) in trainloader:

                optimizer.zero_grad()

                self.encoder.prior = self._original_prior
                (_, y_est, batch_losses, measures) = self.evaluate(x.to(device), y.to(device),
                                                                   current_measures=current_measures,
                                                                   batch=i,
                                                                   with_beta=True)

                self.encoder.prior = self._alternate_prior

                L = batch_losses['total'].mean()

                for x, y in moving_batches:

                    y = torch.zeros_like(y, dtype=int)
                    o = self.evaluate(x.to(device), y.to(device),
                                      current_measures=moving_current_measures,
                                      batch=i,
                                      with_beta=True)

                    _, y_est, batch_losses, measures = o

                    L += alpha * batch_losses['total'].mean()

                L.backward()
                optimizer.clip(self.parameters())
                optimizer.step()

        self.ood_detection_rate()


if __name__ == '__main__':

    C = 10
    shape = (3, 32, 32)

    m = WIMVariationalNetwork(shape, C, type='cvae')
