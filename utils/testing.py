import logging
import numpy as np


def early_stopping(model, strategy='min', which='loss', full_valid=10):
    r""" Returns the epoch at which it should be stopped"""

    if isinstance(model, dict):
        model = model['net']
    mtype = model.type
    history = model.train_history
    has_validation = 'validation_loss' in history

    valid_k = 'validation'
    if not has_validation:
        logging.warning('No validation has been produced for {}'.format(model.job_number))
        valid_k = 'test'

    if valid_k + '_loss' not in history:
        return None

    measures = history[valid_k + '_measures']
    losses = history[valid_k + '_loss']

    metrics = {}

    kl = np.array([_['kl'] for _ in history[valid_k + '_loss']])
    metrics['loss'] = np.array([_['total'] for _ in history[valid_k + '_loss']])
    if mtype in ('cvae', 'vae'):
        sigma = np.array([_['sigma'] for _ in history[valid_k + '_measures']])
        metrics['mse'] = np.array([_['mse'] for _ in history[valid_k + '_measures']])

    validation = metrics[which]
    epoch = {}

    if not len(validation[::full_valid]):
        return None

    epoch['min'] = validation[::full_valid].argmin() * full_valid

    return epoch[strategy]


if __name__ == '__main__':

    pass
