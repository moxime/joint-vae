import logging
import numpy as np


def early_stopping(model, strategy='min', which='loss', full_valid=10):
    r""" Returns the epoch at which it should be stopped"""

    if isinstance(model, dict):
        model = model['net']
    mtype = model.type
    history = model.train_history
    ood_results = model.ood_results
    test_results = model.testing

    epochs = set(ood_results).union(test_results)
    epochs.discard(-1)

    if not epochs:
        return None

    has_validation = 'validation_loss' in history[0]

    valid_k = 'validation'
    if not has_validation:
        logging.warning('No validation has been produced for {}'.format(model.job_number))
        valid_k = 'test'

    if valid_k + '_loss' not in history[0]:
        return None

    measures = {_: history[_][valid_k + '_measures'] for _ in sorted(epochs)}
    losses = {_: history[_][valid_k + '_loss'] for _ in sorted(epochs)}

    metrics = {}

    metrics['loss'] = {_: losses[_]['total'] for _ in losses}
    if mtype in ('cvae', 'vae'):
        metrics['mse'] = {_: measures[_]['mse'] for _ in measures}

    validation = metrics[which]

    epoch = {}
    epoch['min'] = min(validation, key=validation.get)

    return epoch[strategy]


if __name__ == '__main__':

    pass
