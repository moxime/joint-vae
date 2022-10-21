import torch
from torch.nn.functional import one_hot

TEMPS = [None, 1, 5]

NAN_TEMPS = [None, -1, 0]


def log_mean_exp(*tensors, normalize=False):

    t = torch.cat([_.unsqueeze(0) for _ in tensors])

    tref = t.max(axis=0)[0]
    dt = t - tref

    return (dt.exp().mean(axis=0).log() + tref).squeeze(0)


def posterior(logits, axis=0, temps=TEMPS):

    posterior = {}
    nan_temps = [_ for _ in temps if _ in NAN_TEMPS]

    for _ in nan_temps:
        posterior[_] = logits.clone()

    posterior.update({t: (logits / t).softmax(0) for t in temps if t not in nan_temps})

    return posterior


def joint_posterior(*zdist, axis=0, temps=TEMPS):

    z = torch.stack(zdist).sum(0)
    return posterior(-z / 2, axis=axis, temps=temps)


def mean_posterior(*p_x_y, axis=0, temps=TEMPS):

    mean_p_x_y = log_mean_exp(*p_x_y)
    return posterior(mean_p_x_y, axis=axis, temps=temps)


def voting_posterior(*y, temps=[None]):

    one_hot_ = [one_hot(_).T for _ in y]

    p_y_x = sum(one_hot_) / len(y)

    return {t: p_y_x for t in temps}
