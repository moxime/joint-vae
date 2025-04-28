from utils.save_load import find_by_job_number, LossRecorder, available_results
import torch

j = 367028
j = 369332
j = 381599
j = 381986
j = 367028
j = 317032

job_dir = '.test-wim-jobs'
job_dir = '.test-wim-arrays'
job_dir = 'wim-arrays'
job_dir = 'jobs'

dset = 'cifar10'

m = find_by_job_number(j, build_module=True, load_state=False, job_dir=job_dir)

a = available_results(m)

a = a[max(a)]

r_ = LossRecorder.loadall(a['rec_dir'])

k_ = ('zdist', 'total')

normalize = False
normalize = True

for i, s in enumerate(r_):

    loss = {k: r_[s][k].min(0)[0] for k in k_}
    loss.update({k + '*': r_[s][k + '*'].min(0)[0] for k in k_})
    loss.update({k + '~': r_[s][k + '~'] for k in k_ if k + '~' in r_[s]})
    loss.update({'d' + k: loss[k + '*'] - loss[k] for k in k_})
    loss.update({'d' + k + '~': loss[k + '*'] - loss[k + '~'] for k in k_ if k + '~' in r_[s]})

    if normalize:
        if not i:
            norm = {k: (loss[k].mean(), loss[k].std()) for k in loss}

        loss = {k: (loss[k] - norm[k][0]) / norm[k][1] for k in loss}

    for _ in (0, 2, 3, 5):
        loss[_] = _ + torch.randn(1000)

    if not normalize or i:
        if normalize:
            dkl = {k: 0.5 * (loss[k].mean().square() - loss[k].var().log() + loss[k].var() - 1) for k in loss}
        else:
            dkl = {k: 0 for k in loss}
        for k in loss:
            print('{:7} {:7}: {:+9.4g} +/-{:7.3g} {:.2f}'.format(s, k, loss[k].mean(), loss[k].std(), dkl[k]))
