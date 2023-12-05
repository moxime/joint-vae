from utils.save_load import find_by_job_number, LossRecorder, available_results

j = 367028
j = 369332
j = 381599

dset = 'cifar10'

m = find_by_job_number(j, build_module=True, job_dir='.test-wim-arrays')

a = available_results(m)

a = a[max(a)]

r_ = LossRecorder.loadall(a['rec_dir'])

k_ = ('zdist', 'total')
for i, s in enumerate(r_):

    loss = {k: r_[s][k].min(0)[0] for k in k_}
    loss.update({k + '*': r_[s][k + '*'].min(0)[0] for k in k_})
    loss.update({k + '~': r_[s][k + '~'] for k in k_ if k + '~' in r_[s]})
    loss.update({'d' + k: loss[k + '*'] - loss[k] for k in k_})
    loss.update({'d' + k + '~': loss[k + '*'] - loss[k + '~'] for k in k_ if k + '~' in r_[s]})

    if normalize and not i:

    for k in loss:
        print('{:7} {:7}: {:6.4g} +/-{:4.3g}'.format(s, k, loss[k].mean(), loss[k].std()))
