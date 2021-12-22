from utils.testing import worth_computing, testing_plan
from cvae import ClassificationVariationalNetwork as M
from utils.save_load import available_results, make_dict_from_model, LossRecorder, find_by_job_number, load_json
import logging

logging.getLogger().setLevel(logging.DEBUG)

mdir = '/tmp/000033'
mdir = '/tmp/000186'
mdir = '/tmp/151320'
mdir = '/tmp/151024'
mdir = '/tmp/151409'
mdir = '/tmp/151020'
mdir = '/tmp/148722'

print('Loading')
model = M.load(mdir, load_state=False)

new = M((3, 32, 32), 10, type_of_net='vib')

print('Loaded')

"""
acc = {}
acc = {_: model.accuracy(wygiwyu=True, wanted_epoch=_) for _ in (0, 10, 200, 'last')}

print(acc)
"""
# model.trained = 2000

# model.testing[2000].pop('iws')
# model.ood_results.pop(2000)

where = ('json', 'recorders')
available = available_results(model,
                              predict_methods='all',
                              samples_available_by_class=900,
                              min_samples_by_class=800,
                              ood_methods='all',
                              where=where,
                              misclass_methods=[],
                              wanted_epoch=200, epoch_tolerance=5)

# recorders = LossRecorder.loadall(model.saved_dir + '/samples/last')

# plan = testing_plan(model, predict_methods=[], misclass_methods=[], wanted_epoch=30)
# ood = model.ood_detection_rates(epoch=2000)
    
# acc = model.accuracy(epoch=1980, print_result=True, sample_dirs=['/tmp'])
mdict = make_dict_from_model(model, model.saved_dir, wanted_epoch=240)
