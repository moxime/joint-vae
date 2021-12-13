from utils.testing import worth_computing, testing_plan
from cvae import ClassificationVariationalNetwork as M
from utils.save_load import available_results, make_dict_from_model, LossRecorder
import logging

logging.getLogger().setLevel(logging.DEBUG)

mdir = '/tmp/000033'

print('Loading')
model = M.load(mdir, load_state=False)

print('Loaded')

"""
acc = {}
acc = {_: model.accuracy(wygiwyu=True, wanted_epoch=_) for _ in (0, 10, 200, 'last')}

print(acc)
"""
model.trained = 2001

model.testing[2000].pop('iws')

available = available_results(model,
                              predict_methods='all',
                              samples_available_by_compute=15000,
                              ood_methods='all',
                              where=('json', 'recorders', 'compute'),
                              misclass_methods=[],
                              wanted_epoch=2000)


recorders = LossRecorder.loadall(model.saved_dir + '/samples/last')

# plan = testing_plan(model, predict_methods=[], misclass_methods=[], wanted_epoch=30)
model.ood_detection_rates(wanted_epoch=10, wygiwyu=True)
model.accuracy(wanted_epoch=10, wygiwyu=True)
mdict = make_dict_from_model(model, model.saved_dir, wanted_epoch=10)
