import os
import logging
from module.wim.job import WIMJob
from utils.save_load import fetch_models, LossRecorder, available_results, find_by_job_number, make_dict_from_model
from utils.filters import ParamFilter, DictOfListsOfParamFilters, get_filter_keys

JOB_FILE_NAME = 'jobs'


class WIMArray(WIMJob):

    def __init__(self, *a, wanted_components=[], fetch_dir='wim-jobs', **kw):

        super().__init__(*a, **kw)
        self._fecth_dir = fetch_dir
        self._jobs = []
        self._wanted_components = wanted_components

    @property
    def wanted_components(self):
        return self._wanted_components

    @wanted_components.setter
    def wanted_components(self, *components):
        if any(_ not in self.wanted_components for _ in components):
            self._jobs = []

        self._wanted_components = components

    def finetune(self, *a, **kw):

        logging.waringin('WIM array is no meant to be finetuned')

    @classmethod
    def load(cls, dir_name, *a, **kw):
        model = super().load(dir_name, *a, **kw)
        model._jobs = []
        try:
            with open(os.path.join(dir_name, JOB_FILE_NAME), 'r') as f:
                for line in f.readlines():
                    model._jobs.append(line.strip())
        except FileNotFoundError:
            logging.debug('Job file not found in {}'.format(os.path.join(dir_name, JOB_FILE_NAME)))

        logging.debug('{} jobs found'.format(len(model._jobs)))

        return model

    def fetch_jobs(self, flash=False):

        wim_filter_keys = get_filter_keys()
        wim_filter_keys = {_: wim_filter_keys[_] for _ in wim_filter_keys if _.startswith('wim')}
        filter = DictOfListsOfParamFilters()

        self_dict = make_dict_from_model(self, '')

        # print('***', {_: self_dict[_] for _ in self_dict if _.startswith('wim')})
        for k, f in wim_filter_keys.items():
            filter.add(k, ParamFilter(type=f['type'], values=[self_dict[k]]))

        fetched_jobs = fetch_models(self._fetch_dir, flash=False,
                                    build_module=True, filter=filter, load_state=False, show_debug=False)

        logging.debug('Fetched {} models with filters'.format(len(fetched_jobs)))

        fetched_jobs = [_ for _ in fetched_jobs if self == _['net']]

        logging.debug('Kept {} models with eq'.format(len(fetched_jobs)))

        if any(_ not in fetched_jobs for _ in self._jobs):
            logging.warning('Some jobs seemed to have been deleted since last time')
            self._jobs = []

        fetched_jobs = [_ for _ in fetched_jobs if _ not in self._jobs]
        logging.debug('{} new jobs fetched'.format(len(fetched_jobs)))

        self._jobs = [_['dir'] for _ in fetched_jobs]

        return fetched_jobs

    def update_records(self, flash=True):

        updated_jobs = self.fetch_jobs(flash=flash)

        recorders = {}

        for j in updated_jobs:

            a = available_results(j, where=('recorders',))
            a = a[max(a)]
            job_recorders = LossRecorder.loadall(a['rec_dir'])

            for _ in job_recorders:
                if not all(_ in job_recorders[_] for _ in self.wanted_components):
                    continue

                if _ in recorders:
                    recorders[_].merge(job_recorders[_])
                else:
                    recorders[_] = job_recorders[_].copy()


if __name__ == '__main__':

    job_dir = 'fake-jobs'
    job_number = 353527

    logging.getLogger().setLevel(logging.DEBUG)

    wim_model = find_by_job_number(job_number, job_dir=job_dir, build_module=True, load_state=False)
    print('***')
    model = WIMArray.load(wim_model['dir'], build_module=True, load_state=False)
    model._fetch_dir = job_dir
    model.update_records()
