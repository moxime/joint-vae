import logging
from .job import WIMJob
from utils.save_load import fetch_models, LossRecorder, available_results
from utils.filters import ParamFilter, DictOfListsOfParamFilters

JOB_FILE_NAME = 'jobs'


class WIMArray(WIMJob):

    def __init__(*a, fetch_dir='wim-jobs', **kw):

        super().__init__(*a, **kw)
        self._fecth_dir = fetch_dir

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

    def fetch_jobs(self, flash=False):

        filter = DictOfListsOfParamFilters()
        filter.add('wim_hash', ParamFilter(type=int, values=[hash(self)]))
        fetched_jobs = fetch_models(self._fetch_dir, flash=False,
                                    load_net=True, filter=filter, load_state=False)

        logging.debug('Fetched {} models with hash'.format(len(fetched_jobs)))
        fetched_jobs = sorted([_ for _ in fetched_jobs if self == _], key=lambda m: m.get('job_number'))
        logging.debug('Kept {} models with eq'.format(len(fetched_jobs)))

    def update_records(self, flash=True, wanted_components=[]):

        updated_jobs = self.fetch_jobs(flash=flash)

        recorders = {}

        for j in updated_jobs:

            a = available_results(j, where=('recorders',))
            a = a[max(a)]
            job_recorders = LossRecorder.loadall(a['rec_dir'])

            for _ in job_recorders:
                if _ in recorders:
                    recorders[_].merge(job_recorders[_])
                else:
                    recorders[_] = job_recorders[_].copy()
