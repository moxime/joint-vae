import os
import logging
from utils.print_log import turnoff_debug
from module.wim.job import WIMJob
from utils.save_load import fetch_models, LossRecorder, available_results, find_by_job_number
from utils.save_load import make_dict_from_model, model_subdir, save_json
from utils.filters import ParamFilter, DictOfListsOfParamFilters, get_filter_keys
import torch

JOB_FILE_NAME = 'jobs'

wim_job_filter = DictOfListsOfParamFilters()
wim_job_filter.add('wim_from', ParamFilter(type=int, any_value=True))


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

    @classmethod
    def is_wim_array(cls, d):
        return os.path.exists(os.path.join(d, JOB_FILE_NAME))

    def finetune(self, *a, **kw):

        logging.warning('WIM array is no meant to be finetuned')

    def save(self, *a, **kw):
        logging.debug('Saving wim array')
        kw['except_state'] = True
        dir_name = super().save(*a, **kw)
        with open(os.path.join(dir_name, JOB_FILE_NAME), 'w') as f:
            for j in self._jobs:
                f.write(j)
                f.write('\n')
        logging.debug('Model saved in {}'.format(dir_name))
        return dir_name

    @classmethod
    def load(cls, dir_name, *a, load_state=False, **kw):
        model = super().load(dir_name, *a, load_state=load_state, **kw)
        model._jobs = []
        try:
            with open(os.path.join(dir_name, JOB_FILE_NAME), 'r') as f:
                for line in f.readlines():
                    model._jobs.append(line.strip())
        except FileNotFoundError:
            logging.debug('Job file not found in {}'.format(os.path.join(dir_name, JOB_FILE_NAME)))

        logging.debug('{} jobs found'.format(len(model._jobs)))

        return model

    def update_records(self, jobs_to_add, compute_rates=True):

        a = available_results(self, where=('recorders',), min_samples_by_class=0)
        epoch = max(a)
        a = a[epoch]
        if not a['all_sets']['recorders']:
            epoch = None
            array_recorders = {}
        else:
            rec_dir = a['rec_dir']
            array_recorders = LossRecorder.loadall(rec_dir)

        for j in jobs_to_add:

            assert model_subdir(j) not in self._jobs

            # FOR TEST, TBR
            self._jobs.append(model_subdir(j))
            a = available_results(j, where=('recorders',), min_samples_by_class=0)
            if epoch is None:
                epoch = max(a)
                rec_dir = os.path.join(self.saved_dir, 'samples', '{:04d}'.format(epoch))
                try:
                    os.makedirs(rec_dir)
                except FileExistsError:
                    pass

            else:
                assert epoch == max(a)

            a = a[epoch]
            if not a['all_sets']['recorders']:
                logging.warning('No recorders in {}'.format(model_subdir(j)))
                continue
            else:
                logging.debug('Recorders found')
            try:
                job_recorders = LossRecorder.loadall(a['rec_dir'])
            except KeyError:
                logging.error('Will CRASH!')
                return a

            try:
                self.wim_params['array_size'] += 1
            except KeyError:
                self.wim_params['array_size'] = 1

            for _ in job_recorders:
                if not all(c in job_recorders[_] for c in self.wanted_components):
                    continue

                if _ in array_recorders:
                    array_recorders[_].merge(job_recorders[_])
                else:
                    array_recorders[_] = job_recorders[_].copy()

        created_rec_str = ' -- '.join('{} of size {} for {}'.format(_,
                                                                    array_recorders[_].recorded_samples,
                                                                    ','.join(array_recorders[_].keys()))
                                      for _ in array_recorders)

        logging.info('Created recorders {} for {}...{}'.format(created_rec_str,
                                                               self.saved_dir[:20],
                                                               self.saved_dir[-20:]))

        for s, r in array_recorders.items():
            r.save(os.path.join(rec_dir, 'record-{}.pth'.format(s)))
        self.ood_detection_rates(
            #  batch_size=test_batch_size,
            #  testset=testset,
            # oodsets=oodsets,
            # num_batch='all',
            # outputs=outputs,
            # sample_dirs=sample_dirs,
            recorders=array_recorders,
            from_where=('recorders',),
            print_result='*')

        return array_recorders

    def concatenate_samples(sample_dirs=[]):

        for sdir in samples_dir:
            os.makedirs(self.saved_dir, sdir, exist_ok=True)
        for j in self._jobs:
            for sdir in sample_dirs:
                array_sdir = os.path.join(self.saved_dir, sdir)
                job_sdir = os.path.join(j, sdir)
                sample_files = [_ for os.path.listdir(job_sdir) if _.startswith('samples')]

    @ classmethod
    def collect_processed_jobs(cls, job_dir, flash=False):

        logging.info('Collect processed jobs in {}'.format(job_dir))
        jobs = []
        models = fetch_models(job_dir, flash=flash)
        for m in models:
            try:
                with open(model_subdir(m, JOB_FILE_NAME)) as f:
                    for _ in f.readlines():
                        jobs.append(_.strip())

            except FileNotFoundError:
                pass

        logging.debug('Found {} already processed jobs'.format(len(jobs)))

        return jobs


if __name__ == '__main__':

    import sys
    import argparse
    import configparser
    from utils.save_load import find_by_job_number
    from utils.parameters import next_jobnumber, set_log
    from module.optimizers import Optimizer

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--debug', action='store_true')
    conf_parser.add_argument('--verbose', '-v', action='count', default=0)
    conf_parser.add_argument('--config-file', default='config.ini')

    conf_args, remaining_args = conf_parser.parse_known_args()

    config = configparser.ConfigParser()
    config.read(conf_args.config_file)

    config_params = config['wim-default']

    defaults = {}

    defaults.update(config_params)

    parser = argparse.ArgumentParser(parents=[conf_parser],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device')
    parser.add_argument('--array-job-dir')
    parser.add_argument('-J', '--wim-job-dir')
    parser.add_argument('--job-number', '-j', type=int)
    parser.add_argument('--from-job', type=int, nargs='*')

    parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_args)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    job_number = args.job_number
    if not job_number:
        job_number = next_jobnumber()

    log_dir = os.path.join(args.output_dir, 'log')
    log = set_log(conf_args.verbose, conf_args.debug, log_dir, job_number=job_number)

    log.debug('$ ' + ' '.join(sys.argv))

    if args.from_job:
        logging.info('Will only look for jobs/arrays from {}'.format(' '.join(map(str, args.from_job))))
        wim_job_filter.add('wim_from', ParamFilter(type=int, values=args.from_job))

    wim_arrays = fetch_models(args.array_job_dir, filter=wim_job_filter, flash=False, light=True)

    logging.info('Fetched {} wim arrays from {}'.format(len(wim_arrays), args.array_job_dir))

    wim_jobs = fetch_models(args.wim_job_dir, filter=wim_job_filter, flash=False, light=True)

    logging.info('Fetched {} wim jobs from {}'.format(len(wim_jobs), args.wim_job_dir))

    wim_jobs_already_processed = WIMArray.collect_processed_jobs(args.array_job_dir)

    logging.info('{} jobs already processed'.format(len(wim_jobs_already_processed)))

    wim_jobs = [_ for _ in wim_jobs if model_subdir(_) not in wim_jobs_already_processed]

    logging.info('{} jobs remaining'.format(len(wim_jobs)))

    processed_jobs = []

    for i, array in enumerate(wim_arrays):

        if not wim_jobs:
            logging.info('No more wim jobs left for processing')
            break
        with turnoff_debug():
            wim_array = WIMArray.load(model_subdir(array), load_state=False)
        wim_jobs_alike = wim_array.fetch_jobs_alike(models=wim_jobs)
        if not wim_jobs_alike:
            logging.info('Skipping wim array {}, no jobs alike (among {})'.format(i, len(wim_jobs)))
            continue

        """
        Process one wim array among sames (smallest job number)
        """
        wim_arrays_alike = wim_array.fetch_jobs_alike(models=wim_arrays)
        kept_wim_array = min(wim_arrays_alike, key=lambda j: j['job'])
        with turnoff_debug():
            wim_array = WIMArray.load(kept_wim_array['dir'], load_state=False)

        logging.info('Processing {} jobs alike (array {})'.format(len(wim_jobs_alike), i))
        wim_array.update_records([WIMJob.load(_['dir'], build_module=False) for _ in wim_jobs_alike])
        wim_array.save(model_subdir(wim_array))

        """
        Cleaning

        """
        for _ in wim_jobs_alike:
            processed_jobs.append(_)
            wim_jobs.remove(_)

        for _ in wim_arrays_alike:
            if _['job'] != array['job']:
                wim_arrays.remove(_)

    logging.warning('{} processed and {} unprocessed jobs'.format(len(processed_jobs), len(wim_jobs)))

    with open(os.path.join(args.wim_job_dir, 'unprocessed-{}'.format(args.from_job)), 'w') as f:
        for j in wim_jobs:
            f.write(str(j['job']))
            f.write('\n')
