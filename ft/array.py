import os
import logging
from utils.print_log import turnoff_debug
from ft import WIMJob
from ft.job import FTJob
from utils.save_load import fetch_models, LossRecorder, available_results, find_by_job_number
from utils.save_load import make_dict_from_model, model_subdir, save_json, SampleRecorder
from utils.filters import ParamFilter, DictOfListsOfParamFilters, get_filter_keys
import torch
import time

JOB_FILE_NAME = 'jobs'

wim_job_filter = DictOfListsOfParamFilters()
wim_job_filter.add('wim_from', ParamFilter(type=int, any_value=True))


class WIMArray(FTJob):

    def __init__(self, *a, fetch_dir='wim-jobs', **kw):

        super().__init__(*a, **kw)
        self._fecth_dir = fetch_dir
        self._jobs = {'known': set(), 'rec': set()}
        self._rec_dir = None

    @classmethod
    def is_wim_array(cls, d):
        return os.path.exists(os.path.join(d, JOB_FILE_NAME))

    def finetune(self, *a, **kw):

        logging.warning('WIM array is no meant to be fine-tuned')

    def job_files(self, k):

        if not hasattr(self, 'saved_dir'):
            raise FileNotFoundError

        if k == 'known':
            return os.path.join(self.saved_dir, JOB_FILE_NAME)

        if not self._rec_dir:
            raise FileNotFoundError

        if k == 'rec':
            return os.path.join(self._rec_dir, JOB_FILE_NAME)

    def _add_job(self, k, job):

        self._jobs[k].add(model_subdir(job))

    def save(self, *a, **kw):

        logging.debug('Saving wim array')
        kw['except_state'] = True
        dir_name = super().save(*a, **kw)

        for _ in self._jobs:
            with open(self.job_files(_), 'w') as f:
                for j in self._jobs[_]:
                    f.write(j)
                    f.write('\n')
                logging.debug('{} jobs registered as {} in {}'.format(len(self._jobs[_]), _, self._rec_dir))
        return dir_name

    @classmethod
    def load(cls, dir_name, *a, load_state=False, **kw):

        model = super().load(dir_name, *a, load_state=load_state, **kw)

        a = available_results(model, where=('recorders',), min_samples_by_class=0)
        epoch = max(a)
        a = a[epoch]
        if a['all_sets']['recorders']:
            model._rec_dir = a['rec_dir']

        if not hasattr(model, '_jobs'):
            # just a Shell for printing results
            return model
        for _ in model._jobs:
            try:
                fp = model.job_files(_)
                with open(fp, 'r') as f:
                    for line in f.readlines():
                        model._add_job(_, line)
            except FileNotFoundError:
                logging.debug('Job file not found in {}'.format(fp))

            logging.debug('{} {} jobs found'.format(len(model._jobs[_]), _))

        assert model._jobs['rec'].issubset(model._jobs['known']), 'some recorded jobs are not known'

        if not model._jobs['rec']:
            model.wim_params['array_size'] = 0
            model._rec_dir = None
        return model

    def register_jobs(self, *jobs, update_records=True, **kw):

        logging.info('Registering {} jobs (now:{})'.format(len(jobs), len(self._jobs['known'])))

        known_jobs = len(self._jobs['known'])

        for j in jobs:
            self._add_job('known', j)

        registered_jobs = len(self._jobs['known']) - known_jobs

        logging.info('Registered {} job'.format(registered_jobs))

        if update_records:
            self._update_records(**kw)

    def _update_records(self, compute_rates=True):

        jobs_to_add = self._jobs['known'].difference(self._jobs['rec'])

        has_been_updated = False

        if not self._rec_dir:
            array_recorders = {}
        else:
            array_recorders = LossRecorder.loadall(self._rec_dir)

        logging.info('Updating records with {} jobs'.format(len(jobs_to_add)))

        cleaned_records = []
        for j in jobs_to_add:

            self._add_job('rec', j)
            a = available_results(WIMJob.load(j, build_module=False, load_state=False),
                                  where=('recorders',), min_samples_by_class=0)
            epoch = max(a)
            a = a[epoch]
            if not self._rec_dir:
                if not hasattr(self, 'saved_dir'):
                    raise FileNotFoundError('current array not saved')
                self._rec_dir = os.path.join(self.saved_dir, 'samples', '{:04d}'.format(epoch))
                try:
                    os.makedirs(self._rec_dir)
                except FileExistsError:
                    pass

            if not a['all_sets']['recorders']:
                logging.warning('No recorders in {}'.format(model_subdir(j)))
                continue
            else:
                logging.debug('Recorders found')
            job_recorders = LossRecorder.loadall(a['rec_dir'])

            job_recorders_pre = LossRecorder.loadall(os.path.join(a['rec_dir'], 'init'))

            # sanity check
            for s, rec in job_recorders.items():
                for k in rec:
                    if k.endswith('@'):
                        if rec[k].ndim == 2:
                            cleaned_records.append(k)
                            logging.debug('Cleaning rec for {}/{}'.format(s, k))
                            rec._tensors[k] = rec._tensors[k][0, :]

            for s, job_rec in job_recorders_pre.items():
                job_rec._tensors = {'pre-{}'.format(k): job_rec._tensors[k]
                                    for k in job_rec}

                job_recorders[s].merge(job_rec, axis='keys')

            try:
                self.wim_params['array_size'] += 1
            except KeyError:
                self.wim_params['array_size'] = 1

            for _ in job_recorders:

                if _ in array_recorders:
                    array_recorders[_].merge(job_recorders[_])
                else:
                    array_recorders[_] = job_recorders[_].copy()

            has_been_updated = True

        if cleaned_records:
            logging.info('Cleaned {} tensors for {}'.format(len(cleaned_records), ', '.join(set(cleaned_records))))

        created_rec_str = ' -- '.join('{} of size {} for {}'.format(_,
                                                                    array_recorders[_].recorded_samples,
                                                                    ','.join(array_recorders[_].keys()))
                                      for _ in array_recorders)

        logging.info('Created recorders {} for {}...{}'.format(created_rec_str,
                                                               self.saved_dir[:20],
                                                               self.saved_dir[-20:]))

        for s, r in array_recorders.items():
            r.save(os.path.join(self._rec_dir, 'record-{}.pth'.format(s)))

            # print('***', s)
            # for _ in r:
            #     print(_, *r._tensors[_].shape, '--', *r[_].shape)

        if compute_rates and has_been_updated:
            self.ood_detection_rates(
                recorders=array_recorders,
                from_where=('recorders',),
                print_result='*')

        elif not compute_rates:
            logging.info('Does not compute rate')

        else:
            logging.info('Does not compute rate because not updated')

        return array_recorders

    def concatenate_samples(self, *jobs, sample_subdirs=[]):

        for sdir in sample_subdirs:
            array_sdir = model_subdir(self, sdir)
            os.makedirs(array_sdir, exist_ok=True)
            array_sample_rec = {}
            for j in jobs:
                job_sdir = model_subdir(j, sdir)
                job_sample_rec = SampleRecorder.loadall(job_sdir)
                if not array_sample_rec:
                    array_sample_rec = job_sample_rec
                else:
                    for _ in array_sample_rec:
                        array_sample_rec[_].merge(job_sample_rec[_])

            for _ in array_sample_rec:
                spth = os.path.join(array_sdir, 'samples-{}.pth'.format(_))
                array_sample_rec[_].save(spth, append=True)

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
    parser.add_argument('--re', action='store_true')

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

    if args.re:
        wim_jobs_already_processed = []
    else:
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
        wim_array.register_jobs(*[_['dir'] for _ in wim_jobs_alike])
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
