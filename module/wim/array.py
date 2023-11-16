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

    def finetune(self, *a, **kw):

        logging.warning('WIM array is no meant to be finetuned')

    def save(self, dir_name, *a, **kw):
        logging.debug('Saving wim array')
        kw['except_state'] = True
        dir_name = super().save(*a, **kw)
        save_json(self._jobs, dir_name, JOB_FILE_NAME)
        logging.debug('Model saved in {}'.format(dir_name))
        return dir_name

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

    def update_records(self, jobs_to_add):

        recorders = {}

        for j in jobs_to_add:

            # FOR TEST, TBR
            self._jobs.append(model_subdir(j))
            return
            a = available_results(j, where=('recorders',))
            a = a[max(a)]
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

            for _ in job_recorders:
                if not all(_ in job_recorders[_] for _ in self.wanted_components):
                    continue

                if _ in recorders:
                    recorders[_].merge(job_recorders[_])
                else:
                    recorders[_] = job_recorders[_].copy()

        logging.info('Created {} recorders for {}...{}'.format(len(recorders),
                                                               self.saved_dir[:20],
                                                               self.saved_dir[-20:]))
        return recorders

    @classmethod
    def collect_processed_jobs(cls, job_dir, flash=False):

        jobs = []
        models = fetch_models(job_dir, flash=False)
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
    parser.add_argument('--arrays-job-dir')
    parser.add_argument('-J', '--target-job-dir')
    parser.add_argument('--job-number', '-j', type=int)

    parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_args)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    job_number = args.job_number
    if not job_number:
        job_number = next_jobnumber()

    log_dir = os.path.join(args.output_dir, 'log')
    log = set_log(conf_args.verbose, conf_args.debug, log_dir, job_number=job_number)

    log.debug('$ ' + ' '.join(sys.argv))

    wim_jobs = fetch_models(args.target_job_dir, filter=wim_job_filter, flash=False, light=True)

    logging.info('Fetched {} wim jobs from {}'.format(len(wim_jobs), args.target_job_dir))

    wim_arrays = fetch_models(args.arrays_job_dir, filter=wim_job_filter, flash=False, light=True)

    logging.info('Fetched {} wim arrays from {}'.format(len(wim_arrays), args.arrays_job_dir))

    wim_jobs_already_processed = WIMArray.collect_processed_jobs(args.arrays_job_dir)

    logging.info('{} jobs already processed'.format(len(wim_jobs_already_processed)))

    wim_jobs = [_ for _ in wim_jobs if model_subdir(_) not in wim_jobs_already_processed]

    wim_job_by_params = []

    list_of_wim_arrays = []

    for j in wim_arrays:
        wim_array = WIMArray.load(model_subdir(j), load_state=False)
        wim_arrays_alike = wim_array.fetch_jobs_alike(models=wim_arrays)
        for _ in wim_arrays_alike:
            if _['job'] != j['job']:
                wim_arrays.remove(_)

        kept_wim_array = min(wim_arrays_alike, key=lambda j: j['job'])

        wim_jobs_alike = wim_array.fetch_jobs_alike(models=wim_jobs)

        wim_array = WIMArray.load(kept_wim_array['dir'], load_state=False)

        wim_array.update_records([WIMJob.load(_['dir'], build_module=False) for _ in wim_jobs_alike])

    raise KeyboardInterrupt
    for j in wim_jobs:

        mdir = model_subdir(j)
        for dict_of_models in wim_job_by_params:
            if mdir in dict_of_models:
                logging.debug('{} already in list of len {}'.format(mdir[-100:], len(dict_of_models)))
                break
        else:
            logging.debug('{} not yet in lists'.format(mdir[-100:]))

            wim_job = WIMJob.load(mdir, load_state=False)
            jobs_alike = wim_job.fetch_jobs_alike(job_dir=args.target_job_dir, flash=True)

            logging.debug('Found {} jobs alike'.format(len(jobs_alike)))
            wim_job_by_params.append({model_subdir(_): _ for _ in jobs_alike})

    logging.info('Found {} list of jobs of total size {}'.format(len(wim_job_by_params),
                                                                 sum(len(_) for _ in wim_job_by_params)))

    for dict_of_models in wim_job_by_params:

        first_mdir = list(dict_of_models)[0]
        logging.info('Processing wim job {} for a list of {}'.format(first_mdir[-60:], len(dict_of_models)))
        wim_job = WIMJob.load(first_mdir, load_state=False)
        kept_wim_arrays = wim_job.fetch_jobs_alike(models=wim_arrays)

        if not kept_wim_arrays:
            logging.warning('wim array not created for {}'.format(wim_job.wim_params))
            continue

        wim_array_dict = min(kept_wim_arrays, key=lambda d: d.get('job'))

        if len(kept_wim_arrays) > 1:
            logging.warning('{} > 1 arrays with same parameters. Keeping {}'.format(len(kept_wim_arrays),
                                                                                    wim_array_dict['job']))
        wim_array = WIMArray.load(wim_array_dict['dir'], load_state=False)
        wim_array.update_records(dict_of_models.values())

    raise KeyboardInterrupt
    save_dir_root = os.path.join(args.arrays_job_dir, dataset,
                                 model.print_architecture(sampling=False),
                                 'wim')

    save_dir = os.path.join(save_dir_root, f'{job_number:06d}')

    output_file = os.path.join(args.output_dir, f'train-{job_number:06d}.out')

    log.debug(f'Outputs registered in {output_file}')
    outputs = EpochOutput()
    outputs.add_file(output_file)

    model.job_number = job_number
    model.saved_dir = save_dir

    model.encoder.prior.mean.requires_grad_(False)
    alternate_prior_params = model.encoder.prior.params.copy()
    alternate_prior_params['learned_means'] = False

    alternate_prior_params['mean_shift'] = args.prior_means
    if args.prior:
        alternate_prior_params['distribution'] = args.prior
    alternate_prior_params['tau'] = args.tau

    model.set_alternate_prior(**alternate_prior_params)
    model.wim_params['from'] = args.job

    with model.original_prior as p1:
        with model.alternate_prior as p2:
            log.info('WIM from {} to {}'.format(p1, p2))

            if p1.num_priors > 1:
                log.info('Means from {:.3} to {:.3}'.format(p1.mean.std(0).mean(),
                                                            p2.mean.std(0).mean()))

    try:
        model.to(device)
    except Exception:
        log.warning('Something went wrong when trying to send to {}'.format(device))

    optimizer = None

    if args.lr:
        logging.info('New optimizer')
        optimizer = Optimizer(model.parameters(), optim_type='adam', lr=args.lr, weight_decay=args.weight_decay)

    wim_sets = sum((_.split('-') for _ in args.wim_sets), [])
    model.finetune(*wim_sets,
                   train_size=args.train_size,
                   epochs=args.epochs,
                   moving_size=args.moving_size,
                   test_batch_size=args.test_batch_size,
                   alpha=args.alpha,
                   ood_mix=args.mix,
                   optimizer=optimizer,
                   outputs=outputs)

    model.save(model.saved_dir)
