# model/wim

import os
import logging
import torch

from utils.print_log import EpochOutput, turnoff_debug

from utils.save_load import model_subdir, SampleRecorder

from .job import DontDoFineTuning
from .wim import WIMJob
from .scheduler import Scheduler

from .array import WIMArray

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
    conf_parser.add_argument('--job-number', '-j', type=int)
    conf_parser.add_argument('--sampling-seed', '-S', type=int)
    conf_parser.add_argument('--sampling-task', '-T', type=int, default=0)
    conf_parser.add_argument('--sampling-task-shift', type=int, default=0)

    conf_parser.add_argument('--args-from-file', nargs=2)

    conf_args, remaining_args = conf_parser.parse_known_args()

    config = configparser.ConfigParser()
    config.read(conf_args.config_file)

    config_params = config['wim-default']

    defaults = {}

    defaults.update(config_params)

    parser = argparse.ArgumentParser(parents=[conf_parser],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device')
    parser.add_argument('job', type=int)
    parser.add_argument('-J', '--source-job-dir')
    parser.add_argument('-W', '--wim-job-dir')
    parser.add_argument('-A', '--array-job-dir')

    parser.add_argument('--wim-sets', nargs='*', default=[])
    parser.add_argument('--alpha', type=float)

    parser.add_argument('--mix', type=float)

    parser.add_argument('-N', '--train-size', type=int)
    parser.add_argument('-n', '--moving-size', type=int)
    parser.add_argument('--epochs', type=int)

    parser.add_argument('--padding', type=float, nargs='?', const=1.0, default=0.)
    parser.add_argument('--padding-sets', nargs='*', default='')
    parser.add_argument('--mix-padding', type=float, nargs='?', const=1.0, default=0.)

    parser.add_argument('--test-batch-size', type=int)

    parser.add_argument('--prior', choices=['gaussian', 'tilted', 'uniform'])
    parser.add_argument('--prior-means', type=float)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)

    parser.add_argument('-a', '--array', type=int, nargs='*')

    parser.add_argument('--do-not-collect-jobs', action='store_false', dest='collect_jobs')

    parser.add_argument('--inspection', action='store_true')

    parser.set_defaults(**defaults)

    try:
        conf_args.sampling_task += conf_args.sampling_task_shift
    except TypeError:
        pass

    if conf_args.args_from_file:
        arg_str = conf_args.args_from_file
        sch = Scheduler(arg_str[0], index=int(arg_str[1]))
        logging.info('Args from file: {}'.format(sch.line))
        args = parser.parse_args(sch.line.split(), namespace=conf_args)

    else:
        sch = Scheduler(file_path=os.path.join('grid', str(conf_args.sampling_seed))
                        if conf_args.sampling_seed is not None else None,
                        index=conf_args.sampling_task)

        args = parser.parse_args(remaining_args, namespace=conf_args)

    if args.debug and False:
        for k, v in sorted(args.__dict__.items()):
            print(k, v)
        sys.exit()

    sch.start()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    job_number = args.job_number
    if not job_number:
        job_number = next_jobnumber()

    log_dir = os.path.join(args.output_dir, 'log')
    log = set_log(conf_args.verbose, conf_args.debug, log_dir, job_number=job_number)

    log.debug('$ ' + ' '.join(sys.argv))

    if conf_args.args_from_file:
        logging.info('Args from file: {}'.format(sch.line))

    model_dict = find_by_job_number(args.job, job_dir=args.source_job_dir)

    if model_dict is None:
        log.debug('Model not found, reollecting models')
        model_dict = find_by_job_number(args.job, job_dir=args.source_job_dir, flash=False)

    if model_dict is None:
        log.error('Model not found')
        sys.exit(1)

    log.info('Model found of type {}'.format(model_dict['type']))

    dataset = model_dict['set']

    model = WIMJob.load(model_dict['dir'], build_module=True, load_state=True)

    log.info('Job #{}'.format(job_number))

    log.debug('$ ' + ' '.join(sys.argv))

    if args.array is not None:
        sampling_task = 'array'
        is_array = True
    else:
        sampling_task = args.sampling_task
        args.array = []
        is_array = False

    output_file = os.path.join(args.output_dir, f'train-{job_number:06d}.out')

    log.debug(f'Outputs registered in {output_file}')
    outputs = EpochOutput()
    outputs.add_file(output_file)

    model.job_number = job_number
    if args.sampling_seed is None:
        args.sampling_seed = job_number + 7

    model.encoder.prior.mean.requires_grad_(False)
    alternate_prior_params = model.encoder.prior.params.copy()
    alternate_prior_params['learned_means'] = False

    alternate_prior_params['mean_shift'] = 0.
    alternate_prior_params['init_mean'] = args.prior_means

    alternate_prior_params['num_priors'] = 1
    alternate_prior_params['seed'] = args.sampling_seed

    if args.prior:
        alternate_prior_params['distribution'] = args.prior
    alternate_prior_params['tau'] = args.tau

    model.set_alternate_prior(**alternate_prior_params)
    model.ft_params['from'] = args.job

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
    padding_sets = sum((_.split('-') for _ in args.padding_sets), [])

    save_dir_root = os.path.join(args.wim_job_dir, dataset,
                                 model.print_architecture(sampling=False),
                                 'wim')

    save_dir = os.path.join(save_dir_root, f'{job_number:06d}')
    model.saved_dir = save_dir

    sample_recorders = {}
    if args.inspection:
        fake_mu = torch.zeros(args.test_batch_size, model.latent_dim, device=model.device)
        fake_y = torch.zeros(args.test_batch_size, device=model.device, dtype=int)
        fakes = dict(mu=fake_mu, y=fake_y)
        if model.is_cvae:
            fakes['y_nearest'] = fakes['y']
        sample_recorders = {s: SampleRecorder(args.test_batch_size, **fakes) for s in wim_sets}
        sample_recorders[model.training_parameters['set']] = SampleRecorder(args.test_batch_size, **fakes)

        for _ in sample_recorders:
            with model.original_prior as p1:
                with model.alternate_prior as p2:
                    sample_recorders[_].add_auxiliary(centroids=p1.mean, alternate=p2.mean[0])

    try:
        model.finetune(*wim_sets,
                       train_size=args.train_size,
                       epochs=args.epochs,
                       moving_size=args.moving_size,
                       test_batch_size=args.test_batch_size,
                       alpha=args.alpha,
                       ood_mix=args.mix,
                       padding=args.padding,
                       mix_padding=args.mix_padding,
                       padding_sets=padding_sets,
                       optimizer=optimizer,
                       outputs=outputs,
                       seed=args.sampling_seed,
                       task=sampling_task,
                       sample_recorders=sample_recorders
                       )

    except DontDoFineTuning as e:
        is_array = e.continue_as_array
        if not is_array:
            logging.warning('Will stop now')
            sys.exit(0)
        if isinstance(sampling_task, int) and not args.array:
            args.array = True

    if is_array:
        save_dir_root = os.path.join(args.array_job_dir, dataset,
                                     model.print_architecture(sampling=False),
                                     'wim')

        save_dir = os.path.join(save_dir_root, f'{job_number:06d}')
        model.saved_dir = save_dir

        sch.start(block=args.array)

        arrays_alike = model.fetch_jobs_alike(job_dir=args.array_job_dir, flash=False)
        if arrays_alike:
            logging.warning('Already {} similar arrays'.format(len(arrays_alike)))
            logging.info('Similar arrays: {}'.format(','.join(map(str, (_['job'] for _ in arrays_alike)))))
            kept_wim_array = min(arrays_alike, key=lambda j: j['job'])
            array_dir = kept_wim_array['dir']
            logging.warning('Processing array {}'.format(kept_wim_array['job']))

        else:
            array_dir = model.saved_dir
            model.save(model.saved_dir)

        with turnoff_debug():
            wim_array = WIMArray.load(array_dir, load_state=False)

        wim_jobs_already_processed = WIMArray.collect_processed_jobs(args.array_job_dir, flash=True)
        logging.info('{} wim jobs already processed'.format(len(wim_jobs_already_processed)))
        wim_jobs = wim_array.fetch_jobs_alike(args.wim_job_dir)

        wim_jobs = [_ for _ in wim_jobs if model_subdir(_) not in wim_jobs_already_processed]

        logging.info('Processing {} wim jobs alike'.format(len(wim_jobs)))
        array_recorders = wim_array.register_jobs(*[WIMJob.load(_['dir'], build_module=False)
                                                    for _ in wim_jobs])

        sdirs = [os.path.join('samples', '{:04d}'.format(wim_array.trained), _) for _ in ('', 'init')]
        wim_array.concatenate_samples(*wim_jobs, sample_subdirs=sdirs)
        wim_array.save(array_dir)
        logging.info('model saved in {}'.format(wim_array.saved_dir))

        sch.stop()
        sys.exit(0)

    model.save(model.saved_dir)
    logging.info('model saved in {}'.format(model.saved_dir))
    sch.stop()
