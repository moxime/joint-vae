from __future__ import print_function

import numpy as np
import torch
from cvae import ClassificationVariationalNetwork as CVNet
from module.vae_layers import Sigma
import utils.torch_load as torchdl
import os
import sys
import argparse

from utils.parameters import alphanum, list_of_alphanums, get_args, set_log, gethostname, next_jobnumber
from utils.save_load import find_by_job_number, NoModelError, get_submodule
from utils.print_log import EpochOutput
from utils.signaling import SIGHandler
import setproctitle

if __name__ == '__main__':

    # sys.argv[0] = 'training'
    # setproctitle.setproctitle(' '.join(sys.argv))
    setproctitle.setproctitle('training')

    hostname = gethostname()

    args = get_args(what_for='train')

    debug = args.debug
    verbose = args.verbose

    job_dir = args.job_dir
    job_number = args.job_number

    if not job_number:
        job_number = next_jobnumber()

    log_dir = os.path.join(args.output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = set_log(verbose, debug, log_dir, job_number=job_number)

    log.debug('$ ' + ' '.join(sys.argv))

    for k in args.__dict__.items():
        log.debug('%s: %s', *k)

    if job_number:
        log.info(f'Job number {job_number} started')

    if args.force_cpu:
        wanted_device = 'cpu'
    else:
        wanted_device = args.device

    if wanted_device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
        log.info(f'Used device: {device}')
        log.debug(f'CPU asked by user')
    else:
        device = torch.device(wanted_device)
        log.info(f'Used device: {device}')

    # print('*** device:', device)
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version() or np.nan

    log.debug(f'Using cuda v. {cuda_version} and '
              f'cudnn v. {cudnn_version / 1000:.3f}')

    batch_size = args.batch_size
    test_sample_size = args.test_sample_size
    test_batch_size = args.test_batch_size
    dry_run = args.dry_run
    resume = args.resume

    if resume:
        try:
            job_TBR_num = int(resume)
            find_by = 'number'
        except ValueError:
            find_by = 'directory'

        if find_by == 'number':
            try:
                log.info('Looking for job %d to be resumed', job_TBR_num)
                model_dict = find_by_job_number(job_TBR_num, job_dir=args.job_dir, flash=False,
                                                build_module=True, load_state=True)
                if model_dict is None:
                    raise NoModelError
                model = model_dict['net']
                resumed_from = model_dict['dir']
                log.debug('Network loaded in {}'.format(resumed_from))
                done_epochs = model.trained
                if done_epochs == 0:
                    verb = 'will start from scratch.'
                elif done_epochs < args.epochs:
                    verb = f'will resume from {done_epochs}.'
                else:
                    verb = 'is already done.'
                log.info(f'Training {verb}')
            except (NoModelError):
                log.error(f'model #{job_TBR_num} not found!')
                sys.exit(1)

        else:
            try:
                resumed_from = resume
                log.info('Loading network in %s', resume)
                model = CVNet.load(args.resume, load_state=True)
                log.debug('Network loaded in {}'.format(resumed_from))
                done_epochs = model.trained
                if done_epochs == 0:
                    verb = 'will start from scratch.'
                elif done_epochs < args.epochs:
                    verb = f'will resume from {done_epochs}.'
                else:
                    verb = 'is already done.'
                log.info(f'Training {verb}')

            except (FileNotFoundError, NameError):
                log.error(f'network not found in {resume}')
                sys.exit(1)

    else:

        args.optim_params = {
            'optim_type': args.optimizer,
            'lr': args.lr,
            'lr_decay': args.lr_decay,
            'weight_decay': args.weight_decay,
            'grad_clipping': args.grad_clipping
        }

        input_shape, num_labels = torchdl.get_shape_by_name(args.dataset, args.transformer)
        _shape = '-'.join(map(str, input_shape + (num_labels,)))
        log.info('Building network for shape %s (%s with %s)',
                 _shape, args.dataset, args.transformer)

        sdim = input_shape if args.sigma_per_dim else 1

        if isinstance(args.sigma, str):
            sigma_is_learned = args.sigma.startswith('learned')
            sigma_is_coded = args.sigma == 'coded'
            sigma_is_rmse = args.sigma == 'rmse'
            try:
                sigma_value = float(args.sigma.split('--')[-1])
            except ValueError:
                sigma_value = 1.
        else:
            assert isinstance(args.sigma, (float, int))
            sigma_value = args.sigma
            sigma_is_learned = False
            sigma_is_coded = False
            sigma_is_rmse = False

        batch_norm = args.batch_norm if args.batch_norm != 'none' else None
        dropout = args.dropout

        sigma = Sigma(sigma_value,
                      sdim=sdim,
                      input_dim=input_shape if sigma_is_coded else False,
                      reach=False,  # args.sigma_reach,
                      decay=False,  # args.sigma_decay,
                      max_step=None,  # args.sigma_max_step,
                      learned=sigma_is_learned,  # args.sigma_learned,
                      is_rmse=sigma_is_rmse)

        latent_prior_means = args.prior_means,
        latent_prior_variance = args.prior_variance,
        learned_latent_prior_means = args.learned_prior_means,

        if not args.encoder_forced_variance:
            args.encoder_forced_variance = False

        prior = dict(distribution=args.prior,
                     init_mean=args.prior_means,
                     learned_means=args.learned_prior_means,
                     var_dim=args.prior_variance,
                     freeze_means=args.freeze_prior_means
                     )

        if args.prior == 'tilted':
            prior['tau'] = args.tilted_tau

        if args.pretrained_features:
            pretrained_features = get_submodule(args.pretrained_features)
        else:
            pretrained_features = None

        if args.pretrained_upsampler:
            pretrained_upsampler = get_submodule(args.pretrained_upsampler, 'imager')
        else:
            pretrained_upsampler = None

        model = CVNet(input_shape, num_labels,
                      type=args.type,
                      output_distribution=args.output_distribution,
                      features=args.features,
                      pretrained_features=pretrained_features,
                      pretrained_upsampler=pretrained_upsampler,
                      batch_norm=args.batch_norm,
                      dropout=dropout,
                      optimizer=args.optim_params,
                      encoder=args.encoder,
                      encoder_forced_variance=args.encoder_forced_variance,
                      latent_dim=args.latent_dim,
                      prior=prior,
                      latent_sampling=args.latent_sampling,
                      test_latent_sampling=args.test_latent_sampling,
                      activation=args.activation,
                      decoder=args.decoder,
                      upsampler=args.upsampler,
                      classifier=args.classifier,
                      beta=args.beta,
                      gamma=args.gamma,
                      sigma=sigma,
                      output_activation=args.output_activation)

    if args.show:
        print(model)
        sys.exit(0)

    if resume:
        dataset, transformer = model.training_parameters['set'], model.training_parameters['transformer']
        trainset, testset = torchdl.get_dataset(dataset, transformer=transformer)
        validation = model.training_parameters['validation']

        oodsets = [torchdl.get_dataset(n, transformer=transformer, splits=['test'])[1]
                   for n in testset.same_size]

        data_augmentation = model.training_parameters['data_augmentation']
        latent_sampling = model.training_parameters['latent_sampling']

    else:

        dataset, transformer = args.dataset, args.transformer
        trainset, testset = torchdl.get_dataset(dataset, transformer=transformer)  #
        validation = args.validation
        oodsets = [torchdl.get_dataset(n, transformer=transformer, splits=['test'])[1]
                   for n in testset.same_size]

        data_augmentation = args.data_augmentation
        latent_sampling = args.latent_sampling

    if args.oodsets is not None:
        oodsets = [_ for _ in oodsets if _.name in args.oodsets]

    log.debug(f'{trainset.name} dataset loaded')
    log.info('Will test ood for {}'.format(','.join([_.name for _ in oodsets])))

    if not data_augmentation:
        _augment = ''
    else:
        _augment = '--augment='
        data_augmentation.sort()
        _augment += '-'.join(data_augmentation)

    save_dir_root = os.path.join(job_dir, dataset,
                                 model.print_architecture(sampling=False),
                                 f'sigma={model.sigma}' +
                                 f'--optim={model.optimizer}' +
                                 f'--sampling={latent_sampling}' +
                                 _augment)

    save_dir = os.path.join(save_dir_root, f'{job_number:06d}')

    if args.where:
        print(save_dir)
        sys.exit(0)

    output_file = os.path.join(args.output_dir, f'train-{job_number:06d}.out')

    log.debug(f'Outputs registered in {output_file}')
    outputs = EpochOutput()
    outputs.add_file(output_file)

    while os.path.exists(save_dir):
        log.debug(f'{save_dir} exists')
        job_number += 1
        save_dir = os.path.join(save_dir_root, f'{job_number:06d}')

    model.job_number = job_number
    model.saved_dir = save_dir

    if args.resume:
        with open(os.path.join(resumed_from, 'RESUMED'), 'w') as f:
            f.write(str(job_number) + '\n')

    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    with open(os.path.join(job_dir, f'number-{hostname}'), 'w') as f:
        f.write(str(job_number + 1) + '\n')

    log.info('Network built, will be saved in')
    log.info(save_dir)

    log.debug('%s: %s', 'Network architecture',
              model.print_architecture(True, True))

    model.to(device)

    # print('*** .device', jvae.device)

    x, y = torchdl.get_batch(trainset, device=device, batch_size=8)

    if debug:
        log.debug('Trying a first pass')
        log.debug('x in [%.2f, %.2f] with mean (std) %.2f (%.2f)',
                  x.min().item(),
                  x.max().item(),
                  x.mean().item(),
                  x.std().item())
        outs = model(x, y if model.y_is_coded else None)
        log.debug(' -- '.join(map(str, ([tuple(u.shape) for u in outs]))))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    warmup = args.warmup
    if len(warmup) == 1:
        warmup = [0, warmup[0]]
    warmup_gamma = args.warmup_gamma
    if len(warmup_gamma) == 1:
        warmup_gamma = [0, warmup_gamma[0]]

    if not dry_run:
        if model.trained < args.epochs:
            log.info('Training of %s', model.print_architecture())

            # print('t.py l 302 testset:', testset.data[0].shape)

            model.train_model(trainset,
                              transformer=transformer,
                              epochs=args.epochs,
                              batch_size=batch_size,
                              test_batch_size=test_batch_size,
                              full_test_every=2 if debug else args.full_test_every,
                              ood_detection_every=2 if debug else args.full_test_every,
                              validation=validation,
                              device=device,
                              testset=testset,
                              oodsets=oodsets,
                              data_augmentation=data_augmentation,
                              fine_tuning=args.fine_tuning,
                              warmup=warmup,
                              warmup_gamma=warmup_gamma,
                              validation_sample_size=test_sample_size,  # 10000,
                              save_dir=save_dir,
                              outputs=outputs,
                              signal_handler=SIGHandler(2, 3, 15))

            log.info('Done training')
        else:
            log.info('No need to train %s', model.print_architecture())
    else:
        log.info('Dry-run %s', model.print_training(epochs=epochs, set=trainset.name))
