from cvae import ClassificationVariationalNetwork as Net
from utils.parameters import get_filters_args, set_log
import argparse
import sys
import logging
from utils.save_load import LossRecorder, collect_networks
from utils import torch_load as tl
from utils.sample import zsample, sample
from utils.inspection import loss_comparisons
import matplotlib.pyplot as plt

root = 'results/%j/samples'



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--job-dir', default='jobs')
    parser.add_argument('-m', '--batch-size', type=int, default=256)
    parser.add_argument('-W', '--grid-width', type=int, default=0)
    parser.add_argument('--total-width', type=int, default=30)
    parser.add_argument('-N', '--grid-height', type=int, default=0)
    parser.add_argument('-D', '--directory', default=root)
    parser.add_argument('--z-sample', type=int, default=0)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--look-for-missed', type=int, default=0)
    parser.add_argument('--list-jobs-and-quit', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-p', '--plot', nargs='?', const='all')
    
    jobs = [117281,
            129527,
            117250,
            112267
            ]
    
    args_from_file = ['-D', '/tmp/%j/samples',
                      '-N', '10',
                      # '--debug',
                      '-vv',
                      '-m', '64',
                      '--job-num'] + [str(j) for j in jobs]
    
    filter_args, remaining_args = get_filters_args(None if sys.argv[0] else args_from_file)
    
    args = parser.parse_args(args=remaining_args, namespace=filter_args)

    set_log(args.verbose, args.debug, 'jobs/log', name='results')
    
    L = args.grid_width
    N = args.grid_height
    m = args.batch_size
    root = args.directory
    z_sample = args.z_sample
    bins = args.bins
    # cross_sample = args.compare

    filters = args.filters

    device = tl.choose_device()
    logging.debug(device)
    
    """if args.sync:
        r = LossRecorder(1)
        computed_max_batch_size = False
    """
    
    logging.info('Will work in %s', root)

    list_of_networks = sum(collect_networks(args.job_dir, load_net=False, load_state=False), [])

    networks_to_be_studied = []
    for n in list_of_networks:
        filter_results = sum([[f.filter(n[d]) for f in filters[d]] for d in filters], [])
        to_be_studied = all(filter_results)

        if to_be_studied:
            networks_to_be_studied.append(n)
            # for d in filters:
            #   print(d, n[d])
            #   for f in filters[d]:
            #       print(f, f.filter(n[d]))

    if args.list_jobs_and_quit:
        networks_to_be_studied.sort(key=lambda n: n['job'] if isinstance(n['job'], int) else 0)
        for i, n in enumerate(networks_to_be_studied):
            #            print(f'{i:4d}:', n['job'])
            print(n['job'])
        logging.info('%d networks listed', i + 1)
        sys.exit(0)

    if len(networks_to_be_studied) > 2:
        args.plot = False

    logging.info('{} plot will be done'.format('One' if args.plot else 'No'))
    logging.info('{} models will be worked'.format(len(networks_to_be_studied)))

    dict_of_sets = dict()
    for n in networks_to_be_studied:
        model = n['net']
        trainset_ = tuple(model.training_parameters[k] for k in ('set', 'transformer'))
        if trainset_ not in dict_of_sets:
            dict_of_sets[trainset_] = [n]
        else:
            dict_of_sets[trainset_].append(n)

    for (testset, transformer), list_of_nets in dict_of_sets.items():
        logging.info('Going with %s (%s): %d nets to be done',
                     testset, transformer, len(list_of_nets)) 
        x, y = dict(), dict()
        
        _, test_dataset = tl.get_dataset(testset, transformer=transformer)
        x[testset], y[testset] = tl.get_batch(test_dataset, device=device,
                                              batch_size=max(z_sample, N))

        oodsets = test_dataset.same_size

        for o in oodsets:
            _, ood_dataset = tl.get_dataset(o, transformer=transformer)
            x[o], y[o] = tl.get_batch(ood_dataset, device=device, batch_size=max(z_sample, N))

        if not L:
            L = args.total_width // (1 + len(x))

        for n in list_of_nets:

            logging.info('loading state of %s', n['job'])

            model = Net.load(n['dir'])
            model.to(device)
            logging.info('done')
            logging.info('Compute max batch size')

            batch_size = min(m, model.compute_max_batch_size(batch_size=m, which='test'))
            logging.info(f'done ({batch_size})')

            for s in x:
                
                logging.info('sampling %s', s)

                if N:
                    list_of_images = sample(model, x[s][:N], root=root,
                                            directory=s,
                                            N=N, L=L)

                if z_sample:

                    zsample(x[s][:z_sample], model, y=y[s][:z_sample], m, root=root, bins=args.bins, directory=s)

            if N:
                list_of_images = sample(model, root=root,
                                        directory='generate', N=N, L=L)
                
            loss_comparisons(model, root=root, plot=args.plot, bins=args.bins)    

