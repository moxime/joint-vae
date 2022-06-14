from utils.save_load import find_by_job_number
import argparse
import logging
import torch
from utils.print_log import turnoff_debug
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', '-j', nargs='+', type=int, default=[])
    parser.add_argument('-v', action='count', default=0)
    parser.add_argument('--result-dir', default='/tmp')
    parser.add_argument('--when', default='last')
    parser.add_argument('--job-dir', default='./jobs')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-batch', type=int, default=10)
    parser.add_argument('--device', default='cuda')

    args_from_file = ('-vvvv '
                      '--jobs 193080 193082'
                      ).split()
    
    args = parser.parse_args(None if len(sys.argv) > 1 else args_from_file)

    logging.getLogger().setLevel(40 - 10 * args.v)
    models = find_by_job_number(*args.jobs, job_dir=args.job_dir,
                                load_net=True,
                                load_state=True, show_debug=True)

    for _ in models:

        model = models[_]['net']
        dset = models[_]['set']

        num_batch = args.num_batch
        batch_size = args.batch_size
        print('*** Computing accuracy for {} on model # {} with {} images'.format(dset, _, num_batch * batch_size))
        with turnoff_debug():
            with torch.no_grad():
                acc = model.accuracy(batch_size=args.batch_size,
                                     num_batch=args.num_batch,
                                     from_where=('compute'),
                                     print_result='ACC',
                                     update_self_testing=False,
                                     )
        
