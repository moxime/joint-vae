import logging
from . import find_by_job_number, needed_remote_files
import tempfile
import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('jobs', nargs='+')
    parser.add_argument('--job-dir', default='./jobs')
    parser.add_argument('--state', action='store_true')
    parser.add_argument('--optimizer', action='store_true')
    parser.add_argument('--output', default=os.path.join(tempfile.gettempdir(), 'files'))
    parser.add_argument('--rec-files', default='ind')
    parser.add_argument('--register', dest='flash', action='store_false')

    logging.getLogger().setLevel(logging.INFO)

    args = parser.parse_args()

    output_file = args.output

    job_dict = find_by_job_number(*args.jobs, job_dir=args.job_dir, force_dict=True, flash=args.flash)

    logging.info('Will recover jobs {}'.format(', '.join(str(_) for _ in job_dict)))

    mdirs = [job_dict[_]['dir'] for _ in job_dict]

    with open(output_file, 'w') as f:
        for _ in needed_remote_files(*mdirs, which_rec=args.rec_files,
                                     state=args.state, optimizer=args.optimizer,
                                     missing_file_stream=f):
            pass

    with open('/tmp/rsync-files', 'w') as f:
        f.write('rsync -avP --files-from={f} $1 .\n'.format(f=output_file))
