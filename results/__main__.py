from utils.parameters import gethostname, DEFAULT_RESULTS_DIR, DEFAULT_JOBS_DIR
import sys
from .utils import parse_config, make_tables, make_tex

file_ini = None

args_from_file = ['-vv',
                  'jobs/results/manuscrit/tabs/mnist-params.ini',
                  '--keep-auc'
                  ]

tex_output = sys.stdout


if __name__ == '__main__':

    import argparse
    import logging
    from utils.filters import get_filter_keys

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--which', '-c', nargs='*', default=['all'])
    parser.add_argument('--job-dir', default=None)
    parser.add_argument('--result-dir', default='/tmp', const=DEFAULT_RESULTS_DIR, nargs='?')
    parser.add_argument('--texify', default='utils/texify.ini')
    parser.add_argument('--filters-file', default='utils/filters.ini')
    parser.add_argument('--tpr', default=95, type=int)
    parser.add_argument('--register', dest='flash', action='store_false')
    parser.add_argument('--auc', action='store_true')
    parser.add_argument('config_files', nargs='+', default=[file_ini])
    parser.add_argument('-q', action='store_false', dest='show_dfs')

    args = parser.parse_args(None if sys.argv[0] else args_from_file)

    root = args.result_dir

    if args.verbose > 0:
        logging.getLogger().setLevel(logging.WARNING)
    if args.verbose > 1:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    filter_keys = get_filter_keys(args.filters_file, by='key')

    for config_file in args.config_files:

        print(config_file)
        show_dfs = args.show_dfs
        config = parse_config(config_file, root=root, texify_file=args.texify)
        df, df_t, best_vals, best_vals_t = make_tables(config, filter_keys,
                                                       ood_metrics=['fpr', 'auc'], show_dfs=True)

        tab = make_tex(config, df, best=best_vals)
        tab_t = make_tex(config, df_t, best=best_vals_t)

        print('\n\n====')
        print(df_t.to_string(float_format='{:.1f}'.format))
