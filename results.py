from utils.parameters import get_args_from_filters
import argparse


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    root = 'results/%j/samples'
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--batch-size', type=int, default=256)
    parser.add_argument('-W', '--grid-width', type=int, default=0)
    parser.add_argument('-N', '--grid-height', type=int, default=0)
    parser.add_argument('-D', '--directory', default=root)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-v', action='count')
    parser.add_argument('--z-sample', type=int, default=0)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--compare', type=int, default=0)
    parser.add_argument('--look-for-missed', type=int, default=0)

    jobs = [117281,
            117250,
            112267 ]
    
    args_from_file = ['-D', '/tmp/%j/samples',
                      '--compare', '1024',
                      # '--debug',
                      '-vv',
                      '-m', '64',
                      '-job-num', '117281', '117250', '112267',
                      ]

    
    
    filter_args, remaining_args = get_args_from_filters()
    
    args = parser.parse_args(args=remaining_args, namespace=filter_args)


    
