#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:04:58 2021

@author: ziskin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import warnings
from MA_paths import work_david
ml_path = work_david / 'ML'

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = (
        'ignore::UserWarning,ignore::RuntimeWarning')  # Also affect subprocesses


def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def main_nadlan_ML(args):
    from nadlan_ML import cross_validation
    from nadlan_ML import features1
    from nadlan_EDA import load_nadlan_combined_deal
    from nadlan_EDA import apts
    from MA_paths import work_david
    ml_path = work_david /'ML'
    if args.n_splits is not None:
        n_splits = args.n_splits
    else:
        n_splits = 5
    if args.rseed is None:
        seed = 42
    else:
        seed = args.rseed
    if args.pgrid is None:
        pgrid = 'dense'
    else:
        pgrid = args.pgrid
    if args.verbose is None:
        verbose=0
    else:
        verbose = args.verbose
    if args.n_jobs is None:
        n_jobs = -1
    else:
        n_jobs = args.n_jobs
    if args.regressors is not None:
        regressors = args.regressors
    else:
        regressors = features1
    if args.year is not None:
        year = args.year
    else:
        year = 2000
    if args.main_path is not None:
        main_path = args.main_path
    else:
        main_path=work_david
    # load data:
    df = load_nadlan_combined_deal(path=main_path)
    df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
    # scorers = ['roc_auc', 'f1', 'recall', 'precision']
    X = df[[x for x in regressors]]
    X = X.dropna()
    y = df['DEALAMOUNT']
    y = y.loc[X.index]
    X = X[X['year'] == year].drop('year', axis=1)
    y = y.loc[X.index]
    model_name = args.model
    if args.savepath is not None:
        savepath = args.savepath
    else:
        savepath = ml_path
    logger.info(
        'Running {} model with {} nsplits, year= {}, regressors={}'.format(
            model_name, n_splits, year, regressors))
    model = cross_validation(
        X,
        y,
        model_name=model_name,
        n_splits=n_splits,
        verbose=verbose,
        pgrid=pgrid,
        savepath=savepath, n_jobs=n_jobs, year=year)
    print('')
    logger.info('Done!')


def configure_logger(name='general', filename=None):
    import logging
    import sys
    stdout_handler = logging.StreamHandler(sys.stdout)
    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        handlers = [file_handler, stdout_handler]
    else:
        handlers = [stdout_handler]

    logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
            handlers=handlers
            )
    logger = logging.getLogger(name=name)
    return logger

if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    from Migration_main import work_david
    ml_path = work_david / 'ML'
    logger = configure_logger('Nadlan_ML')
    savepath = Path(ml_path)
    parser = argparse.ArgumentParser(
        description='a command line tool for running the ML models tuning for Nadlan deals.')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional.add_argument(
        '--savepath',
        help="a full path to save the gridsearchcv files",
        type=check_path)
    optional.add_argument(
        '--n_splits',
        help='how many splits for the CV',
        type=int)
    optional.add_argument(
        '--main_path',
        help='the path where the nadlan deals are (csv)',
        type=check_path)

    optional.add_argument(
        '--year',
        help='year of the nadlan deals',
        type=int)

    optional.add_argument(
        '--pgrid',
        help='param grids for gridsearchcv object',
        type=str, choices=['light', 'normal', 'dense'])

    optional.add_argument(
        '--n_jobs',
        help='number of CPU threads to do gridsearch and cross-validate',
        type=int)
    optional.add_argument(
        '--rseed',
        help='random seed interger to start psuedo-random number generator',
        type=int)
    optional.add_argument(
        '--verbose',
        help='verbosity 0, 1, 2',
        type=int)
    # optional.add_argument(
    #     '--scorers',
    #     nargs='+',
    #     help='scorers, e.g., r2, r2_adj, etc',
    #     type=str)
#    optional.add_argument('--nsplits', help='select number of splits for HP tuning.', type=int)
    required.add_argument(
        '--model',
        help='select ML model.',
        choices=[
            'SVM',
            'MLP',
            'RF'])
    optional.add_argument('--regressors', help='select features for ML', nargs='+')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()

    # if args.savepath is None:
    #     print('savepath is a required argument, run with -h...')
    #     sys.exit()

    if args.model is None:
        print('model is a required argument, run with -h...')
        sys.exit()
    logger.info('Running ML, CV with {} model'.format(args.model))
    main_nadlan_ML(args)
