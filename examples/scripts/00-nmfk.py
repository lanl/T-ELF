import os

# force SVD to use single core and avoid scheduling problems
# TODO: undo this if n_jobs == 1
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import re
import ast
import sys
import shutil
import pickle
import pathlib
import warnings
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as ss

# load locally defined utility functions
from utils import load_config, load_list, check_path
from utils import process_terms, remove_duplicates, find_param_combinations

# load TELF particulars
from TELF.factorization import NMFk
from TELF.pre_processing import Beaver
from TELF.pre_processing import Vulture 

from TELF.pre_processing.Vulture.modules import SimpleCleaner
from TELF.pre_processing.Vulture.modules import LemmatizeCleaner
from TELF.pre_processing.Vulture.modules import SubstitutionCleaner
from TELF.pre_processing.Vulture.modules import RemoveNonEnglishCleaner

from TELF.pre_processing.Vulture.default_stop_words import STOP_WORDS
from TELF.pre_processing.Vulture.default_stop_phrases import STOP_PHRASES

# constants
CONFIG_PATH = 'input/config.json'
VULTURE_MAP = {
    'SimpleCleaner': SimpleCleaner,
    'LemmatizeCleaner': LemmatizeCleaner,
    'SubstitutionCleaner': SubstitutionCleaner,
    'RemoveNonEnglishCleaner': RemoveNonEnglishCleaner,
}


def load_stop_terms(terms, words=True):
    if terms is None:
        return STOP_WORDS if words else STOP_PHRASES
    if isinstance(terms, str):
        return load_list(terms)
    if isinstance(terms, list):
        return terms

    
def init_vulture_steps(settings):
    steps = []
    for s in settings:
        cleaner = s.get('type')
        if cleaner is None:
            raise ValueError('Config file is missing `type` for Vulture step!')
        if cleaner not in VULTURE_MAP:
            raise ValueError(f'Unknown cleaner "{cleaner}"!')
        
        args = s.get('init_args')
        if args is None:
            raise ValueError('Config file is missing `init_args` for Vulture step!')
        
        # process stop words / stop phrases for SimpleCleaner
        if cleaner == 'SimpleCleaner':
            args['stop_words'] = load_stop_terms(args.get('stop_words'), words=True)
            args['stop_phrases'] = load_stop_terms(args.get('stop_phrases'), words=False)
        
        # create the cleaner object
        steps.append(VULTURE_MAP[cleaner](**args))
    return steps
    
    
def main(file_path, cols, n_nodes, n_jobs, verbose):
    config = load_config(CONFIG_PATH)
    
    # get settings from config 
    output_dir = config['paths']['output']
    
    vulture_steps = init_vulture_steps(config['vulture']['steps'])
    
    nmfk_settings = config['nmfk']['settings']
    nmfk_ks = range(config['nmfk']['ks']['start'], config['nmfk']['ks']['stop'], config['nmfk']['ks']['step'])
    
    beaver_settings = config['beaver']['settings']
    beaver_vocabulary = config['beaver']['vocabulary']
    matrix_is_dense = config['beaver']['dense']
    if beaver_vocabulary is not None:
        beaver_vocabulary = load_list(beaver_vocabulary)
        
    # setup output directory
    file_path = pathlib.Path(file_path)
    file_name = file_path.stem
    output_dir = os.path.join(output_dir, file_name)
    output_dir = check_path(output_dir)
    
    
    # create a separate directory for beaver objects
    beaver_dir = check_path(os.path.join(output_dir, 'beaver'))
    
    # load the input DataFrame
    df = pd.read_csv(file_path)
    if not set(cols).issubset(df.columns):
        raise ValueError('One or more specified columns was not found in the DataFrame!')
    
    # prepare SME terms (if being used)
    terms = config['paths']['terms']
    substitution_map = None
    highlighting_map = None
    if terms is not None:
        substitution_map, highlighting_map = process_terms(terms)
    
    # clean with Vulture
    clean_cols_name = f'clean_{"_".join(cols)}'
    vulture = Vulture(n_jobs=n_jobs, verbose=verbose, parallel_backend='multiprocessing')
    df = vulture.clean_dataframe(df, 
                                 steps = vulture_steps,
                                 columns = cols, 
                                 append_to_original_df = True, 
                                 concat_cleaned_cols = True,
                                 substitutions = substitution_map
                                )
    df.to_csv(os.path.join(beaver_dir, f'clean_{file_name}.csv'), index=False)
        
    # create Beaver object to generate matrices
    beaver = Beaver()

    # get the vocabulary
    settings = {
        "dataset": df,  # which pd.DataFrame object to create vocab on
        "target_column": clean_cols_name, # which column in dataset to use
        "max_df": beaver_settings['max_df'],  # max document frequency (int=num documents, float=percentage documents)
        "min_df": beaver_settings['min_df'],  # min document frequency (int=num documents, float=percentage documents)
        "n_jobs": n_jobs,
        "parallel_backend": 'multiprocessing',
        "sort_alphabetical": False,  # if False sorts by frequency
        "verbose": verbose,
    }
     
    if beaver_vocabulary is None:
        beaver_vocabulary = beaver.get_vocabulary(**settings)
        if not beaver_vocabulary:
            raise ValueError('Could not generate vocabulary!')
        
    # build documents x words matrix
    settings = {
        "dataset":df,
        "target_column": clean_cols_name,
        "options":{"vocabulary": beaver_vocabulary},
        "matrix_type":'tfidf',
        "verbose":verbose,
        "save_path":beaver_dir,
    }
    if highlighting_map is not None:
        settings['highlighting'] = list(highlighting_map.keys())
        settings['weights'] = list(highlighting_map.values())
        beaver_vocabulary = settings['highlighting'] + beaver_vocabulary
        beaver_vocabulary = remove_duplicates(beaver_vocabulary)
    
    # create the documents words matrix
    beaver.documents_words(**settings)
    
    # get matrix
    X = ss.load_npz(os.path.join(beaver_dir, 'documents_words.npz'))
    if matrix_is_dense:
        X = X.toarray()  # convert to dense numpy array

    # words need to be rows, documents cols
    X = X.T  
    
    X_shape = X.shape
    print(f'[TELF]: X={X_shape}')

    # decompose X
    dynamic_settings, all_params = find_param_combinations(nmfk_settings)
    for params in tqdm(all_params, total=len(all_params)):

        # update decomposition params for this instance
        params['save_path'] = output_dir
        params['save_output'] = True
        params['collect_output'] = True,
        params['n_nodes'] = n_nodes
        params['n_jobs'] = n_jobs
        params['verbose'] = verbose

        # set up logging details and run
        note_substr = ', '.join([f'{s}={params[s]}' for s in dynamic_settings])
        note = f"Decomposing X with shape={X_shape} and {note_substr}"
        model = NMFk(**params)

        if X_shape[0] in nmfk_ks:
            warnings.warn(f'[TELF]: You have selected too large of an upper k-limit.\nSetting limit to {X_shape[0]}', RuntimeWarning)
            ks = range(ks.start, X_shape[0], ks.step)
        if X_shape[1] in nmfk_ks:
            warnings.warn(f'[TELF]: Matrix of shape {X_shape} is too small to decompose!', RuntimeWarning)
            continue
        results = model.fit(X, nmfk_ks, file_name, note)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full example of TELF library. A CSV file is cleaned with Vulture to create a " \
                                                 "Documents by Words TF-IDF matrix which is then decomposed by NMFk")
    parser.add_argument("-p", "--path", type=str, required=True, help="The path to the input CSV file")
    parser.add_argument("-c", "--cols", nargs='+', required=True, help="The columns of the DataFrame which to clean and utilize")
    parser.add_argument("-n", "--n_nodes", type=int, required=True, help="Number of nodes (integer > 0)")
    parser.add_argument("-j", "--n_jobs", type=int, required=True, help="Number of jobs (integer > 0)")
    args = parser.parse_args()
    main(args.path, args.cols, args.n_nodes, args.n_jobs, True)