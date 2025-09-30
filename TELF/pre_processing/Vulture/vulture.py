#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
"""
import os
import sys
import uuid
import pickle
import pathlib
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed, parallel_backend

from .modules.simple_clean import SimpleCleaner
from .modules.lemmatize import LemmatizeCleaner
from .modules.substitute import SubstitutionCleaner
from .modules.detect_nonenglish import RemoveNonEnglishCleaner
from .modules.ner import NEDetector
from .default_stop_words import STOP_WORDS
from .default_stop_phrases import STOP_PHRASES
from ...helpers.file_system import check_path

try:
    from mpi4py import MPI
except:
    MPI = None


def chunk_tuple_list(l, n_chunks):
    """
    Splits the given list of (key, value) tuples into sub-lists.

    Parameters
    ----------
    l: list of tuple
        List of (key, value) tuples to split.
    n_chunks: int
        How many sets of sub-lists to create.

    Yields
    ------
    list
        Sub-list containing (key, value) tuples.
    """
    if n_chunks <= 0:
        return

    # determine the chunk size and the remainder
    chunk_size, remainder = divmod(len(l), n_chunks)
    
    start = 0
    for i in range(n_chunks):
        end = start + chunk_size + (i < remainder)
        yield l[start:end]
        start = end


class Vulture:
    """
    Vulture is a parallel, multi-node parallel, and distributed parallel document
    pre-processing tool.
    It is designed to be simple and fast.

    Vultures are natures' cleaners!
    """
    PARALLEL_BACKEND_OPTIONS = {'loky', 'multiprocessing', 'threading'}
    DEFAULT_OPERATOR_PIPELINE = [
        NEDetector(library="en_core_web_trf")
    ]
    DEFAULT_PIPELINE = [
        SimpleCleaner(stop_words = STOP_WORDS,
                      stop_phrases = STOP_PHRASES,
                      order = [
                          'standardize_hyphens',
                          'isolate_frozen',
                          'remove_copyright_statement',
                          'remove_stop_phrases',
                          'make_lower_case',
                          'remove_formulas',
                          'normalize',
                          'remove_next_line',
                          'remove_email',
                          'remove_()',
                          'remove_[]',
                          'remove_special_characters',
                          'remove_nonASCII_boundary',
                          'remove_nonASCII',
                          'remove_tags',
                          'remove_stop_words',
                          'remove_standalone_numbers',
                          'remove_extra_whitespace',
                          'min_characters',
                          'remove_numbers',
                          'remove_alphanumeric',
                          'remove_roman_numerals'
                      ]
                     ),
    ]
    
    
    def __init__(self, *, n_jobs = -1, n_nodes = 1, parallel_backend = "threading", cache = '/tmp', verbose = False):
        self.n_jobs = n_jobs
        self.n_nodes = n_nodes
        self.parallel_backend = parallel_backend
        self.cache = cache
        self.save_path = None
        self.verbose = verbose
        
        # init multi-node (MPI) attributes
        self.comm = None
        self.rank = 0
        self.size = 1
        if self.use_mpi():
            self.mpi = MPI
            self.comm = self.mpi.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

        # generate unique ID only on the root process
        if self.rank == 0:
            self.unique_id = str(uuid.uuid4())
        else:
            self.unique_id = None

        # broadcast unique_id from root process to all other processes
        if self.comm is not None:
            self.unique_id = self.comm.bcast(self.unique_id, root=0)


    def operate(self, documents, steps=None, save_path=None, file_name=""):
        
        if steps is None:
            steps = self.DEFAULT_OPERATOR_PIPELINE.copy()
        
        for step in steps:
            if "module_type" in vars(step):
                assert step.module_type == "OPERATOR", "This method can only be used with a OPERATOR type module."

        # transform documents into list of tuples
        operate_documents = list(documents.items())
        if self.verbose and self.rank == 0:
            print(f'[Vulture]: Cleaning {len(operate_documents)} documents', file=sys.stderr)


        # prepare for MPI by chunking data and saving chunks (assuming DFS)
        if self.use_mpi():
            self._mpi_init(operate_documents)
            self.comm.Barrier()
            operate_documents = self._mpi_load_chunk_from_disk(self.rank, is_clean=False)

        # perform operation
        all_results = []
        for step in steps:
            if save_path is not None:
                self.save_path = os.path.join(save_path, f'{file_name}_{step.__class__.__name__}.p')
            else:
                self.save_path = save_path
            
            curr_operated_documents = self._clean_helper(operate_documents, [step])
            if self.use_mpi():
                self._mpi_save_chunk_to_disk(curr_operated_documents, self.rank, is_clean=True, custom_fn=f'{step.__class__.__name__}')
                self.comm.Barrier()
                curr_operated_documents = self._mpi_combine(custom_fn=f'{step.__class__.__name__}')
                
            if self.save_path is not None:
                self._save_documents(dict(curr_operated_documents))
            else:
                all_results.append((f'{step.__class__.__name__}', dict(curr_operated_documents)))
        
        if self.save_path is None:
            return all_results
        
    def clean(self, documents, steps=None, substitutions=None, save_path=None):
        self.save_path = save_path
        if steps is None:
            steps = self.DEFAULT_PIPELINE.copy()

        for step in steps:
            if "module_type" in vars(step):
                assert step.module_type == "CLEANER", "This method can only be used with a CLEANER type module."

        if substitutions is not None:
            assert isinstance(substitutions, dict), '`substitutions` must be a dict!'
            initial_sub = SubstitutionCleaner(substitutions, permute=True, lower=True, lemmatize=True)
            final_sub = SubstitutionCleaner(substitutions, permute=False, lower=False, lemmatize=True)
            steps = [initial_sub] + steps + [final_sub]
    
        # transform documents into list of tuples
        clean_documents = list(documents.items())
        if self.verbose and self.rank == 0:
            print(f'[Vulture]: Cleaning {len(clean_documents)} documents', file=sys.stderr)
        
        # prepare for MPI by chunking data and saving chunks (assuming DFS)
        if self.use_mpi():
            self._mpi_init(clean_documents)
            self.comm.Barrier()
            clean_documents = self._mpi_load_chunk_from_disk(self.rank, is_clean=False)
        
        # perform cleaning 
        clean_documents = self._clean_helper(clean_documents, steps)
        if self.use_mpi():
            self._mpi_save_chunk_to_disk(clean_documents, self.rank, is_clean=True)
            self.comm.Barrier()
            clean_documents = self._mpi_combine()
        
        # save the clean results or return them
        if self.save_path is not None:
            self._save_documents(dict(clean_documents))
        else:
            return dict(clean_documents)
        
        
    def clean_dataframe(self, df, columns, steps=None, substitutions=None,
                        append_to_original_df=False, concat_cleaned_cols=False):
        
        if not df.index.is_unique:  # DataFrame must have unique indices for mapping
            raise ValueError("DataFrame does not contain unique indices!")
        if not all(col in df.columns for col in columns):  # make sure columns exist
            raise ValueError("One or more columns are invalid!")
        
        if steps is None:
            steps = self.DEFAULT_PIPELINE.copy()

        for step in steps:
            if "module_type" in vars(step):
                assert step.module_type == "CLEANER", "This method can only be used with a CLEANER type module."
        
        # make a copy of the DataFrame to prevent changing original and fill nans with empty strings
        if append_to_original_df:
            df = df.copy()
        else:
            df = df[columns].copy()
        df[columns] = df[columns].fillna('')
    
        # validate that all columns contain strings
        if not all(df[col].apply(lambda x: isinstance(x, str)).all() for col in columns):
            raise ValueError("One or more columns do not contain strings!")
        
        # prepare the document dictionaries
        documents = None
        if concat_cleaned_cols:
            col_name = f'clean_{"_".join(columns)}'
            documents = {col_name: df.apply(lambda x: '. '.join(filter(None, x[columns])), axis=1).to_dict()}
        else:
            documents = {f'clean_{col}': df[col].to_dict() for col in columns}
            
        # clean
        clean_documents = {}
        for col, docs in documents.items():
            clean_docs = self.clean(docs, steps=steps, substitutions=substitutions)
            clean_documents[col] = clean_docs
            
        # add the clean data back to the DataFrame
        for col, clean_docs in clean_documents.items():
            df[col] = df.index.map(clean_docs)
            df[col] = df[col].replace('', np.nan) # set null entries to None
        return df
    
    
    def _mpi_init(self, documents):
        if self.rank == 0:
            for idx, chunk in enumerate(chunk_tuple_list(documents, self.n_nodes)) :
                self._mpi_save_chunk_to_disk(chunk, idx, is_clean=False)

    
    def _mpi_get_name(self, rank, is_clean, custom_fn=None):
        if not custom_fn:
            if is_clean:
                return f'vulture_{self.unique_id}_{rank}_clean.p'
            else:
                return f'vulture_{self.unique_id}_{rank}.p'
        else:
            return f'vulture_{self.unique_id}_{rank}_{custom_fn}.p'
    
    
    def _mpi_save_chunk_to_disk(self, data, rank, *, is_clean, custom_fn=None):
        fn = self._mpi_get_name(rank, is_clean, custom_fn)

        with open(os.path.join(self.cache, fn), 'wb') as fh:
            pickle.dump(data, fh)

                
    def _mpi_load_chunk_from_disk(self, rank, *, is_clean, custom_fn=None):
        fn = self._mpi_get_name(rank, is_clean, custom_fn=custom_fn)
        with open(os.path.join(self.cache, fn), 'rb') as fh:
            return pickle.load(fh)
        

    def _mpi_combine(self, custom_fn=None):
        clean_documents = []
        for rank in range(self.n_nodes):
            clean_documents += self._mpi_load_chunk_from_disk(rank, is_clean=True, custom_fn=custom_fn)
        return clean_documents
    
    
    def _save_documents(self, documents):
        with open(self.save_path, 'wb') as fh:
            pickle.dump(documents, fh)
    
    
    def _clean_helper(self, clean_documents, steps):
        frozen = set()
        if self.n_jobs == 1:
            for cleaner in tqdm(steps, total=len(steps), disable=not self.verbose):
                if self.verbose and self.rank == 0:
                    print(f'[Vulture]: Running {cleaner.__class__.__name__} module', file=sys.stderr)
                frozen |= cleaner.frozen
                cleaner.frozen = frozen
                clean_documents = [cleaner(doc) for doc in tqdm(clean_documents, 
                                   total=len(clean_documents), disable = self.verbose < 10)]
        else:
            clean_documents = self._parallel_helper(clean_documents, steps)
        return clean_documents 
        
        
    def _parallel_helper(self, documents, cleaners):
        """
        Helper function to run processing of given documents in parallel

        Parameters
        ----------
        documents : dict
            Dictionary of documents to clean. In this dictionary, keys are the unique document
            identifiers, and values are the text to clean.
        parameters : dict
            Parameters of the function to use for processing, except documents.
        function : callable
            Processing function to call.
        

        Returns
        -------
        dict
            processed docuements, where keys are the document IDs and values are the text.

        """
        with multiprocessing.Manager() as manager:
            
            # split the documents into chunks
            total_length = len(documents)
            n_jobs = min(total_length, self.n_jobs)
            shared_documents = manager.list(documents)
            chunk_size = total_length // n_jobs
        
            # run processing in parallel with persitent processes
            with parallel_backend(n_jobs=n_jobs, verbose=self.verbose, backend=self.parallel_backend):
                frozen = set()
                for cleaner in tqdm(cleaners, disable=not self.verbose):
                    if self.verbose and self.rank == 0:
                        print(f'[Vulture]: Running {cleaner.__class__.__name__} module', file=sys.stderr)
                    frozen |= cleaner.frozen
                    cleaner.frozen = frozen
                    Parallel()(
                        delayed(self._parallel_worker)(shared_documents, i, min(i + chunk_size, total_length), cleaner)
                        for i in range(0, total_length, chunk_size)
                    )
            return list(shared_documents)
        
        
    @staticmethod
    def _parallel_worker(documents, chunk_start, chunk_end, cleaner):
        for i in range(chunk_start, chunk_end):
            documents[i] = cleaner(documents[i])
        
    def use_mpi(self):
        return 1 < self.n_nodes

    
    ### GETTERS / SETTERS
    
    
    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def n_nodes(self):
        return self._n_nodes
    
    @property
    def parallel_backend(self):
        return self._parallel_backend

    @property
    def cache(self):
        return self._cache
    
    @property
    def save_path(self):
        return self._save_path
    
    @property
    def verbose(self):
        return self._verbose
    
    @n_jobs.setter
    def n_jobs(self, n_jobs):
        if not isinstance(n_jobs, int):
            raise TypeError('n_jobs must be an int!')

        cpu_count = multiprocessing.cpu_count()
        limit = cpu_count + n_jobs
        if (n_jobs == 0) or (limit < 0) or (2 * cpu_count < limit):
            raise ValueError(f'n_jobs must take a value on [-{cpu_count}, -1] or [1, {cpu_count}]!')
        if n_jobs < 0:
            self._n_jobs = cpu_count - abs(n_jobs) + 1
        else:
            self._n_jobs = n_jobs

    @n_nodes.setter
    def n_nodes(self, n_nodes):
        if not isinstance(n_nodes, int):
            raise TypeError('n_nodes must be an int!')
        if 1 > n_nodes:
            raise ValueError('n_nodes must be greater than 0!')
        if 1 < n_nodes and MPI is None:
            raise ImportError('Multiple nodes requested but MPI is not available')
        self._n_nodes = n_nodes
        
    @parallel_backend.setter
    def parallel_backend(self, parallel_backend):
        if not isinstance(parallel_backend, str):
            raise TypeError('parallel_backend must be an str!')
        if parallel_backend not in self.PARALLEL_BACKEND_OPTIONS:
            raise ValueError(f'{parallel_backend} is not a valid parallel_backend option!')
        self._parallel_backend = parallel_backend

    @cache.setter
    def cache(self, cache):
        if not isinstance(cache, (str, pathlib.Path)):
            raise TypeError(f'`cache` must either be a str or a pathlib.Path object!')
        if isinstance(cache, str):
            cache = pathlib.Path(cache)

        if not cache.exists():
            try:
                cache.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f'Failed to create cache directory "{cache}": {str(e)}')
        self._cache = cache

    @save_path.setter
    def save_path(self, save_path):
        if save_path is None:
            self._save_path = None
            return
        if not isinstance(save_path, (str, pathlib.Path)):
            raise TypeError(f'`cache` must either be a str or a pathlib.Path object!')
        if isinstance(save_path, str):
            save_path = pathlib.Path(save_path)
        
        if not save_path.parent.exists():
            raise ValueError(f'The `save_path` directory "{save_path.parent}" does not exist!')
        if save_path.is_dir():
            raise ValueError(f'The `save_path` "{save_path}" is a directory!')
        if save_path.exists():
            warnings.warn(f'The file "{save_path}" already exists and will be overwritten!')
        self._save_path = save_path

        
    @verbose.setter
    def verbose(self, verbose):
        if isinstance(verbose, bool):
            self._verbose = int(verbose)  # convert False to 0, True to 1
        elif isinstance(verbose, int):
            if verbose < 0:
                raise ValueError("Integer values for verbose must be non-negative!")
            self._verbose = verbose
        else:
            raise TypeError("verbose should be of type bool or int!")
