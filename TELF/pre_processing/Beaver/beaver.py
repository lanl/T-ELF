#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 07:01:29 2022

@author: maksim
"""
try:
    from mpi4py import MPI
except:
    MPI = None

import os
import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import multiprocessing
from operator import itemgetter
from collections import defaultdict, Counter

import sparse
from tqdm import tqdm
import scipy.sparse as ss
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfTransformer

from .tenmat import fold, unfold
from .vectorize import tfidf
from .vectorize import count
from .cooccurrence import co_occurrence
from .sppmi import sppmi
from typing import Union


class Beaver():

    SUPPORTED_OUTPUT_FORMATS = {
        'scipy',   # spicy.sparse.csr 
        'pydata',  # sparse.coo
    }

    def __init__(self, n_nodes=1, n_jobs=1) -> None:
        self.n_nodes = n_nodes
        self.n_jobs = n_jobs

        # create a dictionary of supported output formats to their callable output functions
        prefix = '_output_'
        spacing = len(prefix)
        self.output_funcs = {
            name[spacing:]: getattr(self, name) for name in dir(self) if 
            name[spacing:] in Beaver.SUPPORTED_OUTPUT_FORMATS and prefix in name and callable(getattr(self, name))
        }
        
        # create a dictionary of supported output formats to their callable save functions
        prefix = '_save_'
        spacing = len(prefix)
        self.save_funcs = {
            name[spacing:]: getattr(self, name) for name in dir(self) if 
            name[spacing:] in Beaver.SUPPORTED_OUTPUT_FORMATS and prefix in name and callable(getattr(self, name))
        }


    def get_vocabulary(self,
                       dataset: pd.DataFrame,
                       target_column: str=None,
                       split_with: str=None,
                       max_df: Union[int, float]=1.0,
                       min_df: int=1,
                       max_features: int=None,
                       verbose: bool=False,
                       n_jobs: int=None,
                       parallel_backend: str="multiprocessing",
                       save_path: str=None,
                       sort_alphabetical: bool=True,) -> list:
        """
        Builds the vocabulary

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        target_column : str, optional
            Target column name in dataset DataFrame.  
        split_with : str, optional
            Delimeter to use in tokenization, if None split in whitespace. The default is None.
        max_df : int or float, optional
            When building the vocabulary ignore terms that have a document frequency strictly higher 
            than the given threshold (corpus-specific stop words). If float in range [0.0, 1.0], 
            the parameter represents a proportion of documents, integer absolute counts. 
            The default is 1.0.
        min_df : int or float, optional
            When building the vocabulary ignore terms that have a document frequency strictly lower 
            than the given threshold. This value is also called cut-off in the literature. 
            If float in range of [0.0, 1.0], the parameter represents a proportion of documents, 
            integer absolute counts. This parameter is ignored if vocabulary is not None. The default is 1.
        max_features : int, optional
            If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. 
            The default is None.
        verbose : bool, optional
            Vebosity flag. The default is False.
        n_jobs : int, optional
            Number of parallel processes. The default is the Beaver default.
        parallel_backend : str, optional
            Which backend Joblib should use.
        save_path : str, optional
            If not None, saves the outputs. The default is None.
        sort_alphabetical : bool, optional
            If True, sorts the output vocabulary alphabetically. Otherwise the vocabulary is sorted by document frequency.

        Returns
        -------
        List of tokens in the vocabulary.

        """

        assert target_column in dataset, "Target column is not found!"

        # get the target data
        data = dataset[target_column].values.tolist()

        # organize parameters
        if isinstance(max_df, float):
            max_df = int(len(data) * max_df)
        if isinstance(min_df, float):
            min_df = int(len(data) * min_df)

        assert max_df <= len(data), "max_df must be <= data"
        assert min_df > 0, "min_df must be > 0"


        if n_jobs <= 0:
            n_chunks = multiprocessing.cpu_count()
            n_chunks -= (np.abs(n_jobs) + 1)
        else:
            n_chunks = n_jobs

        # chunk the documents to get list of jobs
        if n_jobs is not None:
            self.n_jobs = n_jobs
        n_chunks = self.n_jobs
        data_chunks = list(self._chunk_list(data, n_chunks))

        if split_with == '':
            split_with = None

        # get the stats for the tokens in parallel
        list_results = Parallel(n_jobs=n_chunks, verbose=verbose, backend=parallel_backend)(
            delayed(self._get_vocabulary_helper)(data=chunk, split_with=split_with) for chunk in data_chunks)

        # combine the stats
        token_stats = {}
        for curr_token_stats in tqdm(list_results, disable=not verbose, total=len(list_results)):
            for token, stats in curr_token_stats.items():
                if token in token_stats:
                    token_stats[token]["df"] += stats["df"]
                    token_stats[token]["tf"] += stats["tf"]
                else:
                    token_stats[token] = {"df": 0, "tf": 0}
                    token_stats[token]["df"] += stats["df"]
                    token_stats[token]["tf"] += stats["tf"]
                    
        # filter for document frequencies when selecting the tokens in vocabulary
        vocabulary = set()
        for token, stats in tqdm(token_stats.items(), total=len(token_stats), disable=not verbose):
            if stats["df"] <= max_df and stats["df"] >= min_df:
                vocabulary.add(token)

        # max features
        if max_features:
            max_token_tf_stat = {}
            for token in vocabulary:
                max_token_tf_stat[token] = token_stats[token]["tf"]

            if max_features > len(token_stats):
                warnings.warn(
                    "More features requested than available number of tokens... Reverting to max tokens.")
                max_features = len(token_stats)

            max_token_tf_stat = dict(sorted(max_token_tf_stat.items(), key=itemgetter(1),
                                            reverse=True)[:max_features])

            vocabulary = list(max_token_tf_stat.keys())

        if sort_alphabetical:
            vocabulary = sorted(vocabulary)
        else:  # sort the vocabulary by the document frequency
            vocabulary = sorted(vocabulary, key=lambda x: token_stats[x]['df'], reverse=True)
            
        if save_path:
            np.savetxt(os.path.join(save_path, "Vocabulary.txt"), vocabulary,  fmt="%s", encoding="utf-8")
        return vocabulary


    def coauthor_tensor(self,
                         dataset: pd.DataFrame,
                        target_columns: tuple=("authorIDs", "year"),
                        split_authors_with: str=";",
                        verbose: bool=False,
                        save_path: str=None,
                        n_nodes: int=None,
                        n_jobs:int =None,
                        joblib_backend: str="multiprocessing",
                        authors_idx_map: dict={},
                        time_idx_map: dict={},
                        return_object: bool=True,
                        output_mode: str='pydata',
                        ) -> tuple:
        """
        Create co-author tensor.
        Returns tuple of tensor, authors, and time.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        target_columns : tuple, optional
            Target column names in dataset DataFrame. The default is ("authorIDs", "year").
            When assigning names in this tuple, type order should be preserved, e.g. time column name comes second.
        split_authors_with : str, optional
            What symbol to use to get list of individual authors from string. The default is ";".
        verbose : bool, optional
            Vebosity flag. The default is False.
        save_path : str, optional
            If not None, saves the outputs.. The default is None.
        n_nodes: int, optional
            Number of nodes to use. Default is the Beaver default.
        n_jobs: int, optional
            Number of jobs to use. Default is the Beaver default.
        joblib_backend: str, optional
            Joblib parallel backend. Default is multiprocessing.
        authors_idx_map : dict, optional
            Author to tensor dimension index mapping. Default is {}.
            If not passed, it is created.
        time_idx_map : dict, optional
            Time to tensor dimension index mapping. Default is {}.
            If not passed, it is created.
        return_object : bool, optional
            Flag that determines whether the generated object is returned from this function. In the case of large
            tensors it may be better to save to disk without returning. Default is True.
        output_mode : str, optional
            The type of object returned in the output. See supported options in Beaver.SUPPORTED_OUTPUT_FORMATS. 
            Default is 'pydata'.

        Returns
        -------
        tuple
            Tuple of tensor, author vocabulary, and time vocabulary.

        """
        if n_nodes is not None:
            self.n_nodes = n_nodes
        if n_jobs is not None:
            self.n_jobs = n_jobs
        if self.n_nodes > 1 and MPI is None:
            sys.exit("Attempted to use n_nodes>1 but MPI is not available!")

        assert target_columns[0] in dataset, f'Target column {target_columns[0]} not found'
        assert target_columns[1] in dataset, f'Target column {target_columns[1]} not found'

        # validate mode
        if output_mode not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported mode '{output_mode}'. Modes include {self.SUPPORTED_OUTPUT_FORMATS}")
        elif output_mode == 'scipy':
            raise ValueError('Scipy does not support sparse tensors. Please use "pydata"')
            
        if save_path is None and not return_object:
            warnings.warn('Function does not return object yet no save path has been provided!', RuntimeWarning)
            
        # create authors map
        all_authors = dataset[target_columns[0]].values.tolist()
        if len(authors_idx_map) == 0:
            idx = 0
            for curr_authors_str in sorted(all_authors):
                curr_authors_list = curr_authors_str.split(split_authors_with)
                for aa in curr_authors_list:
                    if aa not in authors_idx_map:
                        authors_idx_map[aa] = idx
                        idx += 1

        # create time map
        times = dataset[target_columns[1]].values.tolist()
        if len(time_idx_map) == 0:
            time_idx_map = {year: i for i, year in enumerate(sorted(dataset[target_columns[1]].unique()))}

        # handle for multiple nodes
        n_nodes = self.n_nodes
        if n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            time_node_chunks = np.array_split(times, n_nodes)
            all_author_node_chunks = np.array_split(all_authors, n_nodes)
            times = time_node_chunks[rank]
            all_authors = all_author_node_chunks[rank]
        else:
            comm = None
            rank = -1

        # split the data into n_jobs chunks
        n_chunks = self.n_jobs
        n_chunks = min(n_chunks, len(all_authors))
        time_chunks = np.array_split(times, n_chunks)
        all_author_chunks = np.array_split(all_authors, n_chunks)
        tensor_dicts_all = Parallel(n_jobs=n_chunks, verbose=verbose, backend=joblib_backend)(delayed(self._coauthor_tensor_helper)
                                                                                             (time_idx_map=time_idx_map,
                                                                                              authors_idx_map=authors_idx_map,
                                                                                              all_authors=curr_doc_authors, 
                                                                                              times=curr_times, 
                                                                                              split_authors_with=split_authors_with)
            for curr_doc_authors, curr_times in zip(all_author_chunks, time_chunks))

        shape = (len(authors_idx_map), len(authors_idx_map), len(time_idx_map))
        X = self._dist_parallel_tensor_build_helper(tensor_dicts_all=tensor_dicts_all,
                                                    verbose=verbose,
                                                    n_nodes=n_nodes,
                                                    comm=comm,
                                                    rank=rank,
                                                    n_chunks=n_chunks,
                                                    shape=shape)

        X = self.output_funcs[output_mode](X)
        if save_path:
            np.savetxt(os.path.join(save_path, "Authors.txt"), list(authors_idx_map.keys()),  fmt="%s", encoding="utf-8")
            np.savetxt(os.path.join(save_path, "Time.txt"), list(time_idx_map.keys()),  fmt="%s", encoding="utf-8")
            self.save_funcs[output_mode](X, os.path.join(save_path, "coauthor.npz"))
        if return_object:
            return (X, list(authors_idx_map.keys()), list(time_idx_map.keys()))


    def cocitation_tensor(self,
                          dataset: pd.DataFrame,
                          target_columns: tuple=("authorIDs", "year", "paper_id", "references"),
                          split_authors_with: str=";",
                          split_references_with: str=";",
                          verbose: bool=False,
                          save_path: str=None,
                          n_nodes: int=None,
                          n_jobs: int=None,
                          joblib_backend: str="multiprocessing",
                          authors_idx_map: dict={},
                          time_idx_map: dict={},
                          return_object: bool=True,
                          output_mode: str='pydata',
                          ) -> tuple:
        """
        Creates an Authors by Authors by Time tensor. An non-zero entry x at author i, 
        author j, year k means that author j cited author i x times in year k. Note that
        x is normalized. This means that for two papers a and b where a cites b, the n 
        authors of a, and a single author from b, the author from b receives 1/n citations 
        from each author on paper a.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        target_columns : tuple, optional
            Target column names in dataset DataFrame. The default is ("authorIDs", "year", "paper_id", "references").
            When assigning names in this tuple, type order should be preserved, e.g. time column name comes second.
        split_authors_with : str, optional
            What symbol to use to get list of individual authors from string. The default is ";".
        split_references_with : TYPE, optional
            What symbol to use to get list of individual references from string. The default is ";".
        verbose : bool, optional
            Vebosity flag. The default is False.
        save_path : str, optional
            If not None, saves the outputs. The default is None.
        n_nodes: int, optional
            Number of nodes to use. Default is is the Beaver default.
        n_jobs : int, optional
            Number of parallel jobs. The default is the Beaver default.
        authors_idx_map : dict, optional
            Author to tensor dimension index mapping. Default is {}.
            If not passed, it is created.
        time_idx_map : dict, optional
            Time to tensor dimension index mapping. Default is {}.
            If not passed, it is created.
        return_object : bool, optional
            Flag that determines whether the generated object is returned from this function. In the case of large
            tensors it may be better to save to disk without returning. Default is True.
        output_mode : str, optional
            The type of object returned in the output. See supported options in Beaver.SUPPORTED_OUTPUT_FORMATS. 
            Default is 'pydata'.

        Returns
        -------
        tuple
            Tuple of tensor, author vocabulary, and time vocabulary.

        """
        if n_nodes is not None:
            self.n_nodes = n_nodes
        if n_jobs is not None:
            self.n_jobs = n_jobs
        if self.n_nodes > 1 and MPI is None:
            sys.exit("Attempted to use n_nodes>1 but MPI is not available!")

        # validate mode
        if output_mode not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported mode '{output_mode}'. Modes include {self.SUPPORTED_OUTPUT_FORMATS}")
        elif output_mode == 'scipy':
            raise ValueError('Scipy does not support sparse tensors. Please use "pydata"')
        
        if save_path is None and not return_object:
            warnings.warn('Function does not return object yet no save path has been provided!', RuntimeWarning)
        
        assert target_columns[0] in dataset, f'Target column {target_columns[0]} not found'
        assert target_columns[1] in dataset, f'Target column {target_columns[1]} not found'
        assert target_columns[2] in dataset, f'Target column {target_columns[2]} not found'
        assert target_columns[3] in dataset, f'Target column {target_columns[3]} not found'

        # create authors map
        all_authors = dataset[target_columns[0]].values.tolist()
        if len(authors_idx_map) == 0:
            idx = 0
            for curr_authors_str in sorted(all_authors):
                curr_authors_list = curr_authors_str.split(split_authors_with)
                for aa in curr_authors_list:
                    if aa not in authors_idx_map:
                        authors_idx_map[aa] = idx
                        idx += 1

        # create time map
        times = dataset[target_columns[1]].values.tolist()
        if len(time_idx_map) == 0:
            time_idx_map = {year: i for i, year in enumerate(sorted(dataset[target_columns[1]].unique()))}

        # document to authors map
        all_doc_ids = dataset[target_columns[2]].values.tolist()
        document_authors_map = {}
        for idx, docID in tqdm(enumerate(all_doc_ids), disable=not verbose, total=len(all_doc_ids)):
            document_authors_map[docID] = all_authors[idx].split(split_authors_with)

        # document to references map
        all_references = dataset[target_columns[3]].values.tolist()
        documents_references_map = {}
        for idx, docID in tqdm(enumerate(all_doc_ids), disable=not verbose, total=len(all_doc_ids)):
            curr_references = all_references[idx]
            curr_references = curr_references.split(split_references_with) if not pd.isna(curr_references) else []
            documents_references_map[docID] = curr_references

        # handle for multiple nodes
        n_nodes = self.n_nodes
        if n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            time_node_chunks = np.array_split(times, n_nodes)
            all_doc_node_chunks = np.array_split(all_doc_ids, n_nodes)
            times = time_node_chunks[rank]
            all_doc_ids = all_doc_node_chunks[rank]
        else:
            comm = None
            rank = -1

        n_chunks = self.n_jobs
        n_chunks = min(n_chunks, len(all_doc_ids))
        time_chunks = np.array_split(times, n_chunks)
        document_ids_chunks = np.array_split(all_doc_ids, n_chunks)
        tensor_dicts_all = Parallel(n_jobs=n_chunks, verbose=verbose, backend=joblib_backend)(delayed(self._cocitation_tensor_helper) 
                                                                                             (all_doc_ids=curr_doc_ids,
                                                                                              times=curr_times,
                                                                                              documents_references_map=documents_references_map,
                                                                                              time_idx_map=time_idx_map,
                                                                                              document_authors_map=document_authors_map,
                                                                                              authors_idx_map=authors_idx_map)


            for curr_doc_ids, curr_times in zip(document_ids_chunks, time_chunks))

        # numpy COO format to Sparse tensor format
        shape = (len(authors_idx_map), len(authors_idx_map), len(time_idx_map))
        X = self._dist_parallel_tensor_build_helper(tensor_dicts_all=tensor_dicts_all,
                                                    verbose=verbose,
                                                    n_nodes=n_nodes,
                                                    comm=comm,
                                                    rank=rank,
                                                    n_chunks=n_chunks,
                                                    shape=shape)
        
        X = self.output_funcs[output_mode](X)
        if save_path:
            np.savetxt(os.path.join(save_path, "Authors.txt"), list(authors_idx_map.keys()),  fmt="%s", encoding="utf-8")
            np.savetxt(os.path.join(save_path, "Time.txt"), list(time_idx_map.keys()),  fmt="%s", encoding="utf-8")
            self.save_funcs[output_mode](X, os.path.join(save_path, "cocitation.npz"))
        if return_object:
            return (X, list(authors_idx_map.keys()), list(time_idx_map.keys()))


    def participation_tensor(self,
                             dataset: pd.DataFrame,
                             target_columns: tuple=("author_ids", "paper_id", "year"),
                             dimension_order: list=[0, 1, 2],
                             split_authors_with: str=";",
                             save_path: str=None,
                             n_nodes: int=None,
                             n_jobs: int=None,
                             joblib_backend: str="multiprocessing",
                             verbose: bool=False,
                             return_object: bool=True,
                             output_mode: str='pydata',
                             ) -> tuple:
        """
        Creates a boolean Authors by Papers by Time tensor. An non-zero entry at author i, 
        paper j, year k means that author i published paper j in year k

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        target_columns : tuple, optional
            Target column names in dataset DataFrame. The default is ("author_ids", "paper_id", "year").
            When assigning names in this tuple, type order should be preserved, e.g. time column name comes last.
        dimension_order: list, optional
            Order in which the dimensions appear. 
            For example, [0,1,2] means it is Authors, Papers, Time
            and [1,0,2] means it is Papers, Authors, Time.
        split_authors_with : str, optional
            What symbol to use to get list of individual elements from string of target_columns[0]. The default is ";".
        save_path : str, optional
            If not None, saves the outputs. The default is None.
        n_nodes: int, optional
            Number of nodes to use. Default is the Beaver default.
        n_jobs: int, optional
            Number of jobs to use. Default is the Beaver default.
        joblib_backend: str, optional
            Joblib parallel backend. Default is multiprocessing.
        verbose : bool, optional
            Vebosity flag. The default is False.
        return_object : bool, optional
            Flag that determines whether the generated object is returned from this function. In the case of large
            tensors it may be better to save to disk without returning. Default is True.
        output_mode : str, optional
            The type of object returned in the output. See supported options in Beaver.SUPPORTED_OUTPUT_FORMATS. 
            Default is 'pydata'.
        """
        if n_nodes is not None:
            self.n_nodes = n_nodes
        if n_jobs is not None:
            self.n_jobs = n_jobs
        if self.n_nodes > 1 and MPI is None:
            sys.exit("Attempted to use n_nodes>1 but MPI is not available!")

        # validate mode
        if output_mode not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported mode '{output_mode}'. Modes include {self.SUPPORTED_OUTPUT_FORMATS}")
        elif output_mode == 'scipy':
            raise ValueError('Scipy does not support sparse tensors. Please use "pydata"')
        
        if save_path is None and not return_object:
            warnings.warn('Function does not return object yet no save path has been provided!', RuntimeWarning)
        
        # quick validation of column names
        assert target_columns[0] in dataset, f'Target column {target_columns[0]} not found'
        assert target_columns[1] in dataset, f'Target column {target_columns[1]} not found'
        assert target_columns[2] in dataset, f'Target column {target_columns[2]} not found'

        # create authors map
        idx = 0
        authors_idx_map = {}
        authors_list = dataset[target_columns[0]].values.tolist()
        for curr_authors_str in sorted(authors_list):
            curr_authors_list = curr_authors_str.split(split_authors_with)
            for auth in curr_authors_list:
                if auth not in authors_idx_map:
                    authors_idx_map[auth] = idx
                    idx += 1

        # create papers map
        papers_list = dataset[target_columns[1]].values.tolist()
        papers_idx_map = {paper: i for i, paper in enumerate(sorted(dataset[target_columns[1]].unique()))}

        # create time map
        time_list = dataset[target_columns[2]].values.tolist()
        time_idx_map = {year: i for i, year in enumerate(sorted(dataset[target_columns[2]].unique()))}

        # handle for multiple nodes
        n_nodes = self.n_nodes
        if n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            # using MPI, grab data intended for current rank
            authors_list = np.array_split(authors_list, n_nodes)[rank]
            papers_list = np.array_split(papers_list, n_nodes)[rank]
            time_list = np.array_split(time_list, n_nodes)[rank]
        else:
            comm = None
            rank = -1

        # determine how many parallel processes are required
        n_chunks = self.n_jobs
        n_chunks = min(n_chunks, len(papers_list))
        
        # compute non-zero coordinates on local node
        authors_chunks = np.array_split(authors_list, n_chunks)
        papers_chunks = np.array_split(papers_list, n_chunks)
        time_chunks = np.array_split(time_list, n_chunks)
        tensor_dicts_all = Parallel(n_jobs=n_chunks, verbose=verbose, backend=joblib_backend)(delayed(self._participation_tensor_helper)
                                                                                              (dimension_order=dimension_order,
                                                                                               authors_idx_map=authors_idx_map,
                                                                                               papers_idx_map=papers_idx_map,
                                                                                               time_idx_map=time_idx_map,
                                                                                               authors_list=curr_authors,
                                                                                               papers_list=curr_papers,
                                                                                               time_list=curr_time,
                                                                                               split_authors_with=split_authors_with)
            for curr_authors, curr_papers, curr_time in zip(authors_chunks, papers_chunks, time_chunks))

        # numpy COO format to Sparse tensor format
        map_lens = [len(authors_idx_map), len(papers_idx_map), len(time_idx_map)]
        shape = tuple([map_lens[x] for x in dimension_order])
        X = self._dist_parallel_tensor_build_helper(tensor_dicts_all=tensor_dicts_all,
                                                    verbose=verbose,
                                                    n_nodes=n_nodes,
                                                    comm=comm,
                                                    rank=rank,
                                                    n_chunks=n_chunks,
                                                    shape=shape)

        X = self.output_funcs[output_mode](X)
        if save_path:
            np.savetxt(os.path.join(save_path, "Authors.txt"), list(authors_idx_map.keys()),  fmt="%s", encoding="utf-8")
            np.savetxt(os.path.join(save_path, "Paper.txt"), list(papers_idx_map.keys()),  fmt="%s", encoding="utf-8")
            np.savetxt(os.path.join(save_path, "Time.txt"), list(time_idx_map.keys()),  fmt="%s", encoding="utf-8")
            self.save_funcs[output_mode](X, os.path.join(save_path, "participation.npz"))
        if return_object:
            return (X, list(authors_idx_map.keys()), list(papers_idx_map.keys()), list(time_idx_map.keys()))


    def citation_tensor(self,
                        dataset: pd.DataFrame,
                        target_columns: tuple=("author_ids", "paper_id", "references", "year"),
                        dimension_order: list=[0, 1, 2],
                        split_authors_with: str=";",
                        split_references_with: str=";",
                        save_path: str=None,
                        n_nodes: int=None,
                        n_jobs: int=None,
                        joblib_backend: str="loky",
                        verbose: bool=False,
                        return_object: bool=True,
                        output_mode: str='pydata',
                       ) -> tuple:
        """
        Creates an Authors by Papers by Time tensor. An non-zero entry x at author i, 
        paper j, year k means that author i cited paper j x times in year k

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        target_columns : tuple, optional
            Target column names in dataset DataFrame. The default is ("author_ids", "paper_id", "references", "year").
            When assigning names in this tuple, type order should be preserved, e.g. time column name comes last.
        dimension_order: list, optional
            Order in which the dimensions appear. 
            For example, [0,1,2] means it is Authors, Papers, Time
            and [1,0,2] means it is Papers, Authors, Time.
        split_authors_with : str, optional
            What symbol to use to get list of individual elements from string of target_columns[0]. The default is ";".
        split_references_with : str, optional
            What symbol to use to get list of individual elements from string of target_columns[2]. The default is ";".
        save_path : str, optional
            If not None, saves the outputs. The default is None.
        n_nodes: int, optional
            Number of nodes to use. Default is the Beaver default.
        n_jobs: int, optional
            Number of jobs to use. Default is the Beaver default.
        joblib_backend: str, optional
            Joblib parallel backend. Default is multiprocessing.
        verbose : bool, optional
            Vebosity flag. The default is False.
        return_object : bool, optional
            Flag that determines whether the generated object is returned from this function. In the case of large
            tensors it may be better to save to disk without returning. Default is True.
        output_mode : str, optional
            The type of object returned in the output. See supported options in Beaver.SUPPORTED_OUTPUT_FORMATS. 
            Default is 'pydata'.
        """
        if n_nodes is not None:
            self.n_nodes = n_nodes
        if n_jobs is not None:
            self.n_jobs = n_jobs
        if self.n_nodes > 1 and MPI is None:
            sys.exit("Attempted to use n_nodes>1 but MPI is not available!")

        # validate mode
        if output_mode not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported mode '{output_mode}'. Modes include {self.SUPPORTED_OUTPUT_FORMATS}")
        elif output_mode == 'scipy':
            raise ValueError('Scipy does not support sparse tensors. Please use "pydata"')
    
        if save_path is None and not return_object:
            warnings.warn('Function does not return object yet no save path has been provided!', RuntimeWarning)
    
        # quick validation of column names
        assert target_columns[0] in dataset, f'Target column {target_columns[0]} not found'
        assert target_columns[1] in dataset, f'Target column {target_columns[1]} not found'
        assert target_columns[2] in dataset, f'Target column {target_columns[2]} not found'

        # create authors map
        idx = 0
        authors_idx_map = {}
        authors_list = dataset[target_columns[0]].values.tolist()
        for curr_authors_str in sorted(authors_list):
            curr_authors_list = curr_authors_str.split(split_authors_with)
            for auth in curr_authors_list:
                if auth not in authors_idx_map:
                    authors_idx_map[auth] = idx
                    idx += 1

        # create papers map
        papers_list = dataset[target_columns[1]].values.tolist()
        papers_idx_map = {paper: i for i, paper in enumerate(sorted(dataset[target_columns[1]].unique()))}

        # create paper to authors map
        assert len(authors_list) == len(papers_list), "Authors & Papers lists cannot be different lengths"
        document_authors_map = {papers_list[i]: authors_list[i].split(split_authors_with) 
                                for i in range(len(papers_list))}

        # create document to references map
        references_list = dataset[target_columns[2]].values.tolist()
        assert len(references_list) == len(papers_list), "References & Papers lists cannot be different lengths"
        document_references_map = {papers_list[i]: references_list[i].split(split_references_with) 
                                for i in range(len(papers_list)) if references_list[i]}

        # create time map
        time_list = dataset[target_columns[3]].values.tolist()
        time_idx_map = {year: i for i, year in enumerate(sorted(dataset[target_columns[3]].unique()))}

        # handle for multiple nodes
        n_nodes = self.n_nodes
        if n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            # using MPI, grab data intended for current rank
            papers_list = np.array_split(papers_list, n_nodes)[rank]
            time_list = np.array_split(time_list, n_nodes)[rank]
        else:
            comm = None
            rank = -1

        # determine how many parallel processes are required
        n_chunks = self.n_jobs
        n_chunks = min(n_chunks, len(papers_list))

        # compute non-zero coordinates on local node
        papers_chunks = np.array_split(papers_list, n_chunks)
        time_chunks = np.array_split(time_list, n_chunks)
        tensor_dicts_all = Parallel(n_jobs=n_chunks, verbose=verbose, backend=joblib_backend)(delayed(self._citation_tensor_helper)
                                                                                             (dimension_order=dimension_order,
                                                                                              authors_idx_map=authors_idx_map,
                                                                                              papers_idx_map=papers_idx_map,
                                                                                              time_idx_map=time_idx_map,
                                                                                              papers_list=curr_papers,
                                                                                              time_list=curr_time,
                                                                                              document_authors_map=document_authors_map,
                                                                                              document_references_map=document_references_map)
            for curr_papers, curr_time in zip(papers_chunks, time_chunks))

        # numpy COO format to Sparse tensor format
        map_lens = [len(authors_idx_map), len(papers_idx_map), len(time_idx_map)]
        shape = tuple([map_lens[x] for x in dimension_order])
        X = self._dist_parallel_tensor_build_helper(tensor_dicts_all=tensor_dicts_all,
                                                    verbose=verbose,
                                                    n_nodes=n_nodes,
                                                    comm=comm,
                                                    rank=rank,
                                                    n_chunks=n_chunks,
                                                    shape=shape)

        X = self.output_funcs[output_mode](X)
        if save_path:
            np.savetxt(os.path.join(save_path, "Authors.txt"), list(authors_idx_map.keys()),  fmt="%s", encoding="utf-8")
            np.savetxt(os.path.join(save_path, "Paper.txt"), list(papers_idx_map.keys()),  fmt="%s", encoding="utf-8")
            np.savetxt(os.path.join(save_path, "Time.txt"), list(time_idx_map.keys()),  fmt="%s", encoding="utf-8")
            self.save_funcs[output_mode](X, os.path.join(save_path, "citation.npz"))
        if return_object:
            return (X, list(authors_idx_map.keys()), list(papers_idx_map.keys()), list(time_idx_map.keys()))


    def cooccurrence_matrix(self,
                           dataset: pd.DataFrame,
                           target_column: str="abstracts",
                           cooccurrence_settings: dict={},
                           sppmi_settings: dict={},
                           save_path: str=None,
                           return_object: bool=True,
                           output_mode: str='scipy',
                           ) -> tuple:
        """
        Generates co-occurance and SPPMI matrix.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        target_column : str, optional
            Target column name in dataset DataFrame. The default is "abstracts".
            Target column should be for text data, where tokens are retrived via empty spaces.
        cooccurrence_settings : dict, optional
            Settings for co-occurance matrix. The default is dict.
            Options are: vocabulary, window_size=20, dense=True, verbose=True, sentences=False
        sppmi_settings : dict, optional
            Settings for SPPMI matrix. The default is dict.
            Options are: shift=4
        save_path : str, optional
            If not None, saves the outputs. The default is None.
        return_object : bool, optional
            Flag that determines whether the generated object is returned from this function. In the case of large
            tensors it may be better to save to disk without returning. Default is True.
        output_mode : str, optional
            The type of object returned in the output. See supported options in Beaver.SUPPORTED_OUTPUT_FORMATS. 
            Default is 'scipy'.

        Returns
        -------
        tuple
            Tuple of co-occurance and SPPMI matrix.

        """

        # Default settings
        if "window_size" not in cooccurrence_settings:
            cooccurrence_settings["window_size"] = 100

        # validate mode
        if output_mode not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported mode '{output_mode}'. Modes include {self.SUPPORTED_OUTPUT_FORMATS}")
        if save_path is None and not return_object:
            warnings.warn('Function does not return object yet no save path has been provided!', RuntimeWarning)
        assert target_column in dataset, "Target column is not found!"

        # get the target documents
        documents = dataset[target_column].values.tolist()

        # Create the matrices
        M = co_occurrence(documents=documents, **cooccurrence_settings)
        SPPMI = sppmi(M, **sppmi_settings)

        # convert to pydata sparse for consistency across all beaver methods
        M = sparse.COO.from_scipy_sparse(M)
        SPPMI = sparse.COO.from_scipy_sparse(SPPMI)
    
        M = self.output_funcs[output_mode](M)
        SPPMI = self.output_funcs[output_mode](SPPMI)
        if save_path:
            self.save_funcs[output_mode](M, os.path.join(save_path, "cooccurrence.npz"))
            self.save_funcs[output_mode](SPPMI, os.path.join(save_path, "SPPMI.npz"))
        if return_object:
            return (M, SPPMI)


    def documents_words(self,
                        dataset: pd.DataFrame,
                        target_column: str="abstracts",
                        options: dict={"min_df": 5, "max_df": 0.5},
                        highlighting: list=[],
                        weights: list=[],
                        matrix_type: str="tfidf",
                        verbose: bool=False,
                        return_object: bool=True,
                        output_mode: str='scipy',
                        save_path: str=None) -> tuple:
        """
        Creates document-words matrix.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        target_column : str, optional
            Target column name in dataset DataFrame.  The default is "abstracts".
            Target column should be for text data, where tokens are retrived via empty spaces.
        options : dict, optional
            Settings for when doing vectorization. The default is {"min_df": 5, "max_df": 0.5}.
        matrix_type : str, optional
            TF-IDF or Count vectorization. The default is "tfidf".
            Other option is "count".
        verbose : bool, optional
            Vebosity flag. The default is False.
        highlighting : list, optional
            The vocabulary or list of tokens to highlight. The default is [].
            Other option is "count".
        weights : list or float or int, optional
            Weights of the highlighted words. The default is [].
        save_path : str, optional
            If not None, saves the outputs. The default is None.
        return_object : bool, optional
            Flag that determines whether the generated object is returned from this function. In the case of large
            tensors it may be better to save to disk without returning. Default is True.
        output_mode : str, optional
            The type of object returned in the output. See supported options in Beaver.SUPPORTED_OUTPUT_FORMATS. 
            Default is 'scipy'.

        Returns
        -------
        tuple
            Tuple of matrix and vocabulary.

        """

        assert matrix_type in ["tfidf", "count"], "Unknown matrix type!"
        assert target_column in dataset, "Target column is not found!"
        assert isinstance(highlighting, list) or isinstance(
            highlighting, np.ndarray), "highlighting should be type list or array!"

        # validate mode
        if output_mode not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported mode '{output_mode}'. Modes include {self.SUPPORTED_OUTPUT_FORMATS}")
        if save_path is None and not return_object:
            warnings.warn('Function does not return object yet no save path has been provided!', RuntimeWarning)
            
        if isinstance(weights, list) or isinstance(weights, np.ndarray):
            assert len(weights) == len(
                highlighting), "length of weights does not match length of highlighting!"
        elif isinstance(weights, int) or isinstance(weights, float):
            weights = [weights] * len(highlighting)

        # get the target documents
        documents = dataset[target_column].values.tolist()
        
        # merge the vocabulary with highliting words
        if (len(highlighting) > 0) and ("vocabulary" in options):
            vocab_options = options["vocabulary"].copy()
            vocab_options = dict(zip(vocab_options, [1]*len(vocab_options)))

            added = False
            for token in highlighting:
                if token not in vocab_options:
                    vocab_options[token] = 1
                    added = True

            if added:
                options["vocabulary"] = sorted(list(vocab_options.keys()))
                warnings.warn("Vocabulary was extended!")

        # vectorize
        if matrix_type == "tfidf":
            X, vocabulary = tfidf(documents, options)

        else:
            X, vocabulary = tfidf(documents, options)

        if len(highlighting) > 0:
            for widx, token in tqdm(enumerate(highlighting), disable=not verbose):
                idxs = np.where(vocabulary == token)[0]
                if len(idxs):
                    X[idxs[0]] = X[idxs[0]] * weights[widx]

        # convert to pydata sparse for consistency across all beaver methods
        X = sparse.COO.from_scipy_sparse(X)
                    
        X = self.output_funcs[output_mode](X)
        if save_path:
            np.savetxt(os.path.join(save_path, "Vocabulary.txt"), vocabulary,  fmt="%s", encoding="utf-8")
            self.save_funcs[output_mode](X, os.path.join(save_path, "documents_words.npz"))
        if return_object:
            return (X, vocabulary)


    def something_words(self,
                        dataset: pd.DataFrame,
                        target_columns: tuple=("authorIDs", "abstracts"),
                        split_something_with: str=";",
                        options: dict={"min_df": 5, "max_df": 0.5},
                        highlighting: list=[],
                        weights: list=[],
                        verbose: bool=False,
                        matrix_type: str="tfidf",
                        return_object: bool=True,
                        output_mode: str='scipy',
                        save_path: str=None) -> tuple:
        """
        Creates a Something by Words matrix. For example, Authors-Words.
        Here something is specified by first index of variable target_columns.
        Individual evelements of target_columns[0] is seperated by split_something_with. 
        For example "autho1;author2" when split_something_with=";".

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        target_columns : tuple, optional
            Target column names in dataset DataFrame. The default is ("authorIDs", "abstracts").
            When assigning names in this tuple, type order should be preserved, e.g. text data column name comes second.
        split_something_with : str, optional
            What symbol to use to get list of individual elements from string of target_columns[0]. The default is ";".
        options : str, optional
            Settings for when doing vectorization. The default is {"min_df": 5, "max_df": 0.5}.
        highlighting : list, optional
            The vocabulary or list of tokens to highlight. The default is [].
            Other option is "count".
        weights : list or float or int, optional
            Weights of the highlighted words. The default is [].
        verbose : bool, optional
            Vebosity flag. The default is False.
        matrix_type : str, optional
            TF-IDF or Count vectorization. The default is "tfidf".
            Other option is "count"
        save_path : TYPE, optional
            If not None, saves the outputs. The default is None.
        return_object : bool, optional
            Flag that determines whether the generated object is returned from this function. In the case of large
            tensors it may be better to save to disk without returning. Default is True.
        output_mode : str, optional
            The type of object returned in the output. See supported options in Beaver.SUPPORTED_OUTPUT_FORMATS. 
            Default is 'scipy'.

        Returns
        -------
        tuple
            Tuple of matrix, vocabulary for somethings (target information specified in target_columns[0]),
            and the vocabulary for words.

        """

        assert matrix_type in ["tfidf", "count"], "Unknown matrix type!"
        assert target_columns[0] in dataset, f'Target column {target_columns[0]} not found'
        assert target_columns[1] in dataset, f'Target column {target_columns[1]} not found'

        # validate mode
        if output_mode not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported mode '{output_mode}'. Modes include {self.SUPPORTED_OUTPUT_FORMATS}")
        if save_path is None and not return_object:
            warnings.warn('Function does not return object yet no save path has been provided!', RuntimeWarning)
            
        # get the target documents
        somethings = dataset[target_columns[0]].values.tolist()
        documents = dataset[target_columns[1]].values.tolist()
        
        somethings_documents_map_temp = defaultdict(lambda: [])
        for doc_idx, curr_something in enumerate(sorted(somethings)):
            individual_curr_something = curr_something.split(split_something_with)
            for individual in individual_curr_something:
                somethings_documents_map_temp[individual].append(documents[doc_idx])
                
        somethings_documents_map = {}
        somethings_documents_map_temp = dict(somethings_documents_map_temp)
        for key, value in somethings_documents_map_temp.items():
            somethings_documents_map[key] = " ".join(value)
        
        somethings = list(somethings_documents_map.keys())

        X, vocabulary = self.documents_words(dataset=pd.DataFrame(list(somethings_documents_map.values())),
                                             target_column=0,
                                             options=options,
                                             highlighting=highlighting,
                                             verbose=verbose,
                                             weights=weights,
                                             matrix_type=matrix_type,
                                             output_mode=output_mode,
                                            )
        
        if save_path:
            np.savetxt(os.path.join(save_path, f'{target_columns[0]}.txt'), somethings,  fmt="%s", encoding="utf-8")
            np.savetxt(os.path.join(save_path, "Vocabulary.txt"), vocabulary,  fmt="%s", encoding="utf-8")            
            self.save_funcs[output_mode](X, os.path.join(save_path, f'{target_columns[0]}_words.npz'))
        if return_object:  # object has already been cast to appropriate mode with call to documents_words 
            return (X, somethings, vocabulary)


    def something_words_time(self,
                             dataset: pd.DataFrame,
                             vocabulary: list,
                             target_columns: tuple=("authorIDs", "abstracts", "year"),
                             split_something_with: str=";",
                             save_path: str=None,
                             tfidf_transformer: bool=False,
                             unfold_at=1,
                             verbose: bool=False,
                             dimension_order: list=[0, 1, 2],
                             return_object: bool=True,
                             output_mode: str='pydata',
                             ) -> tuple:
        """
        Creates a Something by Words by Time tensor. For example, Authors-Words-Time.
        Here something is specified by first index of variable target_columns.
        Individual evelements of target_columns[0] is seperated by split_something_with. 
        For example "autho1;author2" when split_something_with=";".

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        vocabulary : list
            Token vocabulary to use.
        target_columns : tuple, optional
            Target column names in dataset DataFrame. The default is ("authorIDs", "abstracts", "year").
            When assigning names in this tuple, type order should be preserved, e.g. time column name comes last.
        split_something_with : str, optional
            What symbol to use to get list of individual elements from string of target_columns[0]. The default is ";".
        save_path : str, optional
            If not None, saves the outputs. The default is None.
        tfidf_transformer : bool, optional
            If True, performs TF-IDF normalization via unfolding over dimension unfold_at. The default is False.
        unfold_at : int, optional
            Which dimension to unfold the tensor for TF-IDF normalization, when tfidf_transformer=True. The default is 1.
        verbose : bool, optional
            Vebosity flag. The default is False.
        dimension_order: list, optional
            Order in which the dimensions appear. 
            For example, [0,1,2] means it is Something, Words, Time.
            and [1,0,2] means it is Words, Something, Time.
        return_object : bool, optional
            Flag that determines whether the generated object is returned from this function. In the case of large
            tensors it may be better to save to disk without returning. Default is True.
        output_mode : str, optional
            The type of object returned in the output. See supported options in Beaver.SUPPORTED_OUTPUT_FORMATS. 
            Default is 'scipy'.

        Returns
        -------
        tuple
            Tuple of matrix, vocabulary for somethings (target information specified in target_columns[0]),
            the vocabulary for words, and the vocabulary for time.

        """
        # validate mode
        if output_mode not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported mode '{output_mode}'. Modes include {self.SUPPORTED_OUTPUT_FORMATS}")
        elif output_mode == 'scipy':
            raise ValueError('Scipy does not support sparse tensors. Please use "pydata"')
            
        if save_path is None and not return_object:
            warnings.warn('Function does not return object yet no save path has been provided!', RuntimeWarning)
            
        # create time map
        times = dataset[target_columns[2]].values.tolist()
        time_idx_map = {year: i for i, year in enumerate(sorted(dataset[target_columns[2]].unique()))}

        # create vocabulary map
        word_idx_map = {}
        for idx, ww in enumerate(vocabulary):
            word_idx_map[ww] = idx

        # create something map
        idx = 0                
        something_idx_map = {}
        somethings = dataset[target_columns[0]].values.tolist()
        for something in sorted(somethings):
            curr_somethings = something.split(split_something_with)
            for ss in curr_somethings:
                if ss not in something_idx_map:
                    something_idx_map[ss] = idx
                    idx += 1

        # create tensor in dicitonary COO format
        documents = dataset[target_columns[1]].values.tolist()
        tensor_dict = defaultdict(lambda: 0)

        for idx, doc in tqdm(enumerate(documents), disable=not verbose):
            curr_time = times[idx]
            curr_time_idx = time_idx_map[curr_time]
            curr_somethings = somethings[idx].split(split_something_with)
            curr_words = documents[idx].split()

            for word in curr_words:

                # word is not in the vocabulary
                if word not in word_idx_map:
                    continue

                curr_word_idx = word_idx_map[word]
                for ss in curr_somethings:
                    curr_something_idx = something_idx_map[ss]
                    coo_str = ";".join(
                        list(map(str, np.array([curr_something_idx,
                                                curr_word_idx,
                                                curr_time_idx])[dimension_order])))
                    tensor_dict[coo_str] += 1

        # turn dictionary to real COO
        tensor_dict = dict(tensor_dict)
        nnz_coords = []
        nnz_values = []
        for key, value in tqdm(tensor_dict.items(), disable=not verbose):
            indices_str = key.split(";")
            indices = list()
            for idx in indices_str:
                indices.append(int(idx))

            nnz_coords.append(indices)
            nnz_values.append(value)
        nnz_coords = np.array(nnz_coords)
        nnz_values = np.array(nnz_values)

        # get the shape
        shape = list()
        dims = len(nnz_coords[0])
        nnz_coords_arr = np.array(nnz_coords)
        for d in range(dims):
            shape.append(nnz_coords_arr[:, d].max() + 1)

        # Sparse COO format
        X = sparse.COO(nnz_coords.T, nnz_values, shape=tuple(shape))

        if tfidf_transformer:
            X1 = unfold(X, unfold_at)
            X_count = X1.T.tocsr()
            X_tfidf = TfidfTransformer().fit_transform(X_count)
            X_tfidf = sparse.COO(X_tfidf)
            X = fold(X_tfidf.T, unfold_at, (shape[0], shape[1], shape[2]))

        X = self.output_funcs[output_mode](X)
        if save_path:
            np.savetxt(os.path.join(save_path, f'{target_columns[0]}.txt'), list(something_idx_map.keys()),  fmt="%s", encoding="utf-8")
            np.savetxt(os.path.join(save_path, "Words.txt"), list(word_idx_map.keys()),  fmt="%s", encoding="utf-8")
            np.savetxt(os.path.join(save_path, "Time.txt"), list(time_idx_map.keys()),  fmt="%s", encoding="utf-8")
            self.save_funcs[output_mode](X, os.path.join(save_path, f'{target_columns[0]}_words_time.npz'))
        if return_object:
            return (X, list(something_idx_map.keys()), list(word_idx_map.keys()), list(time_idx_map.keys()))


    def _dist_parallel_tensor_build_helper(self,
                                           tensor_dicts_all: list,
                                           verbose: bool,
                                           n_nodes: int,
                                           comm,
                                           rank,
                                           n_chunks,
                                           shape
                                           ):
        """
        Helper to put together distributed and/or parallel tensors into single tensor

        Parameters
        ----------
        tensor_dicts_all : list
            List of tensor in dictionary COO format.
        verbose : bool
            Verbosity flag.
        n_nodes : int
            Number of nodes to use.
        comm : MPI.comm
            MPI communication object.
        rank : int
            Current node.
        n_chunks : int
            Number of chunks.
        shape : tuple
            Shape of the tensor.

        Returns
        -------
        X : sparse.COO
            Tensor in sparse COO format.

        """

        if verbose and rank <= 0:
            print('preparing dictionaries for communication')
        tensor_data_comm = []
        for curr_tensor_dict in tqdm(tensor_dicts_all, disable=not verbose, total=len(tensor_dicts_all)):
            for key, value in curr_tensor_dict.items():
                coords = key.split(";")
                coords = [float(i) for i in coords]

                tensor_data_comm.extend(coords)
                tensor_data_comm.append(float(value))

        tensor_data_comm = np.array(tensor_data_comm, dtype=float)

        # multi-node opreation gather
        if n_nodes > 1:

            # wait for everyone if multiple nodes
            comm.Barrier()

            # chunk the list first so that we can communicate the size
            n = 1000000
            tensor_data_chunks = [tensor_data_comm[i:i + n]
                                  for i in range(0, len(tensor_data_comm), n)]

            # wait for everyone if multiple nodes
            comm.Barrier()

            chunk_sizes = np.array(comm.allgather(len(tensor_data_chunks)))
            maximum_chunk_size = max(chunk_sizes)
            while len(tensor_data_chunks) < maximum_chunk_size:
                tensor_data_chunks.append(np.array([], dtype=float))

            if rank == 0:
                all_chunks = []

            for ii, chunk in enumerate(tensor_data_chunks):

                # gather the sizes first
                sendcounts = np.array(comm.gather(len(chunk), root=0))

                if rank == 0:
                    recvbuf = np.empty(sum(sendcounts), dtype=float)
                else:
                    recvbuf = None

                comm.Gatherv(sendbuf=chunk, recvbuf=(recvbuf, sendcounts), root=0)

                if rank == 0:
                    all_chunks.extend(recvbuf)

            # wait for everyone if multiple nodes
            comm.Barrier()

            if rank == 0:
                tensor_data_comm = all_chunks

            else:
                sys.exit(0)

        # combine the tensors
        if verbose and n_chunks > 1:
            print("Combining the tensors...")

        # first combine all elements into single tensor dictionary
        tensor_dict = defaultdict(lambda: 0)

        for x, y, z, value in tqdm(zip(*[iter(tensor_data_comm)]*4), disable=not verbose, total=len(tensor_data_comm)/4):
            tensor_dict[(int(x), int(y), int(z))] += value

        # numpy COO format to Sparse tensor format
        X = sparse.COO(dict(tensor_dict), shape=shape)
        return X


    def _cocitation_tensor_helper(self,
                                  all_doc_ids: list,
                                  times: list,
                                  documents_references_map: dict,
                                  time_idx_map: dict,
                                  document_authors_map: dict,
                                  authors_idx_map: dict) -> dict:
        """
        Helper function to self.cocitation_tensor, allows parallel tensor creation.

        Parameters
        ----------
        all_doc_ids : list
            list of all document ids to process.
        times : list
            list of years (time) corresponding to the all_doc_ids.
        documents_references_map : dict
            document id to references mapping.
        time_idx_map : dict
            index mapping for time.
        document_authors_map : dict
            document to authors mapping.
        authors_idx_map : dict
            author index mapping.

        Returns
        -------
        sparse.COO
            COO tensor from sparse library.

        """

        # create dictionary coo
        tensor_dict = defaultdict(lambda: 0)
        for idx, docID in enumerate(all_doc_ids):
            curr_references = documents_references_map[docID]
            time = times[idx]
            time_idx = time_idx_map[time]

            # if not citing anyone, skip
            if len(curr_references) == 0:
                continue

            curr_authors = document_authors_map[docID]

            # for each author in the  current document
            for author in curr_authors:

                # for each paper the current document is referencing
                for reference in curr_references:

                    # if reference is not in the corpus, skip
                    if reference not in document_authors_map:
                        continue

                    referenced_authors = document_authors_map[reference]
                    tensor_entry = 1 / len(referenced_authors)
                    author_idx = authors_idx_map[author]

                    for ref_author in referenced_authors:

                        collab_idx = authors_idx_map[ref_author]
                        coo_str = f'{author_idx};{collab_idx};{time_idx}'
                        tensor_dict[coo_str] += tensor_entry

        return dict(tensor_dict)


    def _coauthor_tensor_helper(self,
                                time_idx_map: dict,
                                authors_idx_map: dict,
                                all_authors: list,
                                times: list,
                                split_authors_with: str) -> dict:
        """
        Helper function to self.coauthor_tensor, allows parallel tensor creation.

        Parameters
        ----------
        time_idx_map : dict
            Index mapping for time.
        authors_idx_map : dict
            Index mapping for authors.
        all_authors : list
            List of all authors for each document seperated by split_authors_with.
        times : list
            List of corresponding times for documents, following all_authors.
        split_authors_with : str
            The delimiter for each element in all_author to be used in .split operation.

        Returns
        -------
        X : COO.sparse
            COO tensor from sparse library.

        """

        # build tensor COO dict
        tensor_dict = defaultdict(lambda: 0)

        # for each document
        for idx, curr_authors in enumerate(all_authors):
            curr_authors_list = curr_authors.split(split_authors_with)
            curr_time = times[idx]
            curr_time_idx = time_idx_map[curr_time]

            # for each author in the paper
            for curr_author in curr_authors_list:

                for collaborator in curr_authors_list:

                    # remove self
                    if curr_author != collaborator:

                        author_idx = authors_idx_map[curr_author]
                        collab_idx = authors_idx_map[collaborator]
                        coo_str = f'{author_idx};{collab_idx};{curr_time_idx}'
                        tensor_dict[coo_str] += 1

        return dict(tensor_dict)


    def _participation_tensor_helper(self,
                                     dimension_order: list,
                                     authors_idx_map: dict,
                                     papers_idx_map: dict,
                                     time_idx_map: dict,
                                     authors_list: list,
                                     papers_list: list,
                                     time_list: list,
                                     split_authors_with: str) -> dict:
        """
        Helper function to self.participation_tensor, allows parallel tensor creation.

        Parameters
        ----------
        dimension_order : list
            How to order authors, papers, time
        authors_idx_map : dict
            Index mapping for authors.
        papers_idx_map : dict
            Index mapping for papers.
        time_idx_map : dict
            Index mapping for time.
        authors_list : list
            List of all authors corresponding to each document seperated by split_authors_with.
        papers_list : list
            List of corresponding document unique identifiers 
        time_list : list
            List of corresponding times for documents
        split_authors_with : str
            The delimiter for each element in all_author to be used in .split operation.

        Returns
        -------
        tensor_dict : dict
            Dictionary with keys as coordinates and values representing entries in the tensor

        """
        assert len(authors_list) == len(papers_list) == len(time_list),  \
            "Authors, Papers, Time lists cannot be different lengths"

        # iteratively compute the coordinates fornon-zero values in the tensor
        tensor_dict = {}
        for curr_authors, curr_paper, curr_time in zip(authors_list, papers_list, time_list):
            paper_index = papers_idx_map[curr_paper]
            time_index = time_idx_map[curr_time]
            for curr_author in curr_authors.split(split_authors_with):
                author_index = authors_idx_map[curr_author]
                
                # create coordinate string using the specified dimension order
                indices = [author_index, paper_index, time_index]
                coo_str = ';'.join([str(indices[x]) for x in dimension_order])
                tensor_dict[coo_str] = 1

        return dict(tensor_dict)


    def _citation_tensor_helper(self,
                                dimension_order: list,
                                authors_idx_map: dict,
                                papers_idx_map: dict,
                                time_idx_map: dict,
                                papers_list: list,
                                time_list: list,
                                document_authors_map: dict,
                                document_references_map: dict) -> dict:
        """
        Helper function to self.citation_tensor, allows parallel tensor creation.

        Parameters
        ----------
        dimension_order : list
            How to order authors, papers, time
        authors_idx_map : dict
            Index mapping for authors.
        papers_idx_map : dict
            Index mapping for papers.
        time_idx_map : dict
            Index mapping for time.
        papers_list : list
            List of corresponding document unique identifiers 
        time_list : list
            List of corresponding times for documents
        document_authors_map : dict
            document to author list mapping.
        document_references_map : dict
            document id to reference list mapping.

        Returns
        -------
        tensor_dict : dict
            Dictionary with keys as coordinates and values representing entries in the tensor
        """

        # create dictionary coo
        tensor_dict = defaultdict(lambda: 0)
        for curr_paper, curr_time in zip(papers_list, time_list):
            curr_references = document_references_map.get(curr_paper)
            if curr_references is None:  # if not citing anyone, skip
                continue

            time_index = time_idx_map[curr_time]
            curr_authors = document_authors_map[curr_paper]
            for curr_auth in curr_authors:  # for each author in the  current document
                for curr_ref in curr_references:  # for each paper the current document is referencing
                    paper_index = papers_idx_map.get(curr_ref)
                    if paper_index is None:
                        continue
                    author_index = authors_idx_map[curr_auth]

                    # create coordinate string using the specified dimension order
                    indices = [author_index, paper_index, time_index]
                    coo_str = ';'.join([str(indices[x]) for x in dimension_order])
                    tensor_dict[coo_str] += 1  # update value in dict

        return dict(tensor_dict)

    def _chunk_list(self, l: list, n: int) -> list:
        """
        Yield n number of striped chunks from l.

        Parameters
        ----------
        l : list
            list to be chunked.
        n : int
            number of chunks.

        Yields
        ------
        list
            chunks.

        """
        for i in range(0, n):
            yield l[i::n]

    def _get_vocabulary_helper(self, data: list, split_with: str=None) -> dict:
        """
        Gets the token statistics from a given list of documents.

        Parameters
        ----------
        data : list
            List of documents.
        split_with : str, optional
            Delimeter to use in tokenization, if None split in whitespace. The default is None.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        token_stats = defaultdict(lambda: {"df": 0, "tf": 0})
        for doc in data:
            tokens = doc.split(split_with)
            unique_tokens = set(tokens)
            for tt in unique_tokens:
                # token_stats[tt]["tf"] += tokens.count(tt) not using this for now, may add back later
                token_stats[tt]["df"] += 1
        token_stats = dict(token_stats)
        return token_stats


    def _output_pydata(self, x):
        """
        Return tensor as a sparse.coo object. This is a reflective function

        Parameters
        ----------
        x : sparse.coo
            Tensor object.

        Returns
        -------
        x : sparse.coo
        """
        return x

    def _output_scipy(self, x):
        """
        Return matrix as a scipy.sparse.csr object

        Parameters
        ----------
        x : sparse.coo
            Matrix object.

        Returns
        -------
        x : scipy.sparse.csr
        """
        x = x.to_scipy_sparse()  # convert to scipy coo
        x = ss.csr_matrix(x).astype("float32")  # convert to scipy csr 
        return x
    
    def _save_pydata(self, x, path: str):
        """
        Save a pydata sparse tensor x to path
        
        Parameters
        ----------
        x : sparse.coo
            Tensor object.
        path : str
            path to save x
            
        Returns
        -------
        None
        """
        sparse.save_npz(path, x)

    def _save_scipy(self, x, path: str):
        """
        Save a scipy sparse matrix x to path
            
        Parameters
        ----------
        x : scipy.sparse.csr object
            Matrix object.
        path : str
            path to save x

        Returns
        -------
        None
        """
        ss.save_npz(path, x)

        
    # Getters and Setters
    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_nodes.setter
    def n_nodes(self, n_nodes: int):
        if not isinstance(n_nodes, int) or n_nodes <= 0:
            raise ValueError(f"Unsupported value for n_nodes: '{n_nodes}'")
        self._n_nodes = n_nodes

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        cpu_count = multiprocessing.cpu_count()
        if not isinstance(n_jobs, int):
            raise ValueError(f'n_jobs must be an int')

        limit = cpu_count + n_jobs
        if (n_jobs == 0) or (limit < 0) or (2 * cpu_count < limit):
            raise ValueError(f'n_jobs must take a value on [-{cpu_count}, -1] or [1, {cpu_count}]')
        
        if n_jobs < 0:
            self._n_jobs = cpu_count - abs(n_jobs) + 1
        else:
            self._n_jobs = n_jobs



    def __get_ngrams_helper(self, text, n):
        ngrams = []
        tokens = text.split()
        num_tokens = len(tokens)
        for index in range(num_tokens):
            ngram = []
            for i in range(n):
                
                # build ngram
                if index + i < num_tokens:
                    ngram.append(tokens[index+i])
                
                # test ngram
                len_ngram = len([x for y in ngram for x in y.split('-')])
                if len_ngram == n:
                    ngrams.append(' '.join(ngram))
                    break
                elif len_ngram < n:
                    continue
                else:
                    break
                    
        return ngrams


    def get_ngrams( self,  dataset: pd.DataFrame, target_column: str=None, n: int=1, 
                    limit: int=None,  save_path: str=None) -> list:
        """
        Generates n-grams from a column in a dataset

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe containing the target columns.
        target_column : str, optional
            Target column name in dataset DataFrame. The default is "abstracts".
            Target column should be for text data, where tokens are retrived via empty spaces.
        n : int
            Number of tokens in a gram to generate
        limit : int 
            Restrict number of top n-grams to return
        save_path : str, optional
            If not None, saves the outputs as csv using the column names 'Ngram', 'Count'. The default save_path is None.

        Returns
        -------
        list
            Top ngrams as a list of tuples containing the ngram then count.

        """

        assert target_column in dataset, "Target column is not found!"

        # get the target data
        corpus = dataset[target_column].values.tolist()

        all_ngrams = []
        for text in corpus:
            ngrams = self.__get_ngrams_helper(text, n)
            all_ngrams.extend(ngrams)

        counter = Counter(all_ngrams)
        top_ngrams = counter.most_common(limit)
 
        if save_path:
            counter_df = pd.DataFrame(list(counter.items()), columns=['Ngram', 'Count'])
            counter_df.to_csv(os.path.join(save_path, "top_ngrams.txt"), index=False)

        return top_ngrams
