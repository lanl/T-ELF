#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 17:05:54 2022

@author: maksimekineren
"""

import os
import sys
import ast
import time
import pickle
import warnings
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Union


def add_with_union_of_others(d, s, key):
    """
    Compute the addition of a set associated with a given key and the union of all other sets in a dictionary.

    Parameters:
    -----------
    d: dict
        A dictionary where keys are strings and values are sets.
    s: set
        The set (associated with key) to be added to
    key: str
        The key in the dictionary whose set is to be modified with the union of all other sets.

    Returns:
    --------
    set: 
        The intersection set of the specified key's set and the union of all other sets in the dictionary.

    Raises:
    -------
    ValueError
        If the specified key is not found in the dictionary.
    """
    if key not in d:
        raise ValueError(f'Key {key!r} not found in the dictionary!')

    # dont do anything if dict only contains target key
    if len(d) == 1 and key in d:
        return s
    else:
        
        # create union of all sets (excluding the set for key)
        union_set = set.union(*(other_set for other_key, other_set in d.items() if other_key != key))
        
        # create the modified set
        s_updated = d[key].copy()
        s_updated.update(s & union_set)
        return s_updated


class Cheetah:


    # keys are fields accepted by Cheetah, values are default column names in DataFrame
    COLUMNS = {
               'title': 'title', 
               'abstract': 'abstract',
               'year': 'year',
               'author_ids': 'author_ids',
               'affiliations': 'affiliations',
              }


    def __init__(self, verbose: bool) -> None:
        """ 
        Init an empty Cheetah object

        Parameters
        ----------
        verbose : bool, optional
            Vebosity flag. The default is False.

        Returns
        -------
        None
        """
        self.data = None
        self.indexed = False
        self.verbose = verbose


    @classmethod
    def find_ngram(cls, text:str, query:list, window_size:int=5, ordered:bool=True) -> bool:
        """
        Determine if the tokens in the list query are contained within the string text using 
        a sliding window algorithm with a specified window_size. If ordered is True then the 
        order of tokens appearing in query and text needs to be maintained for a positive 
        match. Returns True is such a match is found.

        Parameters
        ----------
        text : str
            A string of multiple tokens that are separated by whitespace
        query : list
            A list of tokens that should be checked for in text. Duplicate values in query are allowed
            and order will be maintained if ordered=True.
        window_size : int, optional
            Set the size of the sliding window.
            NOTE: if window_size < len(query), no matches can ever be found as the query cannot fit 
                  in the window. Default=5.
        ordered : bool, optional
            If True, preserve the order of tokens in query when searching for match. Default=True.

        Returns
        -------
        bool
            True if ngram query was found in text. False otherwise.
        """
        window = []
        tokens = text.split()
        token_set = dict(Counter(query))  # count occurences of tokens in query
        for token in tokens:
            window.append(token)
            if len(window) > window_size:
                window.pop(0)

            # form a list of tokens for comparison using only tokens seen in the query
            # an additional precaution is taken to only use each token the exact number
            # of times that it is seen in the query at most
            curr_window = []
            window_set = defaultdict(int)
            for t in window:
                if t in token_set and window_set[t] < token_set[t]:
                    window_set[t] += 1
                    curr_window.append(t)
            if len(curr_window) == len(query):
                if ordered and curr_window == query: # compare preserving order
                    return True
                elif not ordered and sorted(curr_window) == sorted(query):
                    return True
        return False


    def index(self, data:pd.DataFrame, columns:dict=None, index_file:str=None, reindex:bool=False, verbose:bool=True) -> None:
        """
        Creates indices for selected columns in data for Cheetah search. author_ids and affiliations are
        expected to use the their respective SLIC data structures. See an example notebook for a sample
        of these data structures. Text data such as 'title' and 'abstract' should be pre-processed using
        Vulture simple clean. The text in these columns is expected to be lowercase with special characters
        removed. Tokens are delimited with a single whitespace.

        Parameters
        ----------
        data: pd.DataFrame
            Pandas DataFrame of papers 
        columns: dict, optional
            Dictionary where the keys are categories that can be mapped by Cheetah and the values are the 
            corresponding columns names for these categories in data. See Cheetah.COLUMNS for an example
            of the structure and all currently supported keys. If columns is None, Cheetah will default
            to the Cheetah.COLUMNS values.
        index_file: str, optional
            Path a to a previously generated Cheetah index file. If no path is passed, Cheetah will generate
            indices for one time use. If index_file is passed but the path does not exist, Cheetah will generate
            indices and save them for future use at the index_file path. If a path is passed and reindex=True,
            new indices will be generated and saved at index_file, overwriting the current contents of index_file
            if it exists.
        reindex: int or float, optional
            If True, overwrite the index_file if it exists
        verbose: bool, optional
            Vebosity flag. The default is False.

        Returns
        -------
        None

        """
        self.data = data.copy().reset_index(drop=True)
        self.data['slic_index'] = data.index
        self.columns = columns
        self.verbose = verbose

        if index_file:
            if not os.path.exists(index_file) or reindex:  # index data
                if self.verbose:
                    if reindex: print("Overwriting existing index.")
                    else: print("Indexing file not found. Creating a new index.")
                indexing_results = self._index_data(self.data)
            else:  # load indexed data from disk
                indexing_results = pickle.load(open(index_file, 'rb'))
        else:
            indexing_results = self._index_data(self.data)
            
        self.abstract_index = indexing_results[0]
        self.title_index = indexing_results[1]
        self.year_index = indexing_results[2]
        self.country_index = indexing_results[3]
        self.author_index = indexing_results[4]
        self.affiliation_index = indexing_results[5]

        if index_file:
            pickle.dump(indexing_results, open(index_file, "wb"))

        self.indexed = True
        self.last_search_result = None


    def search(self, query:list=None, and_search:bool=True, in_title:bool=True, in_abstract:bool=True, 
                     save_path:bool=None, author_filter:list=[], affiliation_filter:list=[], 
                     country_filter:list=[], year_filter:list=[], ngram_window_size:int=5, 
                     ngram_ordered:bool=True, do_results_table=False, link_search=False) -> pd.DataFrame:
        """
        Search a dataset indexed by this Cheetah object. Text can be searched using query and properties
        of the data can be filtered using year_filter, country_filter, author_filter, affiliation_filter.
        If both query and filter(s) are used, the results of the search are intersected. Note that trying
        to use a filter that was never indexed by Cheetah will result in an error. 

        Parameters
        ----------
        query: str, list, dict, NoneType
            A string or a list of strings to lookup. n-grams for n>1 should be split with whitespace.
            Note that query will be pre-processed by converting all characters to lowecase and stripping all
            extra whitespace.
             >>> query = 'laser'                        # a single word to lookup
             >>> query = {'laser': 'node'}              # a single word with negative query
             >>> query = ['laser', 'angle deflection']  # a word and bigram to lookup
             >>> query = [{'laser': ['blue', 'green'],  # a word and bigram to lookup with multiple negative
                           'angle deflection']          #  search terms for the unigram
             >>> query = None                           # no query to lookup (using filters only)
        and_search: bool, optional
            This option applies when multiple queries are being looked up simultenously. If True, the 
            intersection of documents that match all queries is returned. Otherwise, the union. Default=True.
        in_title: bool, optional
            If True, searches for queries in the indexed title text. Default=True.
            NOTE: If in_title and in_abstract are both True, the union between these
                  two query searches is returned
        in_abstract: bool, optional
            If True, searches for queries in the indexed abstract text. Default=True.
            NOTE: If in_title and in_abstract are both True, the union between these
                  two query searches is returned
        save_path: str, optional
            The path at which to save the resulting subset DataFrame. If the path is not defined, the result
            of the search is returned. The default is None.  
        author_filter: list, optional
            List of author ids that papers should be affiliated with. The default is [].
        affiliation_filter: list, optional
            List of affiliation ids that papers should be affiliated with. The default is [].
        country_filter: list, optional
            List of countries that papers should be affiliated with. The default is [].
        year_filter: list, optional
            List of years that papers should be published in. The default is [].
        ngram_window_size: int, optional
            The size of the window used in Cheetah.find_ngram(). This function is called if one or more
            entries in query are n-grams for n>1. ngram_window_size determines how many tokens can be 
            examined at a time.  For example for the text ['aa bb bb cc cc dd'], the query 'aa cc' will 
            be found if the window size is >= 4. Default=5. This value should be greater than the length 
            of the n-gram.
        ngram_ordered: bool
            The order used in Cheetah.find_ngram(). This function is called if one or more entries in 
            query are n-grams for n>1. ngram_ordered determines if the order of tokens in ngram should
            be preserved while looking for a match. Default=True.
        do_results_table: bool, optional
            Flag that determines if a results table should be generated for the search. If True, this table
            will provide explainability for why certain documents were selected by Cheetah. If False, None 
            is returned as the second argument. Default=False
        link_search: bool, optional
            A flag that controls if the queries should be linked in the positive/negative inclusion
            step. For example, take a document that contains the queried text "A" and "B". However 
            positive or negative inclusion partnered with "B" overrides the selection. If this flag
            is set to True then the inclusion step will be ignored since another query, "A", had 
            already selected the document as being on-topic (hence linking the search). Default=False

        Returns
        -------
        return_data: None, pd.DataFrame
            If save_path is not defined, return the search result (pd.DataFrame object). However, if save_path 
            is defined, return None and save result at save_path as a CSV file.
        results_table: None, pd.DataFrame
            If do_results_table is True then this argument will provide explainability for Cheetah filtering. 
            Otherwise this argument is None
        """

        # validate that data has been loaded and indexed
        if not self.indexed:
            raise ValueError('No index found! Call Cheetah.index() first!')

        # validate input to the function
        self.query = query
        self.ngram_ordered = ngram_ordered
        self.ngram_window_size = ngram_window_size

        # begin timing the search
        start = time.time()

        # unite all possible filters into a data structure that will allow easy filtering
        filters = [('year', year_filter, self.year_index),
                   ('affiliations', country_filter, self.country_index),
                   ('author_ids', author_filter, self.author_index),
                   ('affiliations', affiliation_filter, self.affiliation_index)]
        
        ## 1. Filter the data
        # use the filters to generate a set of desirable document ids
        # if the filters exclude all documents, an empty set will be used
        filter_indices_map = {}
        using_filter_flag = False
        for col_filter, filter_values, filter_index in filters:
            if filter_values:  # filter values are None or empty list
                assert col_filter in self.columns, f"Attempted {col_filter} search but {col_filter} column does not exist!"
                filter_indices_map[(col_filter, ",".join([str(x) for x in filter_values]))] = self._filter_search(col_filter, filter_values, filter_index)
                using_filter_flag = True
        
        # intersect the results from each filter
        if filter_indices_map:
            filter_indices = set.intersection(*list(filter_indices_map.values()))
        else:
            filter_indices = set()

        if using_filter_flag and not filter_indices:
            warnings.warn('Selected filters returned an empty set', RuntimeWarning)


        ## 2. Search by query
        # if a query (or several) queries are given, produce the search results for
        # these querie(s) in all documents: with no respect for the filters
        query_indices_map = {}
        if self.query:
            # use reverse index map to quickly lookup 1grams and limit search space for ngrams
            query_indices_map = self._query_search(self.query, in_abstract, in_title, link_search)

            # deal with n_grams
            query_indices_map = self._ngram_check(query_indices_map, self.query, in_abstract, in_title)

            # unite the search results inclusively or exclusively depening on and_search value
            if and_search:  # intersect list of sets to form single set
                query_indices = set.intersection(*list(query_indices_map.values()))
            else:  # union list of sets to form single set
                query_indices = set.union(*list(query_indices_map.values()))

            # intersect results of query search with filter results (if using filtering)
            return_indices = query_indices
            if using_filter_flag:
                return_indices &= filter_indices
        
        else:
            return_indices = filter_indices

        # use the determined indices to select subset of papers from whole data
        return_data = self.data.loc[list(return_indices)].copy().reset_index(drop=True)
        return_data = return_data.set_index('slic_index')
        return_data.index.name = None
        self.last_search_result = return_data.copy()  # update object's last search

        # end timing of search
        end = time.time()

        # print computation time if verbose
        if self.verbose:
            print(f"Found {len(return_data)} papers in {round(end - start, 4)} seconds", file=sys.stderr)

        results_table = None
        if do_results_table:
            results_table = self._create_results_table(filter_indices_map, query_indices_map)
            
        # return results of search if no save_path passed, otherwise save
        if not save_path:
            return return_data, results_table
        else:
            return_data.to_csv(save_path, index=False)
            return None, results_table

        
    def _create_results_table(self, filter_results, query_results):
        """
        Creates a DataFrame that explains why and how documents were selected by Cheetah filters

        Parameters
        ----------
        filter_results: dict
            Map of filters and their associated paper indices
        query_results: list
            Map of queries and their associated paper indices

        Returns
        -------
        pd.DataFrame
            DataFrame that has filters and the papers that they included/excluded
        """
        data = {
            'filter_type': [],
            'filter_value': [],
            'num_papers': [],
            'included_ids': [],
        }
        
        for key, indices in filter_results.items():
            filter_type, filter_value = key
            num_included = len(indices)
            if indices:
                indices = [int(x) for x in indices]
                included_ids = ';'.join([str(x) for x in self.data.loc[indices]['slic_index'].to_list()])
            else:
                included_ids = None
            
            #included_ids = ';'.join([str(x) for x in indices]) if indices else None
            
            # add to DataFrame
            data['filter_type'].append(filter_type)
            data['filter_value'].append(filter_value)
            data['num_papers'].append(num_included)
            data['included_ids'].append(included_ids)
            
        for query, indices in query_results.items():
            num_included = len(indices)
            if indices:
                indices = [int(x) for x in indices]
                included_ids = ';'.join([str(x) for x in self.data.loc[indices]['slic_index'].to_list()])
            else:
                included_ids = None
            
            # add to DataFrame
            data['filter_type'].append('query')
            data['filter_value'].append(query)
            data['num_papers'].append(num_included)
            data['included_ids'].append(included_ids)         
        
        results_table_df = pd.DataFrame.from_dict(data)
        results_table_df = results_table_df.sort_values(by=['filter_type', 'filter_value']).reset_index(drop=True)
        return results_table_df
            
            
    def _filter_search(self, col:str, filters:list, indexing:dict) -> set:
        """
        Searches the index for the given filters in the specified column.

        Parameters
        ----------
        col: str
            The name of the column to be searched.
        filters: list
            A list representing the filters to be applied 
        indexing: dict
            A dictionary representing the index of the text data. 

        Returns
        -------
        set
            A set of record IDs that match the specified filters 
        """
        # pre-process input
        filters = [str(f).lower().strip() for f in filters]

        # get the indices matching filters
        filter_indices = []
        for f in filters:
            filter_indices.append(indexing.get(f, set()))
        
        # take the union of the filter results
        # if multiple values in a filter are given, we combine the results. 
        # EX filters=[2018, 2019, 2020] -> get documents published in this 3 year range
        return set.union(*filter_indices)


    def _query_search(self, queries:list, in_abstract:bool, in_title:bool, link_search:bool) -> dict:
        """
        Searches the text data for the given queries in the specified columns.

        Returns a dictionary of query string -> matching record indices set.
        Ensures that the length of the returned dict matches len(queries).
        """
        if in_abstract:
            assert "abstract" in self.columns, "Attempted abstract search but abstract column does not exist!"
        if in_title:
            assert "title" in self.columns, "Attempted title search but title column does not exist!"

        if not in_abstract and not in_title:
            warnings.warn('Attempting to search a query without any data source.', RuntimeWarning)

        index_map = {}

        for query in queries:
            query_indices = []
            query_proc = []

            for q in query:
                try:
                    term, negatives, positives = q
                    query_proc.append(term)
                    q_indices = set()

                    if in_title:
                        q_indices |= self.title_index.get(term, set())
                        q_indices = self._inclusion_search('title', q_indices, positives, negatives)

                    if in_abstract:
                        q_indices |= self.abstract_index.get(term, set())
                        q_indices = self._inclusion_search('abstract', q_indices, positives, negatives)

                    query_indices.append(q_indices)
                except Exception as e:
                    query_proc.append("<INVALID>")
                    warnings.warn(f"Error processing query part {q}: {e}")
                    query_indices.append(set())

            query_key = " ".join(query_proc)

            try:
                if query_indices:
                    combined = set.intersection(*query_indices) if len(query_indices) > 1 else query_indices[0]
                    index_map[query_key] = combined
                else:
                    index_map[query_key] = set()
            except Exception as e:
                warnings.warn(f"Failed intersection for {query_key}: {e}")
                index_map[query_key] = set()

        # Link search mode
        if link_search:
            old_index_map = index_map.copy()
            index_map = {}
            for i, query in enumerate(queries):
                query_proc = " ".join([x[0] if len(x) > 0 else "<INVALID>" for x in query])
                query_indices = [self.title_index.get(x[0], set()) | self.abstract_index.get(x[0], set()) for x in query]
                try:
                    combined = set.intersection(*query_indices) if len(query_indices) > 1 else query_indices[0]
                except:
                    combined = set()
                index_map[query_proc] = add_with_union_of_others(old_index_map, combined, query_proc)

        return index_map


    
    def _inclusion_search(self, text_index, q_indices, positives, negatives):
        assert text_index in ('title', 'abstract'), f'Invalid text_index {text_index!r}'
        if text_index == 'title':
            index = self.title_index
        if text_index == 'abstract':
            index = self.abstract_index
        
        # remove any terms that are being negated
        for n in negatives:
            q_indices -= index.get(n, set())

        # union the current set of terms with positively included terms
        if positives:
            pos_q_indices = set.union(*[index.get(x, set()) for x in positives])
            q_indices &= pos_q_indices
        
        return q_indices
    
    
    def _ngram_check(self, all_indices, all_queries, in_abstract, in_title) -> list:
        """
        Check for the occurrence of n-grams in a text or document based on some criteria.

        Parameters
        ----------
        all_indices: dict
            A map of query keys and index set values
        all_queries: list
            List of queries by which documents were filtered
        in_abstract: bool
            If True, check for the occurrence of n-grams in the abstract of the document.
        in_title: bool
            If True, check for the occurrence of n-grams in the title of the document.

        Returns
        -------
        list
            True if the specified n-grams are found in the text or document based on the specified criteria, False otherwise.
        """
        # validate input
        if in_abstract:
            assert "abstract" in self.columns, "Attempted abstract search but abstract column does not exist!"
        if in_title:
            assert "title" in self.columns, "Attempted title search but title column does not exist!"
        assert len(all_indices) == len(all_queries), f"Attempted to use {len(all_indices)} indices for {len(all_queries)}"

        for i,q in enumerate(all_queries):
            q = [next(iter(x)) for x in q]
            joined_query = ' '.join(q)
            index, query = all_indices.get(joined_query, set()), q
            index_data = self.data.loc[list(index)]
            if len(query) > 1 and not index_data.empty:
                query_indices = list()
                if in_abstract:  # Join results for title and abstract
                    query_indices.append(self._ngram_check_helper(index_data, query, 'abstract'))
                if in_title:
                    query_indices.append(self._ngram_check_helper(index_data, query, 'title'))

                query_indices = set.intersection(*query_indices)
                all_indices[joined_query] -= query_indices

        return all_indices


    def _ngram_check_helper(self, index_data:pd.DataFrame, query:str, col:str) -> set:
        """
        Perform an n-gram check on the specified DataFrame, query string, and column name.

        Parameters
        ----------
        index_data : pd.DataFrame
            The DataFrame containing the indexed data.
        query : str
            The query string to check for n-grams.
        col : str
            The name of the column to check for n-grams.

        Returns
        -------
        set
            A set of indices of the rows in the DataFrame that contain the specified n-grams.
        """
        # validate input
        assert col in self.columns, f"Attempted abstract search but {col} column does not exist!"
        
        # warn user if window size is too small
        if len(query) > self.ngram_window_size:
            warnings.warn(f"Attempting to find a {len(query)}-token query with a window of size {self.ngram_window_size}", RuntimeWarning)

        # check if ngram is found in each id
        to_remove = set()
        for idx, text in zip(index_data.index.to_list(), index_data[self.columns[col]].to_list()):
            if not self.find_ngram(text, query, self.ngram_window_size, self.ngram_ordered):
                to_remove.add(idx)
        return to_remove


    def _index_text(self, data:dict, column:str) -> dict:
        """
        Indexes the text data in the specified column of a dictionary of records.

        Parameters
        ----------
        data : dict
            A dictionary containing records of data to be indexed. 
        column : str
            The name of the column containing text data to be indexed.

        Returns
        -------
        dict
            A dictionary representing the mapping.

        Raises
        ------
        ValueError
            If the `col` parameter is None.
        """
        if self.verbose:
            print(f"Indexing {column}")
        
        col = self.columns.get(column)
        if col is None:
            raise ValueError(f"Invalid column name for '{column}' provided")

        index_map = {}
        text_list = data[col].to_list()
        for paper_idx, text in tqdm(enumerate(text_list), total=len(text_list), disable=not self.verbose):
            if pd.isna(text):
                continue
            tokens = text.split()
            for word in tokens:
                word = word.strip()  # remove newlines, empty space
                word = word.lower()  # convert to lowercase
                if not word:  # word is an empty string
                    continue

                if word not in index_map:
                    index_map[word] = set()
                index_map[word].add(paper_idx)

        for token, indices in tqdm(index_map.items(), total=len(index_map), disable=not self.verbose):
            index_map[token] = set(indices)
        return index_map


    def _index_year(self, data:dict) -> dict:
        """
        Indexes the year data of a dictionary of records.

        Parameters
        ----------
        data : dict
            A dictionary containing records of data to be indexed. 
       
        Returns
        -------
        dict
            A dictionary representing the year index.

        Raises
        ------
        ValueError
            If the `col` parameter is None.
        """
        if self.verbose:
            print("Indexing years")
        
        col = self.columns.get('year')
        if col is None:
            raise ValueError(f"Invalid column name for 'year' provided")

        year_index = {}
        year_list = data[col].to_list()
        for paper_idx, year in tqdm(enumerate(year_list), total=len(year_list), disable=not self.verbose):
            if pd.isna(year):
                continue
            year = str(int(year)).strip().lower()
            if year not in year_index:
                year_index[year] = set()
            year_index[year].add(paper_idx)

        # for year, indices in tqdm(year_index.items(), total=len(year_index), disable=not self.verbose):
        #     year_index[year] = set(indices)
        return year_index

    
    def _index_author(self, data:dict) -> dict:
        """
        Indexes the author data of a dictionary of records.

        Parameters
        ----------
        data : dict
            A dictionary containing records of data to be indexed. 
       
        Returns
        -------
        dict
            A dictionary representing the author index.

        Raises
        ------
        ValueError
            If the `col` object parameter is None.
        """
        if self.verbose:
            print("Indexing author IDs")
        
        col = self.columns.get('author_ids')
        if col is None:
            raise ValueError(f"Invalid column name for 'author_ids' provided")

        author_index = {}
        author_IDs = data[col].to_list()
        author_index_tmp = {}
        for paper_idx in tqdm(range(len(author_IDs)), disable= not self.verbose):
            curr_info = author_IDs[paper_idx]
            if pd.isna(curr_info):
                continue
            for author_id in curr_info.split(";"):
                author_id = str(author_id).lower().strip()
                if author_id in author_index_tmp:
                    author_index_tmp[author_id].append(paper_idx)
                else:
                    author_index_tmp[author_id] = [paper_idx]

        for token, indices in tqdm(
            author_index_tmp.items(), total=len(author_index_tmp), disable= not self.verbose
        ):
            author_index[token] = set(indices)
            
        return author_index
    
    
    def _index_affiliation_country(self, data:dict) -> tuple:
        """
        Indexes the country affiliation data of a dictionary of records.

        Parameters
        ----------
        data : dict
            A dictionary containing records of data to be indexed. 
       
        Returns
        -------
        dict
            A dictionary representing the country affiliation index.

        Raises
        ------
        ValueError
            If the `col` object parameter is None.
        """
        if self.verbose:
            print("Indexing affiliations and countries")
        
        col = self.columns.get('affiliations')
        if col is None:
            raise ValueError(f"Invalid column name for 'affiliations' provided")

        country_index = {}
        affiliation_index = {}
        
        affiliation_information_str = data[col].to_list()
        affiliation_index_tmp = {}
        country_index_tmp = {}

        for paper_idx, info in tqdm(enumerate(affiliation_information_str), total=len(affiliation_information_str), disable= not self.verbose):
            if pd.isna(info):
                continue
            if isinstance(info, str):
                curr_info_dict = ast.literal_eval(info)
            else:
                curr_info_dict = info
            
            for affil_id, affil_info_dict in curr_info_dict.items():
                affil_id = str(affil_id).strip().lower()
                
                # Check type and country of affiliation
                if not isinstance(affil_info_dict, dict):
                    continue
                country = affil_info_dict.get("country")
                if country:
                    country = country.strip().lower()
                else:
                    country = ''
                
                # affiliation
                if str(affil_id) in affiliation_index_tmp:
                    affiliation_index_tmp[str(affil_id)].append(paper_idx)
                else:
                    affiliation_index_tmp[str(affil_id)] = [paper_idx]
                
                # country
                if str(country) in country_index_tmp:
                    country_index_tmp[str(country)].append(paper_idx)
                else:
                    country_index_tmp[str(country)] = [paper_idx]
        
        for token, indices in tqdm(
            country_index_tmp.items(), total=len(country_index_tmp), disable= not self.verbose
        ):
            country_index[token] = set(indices)
        
                
        for token, indices in tqdm(
            affiliation_index_tmp.items(), total=len(affiliation_index_tmp), disable= not self.verbose
        ):
            affiliation_index[token] = set(indices)
        
        
        return affiliation_index, country_index
        

    def _index_data(self, data:dict):
        """
 
        Parameters
        ----------
        

        Returns
        -------
        
        """
        if "abstract" in self.columns:
            abstract_index = self._index_text(data, 'abstract')
        else:
            abstract_index = {}
            
        if "title" in self.columns:
            title_index = self._index_text(data, 'title')
        else:
            title_index = {}
            
        if "year" in self.columns:
            year_index = self._index_year(data)
        else:
            year_index = {}
            
        if "author_ids" in self.columns:
            author_index = self._index_author(data)
        else:
            author_index = {}
        
        if "affiliations" in self.columns:
            affiliation_index, country_index = self._index_affiliation_country(data)
        else:
            affiliation_index, country_index = {}, {}
        
        return (
            abstract_index,
            title_index,
            year_index,
            country_index,
            author_index,
            affiliation_index,
        )


    # GETTERS 

    @property 
    def ngram_window_size(self) -> int:
        """
        Get the numeric size of the ngram window. 
        
        Parameters
        ----------
        None

        Returns
        -------
        int 
            ngram window size
        """
        return self._ngram_window_size

    @property
    def ngram_ordered(self) -> bool:
        """
        Get the status of ngram_ordered.

        Parameters
        ----------
        None

        Returns
        -------
        bool    
            Status of ngram ordering
        """
        return self._ngram_ordered

    @property
    def query(self) -> Union[list, str, None]:
        """
        Get the last query of the object.

        Parameters
        ----------
        None

        Returns
        -------
        Union[list, str, None]
            The last query of the object, which can be either a list or a string.
        """
        return self._query

    @property
    def columns(self) -> dict:
        """
        Retrieve the columns.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A dictionary containing column names as keys and column values as values.
        """
        return self._columns

    # SETTERS

    @ngram_window_size.setter
    def ngram_window_size(self, ngram_window_size:int) -> None:
        """
        Set or update the size of the n-gram window of the object.

        Parameters
        ----------
        ngram_window_size : int
            The new value for the size of the n-gram window of the object.

        Returns
        -------
        None
        """
        if not isinstance(ngram_window_size, int):
            self.__error_unexpected_type(int, ngram_window_size)

        if ngram_window_size < 1:
            raise ValueError("ngram_window_size cannot be less than 1")
        self._ngram_window_size = ngram_window_size

    @ngram_ordered.setter
    def ngram_ordered(self, ngram_ordered:bool) -> None:
        """
        Set or update a flag that controls whether n-grams should be ordered or not.

        Parameters
        ----------
        ngram_ordered : bool
            If True, n-grams will be ordered; if False, n-grams will be unordered.

        Returns
        -------
        None
        """
        if not isinstance(ngram_ordered, bool):
            self.__error_unexpected_type(bool, ngram_ordered)
        self._ngram_ordered = ngram_ordered
        
        
    @query.setter
    def query(self, query: Union[list, str, dict, None]) -> None:
        """
        Set or update the query of the object.

        Parameters
        ----------
        query : list, dict, str, None
            The new value for the query of the object.

        Returns
        -------
        None
        """
        def process_entry(entry):
            def validate_dependent_terms(terms):
                for term in terms:
                    if len(term.split()) > 1:
                        raise ValueError(f"Expected single token in dependent term, but got {term!r}")

            if isinstance(entry, str):
                return [(word, [], []) for word in entry.split()]
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError('Expected query dict to only contain 1 element')
                term = next(iter(entry))
                dependents = entry[term]
                if isinstance(dependents, str):
                    validate_dependent_terms([dependents])
                    dependents = [dependents]
                elif isinstance(dependents, list):
                    validate_dependent_terms(dependents)
                else:
                    self.__error_unexpected_type({str, list}, dependents)
                
                # process the dependent term(s) to determine which are being 
                # negated and which are being included
                negatives = []
                positives = []
                for dep in dependents:
                    if dep.startswith('+'):  # positive terms are expected to start with '+'
                        positives.append(dep[1:])  # take off the plus sign
                    else:
                        negatives.append(dep)  # negative terms are just the term with no symbols
                
                return [(word, negatives, positives) for word in term.split()]
            else:
                self.__error_unexpected_type({str, list, dict}, entry)
        
        # query is None, an empty string, empty dict, or an empty list
        if not query:
            self._query = None
        else:
            processed = []
            if isinstance(query, str) or isinstance(query, dict):
                processed.append(process_entry(query))
            elif isinstance(query, list):
                for entry in query:
                    processed.append(process_entry(entry))
            else:
                self.__error_unexpected_type({str, list, dict, type(None)}, query)
            self._query = processed


    @columns.setter
    def columns(self, columns:dict) -> None:
        """
        Set or update the columns of a data structure or object.
        
        Parameters
        ----------
        columns : dict
            A dictionary containing column names as keys and column values as values.

        Returns
        -------
        None
        """
        if columns is None:
            columns = Cheetah.COLUMNS
        else:
            if not isinstance(columns, dict):
                self.__error_unexpected_type(dict, columns)

        del_list = []
        data_columns = set(self.data.columns.to_list())
        for col_name, col_value in columns.items():
            if col_value not in data_columns:
                del_list.append(col_name)

        for d in del_list:
            if d in columns:
                warnings.warn(f"'{columns[d]}' not found in DataFrame. Removing the {d} index", RuntimeWarning)
                del columns[d]

        if not columns:
            raise ValueError('No valid columns remain to be indexed')
        else:
            self._columns = columns

    # UTIL

    @classmethod
    def __error_unexpected_type(cls, expected_type, var) -> ValueError:
        """
        Raises an ValueError exception

        Parameters
        ----------
        cls:

        expected_type:
        
        var:

        Returns
        -------
        ValueError
        """
        raise TypeError(f"Expected {expected_type} but instead got {type(var)}: {var}")
