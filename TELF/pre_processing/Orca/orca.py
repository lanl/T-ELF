import os
import ast
import copy
import pickle
import warnings 
import pandas as pd
import networkx as nx
from tqdm import tqdm 

from .AuthorMatcher import AuthorMatcher

class Orca:

    #Code where we have a precomputed duplicates.
    # Pre-computed Scopus duplicates file containing entries.
    # If no duplicates are computed with DAF, this file will be used instead for duplicate removal
    #DUPLICATES_1M = 'scopus_1m_cited_collab_matches.p'

    def __init__(self, duplicates=None, s2_duplicates=None, verbose=False):
        self.slic_df = None
        self.duplicates = duplicates
        self.s2_duplicates = s2_duplicates
        self.verbose = verbose
    
    
    def _run_scopus(self, df):
        """
        Helper function for creating a SLIC map file for a dataset that only contains Scopus information
        """ 
        # generate a map of scopus ids to affiliations
        affiliations_map = self.__generate_affiliations_map(df)
        
        # generate author maps
        scopus_author_map = self.__generate_author_map(df, 'author_ids', 'authors')
            
        # correct duplicates
        duplicates = {}
        for entry in self.duplicates:    
            if not entry & scopus_author_map.keys():
                continue
            
            entry = entry.copy()  # avoid modifying duplicates in place
            best_id = sorted(entry, key=lambda x: len(scopus_author_map.get(x, '')), reverse=True)[0]
            merged_affiliations = self.__merge_scopus_affiliations(entry, affiliations_map)
            affiliations_map[best_id] = merged_affiliations

            # remove old duplicate ids from both maps
            entry.remove(best_id)
            for x in entry:
                del scopus_author_map[x]
                del affiliations_map[x]
                
            # add duplicates for tracking purposes
            duplicates[best_id] = list(entry)
            
        
        ## generate SLIC IDs
        slic_count = 0
        slic_df = {
            'slic_id': [],
            'slic_name': [],
            'scopus_ids': [],
            'scopus_names': [],
            'scopus_affiliations': [],
            's2_ids': [],
            's2_names': [],
        }
    
        for i, scopus_id in enumerate(scopus_author_map):
            scopus_name = scopus_author_map.get(scopus_id)
            scopus_affiliations = affiliations_map.get(scopus_id)
            if scopus_id in duplicates:
                scopus_id = ';'.join([scopus_id] + duplicates[scopus_id])
            
            slic_df['slic_id'].append(f'S{i}')
            slic_df['slic_name'].append(scopus_name)
            slic_df['scopus_ids'].append(scopus_id)
            slic_df['scopus_names'].append(scopus_name)
            slic_df['scopus_affiliations'].append(scopus_affiliations)
            slic_df['s2_ids'].append(None)
            slic_df['s2_names'].append(None)
            
        slic_df = pd.DataFrame.from_dict(slic_df)
        return slic_df
    
    
    def run(self, df, scopus_duplicates=None, s2_duplicates=None, known_matches=None, n_jobs=-1):
        """
        Run Orca and form SLIC ids for a given dataset

        Parameters
        ----------
        df: pandas.DataFrame
            The SLIC dataframe for which author SLIC ids need to be created
        scopus_duplicates: list(set), optional
            A list of sets where each set contains scopus author ids that refer to the same person. In the ideal case, each 
            author only has one scopus id. However, this ideal does not hold up in practice and some authors are represented
            by two or more scopus ids. Duplicate authors can be found using the Orca.DuplicateAuthorFinder tool. If not provided,
            a pre-computed scopus duplicate map is used (pre-computed from 1 million Scopus papers). If provided, this map is
            overriden by the user input. Default is None.
        s2_duplicates: list(set), optional
            A list of sets where each set contains s2 author ids that refer to the same person. If not provided, s2 author ids
            are not scanned for duplicates / only compared against scopus matches as duplicate detection. Default is None.
        known_matches: dict, optional
            A dict of s2 id keys to scopus id values. This dictionary is used to override the author matching if groundtruth 
            is known. This is useful for helping the tool work around edge cases. Default is None.
            
        Returns
        -------
        None
        """
        # process duplicates (if passed)
        if scopus_duplicates is not None:
            self.duplicates = scopus_duplicates
        if s2_duplicates is not None:
            self.s2_duplicates = s2_duplicates
        
        valid, error = self.__verify_df(df)  # make sure that the passed dataframe meets expected 
        if not valid:
            if False:  # TODO: set valid flag to check if Scopus only data here
                raise ValueError(error)
            else:
                self.slic_df = self._run_scopus(df)
                return self.slic_df
        
        # generate a map of scopus ids to affiliations
        affiliations_map = self.__generate_affiliations_map(df)

        # generate author maps
        s2_author_map = self.__generate_author_map(df, 's2_author_ids', 's2_authors')
        scopus_author_map = self.__generate_author_map(df, 'author_ids', 'authors')
        
        # match scopus author ids to s2 author ids
        known_matches = {} if not known_matches else known_matches
        am = AuthorMatcher(df, n_jobs=n_jobs, verbose=self.verbose)
        am_df = am.match(known_matches=known_matches)
    
        # process scopus duplicates
        am_enriched = self.__add_scopus_duplicates(am_df, self.duplicates)
        am_df = pd.concat([am_df, am_enriched], axis=0, ignore_index=True)

        ## generate SLIC IDs
        slic_count = 0
        slic_df = {
            'slic_id': [],
            'slic_name': [],
            'scopus_ids': [],
            'scopus_names': [],
            'scopus_affiliations': [],
            's2_ids': [],
            's2_names': [],
        }
        
        # 1. assign SLIC IDs to author ids that have correspondence between s2 and scopus
        seen_s2 = set()
        seen_scopus = set()
        matches = self.__uncouple_author_matches(am_df, self.s2_duplicates)
        for entry in matches:
            s2_ids = entry['s2']
            s2_names = {s2_author_map.get(x, 'Unknown') for x in s2_ids if x in s2_author_map}
            scopus_ids = entry['scopus']
            scopus_names = {scopus_author_map[x] for x in scopus_ids if x in scopus_author_map}
            scopus_affiliations = self.__merge_scopus_affiliations(scopus_ids, affiliations_map)
            slic_name = entry['name']
            
            seen_s2 |= s2_ids
            seen_scopus |= scopus_ids

            slic_df['slic_id'].append(f'S{slic_count}')
            slic_df['slic_name'].append(slic_name)
            slic_df['scopus_ids'].append(';'.join(scopus_ids))
            slic_df['scopus_names'].append(';'.join(scopus_names))
            slic_df['scopus_affiliations'].append(scopus_affiliations)
            slic_df['s2_ids'].append(';'.join(s2_ids))
            slic_df['s2_names'].append(';'.join(s2_names))
            slic_count += 1
        
        # 2. assign SLIC IDs to scopus ids that did not have correspondence
        df_scopus_authors = {x for y in df.author_ids.to_list() if not pd.isna(y) for x in y.split(';')}
        df_scopus_authors -= seen_scopus
        for scopus_id in df_scopus_authors:
            scopus_name = scopus_author_map.get(scopus_id, None)
            
            slic_df['slic_id'].append(f'S{slic_count}')
            slic_df['slic_name'].append(scopus_name)
            slic_df['scopus_ids'].append(scopus_id)
            slic_df['scopus_names'].append(scopus_name)
            slic_df['scopus_affiliations'].append(affiliations_map.get(scopus_id, None))
            slic_df['s2_ids'].append(None)
            slic_df['s2_names'].append(None)
            slic_count += 1
        
        # 3. assign SLIC Ids to remaining s2 authors
        df_s2_authors = {x for y in df.s2_author_ids.to_list() if not pd.isna(y) for x in y.split(';')}
        df_s2_authors -= seen_s2
        for s2_id in df_s2_authors:
            if s2_id in seen_s2:
                continue
            
            # update the common fields
            slic_df['slic_id'].append(f'S{slic_count}')
            slic_df['scopus_ids'].append(None)
            slic_df['scopus_names'].append(None)
            slic_df['scopus_affiliations'].append(None)
            
            # handle s2 duplicates if they exist
            s2_dup_ids = self.s2_duplicates.get(s2_id)
            if s2_dup_ids is not None:
                s2_ids = {s2_id} | set(s2_dup_ids)
                s2_names = {s2_author_map.get(x, 'Unknown') for x in s2_ids if x in s2_author_map}
                if s2_names == {'Unknown'}:  # handle case where all name missing
                    s2_names = set()
                
                # get the slic name
                try:
                    slic_name = max(s2_names, key=len) 
                    slic_name = None if slic_name == 'Unknown' else slic_name
                except ValueError: 
                    slic_name = None
                
                # update the data map
                slic_df['slic_name'].append(slic_name)
                s2_ids_str = ';'.join(s2_ids) if s2_ids else None
                slic_df['s2_ids'].append(s2_ids_str)
                s2_names = ';'.join(s2_names) if s2_names else None
                slic_df['s2_names'].append(s2_names)
                seen_s2 |= s2_ids
                
            else:
                s2_name = s2_author_map.get(s2_id, None)
                slic_df['slic_name'].append(s2_name)
                slic_df['s2_ids'].append(s2_id)
                slic_df['s2_names'].append(s2_name)
                seen_s2 |= s2_ids
                
            # incremenet the slic id identifier
            slic_count += 1
        
        slic_df = pd.DataFrame.from_dict(slic_df)
        slic_df = slic_df.loc[slic_df.slic_name != 'Unknown'].copy().reset_index(drop=True)
        self.slic_df = slic_df
        return slic_df
    
    
    def apply(self, df, slic_df=None):
        """
        Apply the SLIC id mapping to a SLIC papers dataframe

        Parameters
        ----------
        df: pandas.DataFrame
            The SLIC dataframe for which author SLIC ids need to be created
        slic_df: pandas.DataFrame, optional
            A pre-computed DataFrame with SLIC id mappings. This parameter is provided in the rare cases that a SLIC map is
            being used between multiple datasets (i.e. dataset B is a subset of A and slic_df was computed for A). Be aware that
            setting a value for slic_df is not recommended! If using this parameter, verify that all desired scopus/s2 authors
            have existing SLIC ids. To be sure of the validity of your results, use Orca.run() before using Orca.apply() and 
            do not pass a value for this parameter.
            
        Returns
        -------
        orca_df: pandas.DataFrame
            df with standarized author information (columns for 'SLIC_ids' and 'SLIC_affiliations')
        """
        if slic_df is None and self.slic_df is None:
            return ValueError('No SLIC ID map found. First, compute the map with Orca.run()')
        if slic_df is not None and self.slic_df is not None:
            warnings.warn('[Orca]: slic_df was passed as an argument however this Orca object already has a ' \
                          'stored slic_df object.\n\t\tOverwriting stored slic_df with given argument. If this ' \
                          'message is unexpected, use Orca.apply() without specifying `slic_df`', RuntimeWarning)
            
        if slic_df is not None:
            self.slic_df = slic_df
    
        # verify that paper scopus ids and s2ids are unique
        if 'eid' in df.columns and df.eid.nunique() != len(df.loc[~df.eid.isnull()]):
            df = df[~df['eid'].duplicated(keep='first') | df['eid'].isna()].copy()
            warnings.warn('[Orca]: Encountered duplicate Scopus IDs (`eid`) in df. Dropping duplicate papers.')
        if 's2id' in df.columns and df.s2id.nunique() != len(df.loc[~df.s2id.isnull()]):
            df = df[~df['s2id'].duplicated(keep='first') | df['s2id'].isna()].copy()
            warnings.warn('[Orca]: Encountered duplicate S2 IDs (`s2id`) in df. Dropping duplicate papers.')
        
        # replace scopus and s2 author ids respectively
        if 's2id' in df.columns and 'eid' in df.columns:
            scopus_df = self.__compute_slic_scopus(df)
            s2_df = self.__compute_slic_s2(df)

            # merge and build output dataframe
            df2 = pd.merge(df, scopus_df, on='eid', how='outer')
            df3 = pd.merge(df2, s2_df, on='s2id', how='outer')
            orca_df = df3.copy()
            orca_df['slic_author_ids'] = orca_df['slic_author_ids_x'].combine_first(orca_df['slic_author_ids_y'])
            orca_df = orca_df.drop(columns=['slic_author_ids_x', 'slic_author_ids_y'])
            
        elif 's2id' in df.columns:
            s2_df = self.__compute_slic_s2(df)
            orca_df = pd.merge(df, s2_df, on='s2id', how='outer')
        else:
            scopus_df = self.__compute_slic_scopus(df)
            orca_df = pd.merge(df, scopus_df, on='eid', how='outer')
                
        if orca_df.slic_author_ids.isna().any():
            original_len = len(orca_df)
            orca_df.dropna(subset=['slic_author_ids'], inplace=True)
            warnings.warn(f'[Orca]: Found {original_len - len(orca_df)} papers with missing ' \
                           'SLIC author IDs. Dropping these papers.')
                
        # add a column of slic author names using the matched slic ids
        slic_authors = {k:v for k,v in zip(self.slic_df.slic_id.to_list(), self.slic_df.slic_name.to_list())}
        def map_ids_to_names(ids):
            if pd.isna(ids):
                return None
            names = [slic_authors.get(str(i), '') for i in ids.split(';')]
            names = [name for name in names if name]
            return ';'.join(names)
        orca_df['slic_authors'] = orca_df['slic_author_ids'].apply(map_ids_to_names)
        return orca_df.reset_index(drop=True)
        
        
    def __verify_df(self, df):
        """
        Verify that the given papers dataframe matches the SLIC standard and can be used with Orca

        Parameters
        ----------
        df: pandas.DataFrame
            The SLIC papers DataFrame for which author SLIC ids need to be created

        Returns
        -------
        flag: bool
            If true, df passes the test and can be used with Orca
        error: str, None
            If flag is True, None is returned. Otherwise a string with the encountered error is provided
        """
        must_have = {'eid', 'authors', 'author_ids', 'affiliations', 's2_authors', 's2_author_ids'}
        columns = set(df.columns)
        if columns & must_have != must_have:
            return False, f'The columns {list(must_have - columns)} are missing in `df`'
        
        return True, None
    
    
    def __verify_slic_df(self, slic_df):
        """
        Verify that the given papers dataframe matches the SLIC standard and can be used with Orca

        Parameters
        ----------
        slic_df: pandas.DataFrame
            A pre-computed DataFrame with SLIC id mappings.

        Returns
        -------
        flag: bool
            If true, df passes the test and can be used with Orca
        error: str, None
            If flag is True, None is returned. Otherwise a string with the encountered error is provided
        """
        must_have = {'slic_id', 'slic_name', 'scopus_ids', 'scopus_names', 's2_ids', 's2_names'}
        columns = set(slic_df.columns)
        if columns & must_have != must_have:
            return False, f'The columns {list(must_have - columns)} are missing in `slic_df`'
        
        return True, None
        
        
    def __add_scopus_duplicates(self, auth_df, duplicates):
        """
        Helper function that enriches the results of AuthorMatcher with any previously detected Scopus 
        duplicates. These duplicates are given a shared S2 id that will be used to connect them in the
        succeeding SLIC id creation steps.
        
        Parameters
        ----------
        auth_df: pandas.DataFrame
            The results of AuthorMatcher on the working DataFrame
        duplicates: list(set), optional
            A list of sets where each set contains scopus author ids that refer to the same person. In the ideal case, each 
            author only has one scopus id. However, this ideal does not hold up in practice and some authors are represented
            by two or more scopus ids. Duplicate authors can be found using the Orca.DuplicateAuthorFinder tool. 
            
        Returns
        -------
        out_df: pandas.DataFrame
            DataFrame that matches the shape of auth_df but contains entries to flag scopus duplicates
        """
        out_df = pd.DataFrame(columns=auth_df.columns)
        all_df_ids = set(auth_df.SCOPUS_Author_ID.to_list())
        if self.verbose:
            print('[Orca]: Scanning for Scopus duplicates in dataset. . .')
        for scopus_ids in tqdm(duplicates, total=len(duplicates), disable=not self.verbose):
            if not scopus_ids & all_df_ids:
                continue
            tmp_df = auth_df.loc[auth_df.SCOPUS_Author_ID.isin(scopus_ids)]
            if len(tmp_df.SCOPUS_Author_ID.unique()) > 1:
                row = tmp_df.iloc[0].copy()
                for scopus_id in tmp_df.SCOPUS_Author_ID.unique():
                    if row.SCOPUS_Author_ID == scopus_id:
                        continue
                    else:
                        new_row = row.copy()
                        new_row.SCOPUS_Author_ID = scopus_id
                        out_df = pd.concat([out_df, new_row.to_frame().T], ignore_index=True)
        return out_df
    
    
    def __add_s2_duplicates(self, duplicates):
        """
        Converts a list of sets into a dictionary such that each key in the dictionary 
        is an element from a set, and its value is a list of the other elements in that set.
        Each set should contain s2 author ids that are known duplicates of each other. No 
        two pairs of sets can share an author id. All known duplicate of an s2 author id should
        be contained within a single set. Sets with a single id (non-duplicates) will be ignored.

        Parameters:
        -----------
        duplicates: list(set())
            A list of sets of s2 authors ids to be processed.

        Returns:
        --------
        dict: 
            The processed s2 duplicates which will be resolved in a further processinf step.
            
        Raises:
        -------
        ValueError: 
            If an id appears in more than one set within the list.

        Example:
        --------
        >>> self.__add_s2_duplicates([{1,2}, {3,4}, {5}])
        {1: [2], 2: [1], 3: [4], 4: [3]}
        >>> self.__add_s2_duplicates([{1,2}, {2,3}])
        ValueError
        """
        out_dict = {}
        seen = set()

        for s in duplicates:
            if any(elem in seen for elem in s):
                raise ValueError("Detected multiple entries for s2 duplicates across sets. \
                                  Make sure that all known duplicates are constrained to a single set")
            seen.update(s)
            if len(s) == 1:
                continue
                
            for element in s:
                out_dict[element] = [x for x in s if x != element]
        return out_dict

    
    def __propagate_duplicates(self, a_map, a_duplicates):
        """
        Propagate the ids associated with keys in a_map to their duplicate keys.

        For each key in a_map, if duplicate keys exist in a_duplicates, 
        the associated values from a_map are propagated to these duplicate keys.
        The 'a'/'b' notation is used to keep this function generic so that is can be used 
        to go from s2 --> scopus or scopus  --> s2.

        Parameters:
        -----------
        a_map: dict
            The main author dictionary that is to be updated.
        a_duplicates: dict
            Dictionary mapping keys to lists of their duplicates.

        Returns:
        --------
        None
            a_map is modified in place.
        """

        a_map_update = {}

        # create an update map based on duplicates
        for a_id, b_ids in a_map.items():
            if a_id in a_duplicates:
                for dup_a_id in a_duplicates[a_id]:
                    if dup_a_id not in a_map_update:
                        a_map_update[dup_a_id] = set()
                    a_map_update[dup_a_id] |= set(b_ids)

        # convert set to list for each key in the update map
        for dup_a_id in a_map_update:
            a_map_update[dup_a_id] = list(a_map_update[dup_a_id])

        # update the main map in place
        a_map.update(a_map_update)

    
    def __uncouple_author_matches(self, auth_df, s2_duplicates):
        """
        Helper function that takes the authors DataFrame produced by AuthorMatcher (could be enriched with scopus duplicates
        or not) and finds out which sets of authors ids represent the same individual. This is done by building a graph of
        author ids relationships.

        Take for example the following two maps. In this case letters are scopus IDs and numbers are S2 IDs. Each map presents
        the relationship between scopus and S2 from the perspective of the key dataset. There are only 2 authors but they are 
        represented by 2 scopus / 3 S2 IDs for the first author and 1 scopus / 2 S2 IDs for the second author. 

        >>> scopus_map = {'A': [1,2], 
                          'B': [4], 
                          'C': [3,5]}

        >>> s2_map = {1: ['A'], 
                      2: ['A'], 
                      3: ['C'], 
                      4: ['A'], 
                      5: ['C']}

        For bigger datasets, these relationships can grow complex and are best modeled by a graph. Both scopus and S2 IDs are 
        nodes in this graph and  their relationship can be modeled with egdes between them. This graph is very disconnected as 
        there will be many unique authors in any given SLIC dataset. However, weakly connected components of the graph will 
        signify that all author id nodes in said component belong to the same author.

        Parameters
        ----------
        auth_df: pandas.DataFrame
            The results of AuthorMatcher on the working DataFrame
        s2_duplicates:

        Returns
        -------
        matches: list
            A list of dictionaries. Each dictionary in the list contains 2 keys: 'scopus' and 's2'. The values are sets of 
            corresponding scopus/s2 ids
        """
        # create the two maps necessary for processing
        s2_map = auth_df.groupby('S2_Author_ID')['SCOPUS_Author_ID'].agg(set).to_dict()
        scopus_map = auth_df.groupby('SCOPUS_Author_ID')['S2_Author_ID'].agg(set).to_dict()
        
        # handle s2 duplicates
        self.__propagate_duplicates(s2_map, s2_duplicates)
        
        # ensure that no s2 id == scopus id by coincidence
        s2_map = {f'B_{k}': {f'A_{x}' for x in v} for k,v in s2_map.items()}
        scopus_map = {f'A_{k}': {f'B_{x}' for x in v} for k,v in scopus_map.items()}

        # also create a name map which we will use 
        s2_name_map = auth_df.groupby('S2_Author_ID')['S2_Author_Name'].agg(lambda x: max(x, key=len)).to_dict()

        # setup the graph
        G = nx.DiGraph()
        for k, v_set in scopus_map.items():
            for v in v_set:
                G.add_edge(k, v)

        for k, v_set in s2_map.items():
            for v in v_set:
                G.add_edge(k, v)

        # get the list of components and process them
        components = list(nx.weakly_connected_components(G))
        matches = []
        for component_set in components:
            mdict = {'scopus': set(), 's2': set()}
            for c in component_set:
                if c.startswith('A_'):
                    mdict['scopus'].add(c[2:])
                else:
                    mdict['s2'].add(c[2:])

            # get the longest s2 name to use as slic name
            str_gen = ((pid, s2_name_map[pid]) for pid in mdict['s2'] if pid in s2_name_map)
            _, name = max(str_gen, key=lambda x: len(x[1]), default=(None, 'Unknown'))
            mdict['name'] = name
            matches.append(mdict)
        return matches
    
    
    def __generate_author_map(self, df, id_col, name_col):
        """
        Helper function that generates a map of author ids to author names
        
        Parameters
        ----------
        df: pandas.DataFrame
            The SLIC papers DataFrame for which author SLIC ids need to be created
        id_col: str
            The author ids column. Options are ['author_ids', 's2_author_ids']
        name_col: str
            The author names column. Options are ['authors', 's2_authors']
            
        Returns
        -------
        auth_map: dict
            Map where keys are author ids and values are author names
        """
        if self.verbose:
            print(f'[Orca]: Generating {id_col}-{name_col} map. . .')
        
        auth_map = {}
        tmp_df = df.dropna(subset=[id_col, name_col])
        for id_list, auth_list in tqdm(zip(tmp_df[id_col].to_list(), tmp_df[name_col].to_list()), total=len(tmp_df), disable=not self.verbose):
            for auth_id, name in zip(id_list.split(';'), auth_list.split(';')):
                if auth_id not in auth_map:
                    auth_map[auth_id] = name
        return auth_map
    
    
    def __compute_slic_scopus(self, df):
        """
        Helper function applies the SLIC id map to papers that have scopus information
        
        Parameters
        ----------
        df: pandas.DataFrame
            The SLIC papers DataFrame for which author SLIC ids need to be created
            
        Returns
        -------
        scopus_df: pandas.DataFrame
            papers DataFrame that contains SLIC id and affiliation information 
        """
        tmp_df = df.loc[~df['eid'].isnull()]  # get only scopus papers
        
        # create maps for scopus author an
        scopus_authors, scopus_affiliations = {}, {}
        for eid, author_ids, affiliations in zip(tmp_df['eid'].to_list(), 
                                                 tmp_df['author_ids'].to_list(), 
                                                 tmp_df['affiliations'].to_list()):
            if not pd.isna(author_ids):
                scopus_authors[eid] = author_ids
            if not pd.isna(affiliations):
                if isinstance(affiliations, str):
                    affiliations = ast.literal_eval(affiliations)
                scopus_affiliations[eid] = affiliations
            
        scopus_df = {
            'eid': [],
            'slic_author_ids': [],
            'slic_affiliations': [],
        }
        
        # compute map of scopus author id to slic author id
        scopus_to_slic = {x: k for k,v in zip(self.slic_df.slic_id.to_list(), self.slic_df.scopus_ids.to_list()) 
                          if not pd.isna(v) for x in v.split(';')}
        

        missing_authors = set()
        for eid in tmp_df.eid.to_list():

            slic_author_ids = []  # first replace author_ids information 
            author_ids = scopus_authors.get(eid)
            if author_ids is not None:
                for scopus_id in author_ids.split(';'):
                    scopus_id = str(scopus_id)  # should already be string but hard cast to make sure
                    slic_id = scopus_to_slic.get(scopus_id)
                    if slic_id is None:
                        missing_authors.add(scopus_id)
                    else:
                        slic_author_ids.append(str(slic_id))

            aff_dict, del_dict = {}, []  # next update affiliations structure
            affiliations = scopus_affiliations.get(eid)
            if affiliations is not None:
                for aff_id, aff_info_shallow in affiliations.items():
                    del_list = []  # items to remove
                    aff_info = copy.deepcopy(aff_info_shallow)
                    for i in range(len(aff_info['authors'])):
                        scopus_id = str(aff_info['authors'][i])
                        if scopus_id not in scopus_to_slic:
                            del_list.append(scopus_id)
                            missing_authors.add(scopus_id)
                        else:
                            aff_info['authors'][i] = scopus_to_slic[scopus_id]

                    for d in del_list:
                        if d not in aff_info['authors']:
                            aff_info['authors'].remove(str(d))
                        else:
                             aff_info['authors'].remove(d)

                    if not aff_info['authors']:
                        del_dict.append(aff_id) 
                    aff_dict[aff_id] = aff_info

                for d in del_dict:
                    del aff_dict[d]

            scopus_df['eid'].append(eid)
            if not slic_author_ids:
                scopus_df['slic_author_ids'].append(None)
            else:
                scopus_df['slic_author_ids'].append(";".join(slic_author_ids))
            if not aff_dict:
                scopus_df['slic_affiliations'].append(None)
            else:
                scopus_df['slic_affiliations'].append(aff_dict) 

        if len(missing_authors) > 0:
            warnings.warn(f'[Orca]: {len(missing_authors)} Scopus IDs did not have corresponding SLIC ID and were removed')
        
        scopus_df = pd.DataFrame.from_dict(scopus_df)
        return scopus_df
    
    
    def __compute_slic_s2(self, df):
        """
        Helper function applies the SLIC id map to papers that have S2 information
        
        Parameters
        ----------
        df: pandas.DataFrame
            The SLIC papers DataFrame for which author SLIC ids need to be created
            
        Returns
        -------
        s2_df: pandas.DataFrame
            papers DataFrame that contains SLIC id and affiliation information 
        """
        #tmp_df = df.loc[df['eid'].isnull()]  # get only s2 papers
        tmp_df = df.loc[~df['s2id'].isnull()]
        
        # compute map of s2 author id to slic author id
        s2_to_slic = {x: k for k,v in zip(self.slic_df.slic_id.to_list(), self.slic_df.s2_ids.to_list()) 
                      if not pd.isna(v) for x in v.split(';')}
        
        s2_df = {
            's2id': [],
            'slic_author_ids': [],
        }
        
        missing_authors = set()
        for s2id, s2_author_ids in zip(tmp_df['s2id'].to_list(), tmp_df['s2_author_ids'].to_list()):
            slic_author_ids = []
            if not pd.isna(s2_author_ids):
                for s2_auth_id in s2_author_ids.split(';'):
                    if s2_auth_id not in s2_to_slic:
                        missing_authors.add(s2_auth_id)
                    else:
                        slic_author_ids.append(s2_to_slic[s2_auth_id])
            
            if slic_author_ids:
                s2_df['s2id'].append(s2id)
                s2_df['slic_author_ids'].append(";".join(slic_author_ids))
            else:
                s2_df['s2id'].append(s2id)
                s2_df['slic_author_ids'].append(None)


        if len(missing_authors) > 0:
            warnings.warn(f'[Orca]: {len(missing_authors)} S2 IDs did not have corresponding SLIC ID and were removed')
        
        s2_df = pd.DataFrame.from_dict(s2_df)
        return s2_df


    def __load_pickle(self, fn):
        """
        Helper function for loading pickle files saved in data package
        If upgrading to python >=3.9, change this function make use of importlib.resources

        Parameters
        ----------
        fn: str
            The file name to be loaded
            
        Returns
        -------
        python object stored in the pickle file
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return pickle.load(open((os.path.join(current_dir, 'data', fn)), 'rb'))


    def __generate_affiliations_map(self, df):
        """
        Helper function for computing a map of affiliations for scopus authors. 
        The output map is a dict that adheres to the following structure:
        {
            SCOPUS_AUTHOR_ID:
            {
                SCOPUS_AFFILIATION_ID:
                {
                    'name': NAME,  # the name of the affiliation
                    'country': COUNTRY  # country associated with the affiliation
                    'first_seen': XXXX  # the year when first known paper was published by author with given affiliation
                    'last_seen': XXXX  # the year when last known paper was published by author with given affiliation
                    'papers': XXXX  # list of known papers with this affiliation. NOT guaranteed to contain all papers
                },
                ...
            },
            ...
        }

        Parameters
        ----------
        df: pandas.DataFrame
            The SLIC papers DataFrame for which author SLIC ids need to be created
            
        Returns
        -------
        affiliations_map: dict,
            The created map
        """
        affiliations_map = {}
        for eid, year, affiliations in zip(df.eid.to_list(), df.year.to_list(), df.affiliations.to_list()):
            
            # handle missing / unconverted affiliations
            if pd.isna(affiliations):
                continue
            if isinstance(affiliations, str):
                affiliations = ast.literal_eval(affiliations)
            
            # get the year
            if pd.isna(year):
                year = 0
            else:
                year = int(year)
                
            for aff_id, info in affiliations.items():
                aff_name = info.get('name', 'Unknown')
                aff_country = info.get('country', 'Unknown')

                for auth_id in info.get('authors', []):
                    if auth_id not in affiliations_map:
                        affiliations_map[auth_id] = {}
                    if aff_id not in affiliations_map[auth_id]:
                        affiliations_map[auth_id][aff_id] = {}
                    first_seen = affiliations_map[auth_id][aff_id].get('first_seen', 1*10**4)
                    last_seen = affiliations_map[auth_id][aff_id].get('last_seen', -1)

                    if 'name' not in affiliations_map[auth_id][aff_id]:
                        affiliations_map[auth_id][aff_id]['name'] = aff_name
                        affiliations_map[auth_id][aff_id]['country'] = aff_country
                        affiliations_map[auth_id][aff_id]['first_seen'] = year
                        affiliations_map[auth_id][aff_id]['last_seen'] = year
                        affiliations_map[auth_id][aff_id]['papers'] = {eid}
                    else:
                        affiliations_map[auth_id][aff_id]['papers'].add(eid)
                        if year < first_seen or first_seen == 0:
                            affiliations_map[auth_id][aff_id]['first_seen'] = year
                        elif year > last_seen or last_seen == 0:
                            affiliations_map[auth_id][aff_id]['last_seen'] = year

        # handle the missing years 
        for auth_id in affiliations_map:
            for aff_id, aff_info in affiliations_map[auth_id].items():
                if not aff_info['first_seen']:
                    affiliations_map[auth_id][aff_id]['first_seen'] = None
                if not aff_info['last_seen']:
                    affiliations_map[auth_id][aff_id]['last_seen'] = None
        return affiliations_map


    def __merge_scopus_affiliations(self, scopus_ids, data):
        """
        Helper function for merging the scopus affiliation maps for entries where multiple scopus
        ids correspond to the same author

        Parameters
        ----------
        scopus_ids: list
            A list of scopus author ids
        data: dict
            The scopus affiliations map
            
        Returns
        -------
        merged: dict
            A dictionary that is a result of merging the dictionaries associated with the 
            duplicate author ids. For shared items, 'first_seen' is the earliest and 'last_seen' is 
            the latest among all the dictionaries. For unshared items, their details are kept 
            as is. If 'first_seen' or 'last_seen' is 'Unknown' in one dictionary but has a valid 
            integer value in another, the integer value is considered. If no valid integer value 
            exists for 'first_seen' or 'last_seen', 'Unknown' is set as the value.
        """
        merged = {}
        all_keys = set(k for key in scopus_ids if key in data for k in data[key].keys())
        
        for key in all_keys:
            items = [data[k][key] for k in scopus_ids if k in data and key in data[k]]
            names = [item['name'] for item in items if item['name'] != 'Unknown']
            countries = [item['country'] for item in items if item['country'] != 'Unknown']
            papers = list(set.union(*[item['papers'] for item in items]))
            merged[key] = {
                'name': names[0] if names else 'Unknown',  # use first valid affiliation name, or 'Unknown' if no valid names
                'country': countries[0] if countries else 'Unknown',  # use first valid affiliation country, or 'Unknown' if no valid countries
                'first_seen': min((item['first_seen'] for item in items if isinstance(item['first_seen'], int)), default='Unknown'),
                'last_seen': max((item['last_seen'] for item in items if isinstance(item['last_seen'], int)), default='Unknown'),
                'papers': papers
            }
        merged = merged if merged else None  # {} --> None
        return merged
    
    
    ### GETTERS / SETTERS
    
    
    @property
    def duplicates(self):
        return self._duplicates
    
    @property
    def s2_duplicates(self):
        return self._s2_duplicates

    @duplicates.setter
    def duplicates(self, duplicates):
        if duplicates is None:
            self._duplicates = []
        elif isinstance(duplicates, list):
            self._duplicates =  duplicates
        else:
            raise TypeError(f' {type(duplicates)} is an invalid type for `duplicates`')
        
    """
    Code where we have a precomputed duplicates.
    @duplicates.setter
    def duplicates(self, duplicates):
        if duplicates is None:
            self._duplicates = self.__load_pickle(self.DUPLICATES_1M)
        elif isinstance(duplicates, list):
            self._duplicates = self.__load_pickle(self.DUPLICATES_1M) + duplicates
        else:
            raise TypeError(f' {type(duplicates)} is an invalid type for `duplicates`')
    """
    
    
    @s2_duplicates.setter
    def s2_duplicates(self, s2_duplicates):
        if s2_duplicates is None:
            self._s2_duplicates = {}
        elif isinstance(s2_duplicates, list):
            self._s2_duplicates = self.__add_s2_duplicates(s2_duplicates)
        else:
            raise TypeError(f' {type(s2_duplicates)} is an invalid type for `s2_duplicates`')