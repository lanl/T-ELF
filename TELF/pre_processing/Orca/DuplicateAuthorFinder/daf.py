import os
import sys
import ast
import sparse
import networkx
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from networkx.algorithms.components.connected import connected_components

from ..utils import match_name
from ....helpers.host import verify_n_jobs
from ....helpers.data_structures import gen_chunks

# define DAF globals for shared-mem access
uid_matrix = None
email_matrix = None
affiliation_matrix = None
collaboration_matrix = None
eid_to_name = None

def to_graph(l):
    """
    Convert a list of lists into a graph.
    
    Parameters:
    -----------
    l: list of lists
        Each sublist represents a set of nodes, and also implies 
        edges between each pair of nodes in the sublist.
        
    Returns:
    --------
    networkx.Graph
        A graph with nodes and edges based on the input list `l`.
        
    Example:
    --------
    >>> G = to_graph([[1, 2, 3], [2, 4]])
    >>> G.nodes()
    [1, 2, 3, 4]
    >>> G.edges()
    [(1, 2), (1, 3), (2, 3), (2, 4)]
    """
    G = networkx.Graph()
    for part in l:
        G.add_nodes_from(part)  # each sublist is a bunch of nodes
        G.add_edges_from(to_edges(part))  # it also imlies a number of edges:
    return G


def to_edges(l):
    """ 
    Treat `l` as a Graph and returns its edges 
    
    Parameters:
    -----------
    l: list
        List of nodes. 
        
    Returns:
    --------
    list
        List of edges as tuples where each tuple contains 2 nodes.
    
    Example:
    --------
    >>> list(to_edges(['a','b','c','d']))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    """
    it = iter(l)
    last = next(it)
    for current in it:
        yield last, current
        last = current


class DuplicateAuthorFinder:
    
    """
    TODO: 
    - check if two 'duplicates' show up on the same paper
    - implement email check
    - implement orcid check 
    """

    # the valid search modes for the DuplicateAuthorFinder
    SUPPORTED_SEARCH_OPTIONS = ['emails', 'uid', 'collaborators', 'affiliations', 'name', 'papers']


    def __init__(self, n_jobs=-1, verbose=False):
        
        """
        DAF is a parallel matching tool intended to check if the same author has
        associated duplicate IDs. 

        Parameters
        ----------
        n_jobs: int, optional
            Number of parallel processes. The default is -1.
        verbose: bool, int, optional
            Verbosity level. Can be a bool or an int. Higher int means higher verbosity. Default is False.

        Returns
        -------
        None
        """
        # input check
        assert verbose >= 0, "verbose must be a positive integer"

        self.n_jobs = n_jobs        
        self.verbose = verbose


    def _compute_collaboration_matrix(self, collab_list, id_to_eid):
        rows = []
        cols = []  # generate nonzero value locations for collab matrix
        s = set()
        for aid, collabs in tqdm(collab_list): 
            aid = str(aid)
            if isinstance(collabs, float) or collabs is None:
                continue
            
            if isinstance(collabs, str):
                collabs = ast.literal_eval(collabs)
            for cid in collabs:  # collaborators id
                rows.append(id_to_eid[aid])
                cols.append(id_to_eid[cid])

        coords = np.array([rows, cols])
        X = sparse.COO(coords, 1, shape=(len(id_to_eid), len(id_to_eid)))
        return X

    
    def _compute_affiliation_matrix(self, aff_list, id_to_eid) :
        rows = []
        cols = []  # generate nonzero value locations for matrix

        count = 0
        aff_to_eid = {}
        for aid, affs in tqdm(aff_list): 
            aid = str(aid)
            if isinstance(affs, float) or affs is None:
                continue
            
            if isinstance(affs, str):
                affs = ast.literal_eval(affs)
            for aff in affs:
                afid = aff['afid']
                if afid not in aff_to_eid:
                    aff_to_eid[afid] = len(aff_to_eid)

                rows.append(id_to_eid[aid])
                cols.append(aff_to_eid[afid])

        coords = np.array([rows, cols])
        X = sparse.COO(coords, 1, shape=(len(id_to_eid), len(aff_to_eid)))
        return X

    
    def _compute_attribution_matrix(self, attribute_list, id_to_eid, split=None, lower=False):
        rows = []
        cols = []
        attrib_to_eid = {}
        for aid, attrib in tqdm(attribute_list, disable=not self.verbose): 
            aid = str(aid)
            if pd.isna(attrib):
                continue
                
            # retrive the enumerated index for the author id
            aid_index = id_to_eid[aid]
            
            # if split is specified assume entry is a list of attributes delimited by `split`
            if split is not None:
                attribs = attrib.split(split)
            else:
                attribs = [attrib]
            
            # pre-process attributes
            attribs = [x.strip() for x in attribs]
            if lower:
                attribs = [x.lower() for x in attribs]
            
            for attr in attribs:
                if attr not in attrib_to_eid:
                    attrib_to_eid[attr] = len(attrib_to_eid)
                    
                rows.append(aid_index)
                cols.append(attrib_to_eid[attr])

        coords = np.array([rows, cols])
        X = sparse.COO(coords, 1, shape=(len(id_to_eid), len(attrib_to_eid)))
        return X
    

    def intersect2d(self, target_matrix, neighbor_matrix, neighbor_ids, num_collaborators, thresh):
        thresh_matched =  int(num_collaborators * thresh)
        intersections = np.bitwise_and(neighbor_matrix, target_matrix).sum(axis=1)
        return neighbor_ids[sparse.argwhere(intersections > thresh_matched)]


    def evaluate_collaborations(self, X, target, neighbor_ids=None, min_collab=10, thresh=0.65):
        if neighbor_ids is None:
            neighbor_ids = X[X[target].nonzero()].sum(axis=0).nonzero()[0]
        neighbor_matrix = X[neighbor_ids]
        
        num_collaborators = X[target].nnz
        if num_collaborators < min_collab:
            return np.array([target])  # not enough collaborators, return author alone
        else:
            collaborators = self.intersect2d(X[target], neighbor_matrix, neighbor_ids, num_collaborators, thresh)
            return collaborators.flatten()

    @staticmethod
    def evaluate_membership(X, target, neighbor_ids=None):
        """
        Evaluates membership of the target in matrix X.
        
        Parameters:
        -----------
        X: np.ndarray
            The input matrix.
        target: int
            The target column index to evaluate.
        neighbor_ids: np.ndarray, (optional)
            Array of neighbor indices to intersect with. Defaults to None.
        
        Returns:
        --------
        np.ndarray
            Indices of non-zero members or their intersection with neighbor_ids.
        """
        members = X.T[X[target].nonzero()].sum(axis=0).nonzero()[0]
        if neighbor_ids is None:
            return members
        else:
            return np.intersect1d(neighbor_ids, members)
        
        
    @staticmethod
    def filter_false_positives(matches, matrix):
        """
        Filters out false positive duplicate matches from the matches dictionary.

        Parameters:
        -----------
        matches: dict
            Dictionary where keys are enumerated IDs (eids) and values are sets of duplicate IDs.
        matrix: coo_matrix
            Sparse matrix containing attribute information for entities.
        
        Returns:
        --------
        None
            The function modifies the matches dictionary in place, removing duplicates that are false positives.
        """
        # convert COO to CSR for efficient row slicing
        matrix_csr = matrix.tocsr()
        for eid, duplicates in matches.items():
            if len(duplicates) == 1:  # no duplicates, no need to check
                continue

            eid_mask = matrix_csr[eid].toarray().flatten() != 0
            if not eid_mask.any():  # no UID for this author ID
                continue

            duplicates = np.array(list(duplicates))  # convert to array to track by index

            # check if any detected duplicates have more than one UID
            dup_mask = np.logical_not(np.array([np.array_equal(matrix_csr[idx].toarray().flatten() != 0, eid_mask) for idx in duplicates]))
            dup_indices = np.flatnonzero(dup_mask)
            dup_selected = duplicates[dup_indices]
            for d in dup_selected:
                matches[eid].remove(d)
     
    @staticmethod
    def remove_overlapping_duplicates(matches, matrix):
        """
        Removes overlapping duplicates from the matches dictionary.

        Parameters:
        -----------
        matches: dict
            Dictionary where keys are enumerated IDs (eids) and values are sets of duplicate IDs.
        matrix: coo_matrix
            Sparse matrix containing attribute information for entities.
        
        Returns:
        --------
        None
            The function modifies the matches dictionary in place, removing duplicates that are false positives.
        """
        # convert COO to CSR for efficient row slicing
        matrix_csr = matrix.tocsr()
        for eid, duplicates in matches.items():
            if len(duplicates) == 1:  # no duplicates, no need to check
                continue

            eid_mask = matrix_csr[eid].toarray().flatten() != 0
            if not eid_mask.any():  # no UID for this author ID
                continue

            duplicates = np.array(list(duplicates))  # convert to array to track by index

            # check if any detected duplicates share column values with eid
            dup_mask = np.array([np.any(np.logical_and(matrix_csr[idx].toarray().flatten() != 0, eid_mask)) for idx in duplicates])
            #print(f'{dup_mask=}')
            dup_indices = np.flatnonzero(dup_mask)
            #print(f'{dup_indices=}')
            dup_selected = duplicates[dup_indices]
            #print(f'{dup_selected=}')
            for d in dup_selected:
                if eid == d:
                    continue
                matches[eid].remove(d)


    def __call_(self, df, id_col, subset, fname=None, min_collab_diff_thresh=0.5, 
                min_name_diff_thresh=0.2, cache_dir='/tmp', force_compute=True):
        return self.search(df, id_col, subset, fname, min_collab_diff_thresh, 
                           min_name_diff_thresh, cache_dir, force_compute)
        
        
    def search(self, df, id_col, subset, fname=None, min_collab_diff_thresh=0.5, min_name_diff_thresh=0.2, 
               cache_dir='/tmp', force_compute=True):
        """
        Search for author duplicates in the author's DataFrame. The input file can be generated using
        classes defined in Orca.AuthorsFrames

        Parameters
        ----------
        df: pandas.DataFrame
            The author DataFrame generated by a class in Orca.AuthorsFrames
        auth_id_col: str
            Dataframe column that stores the author ids.
        subset: dict
            A dictionary containing the options to use in the search as keys and  he values as the column 
            names in the source dataframe.  Option keys are ['emails', 'uid', 'collaborators', 'affiliations', 
            'name', 'papers']. Note that name and papers columns should always be passed in subset.
        fname: str, pathlike, (optional)
            The path to the DataFrame if it stored on disk. Should only be used if df is not defined. Default is None.
        min_collab_diff_thresh: float, (optional)
            When comparing collaborators, the intersection of collaborating authors should
            be greater than or equal to this value.This value should be on (0, 1]. Default is 0.5.
        min_name_diff_thresh: float, (optional)
            When comparing names, the normalized difference between the names should be no greater 
            than this value. The difference is evaluated using the Levenshtein distance.
            This value should be on (0, 1]. Default is 0.5.
        cache_dir: str, pathlike, (optional)
            The directory where intermediate results should be cached. Default is '/tmp'.
        force_compute: bool, (optional)
            If True, re-compute the intermediate results even if they exist in cache
            
        Returns
        -------
        matches: list
            A list of sets where each set is matched author ids. If an author has no detected duplicates then the size
            of the set will be equal to 1. 
        """
        global email_matrix
        global affiliation_matrix
        global collaboration_matrix
        global eid_to_name

        if df is not None and fname is not None:
            raise ValueError('`df` and `fname` both defined! Provide one or the other.')
        elif df is None and fname is None:
            raise ValueError('`df` and `fname` both undefined! Define one of these arguments to set the target author DataFrame.')
            
        if not subset:
            raise ValueError(f'Please provide at least one field to search. Supported options include {self.SUPPORTED_SEARCH_OPTIONS}')
        if 'name' not in subset:
            raise ValueError('`name` should always be included in the search options')
        if 'papers' not in subset:
            raise ValueError('`papers` should always be included in the search options')
        if 'collaborators' in subset:
            if 'affiliations' not in subset:
                raise ValueError('`affiliations` must be included when using "collaborators"')
        for search_param in subset:
            if search_param not in self.SUPPORTED_SEARCH_OPTIONS:
                raise ValueError(f'Unsupported search options. Supported options include {self.SUPPORTED_SEARCH_OPTIONS}')

        
        # load the associated dataframe
        if df is None:
            df = pd.read_csv(fname)
        df.dropna(subset=[subset['name'], id_col], inplace=True)

        # compute the eid-id maps
        id_to_eid = {}
        eid_to_id = {}
        for i, auth_id in enumerate(df[id_col].to_list()):
            auth_id = str(auth_id)
            id_to_eid[auth_id] = i
            eid_to_id[i] = auth_id

        # compute the name map
        eid_to_name = {}  # init global
        for auth_id, name in zip(df[id_col].to_list(), df[subset['name']].to_list()):
            auth_id = str(auth_id)
            eid = id_to_eid[auth_id]
            eid_to_name[eid] = name

        ## compute maps for subset options
        # load or generate the papers matrix
        if force_compute or not os.path.exists(os.path.join(cache_dir, 'papers.npz')):
            col_name = subset['papers']
            if self.verbose:
                print('[DAF]: Generating papers matrix. . .', file=sys.stderr)
            paper_list = list(zip(df[id_col].to_list(), [';'.join(x) for x in df[col_name].to_list()]))
            paper_matrix = self._compute_attribution_matrix(paper_list, id_to_eid, split=';', lower=False)
            sparse.save_npz(os.path.join(cache_dir, 'papers.npz'), paper_matrix)
        else:
            if self.verbose:
                print('[DAF]: Loading papers matrix. . .', file=sys.stderr)
            paper_matrix = sparse.load_npz(os.path.join(cache_dir, 'papers.npz'))   
            
        
        # compute necessary maps for collaborations
        if 'collaborators' in subset:
            
            # load or generate the collaboration matrix
            if force_compute or not os.path.exists(os.path.join(cache_dir, 'collaboration.npz')):
                col_name = subset['collaborators']
                if self.verbose:
                    print('[DAF]: Generating collaboration matrix. . .', file=sys.stderr)
                collab_list = list(zip(df[id_col].to_list(), df[col_name].to_list()))
                collaboration_matrix = self._compute_collaboration_matrix(collab_list, id_to_eid)
                sparse.save_npz(os.path.join(cache_dir, 'collaboration.npz'), collaboration_matrix)
            else:
                if self.verbose:
                    print('[DAF]: Loading collaboration matrix. . .', file=sys.stderr)
                collaboration_matrix = sparse.load_npz(os.path.join(cache_dir, 'collaboration.npz'))   

                
        # compute necessary maps for affiliations
        if 'affiliations' in subset:
                
            # load or generate the affiliation matrix
            if force_compute or not os.path.exists(os.path.join(self.path, 'affiliation.npz')):
                col_name = subset['affiliations']
                if self.verbose: 
                    print('[DAF]: Generating affiliation matrix. . .', file=sys.stderr)
                aff_list = list(zip(df[id_col].to_list(), df[col_name].to_list()))
                affiliation_matrix = self._compute_affiliation_matrix(aff_list, id_to_eid)
                sparse.save_npz(os.path.join(cache_dir, 'affiliation.npz'), affiliation_matrix)
            else:
                if self.verbose:
                    print('[DAF]: Loading affiliation matrix. . .', file=sys.stderr)
                affiliation_matrix = sparse.load_npz(os.path.join(cache_dir, 'affiliation.npz'))               

                
        # load or generate email matrix
        if 'emails' in subset:
            if force_compute or not os.path.exists(os.path.join(self.path, 'email.npz')):
                col_name = subset['emails']
                if self.verbose: 
                    print('[DAF]: Generating email matrix. . .', file=sys.stderr)
                email_list = list(zip(df[id_col].to_list(), df[col_name].to_list()))
                email_matrix = self._compute_attribution_matrix(email_list, id_to_eid, split=';', lower=True)
                sparse.save_npz(os.path.join(cache_dir, 'emails.npz'), email_matrix)
            else:
                if self.verbose:
                    print('[DAF]: Loading email matrix. . .', file=sys.stderr)
                email_matrix = sparse.load_npz(os.path.join(cache_dir, 'email.npz'))
                
            # if empty matrix, set to None
            if not email_matrix.size:
                email_matrix = None
        
        # load or generate unique id matrix
        if 'uid' in subset:
            if force_compute or not os.path.exists(os.path.join(self.path, 'uid.npz')):
                col_name = subset['uid']
                if self.verbose: 
                    print('[DAF]: Generating UID matrix. . .', file=sys.stderr)
                uid_list = list(zip(df[id_col].to_list(), df[col_name].to_list()))
                uid_matrix = self._compute_attribution_matrix(uid_list, id_to_eid, lower=True)
                sparse.save_npz(os.path.join(cache_dir, 'uid.npz'), uid_matrix)
            else:
                if self.verbose:
                    print('[DAF]: Loading UID matrix. . .', file=sys.stderr)
                uid_matrix = sparse.load_npz(os.path.join(cache_dir, 'uid.npz'))  
            
            # if empty matrix, set to None
            if not uid_matrix.size:
                uid_matrix = None
            
        # parallellize the search 
        kchunks = gen_chunks(list(eid_to_id.keys()), self.n_jobs)
        func = self._search
        
        if 1 < self.n_jobs:
            jobs = Parallel(n_jobs=self.n_jobs, require=None, verbose=self.verbose)(
                delayed(func)(kc, eid_to_name, uid_matrix, email_matrix, 
                                  collaboration_matrix, affiliation_matrix, 
                                  min_collab_diff_thresh, min_name_diff_thresh) 
                for kc in kchunks)
        else:
            jobs = []
            for kc in kchunks:
                jobs.append(func(kc, eid_to_name, uid_matrix, email_matrix, collaboration_matrix, 
                                 affiliation_matrix, min_collab_diff_thresh, min_name_diff_thresh))
        
        dup_matches = {}
        for j in jobs:
            dup_matches.update(j)
        
        # check that duplicate authors do not appear on the same paper
        # this is also a sign of a false positive duplicate match
        self.remove_overlapping_duplicates(dup_matches, paper_matrix)
        
        # build duplicate graph for unification
        match_graph = to_graph(list(dup_matches.values()))
        eid_matches = list(connected_components(match_graph))
        
        matches = []
        for match in eid_matches:
            matches.append({eid_to_id[eid] for eid in match})
        return matches


    def _search(self, eids, eid_to_name, uid_matrix, email_matrix, collaboration_matrix, affiliation_matrix, 
                min_collab_diff_thresh, min_name_diff_thresh):
        
        matches = {}
        for eid in eids:

            # init the match set
            # an entry will always contain at least the target author id
            # more than one entry in this set means that duplicates were detected
            eid_matches = {eid}
            
            # get potential duplicates from uid (if used)
            if uid_matrix is not None:
                uid_matches = self.evaluate_membership(uid_matrix, eid)
            else:
                uid_matches = np.array([eid])
                
            # get potential duplicates from email (if used)
            if email_matrix is not None:
                email_matches = self.evaluate_membership(email_matrix, eid)
            else:
                email_matches = np.array([eid])

            # get potential duplicates from collaborations/affiliations (if used)
            if collaboration_matrix is not None:
                collab_matches = self.evaluate_collaborations(collaboration_matrix, eid, thresh=min_collab_diff_thresh)
                collab_matches = self.evaluate_membership(affiliation_matrix, eid, collab_matches)
            else:
                collab_matches = np.array([eid])

            # add uid matches to duplicates
            # if two author ids share the same uid (like an orcid id) then they are the same person
            eid_matches |= set(uid_matches)
            
            # process matches that use the same email 
            # these are likely the same author but a name check still needs to be performed since the email could be a associted with 
            # a department or affiliaition and not the author themselves
            # these matches can be processed together with collaboration matches
            potential_duplicates = np.union1d(email_matches, collab_matches)
            
            # use name matching to finalize match
            target_name = eid_to_name[eid]

            # name check 
            for dup_eid in potential_duplicates:
                if eid == dup_eid:
                    continue
                else:
                    norm_diff = match_name(target_name, eid_to_name[dup_eid], normalize=True)
                    if norm_diff >= min_name_diff_thresh:
                        eid_matches.add(dup_eid)

            matches[eid] = eid_matches
    
        # perform a check to see that no matches have different UIDs
        # this is a sign of a false positive duplicate match
        if uid_matrix is not None:
            self.filter_false_positives(matches, uid_matrix)
            
        return matches


    ### GETTERS / SETTERS

    
    @property
    def n_jobs(self):
        return self._n_jobs
    
    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self._n_jobs = verify_n_jobs(n_jobs)

    @property
    def verbose(self):
        return self._verbose
        
    @verbose.setter
    def verbose(self, verbose):
        if isinstance(verbose, bool):
            self._verbose = int(verbose)  # convert False to 0, True to 1
        elif isinstance(verbose, int):
            if verbose < 0:
                raise ValueError('Integer values for `verbose` must be non-negative!')
            self._verbose = verbose
        else:
            raise TypeError('`verbose` should be of type bool or int!')

