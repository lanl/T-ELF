import ast
import sparse
import numpy as np
import pandas as pd
import multiprocessing
import scipy.sparse as ss


def load_file_as_dict(fn):
    """
    Open a text file and process it to return an enumerated dict. Each line of the file will be returned
    read and stored as a value. The line number is the key 

    Parameters
    ----------
    fn: str, Path
        The path to the text file to be processed

    Returns
    -------
    dict
        The enumerated dict as an output
    """
    with open(fn, 'r') as fh:
        return {i: line.strip() for i, line in enumerate(fh)}


def values_have_duplicates(d):
    """
    The keys of a dictionary are guaranteed to be duplicate free but the values
    are not. This helper function checks if a given dictionary contains any
    value duplicates. Note that the values of the dictionary must be hashable for
    this function to work.

    Parameters
    ----------
    d: dict
        The dictionary that should be checked

    Returns
    -------
    bool
        True if duplicate values are found in d 
    """
    if not isinstance(d, dict):
        raise TypeError(f'Expected d to be a dict. Instead recieved {type(d)}.')
    
    values = d.values()
    try:
        return len(values) != len(set(values))
    except TypeError:
        raise TypeError('Values of d must be hashable to be checked for uniqueness.')


def keys_are_sequential(d):
    """
    Check if the keys of a dictionary are sequential integers starting at 0.

    Parameters
    ----------
    d: dict
        The dictionary that should be checked

    Returns
    -------
    bool
        True if the keys of the dictionary are sequential integers starting at 0.
        False otherwise.
    """
    keys = d.keys()
    return set(keys) == set(range(len(keys)))


def verify_matrix(x):
    """
    Checks if x is a 2D matrix and returns a sparse version of x.
    If x is not a 2D matrix, raises a Exception

    Parameters
    ----------
    x: np.ndarray, scipy.sparse, sparse.sparse
        A 2D matrix (either scipy sparse, pydata sparse, or dense)

    Returns
    -------
    x: pydata.sparse
        A PyData sparse version of x.

    Raises
    ------
    ValueError: 
        x is not a 2-dimensional matrix
    TypeError:
        x is not a matrix
    """
    if isinstance(x, np.ndarray):  # dense matrix
        if len(x.shape) != 2:
            raise ValueError(f'Expected X to be a 2D matrix. Got {x.shape} instead.')
        return sparse.COO.from_numpy(x)  

    elif ss.issparse(x):  # scipy sparse matrix
        if len(x.shape) != 2:
            raise ValueError(f'Expected X to be a 2D matrix. Got {x.shape} instead.')
        return sparse.COO.from_scipy_sparse(x)

    elif isinstance(x, sparse.SparseArray):  # pydata sparse matrix
        if len(x.shape) != 2:
            raise ValueError(f'Expected X to be a 2D matrix. Got {x.shape} instead.')
        return x

    else:
        raise TypeError("X must be a 2D matrix.")


def verify_n_jobs(n_jobs):
    """
    Determines what the number of jobs (cores) should be for a given machine. n_jobs can be set
    on [-cpu_count, -1] ∪ [1, cpu_count]. If n_jobs is negative, (cpu_count + 1) - n_jobs cores
    will be used.

    Parameters
    ----------
    n_jobs: int
        Number of cores requested on this machine

    Returns
    -------
    n_jobs: int
        Adjusted n_jobs representing real value 

    Raises
    ------
    ValueError: 
        n_jobs is outside of acceptable range
    TypeError:
        n_jobs is not an int
    """
    cpu_count = multiprocessing.cpu_count()
    if not isinstance(n_jobs, int):
        raise TypeError(f'n_jobs must be an int')

    limit = cpu_count + n_jobs
    if (n_jobs == 0) or (limit < 0) or (2 * cpu_count < limit):
        raise ValueError(f'n_jobs must take a value on [-{cpu_count}, -1] or [1, {cpu_count}]')
    
    if n_jobs < 0:
        return cpu_count - abs(n_jobs) + 1
    else:
        return n_jobs


def verify_stats(stats):
    """
    Verifies the input stats for the graph stat functions. The stats can either be a list of function
    names to call (with default arguments) or a dict with function names as keys and values as key word
    argument dictionaries. 

    Parameters
    ----------
    stats: list, dict
        The input stats to verify.
    
    Returns
    -------
    None

    Raises
    ------
    ValueError: 
        stats is not valid
    TypeError:
        stats is not correct type
    """
    if isinstance(stats, list):
        if not all(isinstance(item, str) for item in stats):
            raise ValueError("stats list must contain all strings")
    elif isinstance(stats, dict):
        if not all(isinstance(key, str) and isinstance(value, dict) and 
                all(isinstance(k, str) for k in value) for key, value in stats.items()):
            raise ValueError("stats dict must contain string function names and kwargs")
    else:
        raise TypeError("stats must be a list of functions to call or a dictionary with kwargs.")

        
def create_attributes(slic_df, attribute_names=["name"]):
    """
    Creates attributes for scientific literature. These attributes can include name. 
    The name is taken from the SLIC map produced by Orca. 

    Parameters
    ----------
    slic_df: pd.DataFrame
        DataFrame that contains SLIC information. 
    attributes : list
        List of names for the attributes that can be found in SLIC map from Orca.
    
    Returns
    -------
    attributes: dict
        Dictionary that contains the SLIC attributes

    Raises
    ------
    ValueError: 
        stats is not valid
    TypeError:
        stats is not correct type
    """
    attributes = {}
    id_to_name = {k:v for k,v in zip(slic_df.slic_id.to_list(), slic_df.slic_name.to_list())}
    id_to_affiliations = {k:v for k,v in zip(slic_df.slic_id.to_list(), slic_df.scopus_affiliations.to_list())}
    
    for slic_id, slic_name in id_to_name.items():
        attributes[slic_id] = {}
        slic_name = slic_name if not pd.isna(slic_name) else 'Unknown'
        slic_aff = id_to_affiliations[slic_id]
        curr_attributes = {}

        if pd.isna(slic_aff):
            for att_name in attribute_names:
                curr_attributes[att_name] = "Unknown"
        else:
            if isinstance(slic_aff, str):
                slic_aff = ast.literal_eval(slic_aff)
            slic_aff_id = max(slic_aff, key=lambda k: len(slic_aff[k]['papers']))
            for att_name in attribute_names:
                curr_attributes[att_name] = slic_aff[slic_aff_id][att_name]
        
        attributes[slic_id]['name'] = slic_name
        attributes[slic_id]['affiliation_id'] = slic_aff_id

        for att_name in attribute_names:
            attributes[slic_id][att_name] = curr_attributes[att_name]
        if "name" in curr_attributes:
            attributes[slic_id]['affiliation_name'] = curr_attributes["name"]

    return attributes