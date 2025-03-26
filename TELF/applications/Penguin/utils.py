import uuid
import warnings
import operator
import rapidfuzz
import numpy as np
import pandas as pd
import multiprocessing
from functools import reduce


def try_int(s):
    """
    Attempt to convert a given value to an integer.

    This function tries to convert the input 's' to an integer. If the conversion
    is successful, the integer value is returned. If the conversion fails due to
    a TypeError or ValueError, return original value.

    Parameters:
    -----------
    s: str, int, or other types
        The value to be converted to an integer.

    Returns:
    --------
    int or original type

    Examples:
    ---------
    >>> try_int('5')
    5
    >>> try_int("abc")
    "abc"
    """
    try:
        return int(s)
    except ValueError:
        return s
    except TypeError:
        return s


def get_from_dict(d, args):
    """ 
    Try to reduce a dictionary using a list of arguments. 
    
    Parameters:
    -----------
    d: dict
        nested dictionary
    args: list
        list of keys by which to access the nested dict
    
    Returns:
    --------
    Value if found, None if error 
    
    Examples:
    ---------
    >>> d = {0: {1: {2: 'foo'}}}
    >>> get_from_dict(d, [0,1,2])
    'foo'
    """
    try:
        val = reduce(operator.getitem, args, d)
    except KeyError:
        return None
    except TypeError: 
        return None
    return val
        

def transform_dict(d):
    """
    Transforms a dictionary into a list of single-entry dictionaries if the dictionary has more than one entry.

    Parameters:
    -----------
    d: dict
        The dictionary to be transformed.

    Returns:
    --------
    dict or list: 
        If the input dictionary has only one entry, it is returned unchanged. If the input dictionary has more than one entry, 
        it is transformed into a list of single-entry dictionaries.

    Examples:
    ---------
    >>> transform_dict({'a': 1})
    {'a': 1}
    >>> transform_dict({'a': 1, 'b': 2})
    [{'a': 1}, {'b': 2}]
    """
    if len(d) <= 1:
        return d
    else:
        return [{k: v} for k, v in d.items()]


def verify_n_jobs(n_jobs):
    """
    Determines what the number of jobs (cores) should be for a given machine. n_jobs can be set
    on [-cpu_count, -1] ∪ [1, cpu_count]. If n_jobs is negative, (cpu_count + 1) - n_jobs cores
    will be used.

    Parameters:
    -----------
    n_jobs: int
        Number of cores requested on this machine

    Returns:
    --------
    n_jobs: int
        Adjusted n_jobs representing real value 

    Raises:
    -------
    ValueError: 
        n_jobs is outside of acceptable range
    TypeError:
        n_jobs is not an int
    """
    cpu_count = multiprocessing.cpu_count()
    if not isinstance(n_jobs, int):
        raise TypeError(f'`n_jobs` must be an int.')

    limit = cpu_count + n_jobs
    if (n_jobs == 0) or (limit < 0) or (2 * cpu_count < limit):
        raise ValueError(f'`n_jobs` must take a value on [-{cpu_count}, -1] or [1, {cpu_count}].')
    
    if n_jobs < 0:
        return cpu_count - abs(n_jobs) + 1
    else:
        return n_jobs
    

def verify_attributes(attributes, default, name='attributes'):
    """
    Validates and merges a given attributes dictionary with a set of default attributes.

    This function ensures that the provided `attributes` dictionary is valid. It checks that:
    - `attributes` is a dictionary.
    - Each key in `attributes` exists within the `default` dictionary.
    - The values associated with each key are either strings, tuples of strings, or lists of strings.
    - For the key 'id', if present, its value must specifically be a string (to enforce uniqueness).

    If `attributes` is None, the function returns the `default` dictionary. If `attributes` is valid,
    it merges `attributes` with `default`, with values from `attributes` taking precedence.

    Parameters:
    -----------
    attributes: dict or None
        The attributes dictionary provided by the user. If None, the function returns the `default` dictionary.
    default: dict
        The default attributes dictionary against which `attributes` is validated and merged.
    name: str, optional
        The name of the attributes dictionary, used in error messages to indicate the source of validation errors.
        Default is 'attributes'.

    Returns:
    --------
    dict
        A validated and merged dictionary of attributes, combining both `attributes` and `default`.

    Raises:
    -------
    ValueError
        If `attributes` is not a dictionary, or if any value in `attributes` is not a string, a tuple of strings, or a list 
        of strings. Additionally, raises a ValueError if the 'id' key is present but its value is not a string.
    KeyError
        If any key in `attributes` is not found in `default`.
    """
    if attributes is None:
        return default

    if not isinstance(attributes, dict):
        raise ValueError(f'`{name}` must be a dictionary.')

    for key, value in attributes.items():
        if key not in default:
            raise KeyError(f'Encountered unknown search field {key!r}. Supported fields are {list(default.keys())}.')

        if not isinstance(value, (str, tuple, list)) or \
                (isinstance(value, (tuple, list)) and not all(isinstance(item, str) for item in value)):
            raise ValueError("Search attribute values must be strings, tuples of strings, or lists of strings.")
        if key == 'id' and not isinstance(value, str):
            raise ValueError("`id` cannot take on multiple values in search query.")
            
    # merge user attributes with default attributes
    validated_attributes = {**default, **attributes}
    return validated_attributes
    

def gen_chunks(l, n):
    """
    Yield n number of sequential chunks from the list l.

    This function divides the list l into n chunks of as equal size as possible. Each chunk is yielded 
    sequentially. If the list cannot be divided evenly, the earlier chunks will have one more element 
    than the later ones.

    Parameters:
    -----------
    l: list
        The list from which to create chunks.
    n: int
        The number of chunks to create.

    Yields:
    -------
    list
        A chunk of the original list l.

    Examples
    --------
    >>> list(gen_chunks([1, 2, 3, 4, 5], 2))
    [[1, 2, 3], [4, 5]]
    >>> list(gen_chunks([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]

        
def match_lists(list_a, list_b, key_a='@afid', key_b='$'):
    """
    Creates a dictionary by matching elements from two lists based on specified keys.

    This function iterates over the elements of the two given lists, `list_a` and `list_b`. For each 
    index, it extracts values corresponding to `key_a` from `list_a` and `key_b` from `list_b`. These 
    values are then  used to form key-value pairs in the resulting dictionary. If an element at a given 
    index is not a dictionary  or does not contain the specified key, None is used as the value for the 
    corresponding key.

    Parameters
    ----------
    list_a: [dict]
        The first list containing dictionaries from which to extract keys.
    list_b: [dict]
        The second list containing dictionaries from which to extract values.
    key_a: hashable
        The key to extract values from the dictionaries in `list_a`.
    key_b: hashable
        The key to extract values from the dictionaries in `list_b`.

    Returns
    -------
    dict
        A dictionary with keys from `list_a` and corresponding values from `list_b`.

    Examples
    --------
    >>> list_a = [{'id': 1, 'value': 'A'}, {'id': 2, 'value': 'B'}]
    >>> list_b = [{'id': 1, 'desc': 'First'}, {'id': 2, 'desc': 'Second'}]
    >>> match_lists(list_a, list_b, 'value', 'desc')
    {'A': 'First', 'B': 'Second'}
    """
    result = {}
    len_a = len(list_a)
    len_b = len(list_b)
    
    for i in range(max(len_a, len_b)):
        value_a = list_a[i][key_a] if i < len_a and isinstance(list_a[i], dict) and key_a in list_a[i] else None
        value_b = list_b[i][key_b] if i < len_b and isinstance(list_b[i], dict) and key_b in list_b[i] else None
        result[value_a] = value_b        
    return result


def drop_duplicates(df, col):
    """
    Drop duplicates from a DataFrame based on a specified column while preserving rows with NaN values 
    in that column. Among duplicates, rows with fewer NaN values in other columns are prioritized.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The input DataFrame from which duplicates are to be removed.
    col: str
        The name of the column based on which duplicates should be identified and dropped.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicates removed 
    """
    df_nan_col = df[df[col].isna()]  # separate out rows where the column has NaN
    df = df.dropna(subset=[col]).copy()
    df['nan_count'] = df.isna().sum(axis=1)  # add helper column for nan counts
    df = df.sort_values(by=[col, 'nan_count'])  # sort by col and then by nan_count
    
    # drop duplicates (prioritizing papers with fewer NaNs due to sorting)
    df = df.drop_duplicates(subset=col, keep='first')
    df = df.drop(columns='nan_count')
    df = pd.concat([df, df_nan_col], axis=0).reset_index(drop=True)
    return df


def reorder_and_add_columns(df, order, fill_value=np.nan):
    """
    Reorders the columns of the dataframe based on the provided column order list.
    Adds columns that do not exist in the original dataframe with the specified fill value.
    Columns in the dataframe that are not in the `order` will be placed at the end.

    Parameters:
    -----------
    df: pd.DataFrame
        The input dataframe.
    order: list
        The desired column order including non-existing columns.
    fill_value: Any
        The value to fill in for the non-existing columns. Default is NaN.

    Returns:
    --------
    pd.DataFrame: 
        The dataframe with columns reordered and non-existing columns added.
    """
    # Ensure all columns in `order` are included in the result, adding missing ones
    columns_to_add = [col for col in order if col not in df.columns]

    # Add missing columns to `df` with the specified fill value
    for col in columns_to_add:
        df[col] = fill_value

    # Determine the final column order
    final_order = [col for col in order if col in df.columns] + \
                  [col for col in df.columns if col not in order]

    # Reorder columns in `df` according to `final_order`
    return df[final_order]


def merge_frames(df1, df2, key):
    """
    Merges two dataframes on a specified key. The common columns between the dataframes (excluding the key) 
    are identified and combined, with preference given to df1.
    
    Parameters:
    -----------
    df1: pd.DataFrame
        The first dataframe.
    df2: pd.DataFrame
        The second dataframe.
    key: str
        The column name to merge the dataframes on. This column should be present in both dataframes.
    
    Returns:
    --------
    pd.DataFrame: 
        A merged dataframe with combined columns from both dataframes
    """
    # find common columns excluding the key
    common = [col for col in df1.columns if col in df2.columns and col != key]
    
    # merge frames
    merged_df = df1.merge(df2, on=key, how='outer', suffixes=('_df1', '_df2'))
    
    # combine common columns prioritizing df1
    for col in common:
        with warnings.catch_warnings():
            # TODO: pandas >= 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
            warnings.filterwarnings("ignore", category=FutureWarning)
            merged_df[col] = merged_df[col + '_df1'].combine_first(merged_df[col + '_df2'])
        merged_df.drop(columns=[col + '_df1', col + '_df2'], inplace=True)
    return merged_df


def match_frames_text(df1, df2, col, min_len=10, threshold=0.05):
    """
    Matches rows from two dataframes based on text similarity in a specified column.
    
    Parameters:
    -----------
    df1: pd.DataFrame
        The first dataframe.
    df2: pd.DataFrame
        The second dataframe.
    col: str
        The column name on which to perform text similarity matching.
    min_len: int 
        The minimum length of a title to be matched. 
    threshold: float
        Minimum normalized Indel similarity score for matching entries. This value should be on [0, 1). 
        The smaller the `threshold`, the better the text match. Default is 0.05 which roughly 
        corresponds to a 95% match based on normalized Indel similarity. 

    Returns:
    --------
    tuple
        Two dataframes with a new column indicating matched keys.
    """
    match_col = f'{col}_key'
    df1[match_col] = [None] * len(df1)  # init empty columns
    df2[match_col] = [None] * len(df2)

    for idx1, row1 in df1.iterrows():
        if pd.isna(row1[col]):
            continue
        elif len(row1[col]) < min_len:
            continue
        extract = rapidfuzz.process.extractOne(row1[col], df2[col], score_cutoff=threshold, 
                                               scorer=rapidfuzz.distance.Indel.normalized_distance)
        if extract is None:
            continue
        else:
            match, score, idx2 = extract
            match_key = str(uuid.uuid4())
            df1.at[idx1, match_col] = match_key
            df2.at[idx2, match_col] = match_key
            
    return df1, df2, match_col


def create_text_query(attributes, targets, join):
    """
    Constructs a MongoDB query to search for specified target terms within given attributes of documents.

    This method generates a query that searches for each term in 'targets' within each field listed in 
    'attributes'. The search is case-insensitive and looks for partial matches of the target terms within 
    the attribute values. The relationship between different target terms can be specified using `join`. 
    This can be 'AND' or 'OR'. Relations between different attributes are always joined with 'OR'. For
    example, if `attributes` is ['title', 'abstract'] then the query will look for target text in either
    the title or the abstract.

    Parameters:
    -----------
    attributes: [str]
        A list of attribute names (fields) within the documents to search for the target terms.
    targets: [str]
        A list of target terms to search for within the specified attributes.
    join: str
        The logical join operator to combine the search queries for different attributes. 
        Supported values are 'OR' and 'AND'.

    Returns:
    --------
    dict
        A MongoDB query dictionary ready to be used with find() or similar methods.

    Notes:
    ------
    - The search is case-insensitive due to the '$options': 'i' flag in the regex queries.
    - Partial matches are allowed, meaning that documents will be matched if they contain 
      any part of the target terms within the specified attributes.
    """
    queries = []
    for attr in attributes:
        attr_queries = []
        for term in targets:
            attr_queries.append({attr: {'$regex': f'.*{term}.*', '$options': 'i'}})
        attr_queries = {f'${join.lower()}': attr_queries}
        queries.append(attr_queries)

    if len(queries) == 1:
        return queries[0]
    else: 
        return {'$or': queries}
    
    
def create_general_query(attributes, targets, join):
    queries = []
    for attr in attributes:
        attr_queries = []
        for term in targets:
            attr_queries.append({attr: {'$regex': f'.*{term}.*', '$options': 'i'}})
        attr_queries = {f'${join.lower()}': attr_queries}
        queries.append(attr_queries)

    if len(queries) == 1:
        return queries[0]
    else: 
        return {'$or': queries} 
