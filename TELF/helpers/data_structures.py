from collections import defaultdict
import pandas as pd
import ast
import numpy as np
import itertools
import operator
from functools import reduce

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

def split_num(x, delim=';'):
    if x is None or isinstance(x, float):
        return 0
    else:
        return len(set(x.split(delim)))

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

def find_param_combinations(params):
    """
    Generate combinations of parameters, separating static and dynamic (list-based) parameters.

    This function processes a dictionary of parameters, segregating them into static (non-list) and 
    dynamic (list) parameters. It then generates combinations of the dynamic parameters while
    keeping the static parameters unchanged.

    Parameters:
    -----------
    params: dict
        A dictionary containing both static and dynamic (list-based) parameters.

    Returns:
    --------
    tuple:
        A tuple containing two elements:
        - A list of keys that had dynamic values.
        - A list of dictionaries, each representing a unique combination of parameters.
    """
    static_params = {}
    dynamic_params = {}
    for k,v in params.items():
        if isinstance(v, list):
            dynamic_params[k] = v
        else:
            static_params[k] = v
    
    all_params = []
    for order in unpack_dynamic_params(dynamic_params):
        for dp in all_combinations(order):
            p = static_params.copy()
            for k,v in dp.items():
                p[k] = v
            all_params.append(p)
    if not all_params:
        return [], [static_params]    
    return list(dynamic_params.keys()), all_params

def unpack_dynamic_params(params):
    """
    Unpack and organize parameters based on their order in a nested list structure.

    This function takes a dictionary where keys map to lists (potentially containing nested lists)
    and organizes these lists based on their depth in the nested structure, as determined by
    'find_order'.

    Parameters:
    -----------
    params: dict
        A dictionary where each key maps to a list, potentially containing nested lists.

    Returns:
    --------
    list:
        A list of dictionaries, each representing parameters organized by their depth.

    Raises:
    -------
    ValueError:
        If any value in the params is not a list.

    Example:
    --------
    >>> params = {'param1': [1, [2, 3]], 'param2': [4, 5]}
    >>>  unpack_dynamic_params(params)
    [{'param1': [1], 'param2': [4, 5]}, {'param1': [2, 3]}]
    """
    combinations = {}
    for key, value in params.items():
        if not isinstance(value, list):
            raise ValueError('find_order() expected a list on input')
        order = find_order(value)
        for k,v in order.items():
            if k not in combinations:
                combinations[k] = {}
            combinations[k][key] = v
    return list(combinations.values())

def find_order(lst):
    """
    Determine the depth of items in a nested list structure.

    This function processes a list, potentially containing nested lists, and determines the depth
    of each element. It utilizes 'find_order_helper' to perform the recursive depth calculation.

    Parameters:
    -----------
    lst: list
        The list to process, which may contain nested lists.

    Returns:
    --------
    dict:
        A dictionary with depths as keys and lists of items at each depth as values.

    Example:
    --------
    >>> lst = [1, [2, [3, 4]], 5]
    >>> find_order(lst)
    {0: [1, 5], 1: [2], 2: [3, 4]}
    """
    depth_dict = {}
    for item in lst:
        depth_dict = _find_order_helper(item, 0, depth_dict)
    return depth_dict

def all_combinations(d):
    """
    Generate all possible combinations of values from a dictionary.

    This function takes a dictionary where each key maps to a list of values and generates
    all possible combinations of these values. Each combination is formed by picking one 
    value from each list associated with a key in the dictionary.

    Parameters:
    -----------
    d: dict
        A dictionary where each key maps to a list of values.

    Returns:
    --------
    list:
        A list of dictionaries, each representing a unique combination of values from
        the input dictionary.

    Example:
    --------
    >>> d = {'a': [1, 2], 'b': [3, 4]},
    >>> all_combinations(d)
    [{'a': 1, 'b': 3}, 
     {'a': 1, 'b': 4},
     {'a': 2, 'b': 3}, 
     {'a': 2, 'b': 4}]
    """
    keys = d.keys()
    values = d.values()
    combinations = list(itertools.product(*values))

    output_dicts = []
    for combination in combinations:
        output_dicts.append(dict(zip(keys, combination)))
    return output_dicts

def remove_duplicates(l):
    """
    Remove duplicate elements from a list and return a new list with unique elements.

    Parameters:
    -----------
    l: list
        The list from which duplicate elements are to be removed.

    Returns:
    --------
    list: 
        A new list containing only the unique elements from the original list.

    Example:
    --------
    >>> remove_duplicates([1, 2, 2, 3, 3, 3, 4])
    [1, 2, 3, 4]
    """
    seen, output = set(), []
    for x in l:
        if x not in seen:
            seen.add(x)
            output.append(x)
    return output

def process_countries(row):
    if pd.isnull(row):
        return np.nan
    
    countries = []
    if isinstance(row, str):
        row = ast.literal_eval(row)
    for k, v in row.items():
        countries.append(v.get('country', 'Unknown'))
    countries = list(set(countries))
    return ';'.join(countries)

def process_affiliations(row):
    if pd.isnull(row):
        return np.nan, np.nan
    
    affil_ids = []
    affil_names = []
    if isinstance(row, str):
        row = ast.literal_eval(row)
    for k, v in row.items():
        affil_ids.append(k)
        affil_names.append(v.get('name', 'Unknown'))
    return ';'.join(affil_ids), ';'.join(affil_names)

def sum_dicts(dict_list, n):
    """
    Sums up the values of a list of dictionaries for each key and returns the top-n key-value pairs.
    
    Parameters:
    -----------
    dict_list: list
        A list of dictionaries. All dictionaries should have integer values.
    n: int 
        Number of top key-value pairs to return based on the summed values.
    
    Returns:
    --------
    dict: 
        A dictionary containing the top-n key-value pairs sorted by their summed values in descending order.
    
    Example:
    --------
    >>> dicts = [{"a": 5, "b": 3}, {"a": 3, "b": 4}, {"a": 1}]
    >>> sum_dicts(dicts, 2)
    {'a': 9, 'b': 7}
    """
    result = defaultdict(int)
    for d in dict_list:
        for key, value in d.items():
            result[key] += value
    results = dict(result)
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True)[:n])


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

def chunk_dict(d, n):
    """ 
    Generator function for chunking dict d into n chunks
    
    Parameters
    ----------
    d: dict
        dict to chunk
    n: int
        number of chunks to chunk to
    Returns
    -------
    Generator of dicts
    """
    size = len(d)
    keys = list(d.keys())
    for i in range(n):
        yield {keys[j]: d[keys[j]] for j in range(i, size, n)}  # yield chunk as a new dict

def get_from_dict(d, args):
    """ 
    Try to reduce a dictionary using a list of arguments. For example, given the following dictionary
    d = {0: {1: {2: 'foo'}}}, get_from_dict(d, [0,1,2]) would return 'foo'. 
    
    Parameters
    ----------
    d: dict
        nested dictionary
    args: list
        list of keys by which to access the nested dict
    Returns
    -------
    Value if found, None if error 
    """
    try:
        val = reduce(operator.getitem, args, d)
    except KeyError:
        return None
    except TypeError: 
        return None
    return val

def chunk_Ks(Ks: list, n_chunks=2) -> list:
    # correct n_chunks if needed
    if len(Ks) < n_chunks:
        n_chunks = len(Ks)

    chunks = list()
    for _ in range(n_chunks):
        chunks.append([])

    for idx, ii in enumerate(Ks):
        chunk_idx = idx % n_chunks
        chunks[chunk_idx].append(ii)

    return chunks

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

def chunk_list(l: list, n: int):
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

def chunk(input_list, chunk_size=10000):
    """
    Splits a list into smaller lists (chunks) of a specified size.
    
    Parameters:
    -----------
    input_list: list
        The list to be divided into chunks.
    chunk_size: int, optional
        The maximum size of each chunk. Default is 10000.
    
    Returns:
    --------
    list of lists: 
        Output chunks
    """
    if len(input_list) <= chunk_size:
        return [input_list]
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]

def _find_order_helper(item, depth, depth_dict):
    """
    A helper function for 'find_order', used to determine the depth of items in nested lists.

    This function recursively explores each item in a nested list and records the depth of each
    element. The depth information is stored in a dictionary.

    Parameters:
    -----------
    item: list or any
        The current item to explore, which could be a list or any other type.
    depth: int
        The current depth level in the nested list structure.
    depth_dict: dict
        A dictionary that records elements and their corresponding depths.

    Returns:
    --------
    dict:
        A dictionary with depths as keys and lists of items at each depth as values.
    """
    if isinstance(item, list):
        depth += 1
        for sub_item in item:
            depth_dict = _find_order_helper(sub_item, depth, depth_dict)
    else:
        if depth in depth_dict:
            depth_dict[depth].append(item)
        else:
            depth_dict[depth] = [item]
    return depth_dict

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