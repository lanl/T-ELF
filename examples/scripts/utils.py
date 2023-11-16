import os
import re
import sys
import yaml
import pathlib
import itertools


def load_config(path):
    """
    Load a configuration file using the YAML format.

    This function reads the provided YAML configuration file and returns its contents
    as a dictionary. The PyYAML library is used to safely parse the file.

    Parameters:
    -----------
    path: str
        The file path to the YAML configuration file.

    Returns:
    --------
    dict: 
        Parsed configuration data from the YAML file.

    Raises:
    -------
    YAMLError: 
        If there's an issue parsing the YAML.
    FileNotFoundError: 
        If the provided file path does not exist.
    """
    with open(path, 'r') as fh:
        config = yaml.safe_load(fh)
    return config


def check_path(path):
    """
    Checks and ensures the given path exists as a directory. If the path does not exist, a new 
    directory will be created. If the path exists but is a file, a ValueError will be raised. 
    A TypeError is raised if the provided path is neither a string nor a `pathlib.Path` object.
    Various exceptions are handled to ensure safety and robustness.

    Parameters:
    -----------
    path: str, pathlib.Path
        The path to be checked and ensured as a directory.
    
    Returns:
    --------
    pathlib.Path:
        The validated path
    
    Raises:
    -------
    TypeError:
        If the provided path is neither a string nor a `pathlib.Path` object.
    ValueError: 
        If the path points to an existing file or contains invalid characters.
    OSError:
        If there are issues with file system permissions, IO errors, or other OS-related errors.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not isinstance(path, pathlib.Path):
        raise TypeError(f'Unsupported type "{type(path)}" for `path`')

    try:
        if path.exists():
            if path.is_file():
                raise ValueError('`path` points to a file instead of a directory')
        else:
            path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise OSError('Insufficient permission to access or create the directory')
    except FileNotFoundError:
        raise ValueError('Invalid path or path contains illegal characters')
    except OSError as e:
        raise OSError(f'Error accessing or creating the directory: {e}')
    return path


def load_list(path, lower=False):
    """
    Reads contents from a specified text file and splits it into a list. The file contents
    are split using the following delimiters: comma, semicolon, and newline. Empty 
    strings and leading/trailing whitespaces are removed from the list.

    Parameters:
    -----------
    path: str
        The path to the file to be read.
    lower: bool
        Flag that determines if the contents of the file should be converted to lowercase

    Returns:
    --------
    list
        The processed file contents
            
    Warnings:
    ---------
    RuntimeWarning
        If the file is not found
        
    Raises:
    -------
    UnicodeDecodeError:
        If file is not a text file
    IsADirectoryError:
        If path is given to a directory, not a file
    """
    try:
        with open(path, 'r') as fh:
            contents = fh.read()
            items = [x.strip() for x in re.split('[,;\n]+', contents) if x.strip()]
            if lower:
                return [x.lower() for x in items]
            else:
                return items
            
    except FileNotFoundError:
        warnings.warn(f'Could not find file at "{path}".', RuntimeWarning)
        return []

    
def process_terms(file_path):
    """
    Processes a CSV file and creates two dictionaries based on the first three columns.
    This function is intended to be used for loading and processing a terms CSV file
    where the columns are term to be substituted, replacement term, higlighting weight.

    Parameters:
    -----------
    file_path: str
        The file path to the CSV file.

    Returns:
    --------
    tuple:
        A tuple containing two dictionaries:
        - The first dictionary maps keys from column 1 to values from column 2.
        - The second dictionary maps keys from column 2 to values from column 3.

    Raises:
    -------
    FileNotFoundError:
        If the CSV file is not found at the specified path.
    """
    substitution_map = {}
    highlighting_map = {}

    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) >= 3:  # Ensure there are at least 3 columns
                    substitution_map[row[0]] = row[1]
                    highlighting_map[row[1]] = row[2]
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: "{file_path}"!')
    return substitution_map, highlighting_map


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