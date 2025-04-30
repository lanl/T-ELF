import pathlib
import os
import warnings
import re
import csv
import shutil
import fnmatch

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
                    highlighting_map[row[1]] = int(row[2])
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: "{file_path}"!')
    return substitution_map, highlighting_map


def find_files(path, start_with, end_with='.csv'):
    """
    Find all files in the given path that start with the specified prefix 
    and end with the specified suffix.

    Parameters:
    -----------
    path: str
        Directory path to search in.
    start_with: str
        Prefix to search for.
    end_with: str
        Suffix to search for.

    Returns:
    --------
    list: 
        List of files that match the criteria.
    """
    matched_files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and \
           file.startswith(start_with) and file.endswith(end_with):
            matched_files.append(os.path.join(path, file))
    return matched_files

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

def list_files(path, file_extension='.csv'):
    """
    Creates a dictionary of file names in a specified directory that have the provided file extension.
    
    Parameters:
    -----------
    path: str
        The path to the directory to be scanned.
    
    file_extension: str
        The file extension to be included in the results (e.g., '.csv').
        The dot should be included in the extension string. Default is '.csv'

    Returns:
    --------
    dict
        Dictionary with integer keys starting from 0 and file names as values.

    Raises:
    -------
    NotADirectoryError:
        If provided path is not a directory.
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f'Provided path "{path}" is not a directory.')
    files = [f for f in os.listdir(path) if f.endswith(file_extension)]
    return {i: f for i, f in enumerate(files)}

def check_path(path):
    """
    Checks and ensures the given path exists as a directory. If path does not exist, a new directory
    will be created. If the path exists but is a file, a ValueError will be raised. A TypeError is
    raised if the provided path is neither a string nor a `pathlib.Path` object.

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
        If the path points to an existing file.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not isinstance(path, pathlib.Path):
        raise TypeError(f'Unsupported type "{type(path)}" for `path`')
    path = path.resolve()
    if path.exists() and path.is_file():  # handle the path already existing as file
        raise ValueError('`path` points to a file instead of a directory')
    if not path.exists():
        path.mkdir(parents=True)  # parents=True ensures all missing parent directories are also created
    else:
        warnings.warn(f'The "{path}" already exists and will be overwritten!')
    return path

def check_path_var(path, var_name):
    """
    Validates and prepares a directory path.

    Ensures that the provided `path` is either a string or a `pathlib.Path` object.
    If the path exists and is a file, raises a ValueError. If the path does not exist,
    it will be created as a directory (including any necessary parent directories).

    Parameters
    ----------
    path : str or pathlib.Path
        The path to validate or create as a directory.
    var_name : str
        The variable name used in error messages for clarity.

    Raises
    ------
    TypeError
        If `path` is not a string or `pathlib.Path` object.
    ValueError
        If the path exists and is a file instead of a directory.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not isinstance(path, pathlib.Path):
        raise TypeError(f'Unsupported type "{type(path)}" for `path`')
    path = path.resolve()
    if path.exists():
        if path.is_file():
            raise ValueError(f'`{var_name}` points to a file instead of a directory')
    else:
        path.mkdir(parents=True, exist_ok=True)

def copy_file(source_dir, dest_dir, file_pattern):
    """
    Copies a file matching a specific pattern from the source directory to the destination directory.

    Parameters:
    -----------
    source_dir: str
        The source directory where the file is located.
    dest_dir: str
        The destination directory to copy the file to.
    file_pattern: str
        The pattern to match the filename

    Raises:
    -------
    FileNotFoundError:
        If the source or destination directory doesn't exist or isn't a directory.
    IOError:
        If an error occurs during file copying.
    RuntimeError:
        If no or multiple files matching the pattern are found.
    """
    if not os.path.isdir(source_dir):  # check if source_dir exists and is dir
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist or is not a directory.")
    if not os.path.isdir(dest_dir):  # check if dest_dir exists and is directory
        raise FileNotFoundError(f"Destination directory '{dest_dir}' does not exist or is not a directory.")

    # find files matching pattern
    matching_files = [f for f in os.listdir(source_dir) if fnmatch.fnmatch(f, file_pattern)]

    # check if exactly one matching file is found
    if len(matching_files) != 1:
        raise RuntimeError(f"Expected one file matching '{file_pattern}', found {len(matching_files)}.")

    # copy file
    try:
        shutil.copy(os.path.join(source_dir, matching_files[0]), dest_dir)
    except IOError as e:
        raise IOError(f"Failed to copy file: {e}")