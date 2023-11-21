import os
import re
import tempfile
import warnings
from collections import deque


FILE_DELIMITER = '*' * 100  # line of 100 asterisks


def format_note(kwargs, spacing=16):
    """
    Formats the values from kwargs dictionary into a single string. The first element in kwargs
    is left aligned while the rest are right aligned. Each element is spaced according to the 
    spacing parameter. 
    
    Parameters:
    -----------
    kwargs: dict 
        The dictionary whose values are to be formatted.
    spacing: int
        The number of spaces between each element in the output string. Default is 16 (4 tabs)

    Returns:
    --------
    str: 
        A string with the formatted values from the dictionary, followed by a newline.
    """
    keys = list(kwargs.keys())
    if keys:
        formatted_str = f"{kwargs[keys[0]]:<{spacing}}"
        for key in keys[1:]:
            formatted_str += f"{kwargs[key]:>{spacing}}"
        return f'{formatted_str}\n'
    else:
        warnings.warn("[tELF]: Empty dictionary passed to format_note()", RuntimeWarning)
        return '\n'
        
    
def take_note(notes, path, lock, name="experiment"):
    """
    Writes key-value pairs from a dictionary into a log file in the format 'key = value'. Each string
    is separated with a newline character
    
    This function is safe for use with multiple threads, processes (or nodes in a DFS application).
    Each write operation is protected by a file lock, ensuring that multiple threads or processes 
    do not write to the file simultaneously.

    Parameters:
    -----------
    notes: dict 
        Dictionary containing key-value pairs to be written to the log file.
    path: str 
        The directory path where the log file is located.
    name: str, optional 
        The base name of the log and lock files. Default is "experiment"

    Returns
    -------
    None
    """
    with lock:
        with open(os.path.join(path, f"{name}.log"), 'a+') as fh:
            for key, value in notes.items():
                fh.write(f"{key} = {value}\n")


def append_to_note(notes, path, lock, name="experiment"):
    """
    Writes string from a list of strings called notes into a log file. Each string is separated 
    with a newline character
    
    This function is safe for use with multiple threads, processes (or nodes in a DFS application).
    Each write operation is protected by a file lock, ensuring that multiple threads or processes 
    do not write to the file simultaneously.

    Parameters:
    -----------
    notes: list 
        A list of strings to enter on the 
    path: str 
        The directory path where the log file is located.
    name: str, optional 
        The base name of the log and lock files. Default is "experiment"

    Returns
    -------
    None
    """
    with lock:
        with open(os.path.join(path, f"{name}.log"), 'a+') as fh:
            for note in notes:
                fh.write(f'{note}{os.linesep}')


def take_note_fmat(path, lock, name="experiment", sort_index=0, spacing=16, **kwargs):
    """
    Records some stats into the log file in a formatted manner. These stats are the values stored
    in the kwargs dictionary. The keys signify the type of stat being recorded and the value is 
    the entry into the log file. The first stat in kwargs is left aligned and the rest of the stats
    are right aligned. This function also takes care to sort the current batch of stats being recorded
    to the log file. The stats are sorted based on the value at index sort_index. This sorting is important
    as using multiple processes may cause the recorded stats to be out of order. 
    
    This function is safe for use with multiple threads, processes (or nodes in a DFS application).
    Each write operation is protected by a file lock, ensuring that multiple threads or processes 
    do not write to the file simultaneously.

    Parameters:
    -----------
    path: str 
        The directory path where the log file is located.
    name: str, optional 
        The base name of the log and lock files. Default is "experiment"
    sort_index: int, optional
        The index of the stat value on which to sort the stats
    spacing: int, optional
        The number of spaces between each element in the output string. Default is 16 (4 tabs)
    kwargs: dict
        The stats to record
    Returns
    -------
    None
    """
    note = format_note(kwargs, spacing)  # create the formatted stats string for recording in the log
    with lock:
        delimiter_line = 0
        with open(os.path.join(path, f"{name}.log"), 'r') as fh:
            lines = fh.readlines()
            for i, line in enumerate(reversed(lines)):
                if line.strip() == FILE_DELIMITER:
                    delimiter_line = len(lines) - i - 1  # record line number when last delimiter encountered
                    break

            stat_lines = deque()
            fh_out = tempfile.NamedTemporaryFile('w', delete=False)
            _, fh_out_path = tempfile.mkstemp(dir=path)
            with open(os.path.join(path, f"{name}.log"), 'r') as fh_in, open(fh_out_path, 'w') as fh_out:
                for i, line in enumerate(fh_in):
                    if i < delimiter_line:
                        fh_out.write(line)
                    else:
                        if line.split()[0].isdigit(): # check if the line starts with a numeric entry
                            stat_lines.appendleft((int(line.split()[sort_index]), line))
                        else:
                            fh_out.write(line)

                # sort the lines and write to fh_out
                if note.split()[0].isdigit(): # check if the line starts with a numeric entry
                    stat_lines.appendleft((int(note.split()[sort_index]), note))
                    sorted_lines = [line for idx, line in sorted(stat_lines)]
                    for sorted_line in sorted_lines:
                        fh_out.write(sorted_line)
                else:
                    fh_out.write(note)

            fh_out.close()
            os.replace(fh_out.name, os.path.join(path, f"{name}.log"))
        