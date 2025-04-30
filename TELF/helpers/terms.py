import warnings
import pathlib

from .file_system import load_list
from ..pre_processing.Vulture.default_stop_words import STOP_WORDS
from ..pre_processing.Vulture.default_stop_phrases import STOP_PHRASES

def load_stop_terms(terms, words=True):
    """
    Load the stop words or stop phrases depending on the type of `terms`
    
    If `terms` is a:
        - None: No stop words or phrases provided, use the Vulture default
        - str/pathlib.Path: The path to where the stop words or phrases file is saved
        - list: A list of the stop word or phrases to be used
    
    Parameters:
    -----------
    terms: str, pathlib.Path, list, None
        The terms argument to be processed
    words: bool
        If True, assume processing if for stop words, if False then stop phrases
        
    Returns:
    --------
    list:
        The loaded stop words or stop phrases
    """
    if terms is None:
        return STOP_WORDS if words else STOP_PHRASES
    if isinstance(terms, (str, pathlib.Path)):
        return load_list(terms)
    if isinstance(terms, list):
        return terms

def form_s2_search(term):
    """
    Create an S2 search query for a given term
    
    Parameters:
    -----------
    term: str, dict
        The term to be processed
    
    Returns:
    --------
    str
        S2 API search for term
    """
    if isinstance(term, str):
        query = f'"{term}"'  # directly process string
    elif isinstance(term, dict): # process dict
        if len(term) != 1:
            raise ValueError('Provided `term` is invalid. Dict can only contain one element!')
        
        main_term = next(iter(term))
        values = term[main_term]
        if not isinstance(values, list):
            raise ValueError('Provided `term` is invalid. Expected value to contain list!')
        
        positive_terms = []
        negative_terms = []

        for value in values:
            if value.startswith('+'):  # positive inclusion
                positive_terms.append(f'"{value[1:]}"')
            else:  # negative inclusion
                negative_terms.append(f'"{value}"')

        # construct the query
        query = f'"{main_term}"'
        if positive_terms:
            query += " + (" + " | ".join(positive_terms) + ")"
        if negative_terms:
            query += ' -' + ' -'.join(negative_terms)
    else:
        raise TypeError('Unexpected type for `term`')

    return query

def form_scopus_search(term, code='TITLE-ABS-KEY', english=True, country=None):
    """
    Create a Scopus search query for a given term
    
    Parameters:
    -----------
    term: str, dict
        The term to be processed
    english: bool
        Add condition to only find English papers. Default is True.
    
    Returns:
    --------
    str
        Scopus API search for term
    """
    query = ''
    if english:
        query += 'LANGUAGE(english) AND '
    if country is not None:
        query += f'AFFILCOUNTRY({country}) AND '
    
    if isinstance(term, str):
        query += f'{code}("{term}")'  # directly process string
    elif isinstance(term, dict): # process dict
        if len(term) != 1:
            raise ValueError('Provided `term` is invalid. Dict can only contain one element!')
        
        main_term = next(iter(term))
        values = term[main_term]
        if not isinstance(values, list):
            raise ValueError('Provided `term` is invalid. Expected value to contain list!')
        
        positive_terms = []
        negative_terms = []

        for value in values:
            if value.startswith('+'):  # positive inclusion
                positive_terms.append(f'{code}("{value[1:]}")')
            else:  # negative inclusion
                negative_terms.append(f'{code}("{value}")')

        # construct the query
        query += f'{code}("{main_term}")'
        if positive_terms:
            query += " AND (" + " OR ".join(positive_terms) + ")"
        if negative_terms:
            query += " AND NOT (" + " OR ".join(negative_terms) + ")"
    else:
        raise TypeError('Unexpected type for `term`')
        
    return query

def print_counts(counts):
    """
    Prints the terms and their counts in a pretty manner.

    Parameters:
    -----------
    counts: dict
        A dictionary where keys are terms (strings) and values are counts (numbers).
    """
    max_term_len = max(len(k) for k in counts)
    for term, count in counts.items():
        formatted_term = term.ljust(max_term_len + 1)
        formatted_count = '{:,}'.format(count).rjust(12)
        print(f'{formatted_term} -- {formatted_count}')

def load_terms(path, lower=False):
    """
    Reads contents from a specified text file and splits it into a list. This function
    assumes that the file follows a specific format for Cheetah search terms. Each line 
    should contain a query (token or ngram) and optionally have negative search terms 
    which should be filtered out. These terms should be split with a colon and should
    use a comma if there are multiple negative search terms. 

    Parameters:
    -----------
    path: str
        The path to the file to be read.

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
        with open(path, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()
        
        results = []
        for line in lines:
            line = line.strip()
            if ':' in line:
                term, negatives = line.split(':', 1)
                negatives = negatives.split(',')
                negatives = [n.strip() for n in negatives]
            else:
                term = line
                negatives = []
            
            if lower:
                term = term.lower()
                negatives = [n.lower() for n in negatives]
            
            if negatives:
                results.append({term: negatives})
            else:
                results.append(term)
        return results
            
    except FileNotFoundError:
        warnings.warn(f'Could not find file at "{path}".', RuntimeWarning)
        return []
    
    
def unite_terms(*args):
    """
    Merge multiple lists containing strings and/or dictionaries into a single list.
    
    Parameters:
    -----------
    *args: lists
        Variable length argument containing lists of terms to be merged.

    Returns:
    --------
    list
        Merged terms list
    """
    result = []
    key_values = {}  # dict to hold terms and their corresponding list of negatives
    unique_strings = set()

    for lst in args:
        for item in lst:
            if isinstance(item, str):
                if item not in key_values:
                    unique_strings.add(item)
            elif isinstance(item, dict):
                key, values = list(item.items())[0]
                if key not in key_values:
                    key_values[key] = values
                else:
                    for value in values:  # ensure negatives are unique
                        if value not in key_values[key]:
                            key_values[key].append(value)

                # if dict key matches string in the set, remove that string
                unique_strings.discard(key)

    for key, values in key_values.items():
        result.append({key: values})
    result.extend(unique_strings)
    return result
