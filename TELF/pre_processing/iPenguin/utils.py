'''
file:    utils.py
author:  Nick Solovyev
date:    1/22/2023
email:   nks@lanl.gov
desc:    iPenguin utility functions
'''
import re
import urllib
import operator
import datetime
from functools import reduce


def multi_urljoin(*parts):
    """
    Join multiple URL parts into a single, unified URL.

    Parameters:
    -----------
    *parts: str 
        Variable length argument list of URL segments. The first part is considered the base URL.

    Returns:
    --------
    str: 
        A fully constructed URL formed by joining the provided segments. 
        Each segment (except for the base URL) is stripped of leading and trailing slashes,
        and special characters are percent-encoded. Spaces are converted to '+'.

    Example:
    --------
    >>> multi_urljoin('https://example.com', 'a', 'b/', 'c', '/d')
    'https://example.com/a/b/c/d'
    """
    return urllib.parse.urljoin(parts[0], "/".join(urllib.parse.quote_plus(part.strip("/"), safe="/") for part in parts[1:]))


def get_query_param(url: str, param: str) -> str:
    """
    Extracts the value of a specified query parameter from a given URL.
    
    Parameters:
    -----------
    url: str 
        The URL from which to extract the parameter
    param: str
        The name of the query parameter to extract
        
    Returns:
    --------
    str: 
        The value of the specified query parameter, or None if the parameter does not exist.
    """
    query_params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return query_params.get(param, [None])[0]


def get_human_readable_timestamp(epoch):
    """
    Convert an epoch timestamp to a human-readable date-time format in the system's local timezone.

    Parameters:
    -----------
    epoch: int
        The epoch timestamp to be converted. Represents the number of seconds since the Unix 
        epoch (1970-01-01 00:00:00 UTC).

    Returns:
    --------
    str: 
        A string representation of the date and time in the format "YYYY-MM-DD HH:MM:SS".
        The timezone is the the system's local timezone

    Example:
    --------
    >>> get_human_readable_timestamp(1693050245)
    '2023-08-26 05:44:05'
    """
    local_time = datetime.datetime.fromtimestamp(epoch)
    return local_time.strftime('%Y-%m-%d %H:%M:%S %Z%z').strip()


def format_pubyear(s):
    """ 
    Process a year of publication string and provide output in the format expected by Scopus API
    Scopus has three ways of dealing with publication year data. For a given input YEAR, these 
    are == YEAR, < YEAR, and > YEAR. To allow iPenguin to specify which of these settings to use,
    the following notation is used:
        YEAR  -->  publication year == YEAR
        YEAR- -->  publication year > year
        -YEAR -->  publication year < year
    This function is used to validate the input string and help form the Scopus search query
    
    Parameters
    ----------
    s: str
        Year input string
    Returns
    -------
    year: int
        The specified year
    fmat: str
        Format mode needed by Scopus API
    """
    if len(s) > 5 or len(s) < 4:  # improperly formatted year input string. expects YEAR, -YEAR, or YEAR-
        return None, None
    pattern = r'(?P<bef>-)?(?P<year>\d{4})(?P<aft>-)?'
    match = re.search(pattern, s)

    if match:
        year = int(match.group('year'))
        current_year = datetime.datetime.now().year
        
        if 1900 < year <= current_year + 1:
            fmat = 'IS'  # IS: 4-digit year (e.g., "2020")

            if match.group('aft'):
                fmat = 'AFT'  # AFT: year followed by a hyphen (e.g., "2020-")
            elif match.group('bef'):
                fmat = 'BEF'  # BEF: year preceded by a hyphen (e.g., "-2020")

            return year, fmat
    return None, None

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

def gen_chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]

        
def try_int(s):
    try:
        return int(s)
    except ValueError:
        return s
    except TypeError:
        return s