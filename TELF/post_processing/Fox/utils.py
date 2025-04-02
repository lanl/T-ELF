'''
File:   utils.py
Author: Nick Solovyev
Email:  nks@lanl.gov
Date:   10/04/23
'''
import os
import re
import sys
import ast
import csv
import pathlib
import warnings
import itertools
import wordcloud
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from itertools import combinations
from collections import Counter, defaultdict
from ...pre_processing.Vulture.modules import SimpleCleaner
from ...pre_processing.Vulture.modules import LemmatizeCleaner
from ...pre_processing.Vulture.modules import SubstitutionCleaner
from ...pre_processing.Vulture.modules import RemoveNonEnglishCleaner
from ...pre_processing.Vulture.default_stop_words import STOP_WORDS
from ...pre_processing.Vulture.default_stop_phrases import STOP_PHRASES


    
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


def merge_frames(df1, df2, col, common_cols, key=None, remove_duplicates=False, 
                 remove_nan=False, order=None):
    """
    Merges two dataframes based on a specified column and combines values of common columns.
    
    This function takes two dataframes `df1` and `df2`, and merges them based on the specified 
    column `col`. For the columns listed in `common_cols`, it combines values, prioritizing 
    non-NaN values from `df1`. The resulting dataframe can optionally be filtered and reordered.

    Parameters:
    -----------
    df1: pd.DataFrame
        Primary DataFrame.
    df2: pd.DataFrame
        Secondary DataFrame
    col: str
        The column name based on which the two dataframes should be merged.
    common_cols: list[str]
        List of column names that exist in both dataframes and should be combined.
    key: str, optional
        Column name based on which rows with NaN values should be dropped and duplicates removed.
        If None, this is not done. Default is None.
    remove_duplicates: bool
        If key is set, remove the duplicates in the column associated with this key
    remove_nan: bool
        If key is set, remove the nans in the column associated with this key
    order: list[str], optional
        List of column names specifying the desired order of columns in the resulting dataframe.
    
    Returns:
    --------
    pd.DataFrame: 
        Merged DataFrame
    """
    df1[col] = df1[col].str.lower()
    df2[col] = df2[col].str.lower()

    merged_df = df1.merge(df2, on=col, how='outer', suffixes=('_df1', '_df2'))

    # use common_cols to combine the columns and drop the temporary ones
    for common_col in common_cols:
        col_df1 = common_col + '_df1'
        col_df2 = common_col + '_df2'
        
        merged_df[common_col] = merged_df[col_df1].combine_first(merged_df[col_df2])
        merged_df = merged_df.drop(columns=[col_df1, col_df2])

    if key is not None:  # remove nans and duplicates (optional)
        if remove_nan:
            merged_df = merged_df.dropna(subset=[key])
        if remove_duplicates:
            merged_df = drop_duplicates(merged_df, key)
    if order is not None:  # reset order (optional)
        merged_df = merged_df[order].reset_index(drop=True)

    return merged_df


def get_papers(df, authors, col):
    """
    Filters a DataFrame for papers based on the presence of a highlighted author from authors
    
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to be filtered.
    authors: set
        A set or any iterable containing author id strings to search for in the DataFrame. If the provided 
        selection is not a set, it will be converted to one.
    col: str
        The column name in the DataFrame where author ids from `authors` should be searched.
    
    Returns:
    --------
    pd.DataFrame: 
        A subset of the original DataFrame containing rows where any highlighted author from authors 
        is found in the specified column's values.
    """
    if not isinstance(authors, set):
        selection = set(authors)
    def is_in_set(s):
        return any(x in authors for x in s.split(';'))
    return df[df[col].apply(is_in_set)].copy()


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


def create_wordcloud(df, col, n=30, save_path=None, figsize=(600,600)):
    """
    Generate and display a word cloud from a specified DataFrame column.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the text data.
    col: str
        The column name in the DataFrame with the text data.
    n: int, optional
        The number of top words to consider for the word cloud. Default is 30.
    save_path: str, optional
        Path to save the word cloud image. If not provided, the image is displayed but not saved.
    figsize: (int, int), optional
        A tuple specifying the size of the displayed image in inches. Default is (600,600).
    
    Returns:
    --------
    None: 
        Displays the word cloud using matplotlib. If save_path is provided, the word cloud is saved to the 
        specified path and not displayed.
    """
    # calculate the frequencies
    tokens    = [x.split() for x in df[col].to_list() if not pd.isna(x)]
    top_words = [dict(Counter(x)) for x in tokens]
    top_words = sum_dicts(top_words, n=n)

    # generate the word cloud image
    figsize_x, figsize_y = figsize
    wc = wordcloud.WordCloud(background_color="white", width=figsize_x*100, height=figsize_y*100)
    wc.generate_from_frequencies(top_words)
    
    # display the word cloud using matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
        
        
def filter_author_map(df, *, country=None, affiliation=None):
    """
    Filter a SLIC author map by country, affiliation id, or both
    
    Parameters:
    -----------
    df: pd.Data
        The SLIC author map being filtered
    country: str
        Some country by which to filter.
    affiliation: str
        The Scopus affiliation by which to filter
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    ids = set(df.index.to_list())
    for index, row in df.iterrows():
        affiliations = row.scopus_affiliations
        if pd.isna(affiliations):
            ids.remove(index)
            continue
            
        if isinstance(affiliations, str):
            affiliations = ast.literal_eval(affiliations)
        
        countries = {x['country'].lower() for x in affiliations.values()}
        matched = False
        for c in countries:
            if country.lower() in c:  # substring comparison
                matched = True
        if not matched:
            ids.remove(index)
        
        affiliation_ids = set(affiliations.keys())
        if affiliation is not None and affiliation not in affiliation_ids:
            ids.remove(index)
    
    return df.iloc[list(ids)]


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


def get_id_to_name(df, name_col, id_col):
    """
    Creates a map of author id to name. 
    The fist occurence of an id, name pair is recorded.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The input DataFrame.
    name_col: str
        The column that contains author names
    id_col: str
        The column contains the author ids
        
    Returns:
    --------
    dict:
        The id to name map
    """
    id_to_name = {} 
    name_list = [x.split(';') for x in df[name_col].to_list()]
    id_list = [x.split(';') for x in df[id_col].to_list()]
    for id_sublist, name_sublist in zip(id_list, name_list):
        for id_, name_ in zip(id_sublist, name_sublist):
            if id_ not in id_to_name:
                id_to_name[id_] = name_
    return id_to_name


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

    
def init_vulture_steps(settings):
    """
    Parse the Vulture JSON to create a list of Vulture cleaning steps
    
    Parameters:
    -----------
    settings: dict
        The Vulture settings
        
    Returns:
    --------
    list:
        List of cleaning steps (in order) to be executed by Vulture
    """
    VULTURE_MAP = {
        'SimpleCleaner': SimpleCleaner,
        'LemmatizeCleaner': LemmatizeCleaner,
        'SubstitutionCleaner': SubstitutionCleaner,
        'RemoveNonEnglishCleaner': RemoveNonEnglishCleaner,
    }
    
    steps = []
    for s in settings:
        cleaner = s.get('type')
        if cleaner is None:
            raise ValueError('Config file is missing `type` for Vulture step!')
        if cleaner not in VULTURE_MAP:
            raise ValueError(f'Unknown cleaner "{cleaner}"!')
        
        args = s.get('init_args')
        if args is None:
            raise ValueError('Config file is missing `init_args` for Vulture step!')
        
        # process stop words / stop phrases for SimpleCleaner
        if cleaner == 'SimpleCleaner':
            args['stop_words'] = load_stop_terms(args.get('stop_words'), words=True)
            args['stop_phrases'] = load_stop_terms(args.get('stop_phrases'), words=False)
        
        # create the cleaner object
        steps.append(VULTURE_MAP[cleaner](**args))
    return steps



def create_authors_graph(df, col):
    G = nx.Graph()
    for index, row in df.iterrows():
        if pd.isna(row[col]):
            continue

        # add edges between every pair of authors for given paper
        authors = str(row[col]).split(';')
        for a, b in combinations(authors, 2):
            if G.has_edge(a, b): # increase weight by 1 if edge already exists
                G[a][b]['weight'] += 1
            else: # new edge, set weight = 1
                G.add_edge(a, b, weight=1)
    return G


def plot_authors_graph(df, id_col='s2_author_ids', name_col='s2_authors', title='Co-Authors Graph',
                       width=900, height=900, max_node_size=50, min_node_size=3):
    G = create_authors_graph(df, id_col)
    pos = nx.spring_layout(G)  # position nodes using networkx's spring layout
    name_map = get_id_to_name(df, name_col, id_col)

    # global normalization for node sizes
    degrees = dict(G.degree())
    min_degree = min(degrees.values())
    max_degree = max(degrees.values())
    if max_degree == min_degree:
        normalized_sizes = [max_node_size for _ in degrees]
    else:
        normalized_sizes = [
            min_node_size + (degrees[node] - min_degree) * (max_node_size - min_node_size) / (max_degree - min_degree)
            for node in G.nodes()
        ]
    size_map = {node: size for node, size in zip(G.nodes(), normalized_sizes)}

    # find connected components, sorted by size in descending order
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    # create traces for each component
    traces = []
    for i, component in enumerate(components):
        component_label = f"Component {i}"

        # edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges(component):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),  #edge color is a shade of gray
            hoverinfo='none',
            mode='lines',
            showlegend=False,
            legendgroup=component_label
        )
        
        node_x = [pos[node][0] for node in component]
        node_y = [pos[node][1] for node in component]
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            text=[f"<b>{name_map.get(id_, 'Unknown')}</b><br>{id_}" for id_ in component],
            hoverinfo='text',
            marker=dict(
                size=[size_map[node] for node in component],
                line_width=2
            ),
            name=component_label,
            legendgroup=component_label,
            showlegend=True
        )

        traces.extend([edge_trace, node_trace])

    # define the figure
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            width=width,
            height=height,
            plot_bgcolor='rgba(0,0,0,0)',  # transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # transparent border
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=12,
                ),
                itemsizing='constant'  # Ensure consistent size in the legend
            )
        )
    )
    return fig


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


def create_country_map(col):
    out = {}
    for row in col:
        if pd.isnull(row):
            continue

        if isinstance(row, str):
            row = ast.literal_eval(row)
        for k, v in row.items():
            if out.get(k, 'Unknown') == 'Unknown':
                out[k] = v.get('country', 'Unknown')        
    return out


def get_top_author_ids(df, col, verbose=False):
    """
    Counts occurrences of semi-colon separated values in an author_ids DataFrame column.

    This function iterates over the entries in the given author_ids column of the DataFrame. It treats 
    each entry as a semi-colon separated list of values and counts the occurrences of each unique value.
    It ignores any NaN or null entries.

    Parameters:
    -----------
    df: pd.DataFrame
        The input DataFrame.
    col: str
        The column name in which to count the occurrences.

    Returns:
    --------
    collections.Counter: 
        A Counter object with keys being the author ids and values being their respective counts.
    """
    c = Counter()
    for author_ids in tqdm(df[col].to_list(), disable=not verbose):
        if pd.isna(author_ids): 
            continue
        for aid in author_ids.split(';'):
            c[aid] += 1
    return c


def get_top_authors(df, author_col='slic_authors'):
    """
    Retrieves the top authors and their corresponding papers counts

    This function assumes there is a pair of columns in the DataFrame that contains the joint author ids
    and author names. The get_top_author_ids() function is used to determine the top author IDs and then maps 
    those IDs to their respective names. If an ID doesn't have a corresponding name, it is labeled as 'Unknown'.

    Parameters:
    -----------
    df: pandas.DataFrame
        The input DataFrame.
    col: str
        The column name containing semi-colon separated author names.

    Returns:
    --------
    list of tuple: 
        A sorted list (based on the count) of tuples where each tuple contains:
          1. The name corresponding to an author ID (or 'Unknown' if not found).
          2. The author ID.
          3. The paper count of that author ID.
    """
    id_col = f'{author_col[:-1]}_ids'
    top_author_ids = get_top_author_ids(df, id_col)
    df = df.dropna(subset=[author_col, id_col])
    
    # create map of id to name
    authID_to_name = get_id_to_name(df, author_col, id_col)
    
    top_authors = []
    for auth in top_author_ids:
        paper_count = top_author_ids[auth]
        top_authors.append((authID_to_name.get(auth, 'Unknown'), auth, paper_count))
    top_authors.sort(key=lambda x: x[2], reverse=True)
    return top_authors


def generate_affiliations_map(df, target_auth_id, auth_col='slic_authors'):
    """
    Helper function for computing a map of affiliations for a single scopus author. 

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the authors papers

    Returns
    -------
    affiliations_map: dict,
        The created map
    """
    
    if auth_col == 'slic_authors':
        auth_id_col = 'slic_author_ids'
        affil_col  = 'slic_affiliations'
    elif auth_col == 'authors':
        auth_id_col = 'author_ids'
        affil_col  = 'affiliations'
    else:
        print(f'Cannot for affiliation map for "{auth_col}" column.', file=sys.stderr)
        return {}
    
    affiliations_map = {}
    df = df.dropna(subset=[auth_id_col, auth_col])
    df = get_papers(df, {target_auth_id}, auth_id_col)
    for eid, year, affiliations, num_citations in zip(df.eid.to_list(), df.year.to_list(), \
                                                      df[affil_col].to_list(), df.num_citations.to_list()):

        # get the number of citations
        num_citations = num_citations if not pd.isna(num_citations) else 0
        
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

        coauthors_set = {x for v in affiliations.values() for x in v.get('authors', [])}
        if target_auth_id not in coauthors_set:
            continue
        coauthors_set.remove(target_auth_id)
        
        # create the coauthors information map
        coauthors = {}
        for aff_id, info in affiliations.items():
            aff_name = info.get('name', 'Unknown')
            aff_country = info.get('country', 'Unknown')
            for auth_id in info.get('authors', []):
                if auth_id not in coauthors_set:
                    continue
                    
                if auth_id not in coauthors:
                    coauthors[auth_id] = {}
                if aff_id not in coauthors[auth_id]:
                    coauthors[auth_id][aff_id] = {
                        'name': aff_name,
                        'country': aff_country,
                        'num_shared_publications': 1,
                    }
                else:
                    coauthors[auth_id][aff_id]['num_shared_publications'] += 1

        
        for aff_id, info in affiliations.items():
            aff_name = info.get('name', 'Unknown')
            aff_country = info.get('country', 'Unknown')
            
            for auth_id in info.get('authors', []):
                if target_auth_id != auth_id:
                    continue
                
                if aff_id not in affiliations_map:
                    affiliations_map[aff_id] = {}
                if year not in affiliations_map[aff_id]:
                    affiliations_map[aff_id][year] = {}

                if 'name' not in affiliations_map[aff_id][year]:
                    affiliations_map[aff_id][year]['name'] = aff_name
                    affiliations_map[aff_id][year]['country'] = aff_country
                    affiliations_map[aff_id][year]['num_publications'] = 1
                    affiliations_map[aff_id][year]['num_citations'] = num_citations
                    affiliations_map[aff_id][year]['coauthors'] = coauthors
                else:
                    affiliations_map[aff_id][year]['num_publications'] += 1
                    affiliations_map[aff_id][year]['num_citations'] += num_citations
                    for c in coauthors:
                        if c in affiliations_map[aff_id][year]['coauthors']:
                            for c_aff_id in coauthors[c]:
                                if c_aff_id in affiliations_map[aff_id][year]['coauthors'][c]:
                                    affiliations_map[aff_id][year]['coauthors'][c][c_aff_id]['num_shared_publications'] += 1
                                else:
                                    affiliations_map[aff_id][year]['coauthors'][c][c_aff_id] = coauthors[c][c_aff_id]
                        else:
                            affiliations_map[aff_id][year]['coauthors'][c] = coauthors[c]
    
    # standarize the affiliation names
    for aff_id in affiliations_map:
        c = Counter([x['name'] for x in affiliations_map[aff_id].values()])
        most_common_name = c.most_common(1)[0][0]
        for year, aff_info in affiliations_map[aff_id].items():
            affiliations_map[aff_id][year]['name'] = most_common_name
    return affiliations_map


def create_author_affil_df(auth_affil_map, name_map):
    data = {
        'Author ID': [],
        'Author Name': [],
        'Affiliation ID': [],
        'Affiliation Name': [],
        'Country': [],
        'Year': [],
        'Num. Publications': [],
        'Num. Citations': [],
        'Num. Coauthors': [],
    }

    for auth_id, aff_auth in auth_affil_map.items():
        for aff_id, aff_year in aff_auth.items():
            for year, aff_info in aff_year.items():
                data['Author ID'].append(auth_id)
                data['Author Name'].append(name_map.get(auth_id, 'Unknown'))
                data['Affiliation ID'].append(aff_id)
                data['Affiliation Name'].append(aff_info['name'])
                data['Country'].append(aff_info['country'])
                data['Year'].append(year)
                data['Num. Publications'].append(aff_info['num_publications'])
                data['Num. Citations'].append(aff_info['num_citations'])
                data['Num. Coauthors'].append(len(aff_info['coauthors']))

    auth_affil_df = pd.DataFrame.from_dict(data)
    return auth_affil_df
    
    
def create_coauthor_affil_df(auth_affil_map, name_map):
    data = {
        'Author ID': [],
        'Author Name': [],
        'Co-Author ID': [],
        'Co-Author Name': [],
        'Affiliation ID': [],
        'Affiliation Name': [],
        'Country': [],
        'Year': [],
        'Num. Shared Publications': [],
    }

    for auth_id, aff_auth in auth_affil_map.items():
        for aff_id, aff_year in aff_auth.items():
            for year, aff_info in aff_year.items():
                for c_auth_id, aff_c_auth in aff_info['coauthors'].items():
                    for c_aff_id, c_aff_info in aff_c_auth.items():
                        data['Author ID'].append(auth_id)
                        data['Author Name'].append(name_map.get(auth_id, 'Unknown'))
                        data['Co-Author ID'].append(c_auth_id)
                        data['Co-Author Name'].append(name_map.get(c_auth_id, 'Unknown'))
                        data['Affiliation ID'].append(c_aff_id)
                        data['Affiliation Name'].append(c_aff_info['name'])
                        data['Country'].append(c_aff_info['country'])
                        data['Year'].append(year)
                        data['Num. Shared Publications'].append(c_aff_info['num_shared_publications'])

    coauth_affil_df = pd.DataFrame.from_dict(data)
    return coauth_affil_df


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


def process_queries(cheetah_files):
    """
    Take some Cheetah files and produce a map of document id to the search query that matched 
    said document

    Parameters:
    -----------
    cheetah_files: str
        List of cheetah file paths that need to be processed

    Returns:
    --------
    dict: 
        The formed map. Scopus paper ids are keys and values are sets of applied queries
    """
    query_map = {}
    for fn in cheetah_files:
        df = pd.read_csv(fn)
        for filter_type, filter_value, selected_scopus_ids in zip(df.filter_type.to_list(), \
                                                                  df.filter_value.to_list(), \
                                                                  df.selected_scopus_ids.to_list()):
            if pd.isna(selected_scopus_ids) or filter_type != 'query':
                continue
            ids = selected_scopus_ids.split(';')
            for eid in ids:
                if eid not in query_map:
                    query_map[eid] = set()
                query_map[eid].add(filter_value)
    return query_map


    
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