import pandas as pd
import ast

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