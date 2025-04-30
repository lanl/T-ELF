import ast
import pandas as pd
from collections import Counter
import sys

from .filters import get_papers


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