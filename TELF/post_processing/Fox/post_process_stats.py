import os
import ast
import sparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from ...pre_processing import Beaver
from ...helpers.file_system import load_file_as_dict

###
### 1. Statistics for Files
###
def process_countries_file_stats(df, col='affiliations'):
    affiliations = [ast.literal_eval(x) for x in df[col].to_list() if isinstance(x, str)]
    countries = []
    for aff_info in affiliations:
        for aff_id, aff in aff_info.items():
            countries.append(aff['country'])
    combined = dict(Counter(countries))
    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))


def process_affiliations_file_stats(df, col='affiliations'):
    affiliations = [ast.literal_eval(x) for x in df[col].to_list() if isinstance(x, str)]
    output = []
    names = {}
    for aff_info in affiliations:
        for aff_id, aff in aff_info.items():
            if aff_id not in names:
                names[aff_id] = []
            
            names[aff_id].append(aff['name'])
            output.append(aff_id)
    
    output = dict(Counter(output))
    combined = {}
    for key, count in output.items():
        most_common_name = Counter(names[key]).most_common(1)[0][0]
        combined[most_common_name] = count
    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))

def organize_top_words(top_words, c, n):
    col_key = str(int(c)) if isinstance(c, float) else str(c)
    if col_key not in top_words.columns:
        raise ValueError(f"Cluster column '{col_key}' not found in top_words_df")
    return '; '.join(list(top_words[col_key].iloc[:n]))



def get_cluster_stat_file(df, top_words_df, summaries_df, n=5):
    data = {
        'cluster': [],
        'label': [],
        f'top_{n}_words': [],
        'summary': [],
        'num_papers': [],
        'num_countries': [],
        'num_affiliations': [],
        'num_citations': [],
        'country_counts': [],
        'affiliation_counts': [],
    }

    # Extract expected clusters from top_words_df columns
    expected_clusters = top_words_df.columns.astype(str).tolist()
    expected_clusters = sorted([int(c) if c.isdigit() else c for c in expected_clusters])

    #  Use cluster column as index for lookup
    if summaries_df is not None:
        summaries_df = summaries_df.copy()
        summaries_df['cluster'] = pd.to_numeric(summaries_df['cluster'], errors='coerce')
        summaries_df = summaries_df.dropna(subset=['cluster'])
        summaries_df['cluster'] = summaries_df['cluster'].astype(int)
        summaries_df = summaries_df.set_index('cluster')

    for c in tqdm(expected_clusters):
        tmp_df = df.loc[df.cluster == c]
        top_words = organize_top_words(top_words_df, c, n)

        if summaries_df is not None and c in summaries_df.index:
            summary = summaries_df.at[c, 'summary']
            label = summaries_df.at[c, 'label']
        else:
            summary = None
            label = None

        if tmp_df.empty:
            countries = {}
            affiliations = {}
            num_citations = 0
        else:
            countries = process_countries_file_stats(tmp_df)
            affiliations = process_affiliations_file_stats(tmp_df)
            num_citations = int(tmp_df.num_citations.sum())

        data['cluster'].append(c)
        data['label'].append(label)
        data[f'top_{n}_words'].append(top_words)
        data['summary'].append(summary)
        data['num_papers'].append(len(tmp_df))
        data['num_countries'].append(len(countries))
        data['num_affiliations'].append(len(affiliations))
        data['num_citations'].append(num_citations)
        data['country_counts'].append(countries)
        data['affiliation_counts'].append(affiliations)

    return pd.DataFrame.from_dict(data)

def create_tensor(df, col, output_dir='/tmp', joblib_backend='loky', n_jobs=-1, verbose=False):
    """
    Create a co-author tensor from the a specified dataframe and column.
    
    Parameters
    ----------
    fn: str, Path
        The path to the text file to be processed

    Returns
    -------
    dict
        The enumerated dict as an output
    """
    if df.empty or col not in df.columns or df[col].dropna().eq('').all():
        raise ValueError(f"No usable data found in column '{col}'.")
    
    # create Beaver object to generate tensor creating
    # the (co-relationship x co-relationship x time) tensor
    beaver = Beaver()
    settings = {
        "dataset":df,
        "target_columns": [col,'year'],
        "split_authors_with": ';',
        "save_path":output_dir,
        "verbose": verbose,
        "n_jobs": n_jobs,
        "authors_idx_map": {},
        "joblib_backend": joblib_backend, #'multiprocessing', 
    }

    beaver.coauthor_tensor(**settings)
    X = sparse.load_npz(os.path.join(output_dir, 'coauthor.npz'))
    time_map = load_file_as_dict(os.path.join(output_dir, 'Time.txt'))
    author_map = load_file_as_dict(os.path.join(output_dir, 'Authors.txt'))
    return X, author_map, time_map


def process_affiliations_coaffiliations(row):
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

########### WOLF END