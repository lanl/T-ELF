import os
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.simplefilter(action='ignore', category=UserWarning)
from ...helpers.file_system import check_path

# constants
CONFIG_PATH = os.path.join("input", "config.json")

def get_core_map(df):
    core_map = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df.loc[df['cluster'] == cluster_id]
        core_map[cluster_id] = len(cluster_df.loc[cluster_df.type == 0])
    return core_map
    
def H_cluster_argmax(H):
    indices, labels, counts = np.unique(np.argmax(H,axis=0), return_counts=True, return_inverse=True)
    arr   = counts/np.sum(counts)
    table = pd.DataFrame({'cluster': indices, 'super_topic':indices, 'counts': counts, 'percentage':np.round(100*arr, 2)})
    return labels, counts, table

def term_frequency(df, term, col):
    term = term.lower()
    return df[col].str.lower().str.count(term).sum()

def document_frequency(df, term, col):
    term = term.lower()
    return df[col].apply(lambda x: term in x.lower() if pd.notnull(x) else False).sum()

def calculate_term_representations(df, terms, col):
    """
    Calculate term frequency, document frequency, and TF-IDF scores
    for a list of terms in a pandas DataFrame.

    Parameters:
    -----------
    df: pd.DataFrame
        Pandas DataFrame with a column col containing the text data.
    col: str
        The column in which to search for terms
    terms: list
        List of terms to calculate TF-IDF scores, term frequency, and document frequency for.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with columns 'Term', 'Term Frequency', 'Document Frequency', 'TF-IDF Score'.
    """
    vectorizer = TfidfVectorizer(vocabulary=terms, dtype=np.float32)
    tfidf_matrix = vectorizer.fit_transform(df[col].dropna())
    
    # get feature names (the terms vocabulary) from vectorizer
    feature_names = list(vectorizer.get_feature_names_out())
    
    # calculate average TF-IDF score for each term
    avg_tfidf_scores = tfidf_matrix.mean(axis=0).tolist()[0]
    
    # prepare results DataFrame
    results_df = {
        'Term': [], 
        'Term Frequency': [], 
        'Document Frequency': [], 
        'TF-IDF Score': []
    }

    # calculate TF, DF for each term
    for term in terms:
        term_index = feature_names.index(term)
        tf = term_frequency(df, term, col)
        df_count = document_frequency(df, term, col)
        tfidf_score = avg_tfidf_scores[term_index]

        results_df['Term'].append(term)
        results_df['Term Frequency'].append(tf)
        results_df['Document Frequency'].append(df_count)
        results_df['TF-IDF Score'].append(tfidf_score)

    return pd.DataFrame.from_dict(results_df)

def best_n_papers(df, path, n):
    """
    Get the top n papers that are closest to the cluster centroid

    Parameters:
    -----------
    df: pd.DataFrame
        The processed DataFrame that contains a `cluster` and `similarity_to_cluster_centroid` columns
    path: str
        Where to save the best papers
    n: int
        How many papers to save
    """
    for cluster_id in sorted(df['cluster'].dropna().unique()):
        save_dir = check_path(os.path.join(path, f'{int(cluster_id)}'))
        cluster_df = df.loc[df['cluster'] == cluster_id].copy()
        cluster_df['similarity_to_cluster_centroid'] = pd.to_numeric(cluster_df['similarity_to_cluster_centroid'], errors='coerce')
        best_df = cluster_df.nlargest(n, ['similarity_to_cluster_centroid'])
        best_df.to_csv(os.path.join(save_dir, f'best_{n}_docs_in_{cluster_id}.csv'), index=False)


def sme_attribution(df, path, terms, col='clean_title_abstract'):
    for cluster_id in sorted(df['cluster'].unique()):
        save_dir = check_path(os.path.join(path, f'{int(cluster_id)}'))
        cluster_df = df.loc[df['cluster'] == cluster_id].copy() 
        attribution_df = calculate_term_representations(cluster_df, terms, col)
        attribution_df.to_csv(os.path.join(save_dir, f'{cluster_id}_attribution.csv'), index=False)
