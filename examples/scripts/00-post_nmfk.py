import os
import re
import ast
import scipy
import pickle
import shutil
import fnmatch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as ss
import matplotlib.pyplot as plt 
from wordcloud import WordCloud

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# load locally defined utility functions
from utils import load_config, load_list, check_path, process_terms

from TELF.pre_processing.Vulture.tokens_analysis.top_words import get_top_words

# constants
CONFIG_PATH = 'input/config.json'


def split_num(x, delim=';'):
    if x is None or isinstance(x, float):
        return 0
    else:
        return len(set(x.split(delim)))

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


def calculate_term_representations(df, terms, col):
    """
    Calculate term frequency, document frequency, and document frequency percentage
    for a list of terms in a pandas DataFrame.

    Parameters:
    -----------
    df: pd.DataFrame
        Pandas DataFrame with a column col containing the text data.
    col: str
        The column in which to search for terms
    terms: lit
        List of terms to calculate frequencies for.
    :return: A new DataFrame with columns 'Term', 'Term Frequency', 'Document Frequency', 'DF Percentage'.
    """
    def term_frequency(df, term, col):
        term = term.lower()
        return df[col].str.lower().str.count(term.lower()).sum()
    def document_frequency(df, term, col):
        return df[col].apply(lambda x: term in x if pd.notnull(x) else False).sum()

    # total number of documents
    total_documents = len(df)

    # prepare results DataFrame
    results_df = {
        'Term': [], 
        'Term Frequency': [], 
        'Document Frequency': [], 
        'DF Percentage': []
    }

    # calculate TF, DF, and DF% for each term
    for term in terms:
        tf = term_frequency(df, term, col)
        df_count = document_frequency(df, term, col)
        df_percentage = round((df_count / total_documents),3) if total_documents > 0 else 0

        results_df['Term'].append(term)
        results_df['Term Frequency'].append(tf)
        results_df['Document Frequency'].append(df_count)
        results_df['DF Percentage'].append(df_percentage)

    return pd.DataFrame.from_dict(results_df)


def top_words(W, vocab, num_top_words):
    """
    Identifies the top words and their corresponding probabilities from W

    Parameters:
    -----------
    W: np.ndarray
        A 2D NumPy array factor from the decomposition
    vocab  numpy.ndarray
        A 1D NumPy array of words corresponding to the rows in `W`.
    num_top_words: int
        The number of top words to extract from each column of `W`.

    Returns:
    --------
    tuple of np.ndarray:
	    words - A 2D array where each column contains the top `num_top_words` for the corresponding 
                 category/topic in `W`. The words are selected from the `vocab` array.
        probabilities - A 2D array of the same shape as `words`, containing the probabilities 
                        associated with these top words in each category/topic.

    Raises:
    -------
    ValueError:
        If `num_top_words` is greater than the number of words in `vocab` or if `W` and `vocab` 
        have mismatched dimensions.
    """
    indices       = np.argsort(-W, axis=0)[:num_top_words,:]
    probabilities = np.take_along_axis(W, indices, axis=0)
    words         = np.take_along_axis(vocab[:,None], indices, axis=0)
    return words, probabilities


def word_cloud(words, probabilities, path, max_words=30, background_color='white', format='png', **kwargs):
    """
    Generates and saves word cloud images based on words and their corresponding probabilities.

    Parameters:
    -----------
    words: list of list of str
        A list of lists, each containing words to be included in the word clouds.
    probabilities: list of list of float
        A list of lists, each containing probabilities corresponding to the words.
    path: str
        The directory path where the word cloud images will be saved.
    max_words: int, optional
        The maximum number of words to include in the word cloud (default is 30).
    background_color : str, optional
        Background color for the word cloud image (default is 'white').
    format: str, optional
        The format of the output image file (e.g., 'png', 'jpeg') (default is 'png').
    **kwargs:
        Additional keyword arguments to pass to the WordCloud constructor.

    Returns:
    --------
	None
    """
    if not all(len(words_row) == len(probs_row) for words_row, probs_row in zip(words, probabilities)):
        raise ValueError("Length of words and probabilities does not match in all rows")
        
    top_words = [{word: prob for word, prob in zip(words_row, probs_row)} for words_row, probs_row in zip(words.T, probabilities.T)]
    
    for k, word_dict in enumerate(top_words):
        wordcloud = WordCloud(max_words=max_words, background_color=background_color, **kwargs).fit_words(word_dict)
        save_dir = check_path(os.path.join(path, f'{k}'))
        wordcloud.to_file(os.path.join(save_dir, f'wordcloud_{k}.{format}'))


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


def H_clustering(H, verbose=False, distance='max') -> (dict, dict):
    """
    Performs argmax H-clustering, and gathers cluster information.
    
    Parameters
    ----------
    H: np.ndarray or scipy.sparse.csr_matrix
        H matrix from NMF
    verbose: bool, default is False
        If True, shows the progress.
    
    Returns
    -------
    (clusters_information, centroid_similarities): tuple of dict and list
    Dictionary carrying information for each cluster, 
    and dictionary carrying information for each document.
    """
    
    # hyper-parameter check
    assert ss.issparse(H) or \
        type(H) == np.ndarray, \
        "H type is not supported. H type: " + str(type(H)) + "\n" \
        "H should be type scipy.sparse.csr_matrix or np.ndarray."
    
    assert type(verbose) is bool, "verbose should be type bool!"
    
    # begin
    cluster_assignments    = np.argmax(H, axis=0)
    total_documents        = H.shape[1]
    clusters_information   = {}
    documents_information  = {}

    # clusters information
    for cluster in tqdm(set(cluster_assignments), disable=not verbose):

        # get the documents in the current cluster
        docs_in_cluster         = np.argwhere(cluster_assignments == cluster).flatten()

        # calculate the cluster information
        num_docs_in_cluster     = len(docs_in_cluster)
        percent_docs_in_cluster = num_docs_in_cluster / total_documents
        percent_docs_in_cluster = 100*np.round(percent_docs_in_cluster,3)

        # calculate the centroid information
        coordinates_in_cluster  = H[:,docs_in_cluster]
        cluster_centroid        = np.sum(coordinates_in_cluster, axis=1) / num_docs_in_cluster

        # save the information
        clusters_information[cluster] = {
            "count": num_docs_in_cluster, 
            "percent": percent_docs_in_cluster,
            "centroid": cluster_centroid
        }

    # documents information, go through document's coordinates and its index
    for doc_idx, doc_coord in tqdm(enumerate(H.transpose()), 
                                   disable = not verbose, total=H.shape[1]):

        # cluster that the document belongs to
        doc_cluster = cluster_assignments[doc_idx]

        # centroid of the cluster that the current document belongs to
        cluster_centroid = clusters_information[doc_cluster]["centroid"]

        # similarity of the coordinate of the document to the cluster centroid
        if distance == 'max':
            doc_sim_centroid = np.max(doc_coord) / np.max(cluster_centroid)
        else:    
            doc_sim_centroid = 1 - scipy.spatial.distance.cosine(doc_coord, cluster_centroid)
        
        documents_information[doc_idx] = {
            "similarity_to_cluster_centroid": doc_sim_centroid,
            "cluster_coordinates": list(doc_coord),
            "cluster": doc_cluster
        }
        
        
    return (clusters_information, documents_information)


def plot_H_clustering(H, name, path):
    """
    Plots the centroids of the H-clusters
    
    Parameters
    ----------
    H: np.ndarray or scipy.sparse.csr_matrix
        H matrix from NMF
    name: File name to save
    
    Returns
    -------
    Matplotlib plots
    """
    labels  = np.argmax(H, axis=0)
    uniques = np.unique(labels)
    fig     = plt.figure(figsize=(6,3*uniques.shape[0]))
    for i,l in enumerate(uniques):
        cluster = H[:,labels==l]
        cluster_means = np.mean(cluster,axis=1)
        cluster_stds = np.std(cluster,axis=1)
        ax = plt.subplot(uniques.shape[0],1,i+1)
        plt.bar(np.arange(H.shape[0]), cluster_means, color=(0.2, 0.4, 0.6, 0.6))
        plt.title(f'{name} cluster {i}')
        plt.tight_layout()
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        save_dir = os.path.join(path, f'{i}')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fig.savefig(os.path.join(save_dir, f'{name}_{i}.png'), bbox_inches=extent.expanded(1.2, 1.3), dpi=200)
    plt.close(fig)


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
    for cluster_id in sorted(df['cluster'].unique()):
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
        
        
def main(name, k, decomposition, num_top_words, num_top_documents):

    # load the config file
    config = load_config(CONFIG_PATH)

    # get settings from config 
    input_dir = config['paths']['output']
    input_dir = os.path.join(input_dir, name)

    # get the columns of the DataFrame which to clean and utilize
    cols = config['input']['text']
    clean_cols_name = f'clean_{"_".join(cols)}'
    
    # get a list of decomposition directories
    decomp_dirs = [x for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, x))]
    decomp_dirs = [x for x in decomp_dirs if x != 'beaver' and not x.startswith('.')]
    if decomposition is not None:
        decomp_dirs = [x for x in decomp_dirs if decomposition in x] 
    decomp_dir = os.path.join(input_dir, decomp_dirs[0])

    # create an output directory
    output_dir = os.path.join(decomp_dir, f'k={k}')
    output_dir = check_path(output_dir)

    file_name = f'WH_k={k}.npz'
    if not os.path.exists(os.path.join(decomp_dir, file_name)):
        raise ValueError(f'File "{file_name}" not found in "{decomp_dir}"')

    data = np.load(os.path.join(decomp_dir, file_name), allow_pickle=True)
    Wtfdf = data["W"]
    Htfdf = data["H"]
    print(f'Wtfdf: {Wtfdf.shape}')
    print(f'Htfdf: {Htfdf.shape}')

    archive_dir = os.path.join(output_dir, 'archive')
    if not os.path.exists(archive_dir):
        os.mkdir(archive_dir)

    # Final sillhoute plot
    try:
        copy_file_with_pattern(decomp_dir, output_dir, 'FINAL*.png')
    except Exception as e:
        warnings.warn(f'Failed to copy final sillhouete plot: {e}')

    # Experiment note
    try:
        copy_file_with_pattern(decomp_dir, output_dir, 'experiment_note.log')
    except Exception as e:
        warnings.warn(f'Failed to copy final sillhouete plot: {e}')

    # H-clustering: and saving the files table and cluster documents map
    labels, counts, table_df = H_cluster_argmax(Htfdf)
    table_df.to_csv(os.path.join(output_dir, 'table_H-clustering.csv'), index=False)
    np.savetxt(os.path.join(archive_dir, 'cluster_documents_map.txt'), np.rint(labels), fmt='%d')

    # Get top words and word clouds
    vocab_file = os.path.join(input_dir, 'beaver', 'Vocabulary.txt')
    if not os.path.exists(vocab_file):
        raise ValueError('Vocabulary file not found!')
    with open(vocab_file, 'r') as fh:
        vocab = fh.read().split('\n') 
    vocab = np.array(vocab)

    # Making the file top-words
    words, probabilities = top_words(Wtfdf, vocab, num_top_words)
    words_df = pd.DataFrame(words)
    probabilities_df = pd.DataFrame(probabilities)
    words_df.to_csv(os.path.join(output_dir, 'top_words.csv'), index=False)
    probabilities_df.to_csv(os.path.join(output_dir, 'probabilities_top_words.csv'), index=False)

    # Making the word clouds figures 
    word_cloud(words, probabilities=probabilities, path=output_dir, max_words=num_top_words, 
               mask=np.zeros((800, 800)), max_font_size=80, contour_width=2)

    # H_clustering 
    clusters_information, documents_information = H_clustering(Htfdf, verbose=True, distance='max')

    # Plot H_Clustering
    plots = plot_H_clustering(Htfdf, name="centroids_H_clustering", path=output_dir)

    # Making the file: cluser; counts ; percent ; centroid
    clusters_information_df = pd.DataFrame(clusters_information).T
    documents_information_df = pd.DataFrame(documents_information).T
    clusters_information_df.to_csv(os.path.join(archive_dir,'clusters_information.csv'), index=False)
    documents_information_df.to_csv(os.path.join(archive_dir,'documents_information.csv'), index=False)

    # Add clustering info to documents
    df_name = [x for x in os.listdir(os.path.join(input_dir, 'beaver')) if fnmatch.fnmatch(x, 'clean_*')]
    df = pd.read_csv(os.path.join(input_dir, 'beaver', df_name[0]))

    # Convert years to ints
    if 'year' in df:
        df['year'].fillna(-1, inplace=True)
        df['year'] = df['year'].astype(int)
        df['year'].replace(-1, np.nan, inplace=True)


    # If type is a column, add the core paper info to H clustering results table
    if 'type' in df:
        core_map = get_core_map(df)
        num_total_core = len(df.loc[df.type == 0])
        core_map_counts = {k: round(100*v / num_total_core, 2) for k,v in core_map.items()}
        table_df['core_count'] = table_df['cluster'].map(core_map)
        table_df['core_percentage'] = table_df['cluster'].map(core_map_counts)
        table_df.to_csv(os.path.join(output_dir, 'table_H-clusterings.csv'), index=False)

    # Add H clustering information to DataFrame
    df['cluster'] = documents_information_df['cluster']
    df['cluster_coordinates'] = documents_information_df['cluster_coordinates']
    df['similarity_to_cluster_centroid'] = documents_information_df['similarity_to_cluster_centroid']

    # Save the post-processed df
    df.to_csv(os.path.join(output_dir, f'cluster_{name}.csv'), index=False)

    # Get the best n samples from each cluster
    best_n_papers(df, output_dir, num_top_documents)
    
    # Get the SME attribution for each cluster
    terms = config['paths']['terms']
    highlighting_map = None
    if terms is not None:
        _, highlighting_map = process_terms(terms)
        terms = list(highlighting_map.keys())
        sme_attribution(df, output_dir, terms, col=clean_cols_name)
    
    # Get bag-of-words topwords 
    for cluster_id in sorted(df['cluster'].unique()):
        save_dir = check_path(os.path.join(output_dir, f'{int(cluster_id)}'))
        cluster_df = df.loc[df['cluster'] == cluster_id].copy() 
        clean_documents = cluster_df[clean_cols_name].to_dict()
        
        top_1grams = get_top_words(clean_documents, 
                             top_n=100, 
                             n_gram=1, 
                             verbose=True,
                             filename=None)
        top_1grams.to_csv(os.path.join(save_dir, f'{cluster_id}_bow_unigrams.csv'), index=False)
        
        top_2grams = get_top_words(clean_documents, 
                             top_n=100, 
                             n_gram=2, 
                             verbose=True,
                             filename=None)
        top_2grams.to_csv(os.path.join(save_dir, f'{cluster_id}_bow_bigrams.csv'), index=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-n", "--name", type=str, required=True, help="The name of the project/data for which NMFk has been performed")
    parser.add_argument("-k", "--k_value", type=int, required=True, help="Chosen k value. Should correspond to an npz file in args.path")
    parser.add_argument("-c", "--decomposition", type=str, default='', help="The name of the decomposition directory " \
                                                                            "(default='', first available decompisition)")
    parser.add_argument("-w", "--num_top_words", type=int, default=30, help="Number of top words (integer > 0, default=30)")
    parser.add_argument("-d", "--num_top_documents", type=int, default=5, help="Number of top documents (integer > 0, default=5)")
    
    args = parser.parse_args()
    main(args.name, args.k_value, args.decomposition, args.num_top_words, args.num_top_documents)
