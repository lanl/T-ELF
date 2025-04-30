from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.sparse as ss
import scipy
from .maps import get_id_to_name

def H_clustering(H, verbose=False, distance='max'):
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
    vocab = np.array(vocab)
    vocab_size = vocab.shape[0]
    
    if W.shape[0] > vocab_size:
        print(f"Warning: W has more rows ({W.shape[0]}) than vocab size ({vocab_size}). Truncating W.")
        W = W[:vocab_size, :]
    
    indices         = np.argsort(-W, axis=0)[:num_top_words, :]
    probabilities   = np.take_along_axis(W, indices, axis=0)
    words           = np.take_along_axis(vocab[:, None], indices.clip(max=vocab_size-1), axis=0)
    return words, probabilities