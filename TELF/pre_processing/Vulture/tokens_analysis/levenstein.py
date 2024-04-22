import pandas as pd
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import os

def levenshtein_distance(s1, s2):
    """
    Calculates the Levenshtein distance between two strings.

    Parameters
    ----------
    s1 : str
        The first string.
    s2 : str
        The second string.

    Returns
    -------
    int
        The Levenshtein distance between s1 and s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compare_keys(key1, key2, threshold=0.95, use_indel=False):
    """
    Check if two strings are Levenshtein similar based on a given threshold. This function can optionally consider
    insertion and deletion costs in the similarity calculation, which is controlled by the 'use_indel' parameter.

    Parameters
    ----------
    key1 : str
        The first string to compare.
    key2 : str
        The second string to compare.
    threshold : float, optional
        The minimum similarity threshold for considering the strings as similar (default is 0.95).
    use_indel : bool, optional
        Whether to include insertion and deletion costs in the similarity calculation (default is False).

    Returns
    -------
    tuple
        A tuple containing a boolean indicating if the strings are similar and the similarity score.
    """
    if use_indel:
        raise ValueError("use_indel is not implemented yet -- pending dependency approval")
    else:
        max_len = max(len(key1), len(key2))
        dist = levenshtein_distance(key1, key2)
        similarity = (max_len - dist) / max_len
        return similarity > threshold, similarity

def process_chunk(pairs, key_frequency, threshold=0.95, use_indel=False):
    """
    Process a chunk of key pairs to determine if they are similar and decide the preferred key based on frequency.

    Parameters
    ----------
    pairs : list of tuple
        A list of tuples each containing two keys to be compared.
    key_frequency : dict
        A dictionary with keys and their corresponding frequency count.
    threshold : float, optional
        The minimum similarity threshold for considering keys as similar (default is 0.95).
    use_indel : bool, optional
        Whether to include insertion and deletion costs in the similarity calculation (default is False).

    Returns
    -------
    list
        A list of tuples, each containing the less preferred key, the preferred key, and the similarity score.
    """
    results = []
    for key1, key2 in pairs:
        similar_bool, similar_score = compare_keys(key1, key2, threshold, use_indel)
        if similar_bool:
            preferred_key = key1 if key_frequency[key1] > key_frequency[key2] else key2
            less_preferred_key = key2 if preferred_key == key1 else key1
            results.append((less_preferred_key, preferred_key, similar_score))
    return results

def replace_similar_keys_levenshtein(dict_list, 
                                     group_by_first_letter=True,
                                     changes_made_save_path=None, 
                                     similarity_threshold=0.95, 
                                     use_indel=False,
                                     n_jobs=-1):
    """
    Replace similar keys in a list of dictionaries based on similarity,
    preferring the key that occurs more often. Optionally uses an alternative similarity calculation method.

    This function can group keys by their first letter before comparing them to reduce computational load, which is
    controlled by the 'group_by_first_letter' parameter. It supports parallel processing through the 'n_jobs' parameter.

    Parameters
    ----------
    dict_list : list
        A list of dictionaries.
    group_by_first_letter : bool, optional
        Whether to group keys by the first letter before comparison (default is True).
    changes_made_save_path : str, optional
        The path to save the changes made (default is None).
    similarity_threshold : float, optional
        The minimum similarity threshold for considering keys as similar (default is 0.95).
    use_indel : bool, optional
        Whether to use an alternative method for similarity comparison, such as including insertions and deletions in the cost (default is False).
    n_jobs : int, optional
        The number of jobs to run in parallel (default is -1, which uses all processors).

    Returns
    -------
    tuple
        A tuple containing the updated list of dictionaries and a DataFrame of changes made.
    """

    all_keys = [key for d in dict_list for key in d.keys()]
    key_frequency = Counter(all_keys)
    similar_keys = {}
    changes = []

    sorted_keys = sorted(set(all_keys))

    # Group keys by the first character
    if group_by_first_letter:
        grouped_keys = {}
        for key in sorted_keys:
            first_char = key[0]
            if first_char not in grouped_keys:
                grouped_keys[first_char] = []
            grouped_keys[first_char].append(key)
        
        # Generate all pairs where the first character matches
        all_pairs = [pair for key_list in grouped_keys.values() for pair in combinations(key_list, 2)]
    else:
        all_pairs = list(combinations(sorted_keys, 2))
        
    num_cpus = os.cpu_count()
    if n_jobs == -1:
        num_cpus = os.cpu_count()  # Get the number of CPUs available
    else:
        # Make sure the thread count passed in is not greater than the number available
        num_cpus = min(n_jobs, num_cpus)
        
    chunk_size = int(len(all_pairs) / num_cpus) + 1
    print(f"chunk_size = {chunk_size}, num_cpus = {num_cpus}, len all_pairs = {len(all_pairs)}")
    chunks = [all_pairs[i:i + chunk_size] for i in range(0, len(all_pairs), chunk_size)]
    progress = tqdm(total=len(chunks), desc="Processing Chunks")

    with ThreadPoolExecutor(max_workers=min(num_cpus,len(chunks))) as executor:
        results = list(executor.map(process_chunk, chunks, [key_frequency]*len(chunks), [similarity_threshold]*len(chunks), [use_indel]*len(chunks)))
        for chunk_result in results:
            for less_preferred_key, preferred_key, similar_score in chunk_result:
                similar_keys[less_preferred_key] = (preferred_key, similar_score)
            progress.update(1)

    progress.close()

    for dict_ in dict_list:
        for less_preferred_key, (preferred_key, score) in similar_keys.items():
            if less_preferred_key in dict_:
                if isinstance(dict_[less_preferred_key], int):
                    dict_[preferred_key] = dict_.get(preferred_key, 0) + dict_.pop(less_preferred_key)
                elif isinstance(dict_[less_preferred_key], str):
                    dict_[preferred_key] = dict_.get(preferred_key, '') + dict_.pop(less_preferred_key)

                changes.append({
                    'Previous Key': less_preferred_key,
                    'New Key': preferred_key,
                    'Similarity Score': score
                })

    changes_df = pd.DataFrame(changes)

    if changes_made_save_path:
        changes_df.to_csv(changes_made_save_path, index=False)

    return dict_list, changes_df
