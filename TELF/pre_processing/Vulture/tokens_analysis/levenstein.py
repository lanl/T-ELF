import pandas as pd
from tqdm import tqdm

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

def is_levenshtein_similar(s1, s2, threshold=0.95):
    """
    Check if two strings are Levenshtein similar based on a given threshold.

    Parameters
    ----------
    s1 : str
        The first string.
    s2 : str
        The second string.
    threshold : float, optional
        The minimum similarity threshold (default is 0.95).

    Returns
    -------
    tuple
        A tuple containing a boolean indicating if the strings are similar and the similarity score.
    """
    max_len = max(len(s1), len(s2))
    dist = levenshtein_distance(s1, s2)
    similarity = (max_len - dist) / max_len
    return similarity >= threshold, similarity

def replace_similar_keys_levenshtein(dict_list, changes_made_save_path=None, similarity_threshold=0.95):
    """
    Replace similar keys in a list of dictionaries based on Levenshtein similarity.

    Parameters
    ----------
    dict_list : list
        A list of dictionaries.
    changes_made_save_path : str, optional
        The path to save the changes made (default is None).
    similarity_threshold : float, optional
        The minimum similarity threshold for considering keys as similar (default is 0.95).

    Returns
    -------
    tuple
        A tuple containing the updated list of dictionaries and a DataFrame of changes made.
    """
    all_keys = set(key for d in dict_list for key in d.keys())
    similar_keys = {}
    changes = []

    sorted_keys = sorted(all_keys)   
    for key1 in tqdm(sorted_keys):
        for key2 in sorted_keys:
            if key1 != key2:
                similar_bool, similar_score = is_levenshtein_similar(key1, key2, similarity_threshold)
                if similar_bool:
                    smaller, larger = sorted([key1, key2], key=len)
                    similar_keys[larger] = (smaller, similar_score)

    for dict_index, dict_ in enumerate(dict_list):
        keys_to_replace = {k: v for k, v in similar_keys.items() if k in dict_}
        for longer_key, (shorter_key, score) in keys_to_replace.items():
            if longer_key in dict_:
                dict_[shorter_key] = dict_.pop(longer_key)
                changes.append({
                    'Index': dict_index,
                    'Previous Word': longer_key,
                    'New Word': shorter_key,
                    'Similarity Score': score
                })

    changes_df = pd.DataFrame(changes)
    if changes_made_save_path:
        changes_df.to_csv(changes_made_save_path, index=False)

    return dict_list, changes_df
