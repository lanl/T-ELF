import pandas as pd
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import os
from TELF.pre_processing.Vulture import Vulture
from TELF.pre_processing.Vulture.modules import SubstitutionOperator

class VocabularyConsolidator:
    """
    A class for processing and replacing similar keys in dictionaries based on Levenshtein distance and suffix processing.
    """

    def __init__(self):
        self.suffixes = ['ingly', 'edly', 'fully', 'ness', 'less', 'ment', 'tion', 'sion',
                         'ship', 'able', 'ible', 'al', 'ial', 'ed', 'ing', 'ly', 'es', 's', 
                         'er', 'tor']
        self.suffixes.sort(key=len, reverse=True)

    @staticmethod
    def levenshtein_distance(s1, s2, length_1, length_2):
        """
        Calculate the Levenshtein distance between two strings s1 and s2.

        Parameters
        ----------
        s1 : str
            The first string.
        s2 : str
            The second string.
        length_1 : int
            The length of the first string.
        length_2 : int
            The length of the second string.

        Returns
        -------
        int
            The Levenshtein distance between s1 and s2.
        """
        if length_1 < length_2:
            return VocabularyConsolidator.levenshtein_distance(s2, s1, length_2, length_1)
        if length_2 == 0:
            return length_1
        previous_row = range(length_2 + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def prefix_process_key(self, 
                       key):
        """
        Preprocess a key by removing suffixes from it exhaustively.

        Parameters
        ----------
        key : str
            The key to preprocess.

        Returns
        -------
        str
            The preprocessed key with the first matched suffix removed.
        """
        original_key = key
        for suffix in self.suffixes:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
                break
        return key

    def compare_keys(self, 
                     key1, 
                     key2, 
                     threshold=0.80, 
                     edge_range=0.1):
        """
        Compare two keys to determine if they are similar based on their Levenshtein distance and a similarity threshold.

        Parameters
        ----------
        key1 : str
            The first key.
        key2 : str
            The second key.
        threshold : float
            The similarity threshold.
        edge_range : float
            The range around the threshold to consider for preprocessing.

        Returns
        -------
        tuple
            A tuple containing a boolean indicating similarity and the similarity score.
        """
        length_1, length_2 = len(key1), len(key2)
        max_len = max(length_1, length_2)
        dist = self.levenshtein_distance(key1, key2, length_1, length_2)
        similarity = (max_len - dist) / max_len

        if threshold <= similarity <= (threshold + edge_range):
            key1_processed = self.prefix_process_key(key1)
            key2_processed = self.prefix_process_key(key2)
            return (key1_processed == key2_processed, similarity)
        return (similarity >= (threshold + edge_range), similarity)

    def process_chunk(self, 
                      pairs, 
                      key_frequency, 
                      threshold=0.90):
        """
        Process a chunk of key pairs to find similar keys.

        Parameters
        ----------
        pairs : list of tuple
            List of key pairs to compare.
        key_frequency : Counter
            Frequency count of all keys.
        threshold : float
            The similarity threshold.

        Returns
        -------
        list
            List of tuples containing less preferred key, preferred key, and similarity score.
        """
        results = []
        for key1, key2 in pairs:
            similar_bool, similar_score = self.compare_keys(key1, key2, threshold)
            if similar_bool:
                preferred_key = key1 if key_frequency[key1] > key_frequency[key2] else key2
                less_preferred_key = key2 if preferred_key == key1 else key1
                results.append((less_preferred_key, preferred_key, similar_score))
        return results

    def replace_similar_keys_levenshtein(self, 
                                         dict_list, 
                                         group_by_first_letter=True, 
                                         group_by_length_difference=True, 
                                         max_length_difference=2,
                                         min_chars=4, 
                                         changes_made_save_path=None, 
                                         similarity_threshold=0.80, 
                                         n_jobs=-1):
        """
        Replace similar keys in a list of dictionaries based on their similarity, optionally grouping them by first letter or length difference.

        Parameters
        ----------
        dict_list : list of dict
            List of dictionaries to process.
        group_by_first_letter : bool
            Whether to group keys by their first letter.
        group_by_length_difference : bool
            Whether to group keys by length difference.
        max_length_difference : int
            The maximum allowable length difference for grouping.
        min_chars : int
            Minimum character count to consider a key.
        changes_made_save_path : str
            Path to save the changes made.
        similarity_threshold : float
            The threshold for considering keys as similar.
        n_jobs : int
            Number of concurrent jobs to run. Uses all available CPUs if set to -1.

        Returns
        -------
        tuple
            A tuple containing the modified list of dictionaries and a DataFrame with the changes made.
        """
        all_keys = [key for d in dict_list for key in d.keys()]
        key_frequency = Counter(all_keys)
        similar_keys = {}
        changes = []

        sorted_keys = sorted(set(all_keys))
        grouped_keys = {}

        # Group keys by the first character
        if group_by_first_letter:
            for key in sorted_keys:
                first_char = key[0]
                if first_char not in grouped_keys:
                    grouped_keys[first_char] = []
                grouped_keys[first_char].append(key)

        # Further grouping by length difference within groups formed by the first letter
        if group_by_length_difference:
            final_grouped_keys = {}
            for key_group, keys in grouped_keys.items():
                temp_grouped_keys = {}
                keys_sorted_by_length = sorted(keys, key=len)

                # Dont pair to check words below a threshold of keys_sorted_by_length
                for index, key in enumerate(keys_sorted_by_length):
                    if len(key) >= min_chars:
                        break
                else:
                    index = -1
                if index != -1:
                    keys_sorted_by_length = keys_sorted_by_length[index:]
                
                # Only pair the terms that are not more different than max_length_difference
                for key in keys_sorted_by_length:
                    added = False
                    for group_key in list(temp_grouped_keys.keys()):
                        if abs(len(group_key) - len(key)) <= max_length_difference:
                            temp_grouped_keys[group_key].append(key)
                            added = True
                            break
                    if not added:
                        temp_grouped_keys[key] = [key]
                final_grouped_keys[key_group] = temp_grouped_keys

            # Flatten the groups correctly
            grouped_keys = {group_key: vals for subdict in final_grouped_keys.values() for group_key, vals in subdict.items()}

        # Generate all pairs for comparison
        all_pairs = [pair for key_list in grouped_keys.values() for pair in combinations(key_list, 2)]

        num_cpus = os.cpu_count() if n_jobs == -1 else min(n_jobs, os.cpu_count())
        chunk_size = int(len(all_pairs) / num_cpus) + 1
        chunks = [all_pairs[i:i + chunk_size] for i in range(0, len(all_pairs), chunk_size)]
        progress = tqdm(total=len(chunks), desc="Processing Chunks")

        with ThreadPoolExecutor(max_workers=min(num_cpus, len(chunks))) as executor:
            results = list(executor.map(self.process_chunk, chunks, [key_frequency]*len(chunks), [similarity_threshold]*len(chunks)))
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
    
    def unique_words_by_id(self, 
                            input_dict):
        """
        Create a list of dictionaries with unique words from the input dictionary.

        Parameters
        ----------
        input_dict : dict of {int: str}
            A dictionary where each key is an integer ID and each value is a string of words.

        Returns
        -------
        list of dict
            A list where each dictionary contains unique words from the input, preserving order.
        """
        output_list = []
        for key, word_string in input_dict.items():
            unique_words_dict = dict.fromkeys(word_string.split(), "")
            output_list.append(unique_words_dict)
        return output_list


    def consolidate_terms(self, 
                          vocabulary=None, 
                          texts=None, 
                          vulture=None, 
                          changes_made_save_path=None,
                          operated_text_save_path=None,
                          ignore_pairs = None):
        """
        Consolidate terms in a vocabulary or a list of texts using a Vulture pre-processing engine.

        Parameters
        ----------
        vocabulary : list of str, optional
            A list of vocabulary terms to process.
        texts : list of str, optional
            A list of texts to process.
        vulture : Vulture, optional
            An instance of the Vulture pre-processing engine.
        changes_made_save_path : str, optional
            Path to save the changes made.
        operated_text_save_path : str, optional
            Path to save the substituted text after word changes.

        Returns
        -------
        list
            Processed texts with consolidated terms.
        """
        if vocabulary and texts:
            raise ValueError("Specify either vocabulary or texts, not both.")
        
        if vocabulary:
            raise ValueError("Not implemented yet")

        if texts:
            output_list = self.unique_words_by_id(texts)
            consolidated_vocab, df_changes = self.replace_similar_keys_levenshtein(output_list, 
                                                                                   changes_made_save_path=changes_made_save_path)
            
            ignore_set = set(ignore_pairs)
            ignore_set.update((b, a) for a, b in ignore_pairs)

            corpus_substitutions = {}
            for p, n in zip(df_changes['Previous Key'], df_changes['New Key']):
                if (p, n) not in ignore_set and (n, p) not in ignore_set:
                    corpus_substitutions[p] = n
            
            if not vulture:
                vulture = Vulture(n_jobs=-1, verbose=True)

            if operated_text_save_path:
                split_path = operated_text_save_path.split(os.path.sep)
                save_path = (os.path.sep).join(split_path[:-1])
                save_file = split_path[-1]
            else:
                save_path = None
                save_file = None
            output = vulture.operate(texts, 
                                     steps=[SubstitutionOperator(document_substitutions=None, 
                                                                 corpus_substitutions=corpus_substitutions,
                                                                   document_priority=False)],
                                    save_path=save_path, 
                                    file_name=save_file)
            return output

    