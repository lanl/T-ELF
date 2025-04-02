# -*- coding: utf-8 -*-
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

def get_top_words(documents,
                  top_n=10,
                  n_gram=1,
                  verbose=True,
                  filename=None) -> pd.DataFrame:
    """
    Collects statistics for the top words or n-grams. Returns a table with columns
    word, tf, df, df_fraction, and tf_fraction.

    - word column lists the words in the top_n.
    - tf is the term-frequency, how many times given word occured in documents.
    - df is the document-frequency, in how documents given word occured.
    - df_fraction is df / len(documents)
    - tf_fraction is tf / (total number of unique tokens or n-grams)

    Parameters
    ----------
    documents : list or dict
        list or dictionary of documents. 
        If dictionary, keys are the document IDs, values are the text.
    top_n : int, optional
        Top n words or n-grams to report. The default is 10.
    n_gram : int, optional
        1 is words, or n-grams when > 1. The default is 1.
    verbose : bool, optional
        Verbosity flag. The default is True.
    filename : str, optional
        If not one, saves the table to the given location.

    Returns
    -------
    pd.DataFrame
        Table for the statistics.

    """

    if isinstance(documents, dict):
        documents = list(documents.values())

    word_stats = defaultdict(lambda: {"tf": 0, "df": 0})

    for doc in tqdm(documents, disable=not verbose):
        tokens = doc.split()
        ngrams = zip(*[tokens[i:] for i in range(n_gram)])
        ngrams = [" ".join(ngram) for ngram in ngrams]

        for gram in ngrams:
            word_stats[gram]["tf"] += 1
        for gram in set(ngrams):
            word_stats[gram]["df"] += 1

    word_stats = dict(word_stats)
    top_words = dict(sorted(word_stats.items(), key=lambda x: x[1]["tf"], reverse=True)[:top_n])
    target_data = {"word": [], "tf": [], "df": [], "df_fraction": [], "tf_fraction": []}

    for word in top_words:
        target_data["word"].append(word)
        target_data["tf"].append(word_stats[word]["tf"])
        target_data["df"].append(word_stats[word]["df"])
        target_data["df_fraction"].append(word_stats[word]["df"] / len(documents))
        target_data["tf_fraction"].append(word_stats[word]["tf"] / len(word_stats))

    # put together the results
    table = pd.DataFrame.from_dict(target_data)

    if filename:
        table.to_csv(filename+".csv", index=False)

    return table

