from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def count(documents, options):
    """
    Count vectorizer.

    Parameters
    ----------
    documents: list
        List of documents. Each entry in the list contains text.
    options: dict
        Parameters for sklearn library.

    Returns
    -------
    X: list
        sparse bag of words.
    vocabulary: list
        List of unqiue words present in the all documents as vocabulary.

    """

    vectorizer = CountVectorizer(**options)
    X = vectorizer.fit_transform(documents)
    vocabulary = vectorizer.get_feature_names_out()

    return X, vocabulary


def tfidf(documents, options):
    """
    TF-IDF

    Parameters
    ----------
    documents: list
        List of documents. Each entry in the list contains text.
    options: dict
        Parameters for sklearn library.

    Returns
    -------
    X: list
        sparse tf-idf matrix.
    vocabulary: list
        List of unqiue words present in the all documents as vocabulary.

    """
    
    vectorizer = TfidfVectorizer(**options)
    X = vectorizer.fit_transform(documents)
    vocabulary = vectorizer.get_feature_names_out()

    return X, vocabulary
