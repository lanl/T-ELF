from pathlib import Path
import pathlib
import numpy as np
import os
import pandas as pd
from wordcloud import WordCloud

def words_probabilities(W:np.ndarray, save_path:str, rows_name:str, rows:list, num_top_words:int):
    words, probabilities = __top_words(W, rows, num_top_words)
    
    words_df = pd.DataFrame(words)
    words_df.to_csv(f'{save_path}/top_{rows_name}.csv', index=False)
    
    probabilities_df = pd.DataFrame(probabilities)
    probabilities_df.to_csv(f'{save_path}/top_{rows_name}_probabilities.csv', index=False)

    for k in range(W.shape[1]):
        k_words_df = pd.DataFrame.from_dict({rows_name:words_df[k], "probabilities":probabilities_df[k]})
        k_words_df.to_csv(f'{os.path.join(save_path, str(k))}/top_{rows_name}_in_cluster_{k}.csv', index=False)
         
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
        save_dir = __check_path(os.path.join(path, f'{k}'))
        wordcloud.to_file(os.path.join(save_dir, f'wordcloud_{k}.{format}'))

def __top_words(W, vocab, num_top_words):
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

def __check_path(path):
    """
    Checks and ensures the given path exists as a directory. If the path does not exist, a new 
    directory will be created. If the path exists but is a file, a ValueError will be raised. 
    A TypeError is raised if the provided path is neither a string nor a `pathlib.Path` object.
    Various exceptions are handled to ensure safety and robustness.

    Parameters:
    -----------
    path: str, pathlib.Path
        The path to be checked and ensured as a directory.

    Returns:
    --------
    pathlib.Path:
        The validated path

    Raises:
    -------
    TypeError:
        If the provided path is neither a string nor a `pathlib.Path` object.
    ValueError: 
        If the path points to an existing file or contains invalid characters.
    OSError:
        If there are issues with file system permissions, IO errors, or other OS-related errors.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not isinstance(path, pathlib.Path):
        raise TypeError(f'Unsupported type "{type(path)}" for `path`')

    try:
        if path.exists():
            if path.is_file():
                raise ValueError('`path` points to a file instead of a directory')
        else:
            path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise OSError('Insufficient permission to access or create the directory')
    except FileNotFoundError:
        raise ValueError('Invalid path or path contains illegal characters')
    except OSError as e:
        raise OSError(f'Error accessing or creating the directory: {e}')
    return path