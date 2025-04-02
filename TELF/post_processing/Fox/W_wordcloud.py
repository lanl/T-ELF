#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 08:31:23 2022

@author: maksim
"""
import random
import matplotlib
from wordcloud import WordCloud
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def _cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}

    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    def reduced_cmap(step): return np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red', 'green', 'blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, %d%%, 0%%)" % random.randint(1, 100)


def _make_wordcloud(
    B1_k,
    top_n,
    words_map,
    max_words=None,
    mask=None,
    background_color="black",
    min_font_size=10,
    max_font_size=None,
    prefer_horizontal=0.95,
    relative_scaling=0.1,
    colormap=None,
    contour_width=0.0,      
):
    """
    Internal helper that returns a WordCloud for the top_n words in B1_k.

    Parameters
    ----------
    B1_k : np.ndarray
        Array (usually a row or column from your matrix).
    top_n : int
        Number of top words to plot based on magnitude in B1_k.
    words_map : dict
        Maps index -> actual word string.
    max_words : int, optional
        Maximum number of words in the word cloud.
    mask : np.ndarray, optional
        A shape mask array for the cloud (e.g. np.zeros((800, 800)) ).
    background_color : str, optional
        Cloud background color, default='black'.
    min_font_size : int, optional
        Minimum font size for the smallest word.
    max_font_size : int, optional
        Maximum font size for the largest word.
    prefer_horizontal : float, optional
        Ratio (0 to 1) indicating how often to layout text horizontally.
    relative_scaling : float, optional
        How much to shrink more frequent words relative to others.
    colormap : matplotlib.colors.Colormap, optional
        A custom colormap. If None, we'll use a default.

    Returns
    -------
    WordCloud
        A generated word cloud for the top_n words in B1_k.
    """

    if colormap is None:
        # Example: a custom tinted colormap
        # or just colormap = plt.get_cmap("RdBu")
        # Here we apply user-defined _cmap_map if desired
        base_cmap = plt.get_cmap("RdBu")
        colormap = _cmap_map(lambda x: x * 0.6, base_cmap)

    # pick top_n largest values in B1_k
    index = np.argpartition(B1_k, -top_n)[-top_n:]
    # Ensure words_map is aligned to vocab and W
    if len(words_map) != len(B1_k):
        words_map = {i: word for i, word in enumerate(words_map[:len(B1_k)])}

    # Pick top_n largest values in B1_k
    index = np.argpartition(B1_k, -top_n)[-top_n:]

    # Build word frequency dictionary
    w_data = {}
    for i in index:
        if i in words_map:  # Avoid KeyError
            w_data[words_map[i]] = float(B1_k[i])


    wordcloud = WordCloud(
        width=1600,
        height=1600,
        background_color=background_color,
        relative_scaling=relative_scaling,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        prefer_horizontal=prefer_horizontal,
        max_words=max_words,
        mask=mask,
        colormap=colormap,
        contour_width=contour_width  
    ).generate_from_frequencies(w_data)

    return wordcloud


def create_wordcloud(
    W,
    vocab,
    top_n=10,
    path="",
    verbose=False,
    max_words=None,
    mask=None,
    background_color="black",
    min_font_size=10,
    max_font_size=None,
    prefer_horizontal=0.95,
    relative_scaling=0.1,
    colormap=None,
    contour_width=0.0,
):
    """
    Creates and saves a word cloud PDF for each column k in W, focusing on top_n words.

    Parameters
    ----------
    W : np.ndarray
        A matrix of shape (num_terms, num_topics).
    vocab : list or array
        The vocabulary list, vocab[i] is the term for row i.
    top_n : int
        Number of top words to show for each topic.
    path : str
        Output directory to save PDFs (e.g. 'results/').
    verbose : bool
        Whether to show a tqdm progress bar.
    max_words : int, optional
        Maximum words to display in the cloud. Default None (no explicit limit).
    mask : np.ndarray, optional
        Shape mask for the cloud.
    background_color : str, optional
        The color behind all words. Default='black'.
    min_font_size : int, optional
        Minimum font size for the smallest word. Default=10.
    max_font_size : int, optional
        Maximum font size for the largest word. Default=None.
    prefer_horizontal : float, optional
        Ratio of how often text is laid horizontally. Default=0.95.
    relative_scaling : float, optional
        Amount to shrink frequent words relative to others. Default=0.1.
    colormap : colormap, optional
        A matplotlib colormap object or name.

    Returns
    -------
    None
    """

    import os
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # map row -> actual word
    # Safely map row indices to actual words
    vocab = np.array(vocab)
    if W.shape[0] > len(vocab):
        print(f"[Warning] Truncating W to match vocab length: {len(vocab)}")
        W = W[:len(vocab), :]

    words_map = {ii: str(word) for ii, word in enumerate(vocab[:W.shape[0]])}
    K = W.shape[1]  # number of topics


    # Ensure the output directory is created
    os.makedirs(path, exist_ok=True)

    for k in tqdm(range(K), disable=not verbose):
        B1_k = W[:, k]
        fig = plt.figure(figsize=(15, 15), dpi=300, constrained_layout=False)

        wordcloud = _make_wordcloud(
            B1_k=B1_k,
            top_n=top_n,
            words_map=words_map,
            max_words=max_words,
            mask=mask,
            background_color=background_color,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            prefer_horizontal=prefer_horizontal,
            relative_scaling=relative_scaling,
            colormap=colormap,
            contour_width=contour_width  # <--- PASS IT HERE
        )

        plt.xticks([])
        plt.yticks([])
        plt.imshow(wordcloud, interpolation="nearest", aspect="equal")
        plt.tight_layout()

        pdf_path = os.path.join(path, f"{k}.pdf")
        plt.savefig(pdf_path)
        plt.close()
