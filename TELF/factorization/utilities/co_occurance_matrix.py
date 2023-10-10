#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:38:26 2021

@author: maksimekineren
"""

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import sparse


def co_occurrence(documents, vocabulary, window_size=20, dense=True, verbose=True, sentences=False):
    """
    Forms co-occurance matrix.

    Parameters
    ----------
    documents: list
        List of documents. Each entry in the list contains text.
        if sentences=True, then documents is a list of lists, where
        each entry in documents is a list of sentences.
    window_size: int, optional
        Number of consecutive words to perform counting.
        If sentences based analysis used, window size indicate the number of sentences.
    vocabulary: list
        List of unqiue words present in the all documents as vocabulary.
    dense: bool, optional
        If True, dense Numpy array is build.
        If False, Sparse CSR matrix is build.
    verbose: bool, optional
        Print progress or not.
    sentences: bool, optional
        If True, documents are list of lists, where each entry in documents
        is a list of sentences from that document. 
        In this case, window size is used as number of sentences, and the matrix is build based
        on the sentences. When False, window size is used, and documents is a list of documents.

    Returns
    -------
    M: np.ndarray or sparse CSR matrix
        Co-occurance matrix.

    """

    d = defaultdict(int)

    # turn the vocab into a hashmap for O(1) lookup
    V_map = dict()
    for ii, word in enumerate(vocabulary):
        V_map[word] = ii

    for document in tqdm(documents, total=len(documents), disable=not (verbose)):
        
        # document sentence based analysis
        if sentences:
            curr_sentences = document
            curr_sentences_combined = []
            tmp = []
            
            # combine sentences based on window size
            for sent_idx, sent in enumerate(curr_sentences):
                tmp.append(sent)
                if (len(tmp) == window_size) or (sent_idx + 1 == len(curr_sentences)):
                    curr_sentences_combined.append(" ".join(tmp))
                    tmp = []
            
            # iterate over sentences
            for sent in curr_sentences_combined:
                tokens = sent.split(" ")
                
                # iterate over words
                for ii in range(len(tokens)):
                    token=tokens[ii]
                    next_token = tokens[ii + 1 : len(tokens)]
                    
                    for t in next_token:
                        key = tuple(sorted([t, token]))
                        d[key] += 1
        
        # document window size based analysis
        else:
            tokens = document.split(" ")

            # iterate over words
            for ii in range(len(tokens)):
                token = tokens[ii]
                next_token = tokens[ii + 1 : min(ii + 1 + window_size, len(tokens))]
                for t in next_token:
                    key = tuple(sorted([t, token]))
                    d[key] += 1

    cols = list()
    rows = list()
    values = list()

    M = list()
    if dense:
        M = np.zeros((len(V_map), len(V_map)), dtype=np.int32)

    for key, value in tqdm(d.items(), disable=not (verbose)):
        try:
            r = V_map[key[0]]
            c = V_map[key[1]]
        except Exception:
            continue

        # if diogonal
        if c == r:
            value = value * 2

        # sparse
        if not dense:

            rows.append(r)
            cols.append(c)
            values.append(value)

            # non diogonal
            if r != c:
                rows.append(c)
                cols.append(r)
                values.append(value)

        # dense
        elif dense:
            M[r, c] = value

            # non diogonal
            if r != c:
                M[c, r] = value

    # sparse matrix from coordinate and values
    if not dense:
        coords = np.vstack([np.array(rows, dtype="int"), np.array(cols, dtype="int")])
        M = sparse.COO(coords, values, shape=(len(V_map), len(V_map)))
        M = M.tocsr()

    return M
