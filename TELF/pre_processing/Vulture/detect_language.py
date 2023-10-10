#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:13:30 2022

@author: maksim
"""
# from langdetect import detect


def get_language(document: str, document_id: str, n_words_use: int) -> tuple:
    """
    This method is currently disabled.
    Performs language detection on the given documents

    Parameters
    ----------
    documents : dict
        Dictionary of documents to clean. In this dictionary, keys are the unique document
        identifiers, and values are the text to clean.
    n_words_use : int
        Number of tokens to use when detecting langauge.

    Returns
    -------
    languages : dict
        List of tuples with document ID and language pairs.

    """
    return (document_id, "unknown")

    """
    # empty text
    if not isinstance(document, str) or len(document) == 0:
        lang = "unknown"

    # get tokens
    tokens = document.split()
    num_tokens = len(tokens)

    if (num_tokens < n_words_use) or (n_words_use == -1):
        target_text = document
    else:
        target_text = " ".join(tokens)[:n_words_use]

    # detect language
    try:
        lang = detect(target_text)
    except Exception:
        lang = "unknown"

    return (document_id, lang)
    """
    