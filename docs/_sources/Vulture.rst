**TELF.pre_processing.Vulture:** Advanced text pre-processing and cleaning tool for NLP and text-mining
---------------------------------------------------------------------------------------------------------

Vulture is a tool for text pre-processing and cleaning. It has multi-processing and distributed computing capabilities, and designed to run fast.

Example
---------------------

First let's load the example dataset (example dataset can be found `here <https://github.com/lanl/T-ELF/tree/main/data>`_):

.. code-block:: python

   import pickle 

   documents = pickle.load(open("../../data/documents.p", "rb"))

Next let's load the stop words (example stop words can be found `here <https://github.com/lanl/T-ELF/tree/main/data>`_):

.. code-block:: python

   file = open("../../data/stop_words.txt", "r")
   stop_words = file.read().split("\n")
   file.close()

Now we can perform pre-processing:

.. code-block:: python
    
    # import libraries
    import os
    import pickle
    import pathlib
    import pandas as pd

    from TELF.pre_processing import Vulture

    from TELF.pre_processing.Vulture.modules import SimpleCleaner
    from TELF.pre_processing.Vulture.modules import LemmatizeCleaner
    from TELF.pre_processing.Vulture.modules import SubstitutionCleaner
    from TELF.pre_processing.Vulture.modules import RemoveNonEnglishCleaner

    from TELF.pre_processing.Vulture.default_stop_words import STOP_WORDS
    from TELF.pre_processing.Vulture.default_stop_phrases import STOP_PHRASES
    
    # load dataset
    DATA_DIR = os.path.join('..', '..', 'data')
    DATA_DIR = pathlib.Path(DATA_DIR).resolve()
    DATA_FILE = 'documents.p'
    documents = pickle.load(open(os.path.join(DATA_DIR, DATA_FILE), 'rb'))
    
    # output directory
    RESULTS_DIR = 'results'
    RESULTS_DIR = pathlib.Path(RESULTS_DIR).resolve()
    RESULTS_FILE = 'clean_documents.p'
    try:
        os.mkdir(RESULTS_DIR)
    except FileExistsError:
        pass
        
    # create a cleaning pipeline
    vulture = Vulture(n_jobs  = 1, 
                  verbose = 10,  # Disable == 0, Verbose >= 1
                 )
                 
    steps = [
    RemoveNonEnglishCleaner(ascii_ratio=0.9, stopwords_ratio=0.25),
    SimpleCleaner(stop_words = STOP_WORDS,
                      stop_phrases = STOP_PHRASES,
                      order = [
                          'standardize_hyphens',
                          'isolate_frozen',
                          'remove_copyright_statement',
                          'remove_stop_phrases',
                          'make_lower_case',
                          'remove_formulas',
                          'normalize',
                          'remove_next_line',
                          'remove_email',
                          'remove_()',
                          'remove_[]',
                          'remove_special_characters',
                          'remove_nonASCII_boundary',
                          'remove_nonASCII',
                          'remove_tags',
                          'remove_stop_words',
                          'remove_standalone_numbers',
                          'remove_extra_whitespace',
                          'min_characters',
                      ]
                     ),
        LemmatizeCleaner('spacy'),
    ]
    
    # clean
    cleaned_documents = vulture.clean(documents, steps=steps)

Available Functions
---------------------

.. currentmodule:: TELF.pre_processing.Vulture.vulture

.. autosummary::
   Vulture.__init__
   Vulture.clean
   Vulture.clean_dataframe
   

Module Contents
-----------------
.. automodule:: TELF.pre_processing.Vulture.vulture
   :members:
   :undoc-members:
   :show-inheritance: