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

   from TELF.pre_processing import Vulture

   settings = {
       # number of parallel jobs
       "n_jobs":-1, 
       # number of nodes
       "n_nodes":1, 
       # verbosity level
       "verbose":1,
       # minimum number of characters in tokens
       "min_characters":2,
       # minimum number of unique characters in tokens
       "min_unique_characters":2,
       # Lemmatize the tokens
       "lemmatize":True,
       # Lemmatize document using Spacy, performed during advance pre-processing
       "lemmatize_spacy":True,
       # nltk stems and substitution
       'lemmatize_slic':True,
       # Stemming of the tokens
       "stem":False,
       # Perform text cleaning
       "clean_text":True,
       # If True, advance text cleaning, otherwise simple only
       "advance_clean":True,
       # Allowed postags for Spacy, used if advance_clean=True
       "allowed_postags":['NOUN', 'ADJ', 'VERB', 'ADV', "PROPN"],
       # Spacy NLP model, used if advance_clean=True
       "spacy_model":"en_core_web_lg",
       # If True, detects language
       "detect_language":True,
       # Number of words to use when detecting language, used if detect_language=True
       "n_words_use_language":10,
       # backend to use
       "parallel_backend":"loky",
       # Settings for simple cleaning
       "simple_clean_settings": { "remove_copyright_with_symbol": True,
                                  "remove_stop_phrases": False,
                                  "make_lower_case": True,
                                  "remove_trailing_dash": True,  # -foo-bar- -> foo-bar
                                  "make_hyphens_words": False,  # foo-bar -> foobar
                                  "remove_next_line": True,
                                  "remove_email": True,
                                  "remove_dash": False,
                                  "remove_between_[]": True,
                                  "remove_between_()": True,
                                  "remove_[]": False,
                                  "remove_()": False,
                                  "remove_\\": False,
                                  "remove_^": False,
                                  "remove_numbers": False,
                                  "remove_nonASCII": False,
                                  "remove_tags": True,  # remove HTML tags
                                  "remove_special_characters": True,  # option to specify these?
                                  "remove_stop_words": True,
                               },
   }
   vulture = Vulture(**settings)
   substitutions = {'pair':'pair_modded','theory':'moddified_theories',}
   vulture.clean(documents, stop_words, filename="results/clean_example", substitutions=substitutions)

Results will be saved under ``results/clean_example``.

Available Functions
---------------------

.. currentmodule:: TELF.pre_processing.Vulture.vulture

.. autosummary::
   Vulture.__init__
   Vulture.clean
   Vulture.dataframe_clean
   Vulture.distributed_clean
   

Module Contents
-----------------
.. automodule:: TELF.pre_processing.Vulture.vulture
   :members:
   :undoc-members:
   :show-inheritance: