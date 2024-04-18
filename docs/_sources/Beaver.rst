**TELF.pre_processing.Beaver:** Fast matrix and tensor building tool
---------------------------------------------------------------------------------------------------

Beaver is a matrix and tensor building tool.

Example
---------------------

Several examples for Beaver can be found `here <https://github.com/lanl/T-ELF/tree/main/examples/Beaver>`_. Here we will show example usage for creating a Document-Words matrix. First let's load the example dataset (example dataset can be found `here <https://github.com/lanl/T-ELF/tree/main/data>`_):

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("../../data/sample.csv")
   df.info()

Next, let's build the vocabulary:

.. code-block:: python

   from TELF.pre_processing import Beaver

   beaver = Beaver()
   settings = {
      "dataset":df,
      "target_column":"clean_abstract",
      "min_df":10,
      "max_df":0.5,
   }

   vocabulary = beaver.get_vocabulary(**settings)

Next we can build the matrix:

.. code-block:: python

   settings = {
      "dataset":df,
      "target_column":"clean_abstract",
      "options":{"min_df": 5, "max_df": 0.5, "vocabulary":vocabulary},
      "matrix_type":"tfidf",
      "highlighting":['aberration', 'ability', 'ablation', 'ablator', 'able'],
      "weights":2,
      "save_path":"./"
   }

   beaver.documents_words(**settings)

We cam then load the matrix:

.. code-block:: python

   import scipy.sparse as ss

   # load the saved file which is in Sparse COO format
   X_csr_sparse = ss.load_npz("documents_words.npz")

Available Functions
---------------------

.. currentmodule:: TELF.pre_processing.Beaver.beaver

.. autosummary::
   Beaver.__init__
   Beaver.get_vocabulary
   Beaver.coauthor_tensor
   Beaver.cocitation_tensor
   Beaver.participation_tensor
   Beaver.citation_tensor
   Beaver.cooccurrence_matrix
   Beaver.documents_words
   Beaver.something_words
   Beaver.something_words_time
   Beaver.get_ngrams

Module Contents
-----------------
.. automodule:: TELF.pre_processing.Beaver.beaver
   :members:
   :undoc-members:
   :show-inheritance: