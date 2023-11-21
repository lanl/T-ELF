**TELF.factorization.RESCALk:** RESCAL with Automatic Model Determination
---------------------------------------------------------------------------------------------------

RECALk is a RESCAL module with the capability to do automatic model determination.

Example
---------------------

First generate synthetic data with pre-determined k. It can be either dense (``np.ndarray``) or sparse matrix (``scipy.sparse._csr.csr_matrix``). Here, we are using the provided scripts for matrix generation (located `here <https://github.com/lanl/T-ELF/tree/main/scripts>`_):

.. code-block:: python

   import sys;
   sys.path.append("../../scripts/")
   from generate_X import gen_data

   X = gen_data(R=4, shape=[1000,1000,8],gen='rescal')["X"]

Now we can factorize the given matrix:

.. code-block:: python

   from TELF.factorization import RESCALk

   params = {
       "n_perturbs": 2,
       "n_iters": 2,
       "epsilon": 0.015,
       "n_jobs": 1,
       "init": "nnsvd", 
       "use_gpu": False,
       "save_path": "../../results/", 
       "save_output": True,
       "collect_output": True,
       "verbose": True,
       "transpose": False,
       "pruned":False,
       "rescal_verbose": False,
       "verbose":True,
       "rescal_method": 'rescal_fro_mu'

   }
   model = RESCALk(**params)
   Ks = range(1, 7, 1)
   name = "RESCALk"
   note = "This is an example run of RESCALk"
   results = model.fit(X, Ks, name, note)

Resulting plots showing the estimation of the matrix rank, or the number of latent factors can be found at ``../../results/``.

Available Functions
---------------------

.. currentmodule:: TELF.factorization.RESCALk

.. autosummary::
   RESCALk.__init__
   RESCALk.fit

Module Contents
-----------------
.. automodule:: TELF.factorization.RESCALk
   :members:
   :undoc-members:
   :show-inheritance: