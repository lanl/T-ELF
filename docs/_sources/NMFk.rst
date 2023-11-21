**TELF.factorization.NMFk:** Non-negative Matrix Factorization with Automatic Model Determination
---------------------------------------------------------------------------------------------------

NMFk is a Non-negative Matrix Factorization module with the capability to do automatic model determination.

Example
---------------------

First generate synthetic data with pre-determined k. It can be either dense (``np.ndarray``) or sparse matrix (``scipy.sparse._csr.csr_matrix``). Here, we are using the provided scripts for matrix generation (located `here <https://github.com/lanl/T-ELF/tree/main/scripts>`_):

.. code-block:: python
      
      import sys; sys.path.append("../../scripts/")
      from generate_X import gen_data,gen_data_sparse

      Xsp = gen_data_sparse(shape=[100, 200], density=0.01)["X"]
      X = gen_data(R=4, shape=[100, 200])["X"]

Now we can factorize the given matrix:

.. code-block:: python

   from TELF.factorization import NMFk

   params = {
       "n_perturbs":36,
       "n_iters":100,
       "epsilon":0.015,
       "n_jobs":-1,
       "init":"nnsvd", 
       "use_gpu":False,
       "save_path":"../../results/", 
       "save_output":True,
       "collect_output":True,
       "predict_k":True,
       "predict_k_method":"sill",
       "verbose":True,
       "nmf_verbose":False,
       "transpose":False,
       "sill_thresh":0.9,
       "pruned":True,
       'nmf_method':'nmf_kl_mu',
       "calculate_error":True,
       "predict_k":True,
       "use_consensus_stopping":0,
       "calculate_pac":False,
       "perturb_type":"uniform"
   }
   Ks = range(1,11,1)
   name = "Example_NMFk"
   note = "This is an example run of NMFk"

   model = NMFk(**params)
   results = model.fit(X, range(1,10,1), name, note)

Resulting plots showing the estimation of the matrix rank, or the number of latent factors can be found at ``../../results/``.


Available Functions
---------------------

.. currentmodule:: TELF.factorization.NMFk

.. autosummary::
   NMFk.__init__
   NMFk.fit

Module Contents
-----------------
.. automodule:: TELF.factorization.NMFk
   :members:
   :undoc-members:
   :show-inheritance: