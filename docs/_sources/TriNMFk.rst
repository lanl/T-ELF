**TELF.factorization.TriNMFk:** NMFk with Automatic Determination of Latent Clusters and Patterns
---------------------------------------------------------------------------------------------------

TriNMFk is a Non-negative Matrix Factorization module with the capability to do automatic model determination for both estimating the number of latent patterns (``Wk``) and clusters (``Hk``).

Example
---------------------

First generate synthetic data with pre-determined k. It can be either dense (``np.ndarray``) or sparse matrix (``scipy.sparse._csr.csr_matrix``). Here, we are using the provided scripts for matrix generation (located `here <https://github.com/lanl/T-ELF/tree/main/scripts>`_):

.. code-block:: python

   import sys; sys.path.append("../../scripts/")
   from generate_X import gen_trinmf_data
   import matplotlib.pyplot as plt

   kwkh=(5,3)
   shape=(20,20)
   data = gen_trinmf_data(shape=shape, 
                          kwkh=kwkh, 
                          factor_wh=(0.5, 1.0), 
                          factor_S=1,
                          random_state=10)

   X = data["X"]
   Wtrue = data["W"]
   Strue = data["S"]
   Htrue = data["H"]


Initilize the model:

.. code-block:: python

   nmfk_params = {
       "n_perturbs":64,
       "n_iters":2000,
       "epsilon":0.01,
       "n_jobs":-1,
       "init":"nnsvd",
       "use_gpu":False,
       "save_path":"../../results/",
       "verbose":True,
       "sill_thresh":0.8,
       "nmf_method":"nmf_fro_mu", 
       "perturb_type":"uniform", 
       "calculate_error":True,
       "pruned":True,
       "predict_k":True,
       "predict_k_method":"sill",
       "transpose":False,
       "mask":None,
       "use_consensus_stopping":False,
       "calculate_pac":True,
       "consensus_mat":True,
       "simple_plot":True,
       "collect_output":True
   }

   tri_nmfk_params = {
       "experiment_name":"TriNMFk",
       "nmfk_params":nmfk_params,
       "nmf_verbose":False,
       "use_gpu":True,
       "n_jobs":-1,
       "mask":None,
       "use_consensus_stopping":False,
       "alpha":(0,0),
       "n_iters":100,
       "n_inits":10,
       "pruned":False,
       "transpose":False,
       "verbose":True
   }

   from TELF.factorization import TriNMFk
   model = TriNMFk(**tri_nmfk_params)

Perform NMFk first:

.. code-block:: python

   Ks = range(1,8,1)
   note = "This the the NMFk portion of the TriNMFk method!"
   results = model.fit_nmfk(X, Ks, note)

Select number of latent patterns ``Wk`` (use the Silhouette score from the plots in ``../../results``) and number of latent clusters ``Hk`` (use the PAC score from the plots in ``../../results``) from the NMFk results.  Next perform TriNFMk


.. code-block:: python

   k1k2=(5,3)
   tri_nmfk_results = model.fit_tri_nmfk(X, k1k2)
   W = tri_nmfk_results["W"]
   S = tri_nmfk_results["S"]
   H = tri_nmfk_results["H"]

Available Functions
---------------------

.. currentmodule:: TELF.factorization.TriNMFk

.. autosummary::
   TriNMFk.__init__
   TriNMFk.fit_nmfk
   TriNMFk.fit_tri_nmfk

Module Contents
-----------------
.. automodule:: TELF.factorization.TriNMFk
   :members:
   :undoc-members:
   :show-inheritance: