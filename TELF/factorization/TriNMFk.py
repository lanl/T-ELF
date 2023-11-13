"""
© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.
"""
from .NMFk import NMFk
from .decompositions.utilities.math_utils import relative_trinmf_error, prune, unprune
from .decompositions.tri_nmf_fro_mu import trinmf as trinmf_fro_mu
from .utilities.organize_n_jobs import organize_n_jobs

import concurrent.futures
from tqdm import tqdm
import numpy as np
import warnings
import scipy.sparse
import numpy as np
import os

try:
    import cupy as cp
    import cupyx.scipy.sparse
except Exception:
    cp = None
    cupyx = None

def _nmf_wrapper(
        init_num:int, 
        nmf, 
        nmf_params:dict, 
        k1k2:tuple, 
        gpuid:int, 
        use_gpu:bool,
        X, 
        mask):

    if mask is not None:
        X[mask] = 0

    np.random.seed(init_num)
    kw,kh = k1k2[0],k1k2[1]
    W, S, H = np.random.rand(X.shape[0], kw), np.random.rand(kw, kh), np.random.rand(kh, X.shape[1])

    if use_gpu:

        with cp.cuda.Device(gpuid):
            # move data and initialization from host to device
            W = cp.array(W)
            S = cp.array(S)
            H = cp.array(H)
            if scipy.sparse.issparse(X):
                Y = cupyx.scipy.sparse.csr_matrix(
                    (cp.array(X.data), cp.array(X.indices), cp.array(X.indptr)),
                    shape=X.shape,
                    dtype=X.dtype,
                )
            else:
                Y = cp.array(X)

            # do optimization on GPU
            W_, S_, H_ = nmf(X=Y, W=W, S=S, H=H, **nmf_params)

            # move solution from device to host
            W = cp.asnumpy(W_)
            S = cp.asnumpy(S_)
            H = cp.asnumpy(H_)

            del S_, W_, H_

            cp._default_memory_pool.free_all_blocks()

    else:
        W, S, H = nmf(X=X, W=W, S=S, H=H, **nmf_params)

    error = relative_trinmf_error(X, W, S, H)

    return W, S, H, error


class TriNMFk():

    def __init__(self,
                 experiment_name="TriNMFk",
                 nmfk_params={},
                 nmf_verbose=False,
                 use_gpu=False,
                 n_jobs=-1,
                 mask=None,
                 use_consensus_stopping=0,
                 alpha=(0,0),
                 n_iters=100,
                 n_inits=10,
                 pruned=True,
                 transpose=False,
                 verbose=True,
                 ):
        """
        TriNMFk is a Non-negative Matrix Factorization module with the capability to do automatic model determination for both estimating the number of latent patterns (``Wk``) and clusters (``Hk``).

        Parameters
        ----------
        experiment_name : str, optional
            Name used for the experiment. Default is "TriNMFk".
        nmfk_params : str, optional
            Parameters for NMFk. See documentation for NMFk for the options.
        nmf_verbose : bool, optional
            If True, shows progress in each NMF operation. The default is False.
        use_gpu : bool, optional
            If True, uses GPU for operations. The default is True.
        n_jobs : int, optional
            Number of parallel jobs. Use -1 to use all available resources. The default is 1.
        mask : ``np.ndarray``, optional
            Numpy array that points out the locations in input matrix that should be masked during factorization. The default is None.
        use_consensus_stopping : str, optional
            When not 0, uses Consensus matrices criteria for early stopping of NMF factorization. The default is 0.
        alpha : tupl, optional
            Error rate used in bootstrap operation. Default is (0, 0).
        n_iters : int, optional
            Number of NMF iterations. The default is 100.
        n_inits : int, optional
            Number of matrix initilization for the bootstrap operation. The default is 10.
        pruned : bool, optional
            When True, removes columns and rows from the input matrix that has only 0 values. The default is True.
        transpose : bool, optional
            If True, transposes the input matrix before factorization. The default is False.
        verbose : bool, optional
            If True, shows progress in each k. The default is False.
 
        Returns
        -------
        None.

        """

        
        # object parameters
        self.experiment_name = experiment_name
        self.nmfk_params = self._organize_nmfk_params(nmfk_params)
        self.nmf_verbose = nmf_verbose
        self.use_gpu = use_gpu
        self.mask = mask
        self.use_consensus_stopping = use_consensus_stopping
        self.alpha = alpha
        self.n_iters = n_iters
        self.n_inits = n_inits
        self.nmfk_fit = False
        self.pruned = pruned
        self.transpose = transpose
        self.save_path = "",
        self.verbose = verbose

        # organize n_jobs
        n_jobs, self.use_gpu = organize_n_jobs(use_gpu, n_jobs)

        if n_jobs > self.n_inits:
            n_jobs = self.n_inits

        self.n_jobs = n_jobs

        # prepare tri_nmf function
        self.nmf_params = {
                "niter": self.n_iters,
                "use_gpu": self.use_gpu,
                "nmf_verbose": self.nmf_verbose,
                "mask": self.mask,
                "use_consensus_stopping": self.use_consensus_stopping,
                "alpha":self.alpha
            }
        self.nmf = trinmf_fro_mu

        # prepare NMFk
        self.nmfk = NMFk(**self.nmfk_params)


    
    def fit_nmfk(self, X, Ks, note=""):
        """
        Factorize the input matrix ``X`` for the each given K value in ``Ks``.

        Parameters
        ----------
        X : ``np.ndarray`` or ``scipy.sparse._csr.csr_matrix`` matrix
            Input matrix to be factorized.
        Ks : list
            List of K values to factorize the input matrix.\n
            **Example:** ``Ks=range(1, 10, 1)``.
        name : str, optional   
            Name of the experiment. Default is "NMFk".
        note : str, optional
            Note for the experiment used in logs. Default is "".
        
        Returns
        -------
        results : dict
            Resulting dict can include all the latent factors, plotting data, predicted latent factors, time took for factorization, and predicted k value depending on the settings specified in ``nmfk_params``.\n
            * If ``get_plot_data=True``, results will include field for ``plot_data``.\n
            * If ``predict_k=True``, results will include field for ``k_predict``. This is an intiger for the automatically estimated number of latent factors.\n
            * If ``predict_k=True`` and ``collect_output=True``, results will include fields for ``W`` and ``H`` which are the latent factors in type of ``np.ndarray``.
            * results will always include a field for ``time``, that gives the total compute time.
        """

        # Do NMFk
        nmfk_results = self.nmfk.fit(X, Ks, self.experiment_name, note)
        self.save_path = os.path.join(self.nmfk.save_path, self.nmfk.experiment_name)

        # Do nmfk here
        self.nmfk_fit = True

        return nmfk_results

    def fit_tri_nmfk(self, X, k1k2:tuple):
        """
        Factorize the input matrix ``X``, after applying ``fit_nmfk()`` to select the ``Wk`` and ``Hk``, to factorize the given matrix with ``k1k2=(Wk, Hk)``.

        Parameters
        ----------
        X : ``np.ndarray`` or ``scipy.sparse._csr.csr_matrix`` matrix
            Input matrix to be factorized.
        k1k2 : tuple
            Tuple of ``Wk`` (number of latent patterns) and ``Hk`` (number of latent clusters), to factorize the matrix ``X`` to.
            **Example:** ``Ks=range(4,3)``.

        Returns
        -------
        results : dict
            Resulting dict will include latent patterns ``W``, ``H``, and mixing matrix ``S`` along with the error from each ``n_inits``.
        """

        
        if not self.nmfk_fit:
            warnings.warn("NMFk needs to be fit first. Use fit_nmfk function!")
            return
        
        if self.transpose:
            if isinstance(X, np.ndarray):
                X = X.T
            elif scipy.sparse.issparse(X):
                X = X.T.asformat("csr")
            else:
                raise Exception("I do not know how to transpose type " + str(type(X)))
            
        #
        # Prune
        #
        if self.pruned:
            X, rows, cols = prune(X, use_gpu=self.use_gpu)
        else:
            rows, cols = None, None

        job_data = {
            "nmf":self.nmf,
            "nmf_params":self.nmf_params,
            "k1k2":k1k2,
            "use_gpu":self.use_gpu,
            "X":X,
            "mask":self.mask
        }
        
        W_all, S_all, H_all, errors  = [], [], [], []
        if self.n_jobs == 1:
            for ninit in tqdm(range(self.n_inits), disable=not self.verbose, total=self.n_inits):
                w, s, h, e = _nmf_wrapper(init_num=ninit, gpuid=0, **job_data)
                W_all.append(w)
                S_all.append(s)
                H_all.append(h)
                errors.append(e)

        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs)
            futures = [executor.submit(_nmf_wrapper, init_num=ninit, gpuid=ninit % self.n_jobs, **job_data) for ninit in range(self.n_inits)]
            all_k_results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), disable=not self.verbose, total=self.n_inits)]
            
            for w, s, h, e, in all_k_results:
                W_all.append(w)
                S_all.append(s)
                H_all.append(h)
                errors.append(e)
            
        
        W_all = np.array(W_all).transpose((1, 2, 0))
        S_all = np.array(S_all).transpose((1, 2, 0))
        H_all = np.array(H_all).transpose((1, 2, 0))
        errors = np.array(errors)

        # select best
        min_err_idx = np.argmin(errors)
        W, S, H = W_all[:,:,min_err_idx], S_all[:,:,min_err_idx], H_all[:,:,min_err_idx]

        # unprune
        if self.pruned:
            W = unprune(W, rows, 0)
            H = unprune(H, cols, 1)

        # final results
        results = {
            "W":W,
            "S":S,
            "H":H,
            "errors":errors
        }

        # save the results
        np.savez_compressed(
                    self.save_path
                    + "/WSH"
                    + "_k="
                    + str(k1k2)
                    + ".npz",
                    **results)

        return results

    def _organize_nmfk_params(self, params):
        params["save_output"] = True
        params["collect_output"] = False
        params["consensus_mat"] = True
        params["calculate_pac"] = True

        return params
