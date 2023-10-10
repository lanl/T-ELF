
from .NMFk import NMFk
from .decompositions.utilities.math_utils import relative_trinmf_error, prune, unprune
from .decompositions.tri_nmf_fro_mu import trinmf as trinmf_fro_mu
from .utilities.organize_n_jobs import organize_n_jobs

from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
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

def nmf_wrapper(
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
                 joblib_backend="multiprocessing",
                 pruned=True,
                 transpose=False,
                 verbose=True,
                 ):
        
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
        self.joblib_backend = joblib_backend
        self.pruned = pruned
        self.transpose = transpose
        self.save_path = "",
        self.verbose = verbose

        # organize n_jobs
        n_jobs, self.use_gpu = organize_n_jobs(use_gpu, n_jobs)
        if self.use_gpu:
            if n_jobs < 0 or n_jobs > 1:
                multiprocessing.set_start_method('spawn', force=True)

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

        # Do NMFk
        nmfk_results = self.nmfk.fit(X, Ks, self.experiment_name, note)
        self.save_path = os.path.join(self.nmfk.save_path, self.nmfk.experiment_name)

        # Do nmfk here
        self.nmfk_fit = True

        return nmfk_results

    def fit_tri_nmfk(self, X, k1k2:tuple):


        
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
        
        W_all, S_all, H_all, errors  = [], [], [], []
        if self.n_jobs == 1:
            for ninit in range(self.n_inits):
                w, s, h, e = nmf_wrapper(
                    init_num=ninit,
                    nmf=self.nmf,
                    nmf_params=self.nmf_params,
                    k1k2=k1k2,
                    gpuid=0,
                    use_gpu=self.use_gpu,
                    X=X,
                    mask=self.mask
                )
                W_all.append(w)
                S_all.append(s)
                H_all.append(h)
                errors.append(e)

        else:
            current_pert_results = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                backend=self.joblib_backend)(delayed(nmf_wrapper)(
                    ninit,
                    self.nmf,
                    self.nmf_params,
                    k1k2,
                    ninit % self.n_jobs,
                    self.use_gpu,
                    X,
                    self.mask
                    ) for ninit in range(self.n_inits))
            
            for w, s, h, e, in current_pert_results:
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
