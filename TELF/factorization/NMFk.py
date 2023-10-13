#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from .utilities.take_note import take_note, take_note_fmat, append_to_note
from .utilities.plot_NMFk import plot_NMFk, plot_consensus_mat, plot_cophenetic_coeff
from .utilities.pvalue_analysis import pvalue_analysis
from .utilities.organize_n_jobs import organize_n_jobs
from .decompositions.nmf_kl_mu import nmf as nmf_kl_mu
from .decompositions.nmf_fro_mu import nmf as nmf_fro_mu
from .decompositions.nmf_recommender import nmf as nmf_recommender
from .decompositions.nmf_fro_mu import H_update
from .decompositions.utilities.nnsvd import nnsvd
from .decompositions.utilities.resample import poisson, uniform_product
from .decompositions.utilities.clustering import custom_k_means, silhouettes
from .decompositions.utilities.math_utils import prune, unprune, relative_error, get_pac
from .decompositions.utilities.concensus_matrix import compute_consensus_matrix, reorder_con_mat
from joblib import Parallel, delayed
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import os
import scipy.sparse
from tqdm import tqdm
import numpy as np
import warnings
import time
import socket
import multiprocessing
from pathlib import Path


try:
    import cupy as cp
    import cupyx.scipy.sparse
except Exception:
    cp = None
    cupyx = None

try:
    from mpi4py import MPI
except Exception:
    MPI = None


def __run_nmf(Y, W, H, nmf, nmf_params, use_gpu:bool, gpuid:int):
    if use_gpu:

        with cp.cuda.Device(gpuid):
            # move data and initialization from host to device
            W = cp.array(W)
            H = cp.array(H)
            if scipy.sparse.issparse(Y):
                Y = cupyx.scipy.sparse.csr_matrix(
                    (cp.array(Y.data), cp.array(Y.indices), cp.array(Y.indptr)),
                    shape=Y.shape,
                    dtype=Y.dtype,
                )
            else:
                Y = cp.array(Y)

            # do optimization on GPU
            W_, H_, other_results = nmf(X=Y, W=W, H=H, **nmf_params)

            # move solution from device to host
            W = cp.asnumpy(W_)
            H = cp.asnumpy(H_)

            del Y, W_, H_

            cp._default_memory_pool.free_all_blocks()

    else:
        W, H, other_results = nmf(X=Y, W=W, H=H, **nmf_params)

    return W, H, other_results

def __perturb_X(X, perturbation:int, epsilon:float, perturb_type:str):

    np.random.seed(perturbation)
    if perturb_type == "uniform":
        Y = uniform_product(X, epsilon)
    elif perturb_type == "poisson":
        Y = poisson(X)

    return Y

def __init_WH(Y, k, mask, init_type:str):
    if init_type == "nnsvd":
        if mask is not None:
            Y[mask] = 0
        W, H = nnsvd(Y, k)
    elif init_type == "random":
        W, H = np.random.rand(Y.shape[0], k), np.random.rand(k, Y.shape[1])

    return W, H

def __H_regression(X, W, mask, use_gpu):
    if use_gpu:
        if scipy.sparse.issparse(X):
            Y = cupyx.scipy.sparse.csr_matrix(
                (cp.array(X.data), cp.array(X.indices), cp.array(X.indptr)),
                shape=X.shape,
                dtype=X.dtype,
            )
        else:
            Y = cp.array(X)

        H_ = H_update(Y, cp.array(W), cp.random.rand(
            W.shape[1], X.shape[1]), use_gpu=use_gpu, mask=mask)
        H = cp.asnumpy(H_)
        del Y, H_
        cp._default_memory_pool.free_all_blocks()

    else:
        H = H_update(X.copy(), W, np.random.rand(
            W.shape[1], X.shape[1]), use_gpu=use_gpu, mask=mask)
        
    return H

def _nmf_parallel_wrapper(
        n_perturbs, 
        nmf, 
        nmf_params,
        init_type="nnsvd", 
        X=None, 
        k=None,
        epsilon=None, 
        gpuid=0, 
        use_gpu=True,
        perturb_type="uniform", 
        calculate_error=True, 
        mask=None, 
        consensus_mat=False,
        predict_k=False,
        predict_k_method="sill",
        pruned=True,
        perturb_rows=None,
        perturb_cols=None,
        save_output=True,
        save_path="",
        experiment_name="",
        collect_output=False,
        logging_stats={},
        start_time=time.time()):

    #
    # run for each perturbations
    #
    W_all, H_all, errors = [], [], []
    for perturbation in range(n_perturbs):
        Y = __perturb_X(X, perturbation, epsilon, perturb_type)
        W, H = __init_WH(Y, k, mask, init_type)
        W, H, other_results = __run_nmf(Y, W, H, nmf, nmf_params, use_gpu, gpuid)

        if calculate_error:
            error = relative_error(X, W, H)
        else:
            error = 0

        W_all.append(W)
        H_all.append(H)
        errors.append(error)

    #
    # organize colutions from each perturbations
    #
    W_all = np.array(W_all).transpose((1, 2, 0))
    H_all = np.array(H_all).transpose((1, 2, 0))
    errors = np.array(errors)

    #
    # cluster the solutions
    #
    W, W_clust = custom_k_means(W_all)
    sils_all = silhouettes(W_clust)

    #
    # concensus matrix
    #
    coeff_k = 0
    reordered_con_mat = None
    if consensus_mat:
        con_mat_k = compute_consensus_matrix(H_all)
        reordered_con_mat, coeff_k = reorder_con_mat(con_mat_k, k)

    #
    # Regress H
    #
    H = __H_regression(X, W, mask, use_gpu)

    # 
    #  reconstruction error
    #
    if calculate_error:
        if mask is not None:
            Xhat = W@H
            X[mask] = Xhat[mask]
        error_reg = relative_error(X, W, H)
    else:
        error_reg = 0

    #
    # calculate columnwise error to predict k
    #
    curr_col_err =  list()
    if predict_k and predict_k_method == "pvalue":
        for q in range(0, X.shape[1]):
            curr_col_err.append(
                relative_error(X[:, q].reshape(-1, 1), W, H[:, q].reshape(-1, 1))
            )

    #
    # unprune
    #
    if pruned:
        W = unprune(W, perturb_rows, 0)
        H = unprune(H, perturb_cols, 1)

    #
    # save output factors and the plot
    #
    if save_output:
        if consensus_mat:
            con_fig_name = f'{save_path}/k_{k}_con_mat.png'
            plot_consensus_mat(reordered_con_mat, con_fig_name)
        
        save_data = {
            "W": W,
            "H": H,
            "sils_all": sils_all,
            "error_reg": error_reg,
            "errors": errors,
            "reordered_con_mat": reordered_con_mat,
            "H_all": H_all,
            "cophenetic_coeff": coeff_k,
            "other_results": other_results
        }
        np.savez_compressed(
            save_path
            + "/WH"
            + "_k="
            + str(k)
            + ".npz",
            **save_data
        )

        # if predict k is True, report "L statistics error"
        
        plot_data = dict()
        for key in logging_stats:
            if key == 'k':
                plot_data["k"] = k
            elif key ==  'sils_min':
                sils_min = np.min(np.mean(sils_all, 1))
                plot_data["sils_min"] = '{0:.3f}'.format(sils_min)
            elif key == 'sils_mean':
                sils_mean = np.mean(np.mean(sils_all, 1))
                plot_data["sils_mean"] = '{0:.3f}'.format(sils_mean)
            elif key == 'err_mean':
                err_mean = np.mean(errors)
                plot_data["err_mean"] = '{0:.3f}'.format(err_mean)
            elif key == 'err_std':
                err_std = np.std(errors)
                plot_data["err_std"] = '{0:.3f}'.format(err_std)
            ### Commenting out PAC calculation because it is invalid when calculated at a single iteration
            ### Need to add a solution for adding PAC calculation after experiment has concluded
            # elif key == 'pac': 
            #     consensus_tensor = np.array([reordered_con_mat])
            #     pac = get_pac(consensus_tensor, use_gpu=use_gpu)[0]
            #     plot_data["pac"] = '{0:.3f}'.format(pac)
            elif key == 'col_error':
                mean_col_err = np.mean(curr_col_err)
                plot_data["col_err"] = '{0:.3f}'.format(mean_col_err)
            elif key == 'time':
                elapsed_time = time.time() - start_time
                elapsed_time = timedelta(seconds=elapsed_time)
                plot_data["time"] = str(elapsed_time).split('.')[0]
            else:
                warnings.warn(f'[tELF]: Encountered unknown logging metric "{key}"', RuntimeWarning)
                plot_data[key] = 'N/A'
        take_note_fmat(save_path, **plot_data)

    #
    # collect results
    #
    results_k = {
        "Ks":k,
        "err_mean":np.mean(errors),
        "err_std":np.std(errors),
        "err_reg":error_reg,
        "sils_min":np.min(np.mean(sils_all, 1)),
        "sils_mean":np.mean(np.mean(sils_all, 1)),
        "sils_std":np.std(np.mean(sils_all, 1)),
        "sils_all":sils_all,
        "cophenetic_coeff":coeff_k,
        "reordered_con_mat":reordered_con_mat,
        "col_err":curr_col_err,
    }

    if collect_output:
        results_k["W"] = W
        results_k["H"] = H
        results_k["other_results"] = other_results

    return results_k


class NMFk:
    def __init__(
            self,
            n_perturbs=20,
            n_iters=100,
            epsilon=0.015,
            perturb_type="uniform",
            n_jobs=1,
            n_nodes=1,
            init="nnsvd",
            use_gpu=True,
            save_path="./",
            save_output=True,
            collect_output=False,
            predict_k=False,
            predict_k_method="pvalue",
            verbose=False,
            nmf_verbose=False,
            transpose=False,
            sill_thresh=0.8,
            nmf_func=None,
            nmf_method="nmf_fro_mu",
            nmf_obj_params={},
            pruned=True,
            calculate_error=True,
            joblib_backend="multiprocessing",
            consensus_mat=False,
            use_consensus_stopping=0,
            mask=None,
            calculate_pac=False,
            get_plot_data=False,
            simple_plot=True,):
        """
        NMFk is a Non-negative Matrix Factorization module with the capability to do automatic model determination.

        Parameters
        ----------
        n_perturbs : int, optional
            Number of bootstrap operations, or random matrices generated around the original matrix. The default is 20.
        n_iters : int, optional
            Number of NMF iterations. The default is 100.
        epsilon : float, optional
            Error amount for the random matrices generated around the original matrix. The default is 0.015.
        perturb_type : str, optional
            Type of error sampling to perform for the bootstrap operation. The default is "uniform".\n
            * ``perturb_type='uniform'`` will use uniform distribution for sampling.\n
            * ``perturb_type='poisson'`` will use Poission distribution for sampling.\n
        n_jobs : int, optional
            Number of parallel jobs. Use -1 to use all available resources. The default is 1.
        n_nodes : int, optional
            Number of HPC nodes. The default is 1.
        init : str, optional
            Initilization of matrices for NMF procedure. The default is "nnsvd".\n
            * ``init='nnsvd'`` will use NNSVD for initilization.\n
            * ``init='random'`` will use random sampling for initilization.\n
        use_gpu : bool, optional
            If True, uses GPU for operations. The default is True.
        save_path : str, optional
            Location to save output. The default is "./".
        save_output : bool, optional
            If True, saves the resulting latent factors and plots. The default is True.
        collect_output : bool, optional
            If True, collectes the resulting latent factors to be returned from ``fit()`` operation. The default is False.
        predict_k : bool, optional
            If True, performs automatic prediction of the number of latent factors. The default is False.

            .. note::

                Even when ``predict_k=False``, number of latent factors can be estimated using the figures saved in ``save_path``.

        predict_k_method : str, optional
            Method to use when performing automatic k prediction. Default is "pvalue".\n
            * ``predict_k_method='pvalue'`` will use L-Statistics with column-wise error for automatically estimating the number of latent factors.\n
            * ``predict_k_method='sill'`` will use Silhouette score for estimating the number of latent factors.

            .. warning::

                ``predict_k_method='pvalue'`` prediction will result in significantly longer processing time! ``predict_k_method='sill'``, on the other hand, will be much faster.

        verbose : bool, optional
            If True, shows progress in each k. The default is False.
        nmf_verbose : bool, optional
            If True, shows progress in each NMF operation. The default is False.
        transpose : bool, optional
            If True, transposes the input matrix before factorization. The default is False.
        sill_thresh : float, optional
            Threshold for the Silhouette score when performing automatic prediction of the number of latent factors. The default is 0.8.
        nmf_func : object, optional
            If not None, and if ``nmf_method=func``, used for passing NMF function. The default is None.
        nmf_method : str, optional
            What NMF to use. The default is "nmf_fro_mu".\n
            * ``nmf_method='nmf_fro_mu'`` will use NMF with Frobenious Norm.\n
            * ``nmf_method='nmf_kl_mu'`` will use NMF with Multiplicative Update rules with KL-Divergence.\n
            * ``nmf_method='func'`` will use the custom NMF function passed using the ``nmf_func`` parameter.\n
            * ``nmf_method='nmf_recommender'`` will use the Recommender NMF method for collaborative filtering.\n
        nmf_obj_params : dict, optional
            Parameters used by NMF function. The default is {}.
        pruned : bool, optional
            When True, removes columns and rows from the input matrix that has only 0 values. The default is True.
        calculate_error : bool, optional
            When True, calculates the relative reconstruction error. The default is True.

            .. warning::
                If ``calculate_error=True``, it will result in longer processing time.

        joblib_backend : str, optional
            Backend used by Joblib for parallel computation. The default is "multiprocessing".
        consensus_mat : bool, optional
            When True, computes the Consensus Matrices for each k. The default is False.
        use_consensus_stopping : str, optional
            When not 0, uses Consensus matrices criteria for early stopping of NMF factorization. The default is 0.
        mask : ``np.ndarray``, optional
            Numpy array that points out the locations in input matrix that should be masked during factorization. The default is None.
        calculate_pac : bool, optional
            When True, calculates the PAC score for H matrix stability. The default is False.
        get_plot_data : bool, optional
            When True, collectes the data used in plotting each intermidiate k factorization. The default is False.
        simple_plot : bool, optional
            When True, creates a simple plot for each intermidiate k factorization which hides the statistics such as average and maximum Silhouette scores. The default is True.
        Returns
        -------
        None.

        """


        # check the save path
        if save_output:
            if not Path(save_path).is_dir():
                Path(save_path).mkdir(parents=True)

        init_options = ["nnsvd", "random"]
        if init not in init_options:
            raise Exception("Invalid init. Choose from:" + str(", ".join(init_options)))

        if n_nodes > 1 and MPI is None:
            sys.exit("Attempted to use n_nodes>1 but MPI is not available!")

        #
        # Object hyper-parameters
        #
        self.n_perturbs = n_perturbs
        self.perturb_type = perturb_type
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.init = init
        self.save_path = save_path
        self.save_output = save_output
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.nmf_verbose = nmf_verbose
        self.transpose = transpose
        self.collect_output = collect_output
        self.sill_thresh = sill_thresh
        self.predict_k = predict_k
        self.predict_k_method = predict_k_method
        self.n_jobs = n_jobs
        self.n_nodes = n_nodes
        self.nmf = None
        self.nmf_method = nmf_method
        self.nmf_obj_params = nmf_obj_params
        self.pruned = pruned
        self.calculate_error = calculate_error
        self.joblib_backend = joblib_backend
        self.consensus_mat = consensus_mat
        self.use_consensus_stopping = use_consensus_stopping
        self.mask = mask
        self.calculate_pac = calculate_pac
        self.simple_plot = simple_plot
        self.get_plot_data = get_plot_data

        # warnings
        assert self.predict_k_method in ["pvalue", "sill"]
        if self.calculate_pac and not self.consensus_mat:
            self.consensus_mat = True
            warnings.warn("consensus_mat was False when calculate_pac was True! consensus_mat changed to True.")

        if self.calculate_error:
            warnings.warn(
                "calculate_error is True! Error calculation can make the runtime longer and take up more memory space!")

        if self.predict_k and self.predict_k_method == "pvalue":
            warnings.warn(
                "predict_k is True with pvalue method! Predicting k can make the runtime significantly longer. Consider using predict_k_method='sill'.")

        # Check the number of perturbations is correct
        if self.n_perturbs < 2:
            raise Exception("n_perturbs should be at least 2!")

        # check that the perturbation type is valid
        assert perturb_type in [
            "uniform", "poisson"], "Invalid perturbation type. Choose from uniform, poisson"

        # organize n_jobs
        self.n_jobs, self.use_gpu = organize_n_jobs(use_gpu, n_jobs)
        if self.use_gpu:
            # multiprocessing on GPU
            if self.n_jobs < 0 or self.n_jobs > 1:
                multiprocessing.set_start_method('spawn', force=True)

        #
        # Save information from the solution
        #
        self.total_exec_seconds = 0
        self.experiment_name = ""

        #
        # Prepare NMF function
        #
        avail_nmf_methods = [
            "nmf_fro_mu", 
            "nmf_kl_mu", 
            "nmf_recommender", 
            "func"
        ]
        if self.nmf_method not in avail_nmf_methods:
            raise Exception("Invalid NMF method is selected. Choose from: " +
                            ",".join(avail_nmf_methods))
        if self.nmf_method == "nmf_fro_mu":
            self.nmf_params = {
                "niter": self.n_iters,
                "use_gpu": self.use_gpu,
                "nmf_verbose": self.nmf_verbose,
                "mask": self.mask,
                "use_consensus_stopping": self.use_consensus_stopping
            }
            self.nmf = nmf_fro_mu

        elif self.nmf_method == "nmf_kl_mu":
            self.nmf_params = {
                "niter": self.n_iters,
                "use_gpu": self.use_gpu,
                "nmf_verbose": self.nmf_verbose,
                "mask": self.mask,
                "use_consensus_stopping": self.use_consensus_stopping
            }
            self.nmf = nmf_kl_mu

        elif self.nmf_method == "func" or nmf_func is not None:
            self.nmf_params = self.nmf_obj_params
            self.nmf = nmf_func

        elif self.nmf_method == "nmf_recommender":
            self.nmf_params = {
                "niter": self.n_iters,
                "use_gpu": self.use_gpu,
                "nmf_verbose": self.nmf_verbose,
            }
            self.nmf = nmf_recommender

        else:
            raise Exception("Unknown NMF method or nmf_func was not passed")

        #
        # Additional NMF settings
        #
        if len(self.nmf_obj_params) > 0:
            for key, value in self.nmf_obj_params.items():
                if key not in self.nmf_params:
                    self.nmf_params[key] = value

        if self.verbose:
            print('Performing NMF with ', self.nmf_method)

    def fit(self, X, Ks, name="NMFk", note=""):
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
            Resulting dict can include all the latent factors, plotting data, predicted latent factors, time took for factorization, and predicted k value depending on the settings specified.\n
            * If ``get_plot_data=True``, results will include field for ``plot_data``.\n
            * If ``predict_k=True``, results will include field for ``k_predict``. This is an intiger for the automatically estimated number of latent factors.\n
            * If ``predict_k=True`` and ``collect_output=True``, results will include fields for ``W`` and ``H`` which are the latent factors in type of ``np.ndarray``.
            * results will always include a field for ``time``, that gives the total compute time.
        """

        if X.dtype != np.dtype(np.float32):
            warnings.warn(
                f'X is data type {X.dtype}. Whic is not float32. Higher precision will result in significantly longer runtime!')

        #
        # Error check
        #
        if len(Ks) == 0:
            raise Exception("Ks range is 0!")

        if max(Ks) >= min(X.shape):
            raise Exception("Maximum rank k to try in Ks should be k<min(X.shape)")

        #
        # MPI
        #
        if self.n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            Ks = self.__chunk_Ks(Ks, n_chunks=self.n_nodes)[rank]
            if self.verbose:
                print("Rank=", rank, "Host=", socket.gethostname(), "Ks=", Ks)
        else:
            comm = None
            rank = 0

        #
        # Setup
        # 
        self.experiment_name = (
            str(name)
            + "_"
            + str(self.n_perturbs)
            + "perts_"
            + str(self.n_iters)
            + "iters_"
            + str(self.epsilon)
            + "eps_"
            + str(self.init)
            + "-init"
        )
        save_path = os.path.join(self.save_path, self.experiment_name)

        if self.n_jobs > len(Ks):
            self.n_jobs = len(Ks)
            
        if self.transpose:
            if isinstance(X, np.ndarray):
                X = X.T
            elif scipy.sparse.issparse(X):
                X = X.T.asformat("csr")
            else:
                raise Exception("I do not know how to transpose type " + str(type(X)))

        # init the stats header 
        # this will setup the logging for all configurations of nmfk
        stats_header = {'k': 'k', 
                        'sils_min': 'Min. Silhouette', 
                        'sils_mean': 'Mean Silhouette'}
        if self.calculate_error:
            stats_header['err_mean'] = 'Mean Error'
            stats_header['err_std'] = 'STD Error'
        if self.predict_k:
            stats_header['col_error'] = 'Mean Col. Error'
        if self.calculate_pac:
            stats_header['pac'] = 'PAC'
        stats_header['time'] = 'Time Elapsed'

        # start the file logging (only root node needs to do this step)
        if self.save_output and ((self.n_nodes == 1) or (self.n_nodes > 1 and rank == 0)):
            if not Path(save_path).is_dir():
                Path(save_path).mkdir(parents=True)

            append_to_note(["#" * 100], save_path)
            append_to_note(["start_time= " + str(datetime.now()),
                            "name=" + str(name),
                            "note=" + str(note)], save_path)

            append_to_note(["#" * 100], save_path)
            object_notes = vars(self).copy()
            del object_notes["total_exec_seconds"]
            del object_notes["nmf"]
            take_note(object_notes, save_path)
            append_to_note(["#" * 100], save_path)

            notes = {}
            notes["Ks"] = Ks
            notes["data_type"] = type(X)
            notes["num_elements"] = np.prod(X.shape)
            notes["num_nnz"] = len(X.nonzero()[0])
            notes["sparsity"] = len(X.nonzero()[0]) / np.prod(X.shape)
            notes["X_shape"] = X.shape
            take_note(notes, save_path)
            append_to_note(["#" * 100], save_path)
            take_note_fmat(save_path, **stats_header)
        
        if self.n_nodes > 1:
            comm.Barrier()
            
        #
        # Prune
        #
        if self.pruned:
            X, perturb_rows, perturb_cols = prune(X, use_gpu=self.use_gpu)
        else:
            perturb_rows, perturb_cols = None, None

        #
        # Begin NMFk
        #
        start_time = time.time()

        job_data = {
            "n_perturbs":self.n_perturbs,
            "nmf":self.nmf,
            "nmf_params":self.nmf_params,
            "init_type":self.init,
            "X":X,
            "epsilon":self.epsilon,
            "use_gpu":self.use_gpu,
            "perturb_type":self.perturb_type,
            "calculate_error":self.calculate_error,
            "mask":self.mask,
            "consensus_mat":self.consensus_mat,
            "predict_k":self.predict_k,
            "predict_k_method":self.predict_k_method,
            "pruned":self.pruned,
            "perturb_rows":perturb_rows,
            "perturb_cols":perturb_cols,
            "save_output":self.save_output,
            "save_path":save_path,
            "experiment_name":self.experiment_name,
            "collect_output":self.collect_output,
            "logging_stats":stats_header,
            "start_time":start_time,
            
        }

        if self.n_jobs == 1:
            all_k_results = []
            for k in tqdm(Ks, total=len(Ks), disable=not self.verbose):
                k_result = _nmf_parallel_wrapper(gpuid=0, k=k, **job_data)
                all_k_results.append(k_result)
        else:   
            all_k_results = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                backend=self.joblib_backend)(delayed(_nmf_parallel_wrapper)(
                    gpuid=kidx % self.n_jobs, k=k, **job_data) for kidx, k in enumerate(Ks))

        #
        # Collect results if multi-node
        #
        if self.n_nodes > 1:
            comm.Barrier()
            all_share_data = comm.gather(all_k_results, root=0)
            all_k_results = []
            if rank == 0:
                for node_k_results in all_share_data:
                    all_k_results.extend(node_k_results)
            else:
                sys.exit(0)

        #
        # Sort results
        #
        collected_Ks = []
        for k_results in all_k_results:
            collected_Ks.append(k_results["Ks"])

        all_k_results_tmp = []
        Ks_sort_indices = np.argsort(np.array(collected_Ks))
        for idx in Ks_sort_indices:
            all_k_results_tmp.append(all_k_results[idx])
        all_k_results = all_k_results_tmp

        #
        # combine results
        #
        combined_result = defaultdict(list)
        for k_results in all_k_results:
            for key, value in k_results.items():
                combined_result[key].append(value)
                
        #
        # revent to original Ks
        #
        if self.n_nodes > 1:
            Ks = np.array(collected_Ks)[Ks_sort_indices]
            
        #
        # Finalize
        # 
        if self.n_nodes == 1 or (self.n_nodes > 1 and rank == 0):

            # holds the final results
            results = {}
            total_exec_seconds = time.time() - start_time
            results["time"] = total_exec_seconds

            # predict k for W
            if self.predict_k:
                if self.predict_k_method == "pvalue":
                    k_predict = pvalue_analysis(
                        combined_result["col_err"], Ks, combined_result["sils_min"], SILL_thr=self.sill_thresh
                    )[0]
                elif self.predict_k_method == "sill":
                    k_predict = Ks[np.max(np.argwhere(
                        np.array(combined_result["sils_min"]) >= self.sill_thresh).flatten())]
            else:
                k_predict = 0
                
            
            # * plot cophenetic coefficients
            combined_result["pac"] = []
            if self.consensus_mat:

                # * save the plot
                if self.save_output:
                    con_fig_name = f'{save_path}/k_{Ks[0]}_{Ks[-1]}_cophenetic_coeff.png'
                    plot_cophenetic_coeff(Ks, combined_result["cophenetic_coeff"], con_fig_name)

                if self.calculate_pac:
                    consensus_tensor = np.array(combined_result["reordered_con_mat"])
                    combined_result["pac"] = np.array(get_pac(consensus_tensor, use_gpu=self.use_gpu))

            # save k prediction
            if self.predict_k:
                results["k_predict"] = k_predict

                if self.collect_output:
                        results["W"] = combined_result["W"][combined_result["Ks"].index(k_predict)]
                        results["H"] = combined_result["H"][combined_result["Ks"].index(k_predict)]

                        if self.nmf_method == "nmf_recommender":
                            results["other_results"] = combined_result["other_results"][combined_result["Ks"].index(k_predict)]

            # final plot
            if self.save_output:
                plot_NMFk(
                    combined_result, 
                    k_predict, 
                    self.experiment_name, 
                    save_path, 
                    plot_predict=self.predict_k,
                    plot_final=True,
                    simple_plot=self.simple_plot
                )
                append_to_note(["#" * 100], save_path)
                append_to_note(["end_time= "+str(datetime.now())], save_path)
                append_to_note(
                    ["total_time= "+str(time.time() - start_time) + " (seconds)"], save_path)
        
            
            if self.get_plot_data:
                results["plot_data"] = combined_result
                
            return results

    def __chunk_Ks(self, Ks: list, n_chunks=2) -> list:
        # correct n_chunks if needed
        if len(Ks) < n_chunks:
            n_chunks = len(Ks)

        chunks = list()
        for _ in range(n_chunks):
            chunks.append([])

        for idx, ii in enumerate(Ks):
            chunk_idx = idx % n_chunks
            chunks[chunk_idx].append(ii)

        return chunks
