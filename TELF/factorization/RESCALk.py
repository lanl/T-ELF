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
from .utilities.plot_NMFk import plot_RESCALk
from .utilities.organize_n_jobs import organize_n_jobs
from .utilities.data_host_transfer_helpers import put_A_cpu, put_A_gpu, put_tensor_X_gpu, put_tensor_R_cpu
from .utilities.run_factorization_helpers import run_rescal
from .utilities.perturbation_helpers import perturb_tensor_X
from .utilities.initialization_helpers import init_A
from .utilities.regression_helpers import R_regression
from .decompositions.rescal_fro_mu import rescal as rescal_fro_mu
from .decompositions.utilities.clustering import custom_k_means
from .decompositions.utilities.silhouettes import silhouettes
from .decompositions.utilities.math_utils import relative_error_rescal


import concurrent.futures
from threading import Lock
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import os
from tqdm import tqdm
import numpy as np
import scipy.sparse
import warnings
import time
import socket
from pathlib import Path


try:
    import cupy as cp
except Exception:
    cp = None
    cupyx = None

try:
    from mpi4py import MPI
except Exception:
    MPI = None
    

def _perturb_parallel_wrapper(
    perturbation,
    gpuid,
    epsilon,
    perturb_type,
    X,
    k,
    use_gpu,
    init_type,
    rescal_params,
    rescal,
    calculate_error):

    # Prepare
    Y = perturb_tensor_X(X, perturbation, epsilon, perturb_type)
    A_init = init_A(Y, k, init_type=init_type)

    # transfer to GPU
    if use_gpu:
        Y = put_tensor_X_gpu(Y, gpuid)
        A_init = put_A_gpu(A_init, gpuid)

    A, R = run_rescal(Y, A_init, rescal, rescal_params, use_gpu, gpuid)

    # transfer to CPU
    if use_gpu:
        A = put_A_cpu(A)
        R = put_tensor_R_cpu(R)
        cp._default_memory_pool.free_all_blocks()

    # error calculation
    if calculate_error:
        error = relative_error_rescal(X, A, R)
    else:
        error = 0
        
    return A, R, error
        

def _rescal_parallel_wrapper(
        n_perturbs, 
        rescal, 
        rescal_params,
        init_type="nnsvd", 
        X=None, 
        k=None,
        epsilon=None, 
        gpuid=0, 
        use_gpu=True,
        perturb_type="uniform", 
        calculate_error=True, 
        pruned=True,
        perturb_rows=None,
        perturb_cols=None,
        save_output=True,
        save_path="",
        logging_stats={},
        start_time=time.time(),
        n_jobs=1,
        perturb_multiprocessing=False,
        perturb_verbose=False,
        lock=None,
        note_name="experiment"):

    #
    # run for each perturbations
    #
    perturb_job_data = {
        "epsilon":epsilon,
        "perturb_type":perturb_type,
        "X":X,
        "use_gpu":use_gpu,
        "rescal_params":rescal_params,
        "rescal":rescal,
        "calculate_error":calculate_error,
        "k":k,
        "init_type":init_type
    }
    
    # single job or parallel over Ks
    A_all, R_all, errors = [], [], []
    if n_jobs == 1 or not perturb_multiprocessing:
        for perturbation in tqdm(range(n_perturbs), disable=not perturb_verbose, total=n_perturbs):
            A, R, error = _perturb_parallel_wrapper(perturbation=perturbation, gpuid=gpuid, **perturb_job_data)
            A_all.append(A)
            R_all.append(R)
            errors.append(error)
            
    # multiple jobs over perturbations
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs)
        futures = [executor.submit(_perturb_parallel_wrapper, gpuid=pidx % n_jobs, perturbation=perturbation, **perturb_job_data) for pidx, perturbation in enumerate(range(n_perturbs))]
        all_perturbation_results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), disable=not perturb_verbose, total=n_perturbs)]
        for A, R, error in all_perturbation_results:
            A_all.append(A)
            R_all.append(A)
            errors.append(error)
    
    #
    # organize colutions from each perturbations
    #
    A_all = np.array(A_all).transpose((1, 2, 0))
    R_all = np.array(R_all)
    errors = np.array(errors)

    #
    # cluster the solutions
    #        
    A, A_clust = custom_k_means(A_all, use_gpu=False)
    sils_all = silhouettes(A_clust)

    #
    # Regress H
    #
    R = R_regression(X, A, use_gpu, gpuid)
    
    if use_gpu:
        cp._default_memory_pool.free_all_blocks()

    # 
    #  reconstruction error
    #
    if calculate_error:
        error_reg = relative_error_rescal(X, A, R)
    else:
        error_reg = 0

    #
    # unprune
    #
    if pruned:
        pass
        #W = unprune(W, perturb_rows, 0)
        #H = unprune(H, perturb_cols, 1)

    #
    # save output factors and the plot
    #
    if save_output:
        
        save_data = {
            "A": A,
            "R": R,
            "sils_all": sils_all,
            "error_reg": error_reg,
            "errors": errors,
        }
        np.savez_compressed( 
            os.path.join(f'{save_path}', f'AR_k={k}.npz'),
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
            elif key == 'time':
                elapsed_time = time.time() - start_time
                elapsed_time = timedelta(seconds=elapsed_time)
                plot_data["time"] = str(elapsed_time).split('.')[0]
            else:
                warnings.warn(f'[tELF]: Encountered unknown logging metric "{key}"', RuntimeWarning)
                plot_data[key] = 'N/A'
        take_note_fmat(save_path, lock=lock, name=note_name, **plot_data)

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
    }

    return results_k


class RESCALk:
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
            save_path="",
            save_output=True,
            verbose=True,
            rescal_verbose=False,
            perturb_verbose=False,
            rescal_func=None,
            rescal_method="rescal_fro_mu",
            rescal_obj_params={},
            pruned=False,
            calculate_error=False,
            perturb_multiprocessing=False,
            get_plot_data=False,
            simple_plot=True,):
        """
        RESCALk is a RESCAL module with the capability to do automatic model determination.

        Parameters
        ----------
        n_perturbs : int, optional
            Number of bootstrap operations, or random matrices generated around the original matrix. The default is 20.
        n_iters : int, optional
            Number of RESCAL iterations. The default is 100.
        epsilon : float, optional
            Error amount for the random matrices generated around the original matrix. The default is 0.015.\n
            ``epsilon`` is used when ``perturb_type='uniform'``.
        perturb_type : str, optional
            Type of error sampling to perform for the bootstrap operation. The default is "uniform".\n
            * ``perturb_type='uniform'`` will use uniform distribution for sampling.\n
            * ``perturb_type='poisson'`` will use Poission distribution for sampling.\n
        n_jobs : int, optional
            Number of parallel jobs. Use -1 to use all available resources. The default is 1.
        n_nodes : int, optional
            Number of HPC nodes. The default is 1.
        init : str, optional
            Initilization of matrices for RESCAL procedure. The default is "nnsvd".\n
            * ``init='nnsvd'`` will use NNSVD for initilization.\n
            * ``init='random'`` will use random sampling for initilization.\n
        use_gpu : bool, optional
            If True, uses GPU for operations. The default is True.
        save_path : str, optional
            Location to save output. The default is "".
        save_output : bool, optional
            If True, saves the resulting latent factors and plots. The default is True.

        verbose : bool, optional
            If True, shows progress in each k. The default is True.
        rescal_verbose : bool, optional
            If True, shows progress in each RESCAL operation. The default is False.
        perturb_verbose : bool, optional
            If True, it shows progress in each perturbation. The default is False.
        rescal_func : object, optional
            If not None, and if ``rescal_method=func``, used for passing RESCAL function. The default is None.
        rescal_method : str, optional
            What RESCAL to use. The default is "rescal_fro_mu".\n
            * ``rescal_method='rescal_fro_mu'`` will use RESCAL with Frobenious Norm.\n                

        rescal_obj_params : dict, optional
            Parameters used by RESCAL function. The default is {}.
        pruned : bool, optional
            When True, removes columns and rows from the input matrix that has only 0 values. The default is False.

            .. warning::
                Pruning is not implemented for RESCALk yet.

        calculate_error : bool, optional
            When True, calculates the relative reconstruction error. The default is False.

            .. warning::
                If ``calculate_error=True``, it will result in longer processing time.

        perturb_multiprocessing : bool, optional
            If ``perturb_multiprocessing=True``, it will make parallel computation over each perturbation. Default is ``perturb_multiprocessing=False``.\n
            When ``perturb_multiprocessing=False``, which is default, parallelization is done over each K (rank).
        get_plot_data : bool, optional
            When True, collectes the data used in plotting each intermidiate k factorization. The default is False.
        simple_plot : bool, optional
            When True, creates a simple plot for each intermidiate k factorization which hides the statistics such as average and maximum Silhouette scores. The default is True.
        Returns
        -------
        None.

        """

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
        self.rescal_verbose = rescal_verbose
        self.perturb_verbose = perturb_verbose
        self.n_jobs = n_jobs
        self.n_nodes = n_nodes
        self.rescal = None
        self.rescal_method = rescal_method
        self.rescal_obj_params = rescal_obj_params
        self.pruned = pruned
        self.calculate_error = calculate_error
        self.simple_plot = simple_plot
        self.get_plot_data = get_plot_data
        self.perturb_multiprocessing = perturb_multiprocessing

        if self.pruned:
            warnings.warn("Pruning for RESCAL is not implemented yet!")

        if self.calculate_error:
            warnings.warn(
                "calculate_error is True! Error calculation can make the runtime longer and take up more memory space!")

        # Check the number of perturbations is correct
        if self.n_perturbs < 2:
            raise Exception("n_perturbs should be at least 2!")

        # check that the perturbation type is valid
        assert perturb_type in [
            "uniform", "poisson"], "Invalid perturbation type. Choose from uniform, poisson"

        # organize n_jobs
        self.n_jobs, self.use_gpu = organize_n_jobs(use_gpu, n_jobs)
        
        # create a shared lock
        self.lock = Lock()

        #
        # Save information from the solution
        #
        self.total_exec_seconds = 0
        self.experiment_name = ""

        #
        # Prepare RESCAL function
        #
        avail_rescal_methods = [
            "rescal_fro_mu", 
            "func"
        ]
        if self.rescal_method not in avail_rescal_methods:
            raise Exception("Invalid RESCAL method is selected. Choose from: " +
                            ",".join(avail_rescal_methods))
        
        if self.rescal_method == "rescal_fro_mu":
            self.rescal_params = {
                "niter": self.n_iters,
                "use_gpu": self.use_gpu,
                "rescal_verbose": self.rescal_verbose,
            }
            self.rescal = rescal_fro_mu

        elif self.rescal_method == "func" or rescal_func is not None:
            self.rescal_params = self.rescal_obj_params
            self.rescal = rescal_func

        else:
            raise Exception("Unknown RESCAL method or rescal_func was not passed")

        #
        # Additional RESCAL settings
        #
        if len(self.rescal_obj_params) > 0:
            for key, value in self.rescal_obj_params.items():
                if key not in self.rescal_params:
                    self.rescal_params[key] = value
                    
        if self.verbose:
            for key, value in vars(self).items():
                print(f'{key}:', value)

    def fit(self, X, Ks, name="RESCALk", note=""):
        """
        Factorize the input matrix ``X`` for the each given K value in ``Ks``.

        Parameters
        ----------
        X : list of symmetric ``np.ndarray`` or list of symmetric ``scipy.sparse._csr.csr_matrix`` matrix
            Input matrix to be factorized.
        Ks : list
            List of K values to factorize the input matrix.\n
            **Example:** ``Ks=range(1, 10, 1)``.
        name : str, optional   
            Name of the experiment. Default is "RESCALk".
        note : str, optional
            Note for the experiment used in logs. Default is "".
        
        Returns
        -------
        results : dict
            Resulting dict can include all the latent factors, plotting data, predicted latent factors, time took for factorization, and predicted k value depending on the settings specified.\n
            * If ``get_plot_data=True``, results will include field for ``plot_data``.\n
            * results will always include a field for ``time``, that gives the total compute time.
        """

        #
        # check X format
        #
        assert type(X) == list, "X sould be list of np.ndarray or scipy.sparse._csr.csr_matrix"
        # make sure csr or numpy array
        expected_type = type(X[0])
        assert expected_type == scipy.sparse._csr.csr_matrix or expected_type == np.ndarray, "X sould be list of np.ndarray or scipy.sparse._csr.csr_matrix"
        # make sure all slices are expected type
        for slice_idx, x in enumerate(X):
            assert expected_type == type(x) or expected_type == type(x), f'X sould be list of all same type (np.ndarray or scipy.sparse._csr.csr_matrix). Matrix at slice index {slice_idx} did not match others.'

        if X[0].dtype != np.dtype(np.float32):
            warnings.warn(
                f'X is data type {X[0].dtype}. Whic is not float32. Higher precision will result in significantly longer runtime!')

        #
        # Error check
        #
        if len(Ks) == 0:
            raise Exception("Ks range is 0!")

        if max(Ks) >= min(X[0].shape):
            raise Exception("Maximum rank k to try in Ks should be k<min(X.shape)")

        #
        # MPI
        #
        if self.n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            Ks = self.__chunk_Ks(Ks, n_chunks=self.n_nodes)[rank]
            note_name = f'{rank}_experiment'
            if self.verbose:
                print("Rank=", rank, "Host=", socket.gethostname(), "Ks=", Ks)
        else:
            note_name = f'experiment'
            comm = None
            rank = 0
            
        #
        # Organize save path
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
        self.save_path_full = os.path.join(self.save_path, self.experiment_name)

        #
        # Setup
        # 
        if (self.n_jobs > len(Ks)) and not self.perturb_multiprocessing:
            self.n_jobs = len(Ks)
        elif (self.n_jobs > self.n_perturbs) and self.perturb_multiprocessing:
            self.n_jobs = self.n_perturbs

        # init the stats header 
        # this will setup the logging for all configurations of rescalk
        stats_header = {'k': 'k', 
                        'sils_min': 'Min. Silhouette', 
                        'sils_mean': 'Mean Silhouette'}
        if self.calculate_error:
            stats_header['err_mean'] = 'Mean Error'
            stats_header['err_std'] = 'STD Error'
        stats_header['time'] = 'Time Elapsed'

        # start the file logging (only root node needs to do this step)
        if self.save_output and ((self.n_nodes == 1) or (self.n_nodes > 1 and rank == 0)):
            try:
                if not Path(self.save_path_full).is_dir():
                    Path(self.save_path_full).mkdir(parents=True)
            except Exception as e:
                print(e)
                
        if self.n_nodes > 1:
            comm.Barrier()
            time.sleep(1)

        # logging
        if self.save_output:

            append_to_note(["#" * 100], self.save_path_full, name=note_name, lock=self.lock)
            append_to_note(["start_time= " + str(datetime.now()),
                            "name=" + str(name),
                            "note=" + str(note)], self.save_path_full, name=note_name, lock=self.lock)

            append_to_note(["#" * 100], self.save_path_full, name=note_name, lock=self.lock)
            object_notes = vars(self).copy()
            del object_notes["total_exec_seconds"]
            del object_notes["rescal"]
            take_note(object_notes, self.save_path_full, name=note_name, lock=self.lock)
            append_to_note(["#" * 100], self.save_path_full, lock=self.lock)

            notes = {}
            notes["Ks"] = Ks
            notes["data_type"] = type(X)
            notes["num_perturbations"] = self.n_perturbs
            notes["epsilon"] = self.epsilon
            notes["init"] = self.init
            notes["n_jobs"] = self.n_jobs
            notes["experiment_name"] = name
            notes["num_iterations"] = self.n_iters
            take_note(notes, self.save_path_full, name=note_name, lock=self.lock)
            append_to_note(["#" * 100], self.save_path_full, name=note_name, lock=self.lock)
            take_note_fmat(self.save_path_full, lock=self.lock, name=note_name, **stats_header)
        
        if self.n_nodes > 1:
            comm.Barrier()
            
        #
        # Prune
        #
        if self.pruned:
            perturb_rows, perturb_cols = None, None
            #X, perturb_rows, perturb_cols = prune(X, use_gpu=self.use_gpu)
        else:
            perturb_rows, perturb_cols = None, None

        #
        # Begin RESCALk
        #
        start_time = time.time()

        job_data = {
            "n_perturbs":self.n_perturbs,
            "rescal":self.rescal,
            "rescal_params":self.rescal_params,
            "init_type":self.init,
            "X":X,
            "epsilon":self.epsilon,
            "use_gpu":self.use_gpu,
            "perturb_type":self.perturb_type,
            "calculate_error":self.calculate_error,
            "pruned":self.pruned,
            "perturb_rows":perturb_rows,
            "perturb_cols":perturb_cols,
            "save_output":self.save_output,
            "save_path":self.save_path_full,
            "logging_stats":stats_header,
            "start_time":start_time,
            "n_jobs":self.n_jobs,
            "perturb_multiprocessing":self.perturb_multiprocessing,
            "perturb_verbose":self.perturb_verbose,
            "lock":self.lock,
            "note_name":note_name
        }
        
        # Single job or parallel over perturbations
        if self.n_jobs == 1 or self.perturb_multiprocessing:
            all_k_results = []
            for k in tqdm(Ks, total=len(Ks), disable=not self.verbose):
                k_result = _rescal_parallel_wrapper(gpuid=0, k=k, **job_data)
                all_k_results.append(k_result)
        
        # multiprocessing over each K
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs)
            futures = [executor.submit(_rescal_parallel_wrapper, gpuid=kidx % self.n_jobs, k=k, **job_data) for kidx, k in enumerate(Ks)]
            all_k_results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(Ks), disable=not self.verbose)]

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

            # final plot
            if self.save_output:
                plot_RESCALk(
                    combined_result, 
                    0, 
                    self.experiment_name, 
                    self.save_path_full, 
                    plot_predict=False,
                    plot_final=True,
                    simple_plot=self.simple_plot
                )
                append_to_note(["#" * 100], self.save_path_full, name=note_name, lock=self.lock)
                append_to_note(["end_time= "+str(datetime.now())], self.save_path_full, name=note_name, lock=self.lock)
                append_to_note(
                    ["total_time= "+str(time.time() - start_time) + " (seconds)"], self.save_path_full, name=note_name, lock=self.lock)
        
            
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
