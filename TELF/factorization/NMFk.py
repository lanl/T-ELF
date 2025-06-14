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
from .utilities.organize_n_jobs import organize_n_jobs, organize_devices
from .utilities.data_host_transfer_helpers import put_X_gpu, put_A_gpu, put_A_cpu, put_other_results_cpu
from .utilities.run_factorization_helpers import run_nmf, run_thresholding
from .utilities.perturbation_helpers import perturb_X
from .utilities.initialization_helpers import init_WH
from .utilities.regression_helpers import H_regression
from .utilities.bst_helper import BST
from .utilities.factor_thresholding import (find_thres_WH, coord_desc_thresh, 
                                            coord_desc_thresh_onefactor, otsu_thresh, 
                                            otsu_thresh_onefactor, kmeans_thresh_onefactor, kmeans_thresh)
from .decompositions.nmf_kl_mu import nmf as nmf_kl_mu
from .decompositions.nmf_fro_mu import nmf as nmf_fro_mu
from .decompositions.wnmf import nmf as wnmf
from .decompositions.bnmf import nmf as bnmf
from .decompositions.nmf_recommender import nmf as nmf_recommender
from .decompositions.utilities.clustering import custom_k_means, custom_bool_clustering
from .decompositions.utilities.silhouettes import silhouettes, silhouettes_with_distance
from .decompositions.utilities.math_utils import prune, unprune, relative_error, get_pac
from .decompositions.utilities.concensus_matrix import compute_consensus_matrix, reorder_con_mat
from ..helpers.data_structures import chunk_Ks

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
from pathlib import Path
import concurrent.futures
from threading import Lock
import pickle
import fcntl
import tempfile

try:
    import cupy as cp
except Exception:
    cp = None
    cupyx = None

try:
    from mpi4py import MPI
except Exception:
    MPI = None

def _HPC_K_search_settings_update(n_nodes, my_rank, K_search_settings, save_path):
    for node_rank in range(n_nodes): 
        if (node_rank == my_rank) or (not Path(os.path.join(f'{save_path}', f'{node_rank}-K_search_settings.p')).is_file()):
            continue
        
        with open(os.path.join(f'{save_path}', f'{node_rank}-K_search_settings.p'), 'rb') as file:
            fcntl.flock(file, fcntl.LOCK_SH)
            node_settings = pickle.load(file)
            fcntl.flock(file, fcntl.LOCK_UN)

        if node_settings["k_min"] > K_search_settings["k_min"]:
            K_search_settings["k_min"] = node_settings["k_min"]

        if node_settings["k_max"] < K_search_settings["k_max"]:
            K_search_settings["k_max"] = node_settings["k_max"]

def _perturb_parallel_wrapper(
    perturbation,
    gpuid,
    epsilon,
    perturb_type,
    X,
    k,
    mask,
    use_gpu,
    init_type,
    nmf_params,
    nmf,
    calculate_error,
    K_search_settings,
    n_nodes,
    my_rank,
    save_path,
    factor_thresholding_obj_params,
    thresholding_function,
    ):
    
    if K_search_settings["k_search_method"] != "linear":
        with K_search_settings['lock']:
            
            if n_nodes > 1:
                _HPC_K_search_settings_update(n_nodes, my_rank, K_search_settings, save_path)

            if k <= K_search_settings["k_min"] or k >= K_search_settings["k_max"]:
                return {"exit_early": True, "perturbation":perturbation}

    # Prepare
    Y = perturb_X(X, perturbation, epsilon, perturb_type)
    W_init, H_init = init_WH(Y, k, mask=mask, init_type=init_type)

    # transfer to GPU
    if use_gpu:
        Y = put_X_gpu(Y, gpuid)
        W_init, H_init = put_A_gpu(W_init, gpuid), put_A_gpu(H_init, gpuid)
        
        if "MASK" in nmf_params and nmf_params["MASK"] is not None:
            nmf_params = nmf_params.copy()
            nmf_params["MASK"] = put_X_gpu(nmf_params["MASK"], gpuid)

    W, H, other_results = run_nmf(Y, W_init, H_init, nmf, nmf_params, use_gpu, gpuid)
    del W_init
    del H_init

    # boolean thresholding if specified
    if thresholding_function is not None:
        W, H = run_thresholding(Y, W, H, factor_thresholding_obj_params, thresholding_function, use_gpu, gpuid)

    # transfer to CPU
    del Y # done with this
    if use_gpu:
        W, H = put_A_cpu(W), put_A_cpu(H)
        other_results = put_other_results_cpu(other_results)
        cp._default_memory_pool.free_all_blocks()

    # error calculation
    if calculate_error:
        if use_gpu:
            if "MASK" in nmf_params and nmf_params["MASK"] is not None:
                nmf_params["MASK"] = put_A_cpu(nmf_params["MASK"])

        error = relative_error(
            X=X, W=W, H=H, 
            MASK=nmf_params["MASK"] if "MASK" in nmf_params else None)
    else:
        error = 0
        
    return {"exit_early":False, "W":W, "H":H, "error":error, "other_results":other_results}

def _take_exit_note(k, n_perturbs, logging_stats, start_time, save_path, note_name, lock):
    note_data = dict()
    for key in logging_stats:
        if key == 'k':
            note_data["k"] = k
        elif key ==  'sils_min_W':
            note_data["sils_min_W"] = "--"
        elif key == 'sils_mean_W':
            note_data["sils_mean_W"] = "--"
        elif key ==  'sils_min_H':
            note_data["sils_min_H"] = "--"
        elif key == 'sils_mean_H':
            note_data["sils_mean_H"] = "--"
        elif key == 'err_mean':
            note_data["err_mean"] = "--"
        elif key == 'err_std':
            note_data["err_std"] = "--"
        elif key == 'col_error':
            note_data["col_err"] = "--"
        elif key == "Done":
                note_data["Done"] = "N"
        elif key == "Perturbs":
                note_data["Perturbs"] = f'{n_perturbs}'
        elif key == 'time':
            elapsed_time = time.time() - start_time
            elapsed_time = timedelta(seconds=elapsed_time)
            note_data["time"] = str(elapsed_time).split('.')[0]
        else:
            warnings.warn(f'[tELF]: Encountered unknown logging metric "{key}"', RuntimeWarning)
            note_data[key] = 'N/A'
    take_note_fmat(save_path, name=note_name, lock=lock, **note_data)
        

def _nmf_parallel_wrapper(
        n_perturbs, 
        nmf, 
        nmf_params,
        clustering_obj_params,
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
        predict_k_method="WH_sill",
        pruned=True,
        pruned_rows=None,
        pruned_cols=None,
        save_output=True,
        save_path="",
        collect_output=False,
        logging_stats={},
        start_time=time.time(),
        n_jobs=1,
        perturb_multiprocessing=False,
        perturb_verbose=False,
        lock=None,
        note_name="experiment",
        K_search_settings=None,
        n_nodes=1,
        rank=0,
        factor_thresholding_obj_params={},
        factor_thresholding_H_regression_obj_params={},
        clustering_method="kmeans",
        thresholding_function=None,
        factor_thresholding_H_regression=None
        ):

    #
    # run for each perturbations
    #
    perturb_job_data = {
        "epsilon":epsilon,
        "perturb_type":perturb_type,
        "X":X,
        "use_gpu":use_gpu,
        "nmf_params":nmf_params,
        "nmf":nmf,
        "calculate_error":calculate_error,
        "k":k,
        "mask":mask,
        "init_type":init_type,
        "n_nodes":n_nodes,
        "my_rank":rank,
        "save_path":save_path,
        "factor_thresholding_obj_params":factor_thresholding_obj_params,
        "thresholding_function":thresholding_function,
    }
    
    # single job or parallel over Ks
    W_all, H_all, errors, other_results_all = [], [], [], []
    if n_jobs == 1 or not perturb_multiprocessing:
        for perturbation in tqdm(range(n_perturbs), disable=not perturb_verbose, total=n_perturbs):

            curr_perturbation_results = _perturb_parallel_wrapper(
                perturbation=perturbation, 
                gpuid=gpuid, 
                K_search_settings=K_search_settings,
                **perturb_job_data)

            # check for early termination
            if curr_perturbation_results["exit_early"]:
                if save_output:
                    _take_exit_note(k, perturbation, logging_stats, start_time, save_path, note_name, lock)
                return {"Ks":k, "exit_early":True}

            # did not exit early
            W = curr_perturbation_results["W"]
            H = curr_perturbation_results["H"]
            error = curr_perturbation_results["error"]
            other_results_curr = curr_perturbation_results["other_results"]

            W_all.append(W)
            H_all.append(H)
            errors.append(error)
            other_results_all.append(other_results_curr)
            
    # multiple jobs over perturbations
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs)
        futures = [executor.submit(
            _perturb_parallel_wrapper, 
            gpuid=pidx % n_jobs, 
            perturbation=perturbation, 
            K_search_settings=K_search_settings, 
            **perturb_job_data) for pidx, perturbation in enumerate(range(n_perturbs))]
        all_perturbation_results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), disable=not perturb_verbose, total=n_perturbs)]

        # check for early termination
        for curr_perturbation_results in all_perturbation_results:
            if curr_perturbation_results["exit_early"]:
                if save_output:
                    exit_perturbation = curr_perturbation_results["perturbation"]
                    _take_exit_note(k, exit_perturbation, logging_stats, start_time, save_path, note_name, lock)
                return {"Ks":k, "exit_early":True}

        # did not exit early
        for curr_perturbation_results in all_perturbation_results:
            W_all.append(curr_perturbation_results["W"])
            H_all.append(curr_perturbation_results["H"])
            errors.append(curr_perturbation_results["error"])
            other_results_all.append(curr_perturbation_results["other_results"])
    
    #
    # Organize other results
    #
    other_results = {}
    for other_results_curr in other_results_all:
        for key, value in other_results_curr.items():
            if key not in other_results:
                other_results[key] = value
            else:
                other_results[key] = (other_results[key] + value) / 2
    
    #
    # organize colutions from each perturbations
    #
    W_all = np.array(W_all).transpose((1, 2, 0))
    H_all = np.array(H_all).transpose((1, 2, 0))
    errors = np.array(errors)

    #
    # cluster the solutions
    #
    if clustering_method == "bool" or clustering_method == "boolean":
        W, W_clust = custom_bool_clustering(W_all, use_gpu=False, **clustering_obj_params)
        sils_all_W = silhouettes_with_distance(W_all, distance=clustering_obj_params["distance"], use_gpu=False)
        sils_all_H = silhouettes_with_distance(W_all, distance=clustering_obj_params["distance"], use_gpu=False)
        
    elif clustering_method == "kmeans":
        W, W_clust = custom_k_means(W_all, use_gpu=False, **clustering_obj_params)
        sils_all_W = silhouettes(W_clust, use_gpu=False)
        sils_all_H = silhouettes(np.array(H_all).transpose((1, 0, 2)), use_gpu=False)
    
    else:
        raise Exception("Unknown clustering method!")

    #
    # concensus matrix
    #
    coeff_k = 0
    reordered_con_mat = None
    if consensus_mat:
        con_mat_k = compute_consensus_matrix(H_all, pruned=pruned, pruned_cols=pruned_cols)
        reordered_con_mat, coeff_k = reorder_con_mat(con_mat_k, k)

    #
    # Regress H
    #
    H = H_regression(X, W, mask, use_gpu, gpuid)

    #
    # Thresholding
    #
    if factor_thresholding_H_regression is not None:
        H_dtype = H.dtype
    
        if factor_thresholding_H_regression == "otsu_thresh":
            H_thresh_result = otsu_thresh_onefactor(X, H.T, W, use_gpu=False, **factor_thresholding_H_regression_obj_params)["wt"].T
        elif factor_thresholding_H_regression == "coord_desc_thresh":
            H_thresh_result = coord_desc_thresh_onefactor(X.T, H.T, W.T, use_gpu=False, **factor_thresholding_H_regression_obj_params)["wt"]
        elif factor_thresholding_H_regression == "kmeans_thresh": 
            H_thresh_result = kmeans_thresh_onefactor(X, W, H, use_gpu=False, **factor_thresholding_H_regression_obj_params)["wt"]

        H = (H >= H_thresh_result).astype(H_dtype)

    if use_gpu:
        cp._default_memory_pool.free_all_blocks()

    # 
    #  reconstruction error
    #
    if calculate_error:
        if mask is not None:
            Xhat = W@H
            X[mask] = Xhat[mask]

        error_reg = relative_error(
            X=X, W=W, H=H, 
            MASK=nmf_params["MASK"] if "MASK" in nmf_params else None)
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
        # latent factors
        W = unprune(W, pruned_rows, 0)
        H = unprune(H, pruned_cols, 1)

        # biases
        if "bu" in other_results:
            other_results["bu"] = unprune(other_results["bu"], pruned_rows, 1)[0]
        if "bi" in other_results:
            other_results["bi"] = unprune(other_results["bi"], pruned_cols, 1)[0]

    #
    # Calculate statistics on results
    #
    err_mean = np.mean(errors)
    err_std = np.std(errors)

    sils_W_all_mean = np.mean(sils_all_W, 1)
    sils_H_all_mean = np.mean(sils_all_H, 1)
    
    sils_min_W = np.min(sils_W_all_mean)
    sils_mean_W = np.mean(sils_W_all_mean)
    sils_std_W = np.std(sils_W_all_mean)
    sils_min_H = np.min(sils_H_all_mean)
    sils_mean_H = np.mean(sils_H_all_mean)
    sils_std_H = np.std(sils_H_all_mean)

    #
    # Decisions on K search
    #
    if K_search_settings["k_search_method"] != "linear":
        with K_search_settings['lock']:
            
            if predict_k_method in ["WH_sill", "sill"]:
                curr_score = min(sils_min_W, sils_min_H)
            elif predict_k_method == "W_sill":
                curr_score = sils_min_W
            elif predict_k_method == "H_sill":
                curr_score = sils_min_H
            elif predict_k_method == "pvalue":
                curr_score = sils_min_W
            else:
                raise Exception("Unknown predict_k_method!")

            if curr_score >= K_search_settings["sill_thresh"]:
                K_search_settings['k_min'] = k
            if K_search_settings["H_sill_thresh"] is not None and (sils_min_H <= K_search_settings["H_sill_thresh"]):
                K_search_settings['k_max'] = k

            if n_nodes > 1:
                # save the changed state for multi-node operations
                save_settings = {
                    "k_min": K_search_settings['k_min'],
                    "k_max": K_search_settings['k_max'],
                }
                
                # atomic safe write
                dir_name = os.path.dirname(f'{save_path}')
                temp_file = tempfile.NamedTemporaryFile(delete=False, dir=dir_name, mode='wb')
                try:
                    # Write data to the temporary file
                    pickle.dump(save_settings, temp_file)
                finally:
                    temp_file.close()
                
                # Atomically replace the old file with the new file 
                os.replace(temp_file.name, os.path.join(f'{save_path}', f'{rank}-K_search_settings.p'))

    #
    # save output factors and the plot
    #
    if save_output:
        if consensus_mat: 
            con_fig_name = os.path.join(f'{save_path}', f'k_{k}_con_mat.png')
            plot_consensus_mat(reordered_con_mat, con_fig_name)
        
        save_data = {
            "W": W,
            "H": H,
            "sils_all_W": sils_all_W,
            "sils_all_H": sils_all_H,
            "error_reg": error_reg,
            "errors": errors,
            "reordered_con_mat": reordered_con_mat,
            "H_all": H_all,
            "W_all": W_all,
            "cophenetic_coeff": coeff_k,
            "pruned_rows":pruned_rows,
            "pruned_cols":pruned_cols,
        }
        for key, value in other_results.items():
            save_data[key] = value
            
        np.savez_compressed( 
            os.path.join(f'{save_path}', f'WH_k={k}.npz'),
            **save_data
        )

        note_data = dict()
        for key in logging_stats:
            if key == 'k':
                note_data["k"] = k
            elif key ==  'sils_min_W':
                note_data["sils_min_W"] = '{0:.3f}'.format(sils_min_W)
            elif key == 'sils_mean_W':
                note_data["sils_mean_W"] = '{0:.3f}'.format(sils_mean_W)
            elif key ==  'sils_min_H':
                note_data["sils_min_H"] = '{0:.3f}'.format(sils_min_H)
            elif key == 'sils_mean_H':
                note_data["sils_mean_H"] = '{0:.3f}'.format(sils_mean_H)
            elif key == 'err_mean':
                note_data["err_mean"] = '{0:.3f}'.format(err_mean)
            elif key == 'err_std':
                note_data["err_std"] = '{0:.3f}'.format(err_std)
            elif key == 'col_error':
                note_data["col_err"] = '{0:.3f}'.format(np.mean(curr_col_err))
            elif key == 'time':
                elapsed_time = time.time() - start_time
                elapsed_time = timedelta(seconds=elapsed_time)
                note_data["time"] = str(elapsed_time).split('.')[0]
            elif key == "Done":
                note_data["Done"] = "Y"
            elif key == "Perturbs":
                note_data["Perturbs"] = n_perturbs
            else:
                warnings.warn(f'[tELF]: Encountered unknown logging metric "{key}"', RuntimeWarning)
                note_data[key] = 'N/A'
        take_note_fmat(save_path, name=note_name, lock=lock, **note_data)

    #
    # collect results
    #
    results_k = {
        "Ks":k,
        "err_mean":err_mean,
        "err_std":err_std,
        "err_reg":error_reg,

        "sils_min_W":sils_min_W,
        "sils_mean_W":sils_mean_W,
        "sils_std_W":sils_std_W,
        "sils_all_W":sils_all_W,

        "sils_min_H":sils_min_H,
        "sils_mean_H":sils_mean_H,
        "sils_std_H":sils_std_H,
        "sils_all_H":sils_all_H,

        "cophenetic_coeff":coeff_k,
        "col_err":curr_col_err,
        "exit_early":False
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
            save_path="",
            save_output=True,
            collect_output=False,
            predict_k=False,
            predict_k_method="WH_sill",
            verbose=True,
            nmf_verbose=False,
            perturb_verbose=False,
            transpose=False,
            sill_thresh=0.8,
            nmf_func=None,
            nmf_method="nmf_fro_mu",
            clustering_method="kmeans",
            nmf_obj_params={},
            clustering_obj_params={},
            pruned=True,
            calculate_error=True,
            perturb_multiprocessing=False,
            consensus_mat=False,
            use_consensus_stopping=0,
            mask=None,
            calculate_pac=False,
            get_plot_data=False,
            simple_plot=True,
            k_search_method="linear",
            H_sill_thresh=None,
            factor_thresholding=None,
            factor_thresholding_H_regression=None,
            factor_thresholding_obj_params={},
            factor_thresholding_H_regression_obj_params={},
            device=-1,
            ):
        """
        NMFk is a Non-negative Matrix Factorization module with the capability to do automatic model determination.

        Parameters
        ----------
        n_perturbs : int, optional
            Number of bootstrap operations, or random matrices generated around the original matrix. The default is 20.
        n_iters : int, optional
            Number of NMF iterations. The default is 100.
        epsilon : float or tuple of two elements, optional
            Error amount for the random matrices generated around the original matrix. The default is 0.015.\n
            The default when ``perturb_type='bool'`` or ``perturb_type='boolean'`` is (epsilon, epsilon).\n
            ``epsilon`` is used when ``perturb_type='uniform'`` or ``perturb_type='bool'`` or ``perturb_type='boolean'``.

            .. note::

                If ``perturb_type='bool'`` or ``perturb_type='boolean'``, use ```epsilon=tuple()``` where\n
                positive noise: flip 0s to 1s (additive noise), negative noise: flip 1s to 0s (subtractive noise).

        perturb_type : str, optional
            Type of error sampling to perform for the bootstrap operation. The default is "uniform".\n
            * ``perturb_type='uniform'`` will use uniform distribution for sampling.\n
            * ``perturb_type='poisson'`` will use Poission distribution for sampling.\n
            * ``perturb_type='bool'`` or ``perturb_type='boolean'`` will use Boolean perturbations.\n

            .. note::

                If ``perturb_type='bool'`` or ``perturb_type='boolean'``, use ```epsilon=tuple()``` where\n
                positive noise: flip 0s to 1s (additive noise), negative noise: flip 1s to 0s (subtractive noise).

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
            Location to save output. The default is "".
        save_output : bool, optional
            If True, saves the resulting latent factors and plots. The default is True.
        collect_output : bool, optional
            If True, collectes the resulting latent factors to be returned from ``fit()`` operation. The default is False.
        predict_k : bool, optional
            If True, performs automatic prediction of the number of latent factors. The default is False.

            .. note::

                Even when ``predict_k=False``, number of latent factors can be estimated using the figures saved in ``save_path``.

        predict_k_method : str, optional
            Method to use when performing automatic k prediction. Default is "WH_sill".\n
            * ``predict_k_method='pvalue'`` will use L-Statistics with column-wise error for automatically estimating the number of latent factors.\n
            * ``predict_k_method='WH_sill'`` will use Silhouette scores from minimum of W and H latent factors for estimating the number of latent factors.\n
            * ``predict_k_method='W_sill'`` will use Silhouette scores from W latent factor for estimating the number of latent factors.\n
            * ``predict_k_method='H_sill'`` will use Silhouette scores from H latent factor for estimating the number of latent factors.\n
            * ``predict_k_method='sill'`` will default to ``predict_k_method='WH_sill'``.
            
            .. warning::

                ``predict_k_method='pvalue'`` prediction will result in significantly longer processing time, altough it is more accurate! ``predict_k_method='WH_sill'``, on the other hand, will be much faster.

        verbose : bool, optional
            If True, shows progress in each k. The default is True.
        nmf_verbose : bool, optional
            If True, shows progress in each NMF operation. The default is False.
        perturb_verbose : bool, optional
            If True, it shows progress in each perturbation. The default is False.
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
            * ``nmf_method='wnmf'`` will use the Weighted NMF for missing value completion.\n
            * ``nmf_method='bnmf'`` will use the Boolean NMF for missing value completion on boolean matrix.\n

            .. note::

                When using ``nmf_method='nmf_recommender'``, RNMFk prediction method can be done using ``from TELF.factorization import RNMFk_predict``.\n
                Here ``RNMFk_predict(W, H, global_mean, bu, bi, u, i)``, ``W`` and ``H`` are the latent factors, ``global_mean``, ``bu``, and ``bi`` are the biases returned from ``nmf_recommender`` method.\n
                Finally, ``u`` and ``i`` are the indices to perform prediction on.

            .. note::
                When using ``nmf_method='wnmf'``, pass ``nmf_obj_params={"WEIGHTS":P}`` where ``P`` is a matrix of size ``X`` and carries the weights for each item in ``X``.\n
                For example, here ``P`` can be used as a mask where 1s in ``P`` are the known entries, and 0s are the missing values in ``X`` that we want to predict (i.e. a recommender system).\n
                Note that ``nmf_method='wnmf'`` does not support sparse matrices currently.

            .. note::
                When using ``nmf_method='bnmf'``, pass ``nmf_obj_params={"MASK":P}`` where ``P`` is a mask matrix of size ``X`` where 0s and 1s in ``P`` are the known and unknown locations in ``X``.\n
                0s in ``P`` are the places we would like to predict.\n
                Note that ``nmf_method='bnmf'`` does not support sparse matrices currently.\n
                When ``nmf_method='bnmf'``, ``perturb_type='bool'`` or ``perturb_type='boolean'`` is recommended to use. It will not set it automatically but raise warning if not used.\n
        
        clustering_method : str, optional
            Clustering used on the W patterns. Default is "kmeans".
            Options are "kmeans" and "bool" or "boolean"

        nmf_obj_params : dict, optional
            Parameters used by NMF function. The default is {}.

        clustering_obj_params: dict, optinal
            Parameters used by custom clustering functions. The default is {}.\n
            When ``nmf_method='bnmf'``, ``max_iters:int`` and ``distance:str`` can be passed here. ``distance`` can be ``'hamming'``, ``'FN'``, or ``'FP'``. Default is ``hamming``.\n 
            For all nmf methods passed in ``nmf_method``, ``max_iters`` can also be passed.
        
        pruned : bool, optional
            When True, removes columns and rows from the input matrix that has only 0 values. The default is True.

            .. warning::
                * If after pruning decomposition is not possible (for example if the number of samples left is 1, or K range is empty based on the rule ``k < min(X.shape)``, ``fit()`` will return ``None``.

        calculate_error : bool, optional
            When True, calculates the relative reconstruction error. The default is True.

            .. warning::
                If ``calculate_error=True``, it will result in longer processing time.

        perturb_multiprocessing : bool, optional
            If ``perturb_multiprocessing=True``, it will make parallel computation over each perturbation. Default is ``perturb_multiprocessing=False``.\n
            When ``perturb_multiprocessing=False``, which is default, parallelization is done over each K (rank).
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
        k_search_method : str, optional
            Which approach to use when searching for the rank or k. The default is "linear".\n
            * ``k_search_method='linear'`` will linearly visit each K given in ``Ks`` hyper-parameter of the ``fit()`` function.\n
            * ``k_search_method='bst_post'`` will perform post-order binary search. When an ideal rank is found, determined by the selected ``predict_k_method``, all lower ranks are pruned from the search space.\n
            * ``k_search_method='bst_pre'`` will perform pre-order binary search. When an ideal rank is found, determined by the selected ``predict_k_method``, all lower ranks are pruned from the search space.\n
            * ``k_search_method='bst_in'`` will perform in-order binary search. When an ideal rank is found, determined by the selected ``predict_k_method``, all lower ranks are pruned from the search space.
        H_sill_thresh : float, optional
            Setting for removing higher ranks from the search space.\n
            When searching for the optimal rank with binary search using ``k_search='bst_post'`` or ``k_search='bst_pre'``, this hyper-parameter can be used to cut off higher ranks from search space.\n
            The cut-off of higher ranks from the search space is based on threshold for H silhouette. When a H silhouette below ``H_sill_thresh`` is found for a given rank or K, all higher ranks are removed from the search space.\n
            If ``H_sill_thresh=None``, it is not used. The default is None.
        factor_thresholding : str, optional
            If not None, W and H factors are thresholded using a thresholding method specified to be boolean. Default is None.\n
            Options are ``WH_thresh`` and ``coord_desc_thresh`` and ``otsu_thresh`` and ``kmeans_thresh``.\n
            If ``nmf_method='bnmf'``, ``factor_thresholding='otsu_thresh'`` is used by default.
        factor_thresholding_H_regression : str, optional
            If not None, H factor is thresholded using a thresholding method specified to be boolean. Default is None.\n
            Options are ``coord_desc_thresh``, ``otsu_thresh``, and ``kmeans_thresh``.\n
            If ``nmf_method='bnmf'``, ``factor_thresholding='kmeans_thresh'`` is used by default.
        factor_thresholding_obj_params : dict, optional
            Extra settings used for the thresholding used in ``factor_thresholding``. Default is {}.\n
            For ``factor_thresholding='coord_desc_thresh'``, options include ``max_iter:int``, ``wt``, and ``ht``.\n
            For ``factor_thresholding='WH_thresh'``, options include ``npoint``.
        factor_thresholding_H_regression_obj_params : dict, optional
            Extra settings used for the thresholding used in ``factor_thresholding_H_regression``. Default is {}.\n
            For ``factor_thresholding='coord_desc_thresh'``, options include ``max_iter:int``, ``wt``, and ``ht``.\n
            For ``factor_thresholding='WH_thresh'``, options include ``npoint``.
        device : int or list, optional
            CUDA device or list of CUDA devices (GPUs) to use. Default is -1.\n
            When device is a positive integer such as ``device=0`` it will use the given GPU with the id.\n
            When device is -1, it will use all devices.\n
            When device is a list of devices, it will use those devices.\n
            If device is negative integer other than -1, it will use number if GPUs minues the device + 1.
        
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
        self.nmf_verbose = nmf_verbose
        self.perturb_verbose = perturb_verbose
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
        self.clustering_obj_params = clustering_obj_params
        self.pruned = pruned
        self.calculate_error = calculate_error
        self.consensus_mat = consensus_mat
        self.use_consensus_stopping = use_consensus_stopping
        self.mask = mask
        self.calculate_pac = calculate_pac
        self.simple_plot = simple_plot
        self.get_plot_data = get_plot_data
        self.perturb_multiprocessing = perturb_multiprocessing
        self.k_search_method = k_search_method
        self.H_sill_thresh = H_sill_thresh
        self.factor_thresholding = factor_thresholding
        self.factor_thresholding_H_regression = factor_thresholding_H_regression
        self.factor_thresholding_obj_params = factor_thresholding_obj_params
        self.factor_thresholding_H_regression_obj_params = factor_thresholding_H_regression_obj_params
        self.clustering_method = clustering_method
        self.device = device

        # warnings
        assert self.k_search_method in ["linear", "bst_pre", "bst_post", "bst_in"], "Invalid k_search_method method. Choose from linear, bst_pre, bst_in, or bst_post."
        assert self.predict_k_method in ["pvalue", "WH_sill", "W_sill", "H_sill", "sill"], "Invalid predict_k_method method. Choose from pvalue, WH_sill, W_sill, H_sill, or sill. sill defaults to WH_sill."
        assert self.clustering_method in ["kmeans", "bool", "boolean"]
        
        if self.predict_k_method == "sill":
            self.predict_k_method = "WH_sill"
            warnings.warn("predict_k_method is defaulted to WH_sill!")

        if self.calculate_pac and not self.consensus_mat:
            self.consensus_mat = True
            warnings.warn("consensus_mat was False when calculate_pac was True! consensus_mat changed to True.")

        if self.calculate_pac:
            warnings.warn("calculate_pac is True. PAC calculation for large matrices can take long time. For large matrices, instead use consensus_mat=True and calculate_pac=False.")

        if self.calculate_pac and not self.save_output:
            self.save_output = True
            warnings.warn("save_output was False when calculate_pac was True! save_output changed to True.")

        if (self.n_nodes > 1 and self.k_search_method != "linear") and not self.save_output:
            self.save_output = True
            warnings.warn("save_output was False when n_nodes > 1 and k_search_method != 'linear'! save_output changed to True.")

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
        perturb_options = ["uniform", "poisson", "bool", "boolean"]
        assert self.perturb_type in perturb_options, f"Invalid perturbation type. Choose from {', '.join(perturb_options)}."

        if self.perturb_type in ["bool", "boolean"] and (isinstance(self.epsilon, tuple) or isinstance(self.epsilon, list) or isinstance(self.epsilon, np.ndarray)):
            assert len(self.epsilon) == 2, f"When using boolean perturbation, epsilon should be either a single float/integer, or a tuple/list of two elements."

        if self.perturb_type in ["bool", "boolean"] and not any([isinstance(self.epsilon, tuple) or isinstance(self.epsilon, list) or isinstance(self.epsilon, np.ndarray)]):
            self.epsilon = (self.epsilon, self.epsilon)
            warnings.warn(f"epsilon was single element while using boolean perturbation. epsilon is now set to {self.epsilon}.")


        # organize n_jobs
        self.n_jobs, self.use_gpu = organize_n_jobs(use_gpu, n_jobs)

        # organize devices
        self.device = organize_devices(self.n_jobs, self.device, self.use_gpu)
           
        # create a shared lock
        self.lock = Lock()

        # settings for K search
        self.K_search_settings = {
            "lock": Lock(),
            "k_search_method":self.k_search_method,
            "sill_thresh":self.sill_thresh,
            "H_sill_thresh":self.H_sill_thresh,
            "k_min":-1,
            "k_max":float('inf')
        }

        #
        # Save information from the solution
        #
        self.total_exec_seconds = 0
        self.experiment_name = ""

        #
        # Check the clustering parameters
        #
        if "distance" in self.clustering_obj_params:
            assert self.nmf_method is "bnmf", "distance can only be set for nmf_method='bnmf'!"
            assert self.clustering_obj_params["distance"] in ["hamming", "FN", "FP"], "Unknown clustering distance parameter."
        
        elif "distance" not in self.clustering_obj_params and self.clustering_method in ["bool", "boolean"]:
            self.clustering_obj_params["distance"] = "hamming"

        #
        # Prepare NMF function
        #
        avail_nmf_methods = [
            "nmf_fro_mu", 
            "nmf_kl_mu", 
            "nmf_recommender", 
            "wnmf",
            "bnmf",
            "func"
        ]
        if "WEIGHTS" in self.nmf_obj_params:
            self.nmf_obj_params["MASK"] = self.nmf_obj_params["WEIGHTS"].copy()
            del self.nmf_obj_params["WEIGHTS"]

        if self.nmf_method not in avail_nmf_methods:
            raise Exception("Invalid NMF method is selected. Choose from: " +
                            ",".join(avail_nmf_methods))
        
        if self.nmf_method == "bnmf" and self.perturb_type not in ["bool", "boolean"]:
            warnings.warn(f"nmf_method='bnmf' but perturb_type={self.perturb_type} instead of a boolean perturbation.")

        if self.nmf_method == "bnmf" and self.clustering_method not in ["bool", "boolean"]:
            warnings.warn(f"nmf_method='bnmf' but clustering_method={self.clustering_method} instead of a boolean clustering.")
        
        if self.nmf_method == "bnmf" and self.factor_thresholding is None:
            self.factor_thresholding = "otsu_thresh"

        if self.nmf_method == "bnmf" and self.factor_thresholding_H_regression is None:
            self.factor_thresholding_H_regression = "kmeans_thresh"
        
        if self.factor_thresholding_H_regression is not None:
            assert self.factor_thresholding_H_regression in ["coord_desc_thresh", "otsu_thresh", "kmeans_thresh"], f"Unknown factor_thresholding_H_regression! Select from {', '.join(['coord_desc_thresh', 'otsu_thresh', 'kmeans_thresh'])}"

        if self.factor_thresholding is not None:
            assert self.factor_thresholding in ["coord_desc_thresh", "WH_thresh", "otsu_thresh", "kmeans_thresh"], f"Unknown factor_thresholding! Select from {', '.join(['coord_desc_thresh', 'WH_thresh', 'otsu_thresh'])}"
        
        if self.factor_thresholding == "coord_desc_thresh":
            self.thresholding_function = coord_desc_thresh
        elif factor_thresholding == "WH_thresh":
            self.thresholding_function = find_thres_WH
        elif factor_thresholding == "otsu_thresh":
            self.thresholding_function = otsu_thresh
        elif factor_thresholding == "kmeans_thresh":
            self.thresholding_function = kmeans_thresh
        else:
            self.thresholding_function = None

        if self.nmf_method in ["wnmf", "bnmf", "nmf_recommender"] and "MASK" not in self.nmf_obj_params:
                warnings.warn(f"When using {self.nmf_method}, use nmf_obj_params={'MASK':P}, where P is the mask matrix. Otherwise P will have 1s where X>0. P should have 0's at places in X that we do not know, i.e. NaNs, and 1s at places we know in X.")

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

        elif self.nmf_method == "wnmf":
            self.nmf_params = {
                "niter": self.n_iters,
                "use_gpu": self.use_gpu,
                "nmf_verbose": self.nmf_verbose,
            }
            self.nmf = wnmf

        elif self.nmf_method == "bnmf":        
            self.nmf_params = {
                "niter": self.n_iters,
                "use_gpu": self.use_gpu,
                "nmf_verbose": self.nmf_verbose,
            }
            self.nmf = bnmf    

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
            for key, value in vars(self).items():
                print(f'{key}:', value)

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


        # Prepare Ks
        Ks = list(Ks)
        Ks.sort()
        if self.K_search_settings["k_search_method"] != "linear":
            node = BST.sorted_array_to_bst(Ks)
            if self.K_search_settings["k_search_method"] == "bst_pre": 
                Ks = list(node.preorder())
            elif self.K_search_settings["k_search_method"] == "bst_post": 
                Ks = list(node.postorder())
            elif self.K_search_settings["k_search_method"] == "bst_in": 
                Ks = list(node.inorder())
            else:
                raise Exception("Unknown k_search_method!")
            
            if self.verbose:
                print(f'Performing K search with {self.K_search_settings["k_search_method"]}. Ks={Ks}')

        
        #
        # check X format
        #
        assert scipy.sparse._csr.csr_matrix == type(X) or np.ndarray == type(X), "X sould be np.ndarray or scipy.sparse._csr.csr_matrix"

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
        
        if min(Ks) <= 0:
            raise Exception("Minimum Ks needs to be more than 0.")

        #
        # MPI
        #
        if self.n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            Ks = chunk_Ks(Ks, n_chunks=self.n_nodes)[rank]
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
            
        if self.transpose:
            if isinstance(X, np.ndarray):
                X = X.T
            elif scipy.sparse.issparse(X):
                X = X.T.asformat("csr")
            else:
                raise Exception("I do not know how to transpose type " + str(type(X)))

        # init the stats header 
        # this will setup the logging for all configurations of nmfk
        stats_header = {
                        'Done': 'Done',
                        'k': 'k',
                        'Perturbs': 'Perturbs',
                        'sils_min_W': 'W Min. Sill', 
                        'sils_mean_W': 'W Mean Sill',
                        'sils_min_H': 'H Min. Sill', 
                        'sils_mean_H': 'H Mean Sill',
                        }
        if self.calculate_error:
            stats_header['err_mean'] = 'Mean Err'
            stats_header['err_std'] = 'STD Err'
        if self.predict_k:
            stats_header['col_error'] = 'Mean Col. Err'
        if self.calculate_pac:
            stats_header['pac'] = 'PAC'
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
            del object_notes["nmf"]
            take_note(object_notes, self.save_path_full, name=note_name, lock=self.lock)
            append_to_note(["#" * 100], self.save_path_full, name=note_name, lock=self.lock)

            notes = {}
            notes["Ks"] = Ks
            notes["data_type"] = type(X)
            notes["num_elements"] = np.prod(X.shape)
            notes["num_nnz"] = len(X.nonzero()[0])
            notes["sparsity"] = len(X.nonzero()[0]) / np.prod(X.shape)
            notes["X_shape"] = X.shape
            take_note(notes, self.save_path_full, name=note_name, lock=self.lock)
            append_to_note(["#" * 100], self.save_path_full, name=note_name, lock=self.lock)
        
        if self.n_nodes > 1:
            comm.Barrier()
            
        #
        # Prune
        #
        if self.pruned:
            X, pruned_rows, pruned_cols, self.nmf_params = prune(X, 
                                                use_gpu=self.use_gpu, 
                                                other=self.nmf_params, keys_to_check_other=["MASK"])

            # check for K after prune and adjust if needed
            if max(Ks) >= min(X.shape):
                Ks = range(min(Ks), min(X.shape), 1)
                warnings.warn(f'Ks range re-adjusted after pruning. New Ks range: {Ks}')

            if self.save_output:
                prune_notes = {}
                prune_notes["Ks_pruned"] = Ks
                prune_notes["X_shape_pruned"] = X.shape
                take_note(prune_notes, self.save_path_full, name=note_name, lock=self.lock)
            
            # Check if we can decompose after pruning
            if len(Ks) == 0 or min(X.shape) <= 1:

                if self.save_output:
                    take_note({"Prune_status":"Decomposition not possible."}, self.save_path_full, name=note_name, lock=self.lock)
                    append_to_note(["#" * 100], self.save_path_full, name=note_name, lock=self.lock)
                
                warnings.warn(f'Decomposition is not possible after pruning. X shape is {X.shape} and Ks is {Ks} after pruning. Returning None.')
                return None
            
            else:
                if self.save_output:
                    take_note({"Prune_status":"Decomposition possible."}, self.save_path_full, name=note_name, lock=self.lock)
                    append_to_note(["#" * 100], self.save_path_full, name=note_name, lock=self.lock)

        else:
            pruned_rows, pruned_cols = None, None

        if self.save_output:
            take_note_fmat(self.save_path_full, name=note_name, lock=self.lock, **stats_header)

        #
        # Begin NMFk
        #
        start_time = time.time()

        job_data = {
            "n_perturbs":self.n_perturbs,
            "nmf":self.nmf,
            "nmf_params":self.nmf_params,
            "clustering_obj_params":self.clustering_obj_params,
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
            "pruned_rows":pruned_rows,
            "pruned_cols":pruned_cols,
            "save_output":self.save_output,
            "save_path":self.save_path_full,
            "collect_output":self.collect_output,
            "logging_stats":stats_header,
            "start_time":start_time,
            "n_jobs":self.n_jobs,
            "perturb_multiprocessing":self.perturb_multiprocessing,
            "perturb_verbose":self.perturb_verbose,
            "lock":self.lock,
            "note_name":note_name,
            "n_nodes":self.n_nodes,
            "rank":rank,
            "factor_thresholding_obj_params":self.factor_thresholding_obj_params,
            "factor_thresholding_H_regression_obj_params":self.factor_thresholding_H_regression_obj_params,
            "factor_thresholding_H_regression":self.factor_thresholding_H_regression,
            "clustering_method":self.clustering_method,
            "thresholding_function":self.thresholding_function,
        }
        
        # Single job or parallel over perturbations
        if self.n_jobs == 1 or self.perturb_multiprocessing:
            all_k_results = []
            for kidx, k in tqdm(enumerate(Ks), total=len(Ks), disable=not self.verbose):
                k_result = _nmf_parallel_wrapper(gpuid=self.device[kidx % len(self.device)], k=k, K_search_settings=self.K_search_settings, **job_data)
                all_k_results.append(k_result)
        
        # multiprocessing over each K
        else:   
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs)
            futures = [executor.submit(
                _nmf_parallel_wrapper, gpuid=self.device[kidx % len(self.device)], k=k, K_search_settings=self.K_search_settings, **job_data
                ) for kidx, k in enumerate(Ks)]
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
        # Which Ks computed
        #    
        Ks_not_computed = []
        if self.K_search_settings != "linear":
            Ks_computed = []
            for idx, flag in enumerate(combined_result["exit_early"]):
                if flag is False:
                    Ks_computed.append(combined_result["Ks"][idx])
                else:
                    Ks_not_computed.append(combined_result["Ks"][idx])        
            
            combined_result["Ks"] = Ks_computed
            Ks = Ks_computed
            
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
                        combined_result["col_err"], Ks, combined_result["sils_min_W"], SILL_thr=self.sill_thresh
                    )[0]
                
                else:
                    if self.predict_k_method in ["WH_sill", "sill"]:
                        curr_sill_max_score = min([max(combined_result["sils_min_W"]), max(combined_result["sils_min_H"])])
                    
                    elif self.predict_k_method == "W_sill":
                        curr_sill_max_score = max(combined_result["sils_min_W"])
                    
                    elif self.predict_k_method == "H_sill":
                        curr_sill_max_score = max(combined_result["sils_min_H"])

                    # check if that sill threshold exist
                    if self.sill_thresh > curr_sill_max_score:
                        self.sill_thresh = curr_sill_max_score
                        warnings.warn(f'W or H Silhouettes were all less than sill_thresh. Setting sill_thresh to minimum for K prediction. sill_thresh={round(self.sill_thresh, 3)}')
                    
                    if self.predict_k_method in ["WH_sill", "sill"]:
                        k_predict_W = Ks[np.max(np.argwhere(np.array(combined_result["sils_min_W"]) >= self.sill_thresh).flatten())]
                        k_predict_H = Ks[np.max(np.argwhere(np.array(combined_result["sils_min_H"]) >= self.sill_thresh).flatten())]
                        k_predict = min(k_predict_W, k_predict_H)
                    elif self.predict_k_method == "W_sill":
                        k_predict = Ks[np.max(np.argwhere(np.array(combined_result["sils_min_W"]) >= self.sill_thresh).flatten())]
                    elif self.predict_k_method == "H_sill":
                        k_predict = Ks[np.max(np.argwhere(np.array(combined_result["sils_min_H"]) >= self.sill_thresh).flatten())]
            
            else:
                k_predict = 0
                
            
            # * plot cophenetic coefficients
            combined_result["pac"] = []
            if self.consensus_mat:

                # * save the plot
                if self.save_output: 
                    con_fig_name = os.path.join(f'{self.save_path_full}', f'k_{Ks[0]}_{Ks[-1]}_cophenetic_coeff.png')
                    plot_cophenetic_coeff(Ks, combined_result["cophenetic_coeff"], con_fig_name)

                if self.calculate_pac:
                    
                    # load reordered consensus matrices from each k
                    reordered_con_matrices = []
                    for curr_k_to_load in Ks: 
                        reordered_con_matrices.append(np.load(os.path.join(f'{self.save_path_full}', f'WH_k={curr_k_to_load}.npz'))["reordered_con_mat"])
                    consensus_tensor = np.array(reordered_con_matrices)
                    combined_result["pac"] = np.array(get_pac(consensus_tensor, use_gpu=self.use_gpu, verbose=self.verbose))
                    consensus_tensor, reordered_con_matrices = None, None

            # save k prediction
            if self.predict_k:
                results["k_predict"] = k_predict

                if self.collect_output:
                        for key in ["W", "H", "other_results"]:
                            if key in combined_result:
                                results[key] = combined_result[key][combined_result["Ks"].index(k_predict)]
            # final plot
            if self.save_output:
                plot_NMFk(
                    combined_result, 
                    k_predict, 
                    self.experiment_name, 
                    self.save_path_full, 
                    plot_predict=self.predict_k,
                    plot_final=True,
                    simple_plot=self.simple_plot,
                    calculate_error=self.calculate_error,
                    Ks_not_computed=Ks_not_computed
                )
                append_to_note(["#" * 100], self.save_path_full, name=note_name, lock=self.lock)
                append_to_note(["end_time= "+str(datetime.now())], self.save_path_full, name=note_name, lock=self.lock)
                append_to_note(
                    ["total_time= "+str(time.time() - start_time) + " (seconds)"], self.save_path_full, name=note_name, lock=self.lock)
        
            
            if self.get_plot_data:
                results["plot_data"] = combined_result
                
            return results
