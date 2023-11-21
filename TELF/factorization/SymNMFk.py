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
import os
import sys
import time
import socket
import warnings
import numpy as np
import scipy.sparse
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import concurrent.futures
from collections import defaultdict
from joblib import Parallel, delayed
from datetime import datetime, timedelta
from scipy.spatial.distance import pdist, squareform
from threading import Lock

from .utilities.take_note import take_note, take_note_fmat, append_to_note
from .utilities.plot_NMFk import plot_SymNMFk, plot_consensus_mat, plot_cophenetic_coeff
from .utilities.organize_n_jobs import organize_n_jobs
from .decompositions.sym_nmf import sym_nmf_newt
from .decompositions.utilities.nnsvd import nnsvd
from .decompositions.utilities.resample import poisson, uniform_product
from .decompositions.utilities.math_utils import prune, unprune, get_pac
from .decompositions.utilities.concensus_matrix import compute_connectivity_mat, compute_consensus_matrix, reorder_con_mat
from .decompositions.utilities.similarity_matrix import build_similarity_matrix, get_connectivity_matrix, dist2, scale_dist3


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


def __put_X_gpu(X, gpuid:int):
    with cp.cuda.Device(gpuid):
        if scipy.sparse.issparse(X):
            Y = cupyx.scipy.sparse.csr_matrix(
                (cp.array(X.data), cp.array(X.indices), cp.array(X.indptr)),
                shape=X.shape,
                dtype=X.dtype,
            )
        else:
            Y = cp.array(X)
    return Y

def __put_W_gpu(W, gpuid:int):
    with cp.cuda.Device(gpuid):
        W = cp.array(W)
    return W


def __run_symnmf(Y, W, nmf, nmf_params, use_gpu:bool, gpuid:int):
    if use_gpu:
        with cp.cuda.Device(gpuid):
            W, obj = nmf(Y, W=W, **nmf_params)
    else:
        W, obj = nmf(Y, W=W, **nmf_params)
    return W, obj


def __perturb_X(X, perturbation:int, epsilon:float, perturb_type:str):
    if perturb_type == "uniform":
        Y = uniform_product(X, epsilon)
    elif perturb_type == "poisson":
        Y = poisson(X)
    return Y


def __init_W(Y, k, mask, init_type:str):
    # REMOVED SEEDING
    if init_type == "nnsvd":
        if mask is not None:
            Y[mask] = 0
        W, _ = nnsvd(Y, k, use_gpu=False)
    elif init_type == "random":
        W = 2 * np.sqrt(np.mean(Y) / k) * np.random.rand(Y.shape[0], k)
    return W


def _perturb_parallel_wrapper(
    perturbation,
    gpuid,
    epsilon,
    perturb_type,
    graph_type,
    similarity_type,
    nearest_neighbors,
    X,
    k,
    mask,
    use_gpu,
    init_type,
    nmf_params,
    nmf):

    # Perturb X
    Xq = __perturb_X(X, perturbation, epsilon, perturb_type)

    # Compute the similarity matrix
    if graph_type == 'full' and similarity_type == 'gaussian':
        Dq = dist2(Xq, Xq)
        Aq = scale_dist3(Dq, nearest_neighbors)
    else:
        raise ValueError('Unknown graph_type and/or similarity_type')

    # Initialize W
    Wq = __init_W(Aq, k, mask=mask, init_type=init_type)
    
    
    # transfer to GPU
    if use_gpu:
        Aq = __put_X_gpu(Aq, gpuid)
        Wq = __put_W_gpu(Wq, gpuid)

    Wq, obj = __run_symnmf(Aq, Wq, nmf, nmf_params, use_gpu, gpuid)
    
    # transfer to CPU
    if use_gpu:
        Wq =  cp.asnumpy(Wq)
        obj = cp.asnumpy(obj)
        cp._default_memory_pool.free_all_blocks()
    
    return Wq, obj


def _symnmf_parallel_wrapper(
        n_perturbs, 
        nmf, 
        nmf_params,
        X=None, 
        k=None,
        epsilon=None, 
        gpuid=0, 
        use_gpu=True,
        perturb_type="uniform",
        init_type="random",
        graph_type="full",
        similarity_type="gaussian",
        nearest_neighbors=7,
        mask=None, 
        consensus_mat=False,
        save_output=True,
        save_path="",
        experiment_name="",
        collect_output=False,
        logging_stats={},
        start_time=time.time(),
        n_jobs=1,
        perturb_multiprocessing=False,
        perturb_verbose=False,
        lock=None,
        note_name="experiment",
):

    assert graph_type in {'full'}, 'Supported graph types are ["full"]'
    assert similarity_type in {'gaussian'}, 'Supported similarity metrics are ["gaussian"]'
    
    #
    # run for each perturbations
    #
    perturb_job_data = {
        "epsilon":epsilon,
        "perturb_type":perturb_type,
        "graph_type": graph_type,
        "similarity_type": similarity_type,
        "nearest_neighbors": nearest_neighbors,
        "X":X,
        "k":k,
        "mask":mask,
        "use_gpu":use_gpu,
        "init_type":init_type,
        "nmf_params":nmf_params,
        "nmf":nmf,
    }
    
    W_all, obj_all = [], []
    connectivity_matrices = []
    
    # single job or parallel over Ks
    if n_jobs == 1 or not perturb_multiprocessing:
        for perturbation in tqdm(range(n_perturbs), disable=not perturb_verbose, total=n_perturbs):
            W, obj = _perturb_parallel_wrapper(perturbation=perturbation, gpuid=gpuid, **perturb_job_data)
            W_all.append(W)
            obj_all.append(obj)
            B = get_connectivity_matrix(np.argmax(W, 1))
            connectivity_matrices.append(B)
            
    # multiple jobs over perturbations
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs)
        futures = [executor.submit(_perturb_parallel_wrapper, gpuid=pidx % n_jobs, perturbation=perturbation, **perturb_job_data) for pidx, perturbation in enumerate(range(n_perturbs))]
        all_perturbation_results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), disable=not perturb_verbose, total=n_perturbs)]

        for W, obj in all_perturbation_results:
            W_all.append(W)
            obj_all.append(obj)
            B = get_connectivity_matrix(np.argmax(W, 1))
            connectivity_matrices.append(B)
    
    #
    # organize solutions from each perturbations
    #
    obj_all = np.array(obj_all)
    avg_W = np.mean(np.stack(W_all, axis=0), axis=0)
    
    #
    # get the consensus matrix
    #
    coeff_k = 0
    con_mat_k = None
    reordered_con_mat = None
    if consensus_mat:
        con_mat_k = np.stack(connectivity_matrices, axis=0)
        con_mat_k = np.mean(con_mat_k, axis=0)
        reordered_con_mat, coeff_k = reorder_con_mat(con_mat_k, k)
        
    #
    # save output factors and the plot
    #
    if save_output:
        con_fig_name = f'{save_path}/k_{k}_con_mat.png'
        plot_consensus_mat(reordered_con_mat, con_fig_name)
        
        save_data = {
            "avg_W": avg_W,
            "avg_obj": np.mean(obj_all),
            "reordered_con_mat": reordered_con_mat,
            "cophenetic_coeff": coeff_k
        }
        np.savez_compressed(
            save_path
            + "/W"
            + "_k="
            + str(k)
            + ".npz",
            **save_data
        )
        
        plot_data = dict()
        for key in logging_stats:
            if key == 'k':
                plot_data["k"] = k
            elif key == 'err_mean':
                err_mean = np.mean(obj_all)
                plot_data["err_mean"] = '{0:.3f}'.format(err_mean)
            elif key == 'err_std':
                err_std = np.std(obj_all)
                plot_data["err_std"] = '{0:.3f}'.format(err_std)
            elif key == 'time':
                elapsed_time = time.time() - start_time
                elapsed_time = timedelta(seconds=elapsed_time)
                plot_data["time"] = str(elapsed_time).split('.')[0]
            else:
                warnings.warn(f'[tELF]: Encountered unknown logging metric "{key}"', RuntimeWarning)
                plot_data[key] = 'N/A'
        take_note_fmat(save_path, name=note_name, lock=lock, **plot_data)

    #
    # collect results
    #
    results_k = {
        "Ks":k,
        "W":avg_W,
        "err_mean":np.mean(obj_all),
        "err_std":np.std(obj_all),
        "cophenetic_coeff":coeff_k,
        "reordered_con_mat":reordered_con_mat
    }
    return results_k


class SymNMFk:
    def __init__(
            self,
            n_perturbs=20,
            n_iters=1000,
            epsilon=0.015,
            perturb_type="uniform",
            n_jobs=1,
            n_nodes=1,
            use_gpu=False,
            save_path="./",
            save_output=True,
            collect_output=False,
            verbose=False,
            nmf_verbose=False,
            perturb_verbose = False,
            transpose=False,
            nmf_method="newton",
            nmf_obj_params={},
            graph_type="full",
            similarity_type="gaussian",
            nearest_neighbors=7,
            use_consensus_stopping=False,
            perturb_multiprocessing=False,
            calculate_pac=True,
            mask=None,
            pac_thresh=0,
            get_plot_data=False):

        if n_nodes > 1 and MPI is None:
            sys.exit("Attempted to use n_nodes>1 but MPI is not available!")

        # overwrite NMF params with higher level definitions
        nmf_obj_params['n_iters'] = n_iters
        nmf_obj_params['use_consensus_stopping'] = use_consensus_stopping
            
        #
        # Object hyper-parameters
        #
        self.pac_thresh=pac_thresh
        self.n_perturbs = n_perturbs
        self.perturb_type = perturb_type
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.save_path = save_path
        self.save_output = save_output
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.nmf_verbose = nmf_verbose
        self.perturb_verbose = perturb_verbose
        self.transpose = transpose
        self.collect_output = collect_output
        self.n_jobs = n_jobs
        self.n_nodes = n_nodes
        self.nmf = None
        self.nmf_method = nmf_method
        self.nmf_obj_params = nmf_obj_params
        self.graph_type=graph_type
        self.similarity_type=similarity_type
        self.nearest_neighbors=nearest_neighbors
        self.consensus_mat = True
        self.use_consensus_stopping = use_consensus_stopping
        self.mask = mask
        self.calculate_pac = calculate_pac
        self.get_plot_data = get_plot_data
        self.perturb_multiprocessing = perturb_multiprocessing

        # warnings
        if self.calculate_pac and not self.consensus_mat:
            self.consensus_mat = True
            warnings.warn("consensus_mat was False when calculate_pac was True! consensus_mat changed to True.")

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
        # Prepare NMF function
        #
        avail_nmf_methods = ["newton"]
        if self.nmf_method not in avail_nmf_methods:
            raise ValueError(f"Invalid NMF method is selected. Choose from: " \
                             f"{','.join(avail_nmf_methods)}")
        if self.nmf_method == "newton":
            self.nmf = sym_nmf_newt
            self.nmf_obj_params['n_iters'] = self.n_iters
            self.nmf_obj_params['use_gpu'] = self.use_gpu
            self.nmf_obj_params['use_consensus_stopping'] = self.use_consensus_stopping
            supported_args = {'n_iters', 'use_gpu', 'tol', 'sigma', 'beta', 'use_consensus_stopping', 'debug'}
            assert set(self.nmf_obj_params.keys()).issubset(supported_args), \
                   f"nmf_obj_params contains unexpected arguments for {self.nmf_method} method"
            
        if self.verbose:
            print('Performing NMF with ', self.nmf_method)

            
    def fit(self, X, Ks, name="SymNMFk", note=""):

        if X.dtype != np.dtype(np.float32):
            warnings.warn(
                f'X is data type {X.dtype}. Whic is not float32. Higher precision will result in significantly longer runtime!')

        #
        # Error check
        #
        if len(Ks) == 0:
            raise Exception("Ks range is 0!")

        if max(Ks) >= X.shape[0]:
            raise Exception("Maximum rank k to try in Ks should be k<X.shape[0]")

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
        )
        save_path = os.path.join(self.save_path, self.experiment_name)

        if self.n_jobs > len(Ks):
            warnings.warn(f'Requested {self.n_jobs} jobs but only processing {len(Ks)} k values!')
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
                        'err_mean': 'Error Mean', 
                        'err_std': 'Error STD-DEV'}
        stats_header['time'] = 'Time Elapsed'
        
        # start the file logging (only root node needs to do this step)
        if self.save_output and ((self.n_nodes == 1) or (self.n_nodes > 1 and rank == 0)):
            try:
                if not Path(save_path).is_dir():
                    Path(save_path).mkdir(parents=True)
            except Exception as e:
                print(e)
                
        if self.n_nodes > 1:
            comm.Barrier()
            time.sleep(1)

        # logging
        if self.save_output:
            if not Path(save_path).is_dir():
                Path(save_path).mkdir(parents=True)

            append_to_note(["#" * 100], save_path, name=note_name, lock=self.lock)
            append_to_note(["start_time= " + str(datetime.now()),
                            "name=" + str(name),
                            "note=" + str(note)], save_path, name=note_name, lock=self.lock)

            append_to_note(["#" * 100], save_path, name=note_name, lock=self.lock)
            object_notes = vars(self).copy()
            del object_notes["total_exec_seconds"]
            del object_notes["nmf"]
            take_note(object_notes, save_path, name=note_name, lock=self.lock)
            append_to_note(["#" * 100], save_path, name=note_name, lock=self.lock)

            notes = {}
            notes["Ks"] = Ks
            notes["data_type"] = type(X)
            notes["num_elements"] = np.prod(X.shape)
            notes["num_nnz"] = len(X.nonzero()[0])
            notes["sparsity"] = len(X.nonzero()[0]) / np.prod(X.shape)
            notes["X_shape"] = X.shape
            take_note(notes, save_path, name=note_name, lock=self.lock)
            append_to_note(["#" * 100], save_path, name=note_name, lock=self.lock)
            take_note_fmat(save_path, name=note_name, lock=self.lock, **stats_header)
        
        if self.n_nodes > 1:
            comm.Barrier()

        #
        # Begin SymNMFk
        #
        start_time = time.time()

        job_data = {
            "n_perturbs":self.n_perturbs,
            "nmf":self.nmf,
            "nmf_params":self.nmf_obj_params,
            "X":X,
            "epsilon":self.epsilon,
            "use_gpu":self.use_gpu,
            "perturb_type":self.perturb_type,
            "mask":self.mask,
            "consensus_mat":self.consensus_mat,
            "save_output":self.save_output,
            "save_path":save_path,
            "experiment_name":self.experiment_name,
            "collect_output":self.collect_output,
            "logging_stats":stats_header,
            "start_time":start_time,
            "graph_type": self.graph_type,
            "similarity_type": self.similarity_type,
            "nearest_neighbors": self.nearest_neighbors,
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
                k_result = _symnmf_parallel_wrapper(gpuid=0, k=k, **job_data)
                all_k_results.append(k_result)

        # multiprocessing over each K
        else:   
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs)
            futures = [executor.submit(_symnmf_parallel_wrapper, gpuid=kidx % self.n_jobs, k=k, **job_data) for kidx, k in enumerate(Ks)]
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
        # revert to original Ks
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
            
            # * plot cophenetic coefficients
            combined_result["pac"] = []
            if self.consensus_mat:

                # # * save the plot
                # if self.save_output:
                #     con_fig_name = f'{save_path}/k_{Ks[0]}_{Ks[-1]}_cophenetic_coeff.png'
                #     plot_cophenetic_coeff(Ks, combined_result["cophenetic_coeff"], con_fig_name)

                if self.calculate_pac:
                    consensus_tensor = np.array(combined_result["reordered_con_mat"])
                    combined_result["pac"] = np.array(get_pac(consensus_tensor, use_gpu=self.use_gpu))
                    argmin = np.max(np.argwhere(
                        np.array(combined_result["pac"]) <= self.pac_thresh).flatten())
                    results["clusters"] = np.argmax(combined_result["W"][argmin], 1)
                    
            # final plot
            if self.save_output:
                plot_SymNMFk(
                    combined_result, 
                    self.experiment_name, 
                    save_path, 
                    plot_final=True,
                )
                append_to_note(["#" * 100], save_path, name=note_name, lock=self.lock)
                append_to_note(["end_time= "+str(datetime.now())], save_path, name=note_name, lock=self.lock)
                append_to_note(
                    ["total_time= "+str(time.time() - start_time) + " (seconds)"], save_path, name=note_name, lock=self.lock)
        
            
            if self.get_plot_data:
                results["plot_data"] = combined_result
            
            results["W"] = combined_result["W"]
            results["reordered_con_mat"] = combined_result["reordered_con_mat"]
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
