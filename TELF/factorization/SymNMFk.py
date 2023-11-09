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
from .utilities.plot_NMFk import plot_SymNMFk, plot_consensus_mat, plot_cophenetic_coeff
from .utilities.organize_n_jobs import organize_n_jobs
from .decompositions.sym_nmf import sym_nmf_newt
from .decompositions.utilities.nnsvd import nnsvd
from .decompositions.utilities.resample import poisson, uniform_product
from .decompositions.utilities.math_utils import prune, unprune, get_pac
from .decompositions.utilities.concensus_matrix import compute_connectivity_mat, compute_consensus_matrix, reorder_con_mat
from .decompositions.utilities.similarity_matrix import build_similarity_matrix, get_connectivity_matrix, dist2, scale_dist3
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

from scipy.spatial.distance import pdist, squareform


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


def run_symnmf(Y, W, nmf, nmf_params, use_gpu:bool, gpuid:int):
    if use_gpu:
        with cp.cuda.Device(gpuid):
            # move data and initialization from host to device
            W = cp.array(W)
            if scipy.sparse.issparse(Y):
                Y = cupyx.scipy.sparse.csr_matrix(
                    (cp.array(Y.data), cp.array(Y.indices), cp.array(Y.indptr)),
                    shape=Y.shape,
                    dtype=Y.dtype,
                )
            else:
                Y = cp.array(Y)

            # do optimization on GPU
            W_, obj_ = nmf(A=Y, W=W, **nmf_params)

            # move solution from device to host
            W = cp.asnumpy(W_)
            obj = cp.asnumpy(obj_)
            
            # cleanup
            del Y, W_
            cp._default_memory_pool.free_all_blocks()

    else:
        W, obj = nmf(A=Y, W=W, **nmf_params)
    return W, obj


def perturb_X(X, perturbation:int, epsilon:float, perturb_type:str):

    np.random.seed(perturbation)
    if perturb_type == "uniform":
        Y = uniform_product(X, epsilon)
    elif perturb_type == "poisson":
        Y = poisson(X)
    return Y


def init_W(Y, k, mask, init_type:str, seed=42):
    if init_type == "nnsvd":
        if mask is not None:
            Y[mask] = 0
        W, _ = nnsvd(Y, k)
    elif init_type == "random":
        np.random.seed(seed)
        W = 2 * np.sqrt(np.mean(Y) / k) * np.random.rand(Y.shape[0], k)
    return W


def symnmf_parallel_wrapper(
        n_perturbs, 
        nmf, 
        nmf_params,
        X=None, 
        k=None,
        epsilon=None, 
        gpuid=0, 
        use_gpu=False,
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
        start_time=time.time()):

    assert graph_type in {'full'}, 'Supported graph types are ["full"]'
    assert similarity_type in {'gaussian'}, 'Supported similarity metrics are ["gaussian"]'
    
    #
    # run for each perturbations
    #
    all_W, all_obj = [], []
    connectivity_matrices = []
    for q in range(n_perturbs):
        Xq = perturb_X(X, q, epsilon, perturb_type)
        
        if graph_type == 'full' and similarity_type == 'gaussian':
            Dq = dist2(Xq, Xq, use_gpu=use_gpu)
            Aq = scale_dist3(Dq, 7, use_gpu=use_gpu)
        else:
            raise ValueError('Unknown graph_type and/or similarity_type')

        Wq = init_W(Aq, k, mask, init_type, seed=q)        
        Wq, obj = run_symnmf(Aq, Wq, nmf, nmf_params, use_gpu, gpuid)
        
        IDq = np.argmax(Wq, 1)
        Bq = get_connectivity_matrix(IDq)
        
        all_W.append(Wq)
        all_obj.append(obj)
        connectivity_matrices.append(Bq)
    
    #
    # organize solutions from each perturbations
    #
    all_obj = np.array(all_obj)
    avg_W = np.mean(np.stack(all_W, axis=0), axis=0)
    
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
            "avg_obj": np.mean(all_obj),
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
                err_mean = np.mean(all_obj)
                plot_data["err_mean"] = '{0:.3f}'.format(err_mean)
            elif key == 'err_std':
                err_std = np.std(all_obj)
                plot_data["err_std"] = '{0:.3f}'.format(err_std)
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
        "W":avg_W,
        "err_mean":np.mean(all_obj),
        "err_std":np.std(all_obj),
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
            transpose=False,
            nmf_method="newton",
            nmf_obj_params={},
            graph_type="full",
            similarity_type="gaussian",
            nearest_neighbors=7,
            use_consensus_stopping=False,
            calculate_pac=True,
            mask=None,
            get_plot_data=False):

        # check the save path
        if save_output:
            if not Path(save_path).is_dir():
                Path(save_path).mkdir(parents=True)

        if n_nodes > 1 and MPI is None:
            sys.exit("Attempted to use n_nodes>1 but MPI is not available!")

        # overwrite NMF params with higher level definitions
        nmf_obj_params['n_iters'] = n_iters
        nmf_obj_params['use_consensus_stopping'] = use_consensus_stopping
            
        #
        # Object hyper-parameters
        #
        self.n_perturbs = n_perturbs
        self.perturb_type = perturb_type
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.save_path = save_path
        self.save_output = save_output
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.nmf_verbose = nmf_verbose
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
        # Begin NMFk
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
        }

        if self.n_jobs == 1:
            all_k_results = []
            for k in tqdm(Ks, total=len(Ks), disable=not self.verbose):
                k_result = symnmf_parallel_wrapper(gpuid=0, k=k, **job_data)
                all_k_results.append(k_result)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs)
            futures = [executor.submit(symnmf_parallel_wrapper, gpuid=kidx % self.n_jobs, k=k, **job_data) for kidx, k in enumerate(Ks)]
            all_k_results = [future.result() for future in concurrent.futures.as_completed(futures)]

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
                    argmin = np.argmin(combined_result["pac"])
                    results["clusters"] = np.argmax(combined_result["W"][argmin], 1)
                    
            # final plot
            if self.save_output:
                plot_SymNMFk(
                    combined_result, 
                    self.experiment_name, 
                    save_path, 
                    plot_final=True,
                )
                append_to_note(["#" * 100], save_path)
                append_to_note(["end_time= "+str(datetime.now())], save_path)
                append_to_note(
                    ["total_time= "+str(time.time() - start_time) + " (seconds)"], save_path)
        
            
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
