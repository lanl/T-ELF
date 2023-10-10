#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from .decompositions.utilities.math_utils import relative_error, fro_norm, relative_error_rescal
from .decompositions.utilities.generic_utils import grid_eval
from .decompositions.utilities.clustering import custom_k_means, silhouettes
from .decompositions.rescal_fro_mu import R_update
from .decompositions.utilities.resample import uniform_product
from .decompositions.utilities.nnsvd import nnsvd
from .decompositions.rescal_fro_mu import rescal as rescal
from .utilities.plot_NMFk import plot_NMFk
from .utilities.take_note import take_note

from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
import GPUtil
import time
import warnings
import numpy as np
from tqdm import tqdm
import scipy.sparse
import os
from tqdm import tqdm
import sys
import socket
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


def rescal_wrapper(
        perturbation, rescal, rescal_params, init="nnsvd", X=None, k=None, epsilon=None, gpuid=0, use_gpu=True,
        calculate_error=True):
    """


    Parameters
    ----------
    perturbation : TYPE
        DESCRIPTION.
    X : TYPE, optional
        DESCRIPTION. The default is None.
    k : TYPE, optional
        DESCRIPTION. The default is None.
    epsilon : TYPE, optional
        DESCRIPTION. The default is None.
    gpu_queue : TYPE, optional
        DESCRIPTION. The default is None.
    opts : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    W : TYPE
        DESCRIPTION.
    H : TYPE
        DESCRIPTION.
    error : TYPE
        DESCRIPTION.

    """

    np.random.seed(perturbation)

    # perturbation or resampling
    Y = [uniform_product(X_, epsilon) for X_ in X]

    # initialization
    if init == "nnsvd":
        if scipy.sparse.issparse(Y[0]):
            t1 = scipy.sparse.hstack((Y))
            t2 = scipy.sparse.hstack(([i.T for i in Y]))
            tmp = scipy.sparse.hstack((t1, t2))
        else:
            t1 = np.hstack((Y))
            t2 = np.hstack(([i.T for i in Y]))
            tmp = np.hstack((t1, t2))
        A, _ = nnsvd(tmp, k)

    elif init == "random":
        A = np.random.rand(Y[0].shape[0], k)

    if use_gpu:
        # get a gpu
        # if gpuid==-1:
        #   gpuid = job_queue.get()
        with cp.cuda.Device(gpuid):
            # move data and initialization from host to device
            A = cp.array(A)
            if scipy.sparse.issparse(Y[0]):
                Y = [cupyx.scipy.sparse.csr_matrix((cp.array(Y1.data), cp.array(Y1.indices), cp.array(Y1.indptr)),
                                                   shape=Y1.shape, dtype=Y1.dtype) for Y1 in Y]
            else:
                Y = [cp.array(Y1) for Y1 in Y]
            R = R_update(Y, A, [cp.random.rand(k, k) for _ in range(len(X))])
            # do optimization on GPU
            A_, R_ = rescal(Y, A, R, **rescal_params)

            # W_, H_ = nmf(X=Y, W=W, H=H, **nmf_params)

            # move solution from device to host
            A = cp.asnumpy(A_)
            R = [cp.asnumpy(h_) for h_ in R_]

            # release the GPU so other processes can use
            del Y, A_, R_
            cp._default_memory_pool.free_all_blocks()

        # release the GPU so other processes can use
        # job_queue.put(gpuid)
    else:
        # job_id = job_queue.get()
        R = R_update(Y, A, [np.random.rand(k, k) for _ in range(len(X))])
        A, R = rescal(X=Y, A=A, R=R, **rescal_params)
        # job_queue.put(job_id)

    if calculate_error:
        error = relative_error_rescal(X, A, R)
    else:
        error = 0
    return A, R, error


class RESCALk:
    def __init__(
            self,
            n_perturbs=20,
            n_iters=100,
            epsilon=0.015,
            n_jobs=1,
            n_nodes=1,
            init="nnsvd",
            use_gpu=True,
            save_path="./",
            save_output=True,
            get_plot_data=False,
            collect_output=False,
            predict_k=False,
            verbose=False,
            rescal_verbose=False,
            elegant_verbose=True,
            transpose=False,
            sill_thresh=0.8,
            rescal_func=None,
            rescal_method="rescal_fro_mu",
            rescal_obj_params={},
            pruned=True,
            calculate_error=True,
            joblib_backend="multiprocessing",
    ):
        """


        Parameters
        ----------
        n_perturbs : TYPE, optional
            DESCRIPTION. The default is 20.
        n_iters : TYPE, optional
            DESCRIPTION. The default is 100.
        epsilon : TYPE, optional
            DESCRIPTION. The default is 0.015.
        n_jobs : TYPE, optional
            DESCRIPTION. The default is 1.
        init : TYPE, optional
            DESCRIPTION. The default is "nnsvd".
        use_gpu : TYPE, optional
            DESCRIPTION. The default is True.
        save_path : TYPE, optional
            DESCRIPTION. The default is ".".
        save_output : TYPE, optional
            DESCRIPTION. The default is True.
        collect_output : TYPE, optional
            DESCRIPTION. The default is False.
        predict_k : TYPE, optional
            DESCRIPTION. The default is True.
        verbose : TYPE, optional
            DESCRIPTION. The default is True.
        transpose : TYPE, optional
            DESCRIPTION. The default is False.
        sill_thresh : TYPE, optional
            DESCRIPTION. The default is 0.8.
         : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # check the save path
        if save_output:
            if not os.path.isdir(save_path):
                raise Exception("Directory in save_path parameter does not exist!")

        init_options = ["nnsvd", "random"]
        if init not in init_options:
            raise Exception("Invalid init. Choose from:" + str(", ".join(init_options)))

        if n_nodes > 1 and MPI is None:
            sys.exit("Attempted to use n_nodes>1 but MPI is not available!")

        #
        # Object hyper-parameters
        #
        self.n_perturbs = n_perturbs
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.init = init
        self.save_path = save_path
        self.save_output = save_output
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.rescal_verbose = rescal_verbose
        self.transpose = transpose
        self.collect_output = collect_output
        self.sill_thresh = sill_thresh
        self.get_plot_data = get_plot_data
        self.n_jobs = n_jobs
        self.n_nodes = n_nodes
        self.rescal = None
        self.rescal_method = rescal_method
        self.rescal_obj_params = rescal_obj_params
        self.pruned = pruned
        self.elegant_verbose = elegant_verbose
        self.predict_k = predict_k
        self.joblib_backend = joblib_backend
        self.calculate_error = calculate_error

        # check if GPUs available if requested
        if self.use_gpu:
            if len(GPUtil.getGPUs()) <= 0:
                warnings.warn("No GPU found! Using CPUs")
                self.use_gpu = False
                self.n_jobs = -1
            # multiprocessing on GPU
            if n_jobs < 0 or n_jobs > 1:
                multiprocessing.set_start_method('spawn', force=True)

        # if resources requested
        if n_jobs < 0:

            # gpu
            if self.use_gpu:
                resources = len(GPUtil.getGPUs())
            # cpu
            else:
                resources = multiprocessing.cpu_count()
            n_jobs = resources + (n_jobs + 1)

        # 0 or less resources requested
        if n_jobs <= 0:
            raise Exception("Number of GPUs or CPUs must be 1 or more.")

        # too many GPUs requested
        if self.use_gpu:
            if n_jobs > len(GPUtil.getGPUs()) and self.use_gpu:
                n_jobs = len(GPUtil.getGPUs())
                warnings.warn(
                    "Too mang GPUs requested. Reverting to max available:" + str(n_jobs)
                )
        else:
            # too many CPUs requested
            if n_jobs > multiprocessing.cpu_count() and not self.use_gpu:
                n_jobs = multiprocessing.cpu_count()
                warnings.warn(
                    "Too mang CPUs requested. Reverting to max available:" + str(n_jobs)
                )

            if n_jobs > self.n_perturbs:
                n_jobs = self.n_perturbs

        self.n_jobs = n_jobs

        #
        # Save information from the solution
        #
        self.total_exec_seconds = 0
        self.experiment_name = ""

        #
        # Prepare NMF function
        #
        avail_rescal_methods = ["rescal_fro_mu", "func"]
        if self.rescal_method not in avail_rescal_methods:
            raise Exception("Invalid RESCAL method is selected. Choose from: " +
                            ",".join(avail_rescal_methods))
        print(rescal_method)
        if self.rescal_method == "rescal_fro_mu":
            self.rescal_params = {
                "niter": self.n_iters,
                "use_gpu": self.use_gpu,
                "rescal_verbose": self.rescal_verbose
            }
            self.rescal = rescal

        elif self.rescal_method == "func" or rescal_func is not None:
            self.rescal_params = self.rescal_obj_params
            self.rescal = rescal_func

        else:
            raise Exception("Unknown RESCAL method or rescal_func was not passed")

    def fit(self, X, Ks, name="RESCALk", note=""):
        """


        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        Ks : TYPE
            DESCRIPTION.
        name : TYPE, optional
            DESCRIPTION. The default is "NMFk".
        note : TYPE, optional
            DESCRIPTION. The default is "".

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.

        """

        #
        # Error check
        #
        if len(Ks) == 0:
            raise Exception("Ks range is 0!")

        #
        # MPI
        #
        if self.n_nodes > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            Ks = self.chunk_Ks(Ks, n_chunks=self.n_nodes)[rank]
            if self.verbose:
                print("Rank=", rank, "Host=", socket.gethostname(), "Ks=", Ks)

        self.experiment_name = (
            str(name)
            + "_"
            + str(self.n_perturbs)
            + "perts_"
            + str(self.n_iters)
            + "iters_"
            + str(self.epsilon)
            + "eps"
        )
        save_path = os.path.join(self.save_path, self.experiment_name)
        if self.save_output and ((self.n_nodes == 1) or (self.n_nodes > 1 and rank == 0)):
            # if self.save_output:
            try:
                os.mkdir(save_path)
            except Exception:
                pass

            notes = dict()
            notes["Ks"] = Ks
            notes["num_perturbations"] = self.n_perturbs
            notes["epsilon"] = self.epsilon
            notes["init"] = self.init
            notes["n_jobs"] = self.n_jobs
            notes["experiment_name"] = name
            notes["num_iterations"] = self.n_iters
            notes["note"] = note
            take_note(notes, save_path)

        sils_min = []
        sils_mean = []
        sils_std = []
        err_reg = []
        err_mean = []
        err_std = []
        decomp_data = list()
        col_err = list()
        #
        # Prune
        #
        if self.pruned:
            warnings.warn("Pruning for RESCAL is not implemented yet!")
        else:
            rows, cols = None, None
        #
        # Begin
        #
        start_time = time.time()

        if self.elegant_verbose:
            print("NMFk:")

        for i, k in tqdm(enumerate(Ks), total=len(Ks), disable=self.verbose == False):

            # solve for current k
            if self.n_jobs == 1:
                A_all, R_all, errors = [], [], []
                for p in range(self.n_perturbs):
                    w, h, e = rescal_wrapper(
                        p,
                        rescal=self.rescal,
                        rescal_params=self.rescal_params,
                        calculate_error=self.calculate_error,
                        init=self.init,
                        X=X,
                        k=k,
                        epsilon=self.epsilon,
                        use_gpu=self.use_gpu
                    )
                    A_all.append(w)
                    R_all.append(h)
                    errors.append(e)

                A_all = np.array(A_all).transpose((1, 2, 0))
                R_all = np.array(R_all)  # np.stack(R_all,axis=0).transpose((1, 2, 0))
                errors = np.array(errors)
            else:
                current_pert_results = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    backend=self.joblib_backend)(delayed(rescal_wrapper)(
                        pert,
                        self.rescal,
                        self.rescal_params,
                        self.init,
                        X,
                        k,
                        self.epsilon,
                        pert % self.n_jobs,
                        self.use_gpu,
                        self.calculate_error
                    ) for pert in range(self.n_perturbs))

                A_all, R_all, errors = [], [], []
                for w, h, e, in current_pert_results:
                    A_all.append(w)
                    R_all.append(h)
                    errors.append(e)

                A_all = np.array(A_all).transpose((1, 2, 0))
                R_all = np.array(R_all)  # np.stack(R_all,axis=0).transpose((1, 2, 0))
                errors = np.array(errors)

            err_mean.append(np.mean(errors))
            err_std.append(np.std(errors))

            # cluster the solutions
            A, A_clust = custom_k_means(A_all)
            sils_all = silhouettes(A_clust)
            sils_min.append(np.min(np.mean(sils_all, 1)))
            sils_mean.append(np.mean(np.mean(sils_all, 1)))
            sils_std.append(np.std(np.mean(sils_all, 1)))

            if self.use_gpu:
                # do regression and compute the relative error
                if scipy.sparse.issparse(X[0]):
                    Y = [cupyx.scipy.sparse.csr_matrix(
                        (cp.array(x.data), cp.array(x.indices), cp.array(x.indptr)),
                        shape=x.shape,
                        dtype=x.dtype,
                    ) for x in X]
                else:
                    Y = [cp.array(x) for x in X]

                R_ = R_update(Y, cp.array(A), [cp.random.rand(k, k)
                              for _ in range(len(X))], use_gpu=self.use_gpu)
                R = [cp.asnumpy(r) for r in R_]
                del Y, R_
                cp._default_memory_pool.free_all_blocks()

            else:
                R = R_update(X, A, [np.random.rand(k, k)
                             for _ in range(len(X))], use_gpu=self.use_gpu)

            # reconstruction error
            if self.calculate_error:
                error = relative_error_rescal(X, A, R)
            else:
                error = 0
            err_reg.append(error)

            if self.elegant_verbose:
                print("Error Mean=" + str(err_mean[i]) + ", Regress Error=" + str(err_reg[i]) + ", Sill Min=" + str(
                    sils_min[i]))

            # unprune
            if self.pruned:
                pass

            # save output factors and the plot
            if self.save_output:
                save_data = {
                    "A": A,
                    "R": R,
                    "sils_all": sils_all,
                    "error": error,
                    "errors": errors,
                }
                np.savez_compressed(
                    save_path
                    + "/AR"
                    + "_k="
                    + str(k)
                    + ".npz",
                    **save_data
                )

                plot_data = dict()
                plot_data["Ks"] = Ks[:i + 1]
                plot_data["sils_min"] = sils_min
                plot_data["sils_mean"] = sils_mean
                plot_data["sils_std"] = sils_std
                plot_data["err_reg"] = err_reg
                plot_data["err_mean"] = err_mean
                plot_data["err_std"] = err_std
                plot_NMFk(
                    plot_data, k, self.experiment_name, save_path
                )

            # collect output to be returned
            if self.collect_output:
                decomp_data.append({"A": A, "R": R, "k": k})

        # wait for everyone if MPI
        if self.n_nodes > 1:

            # put together the data that we will collect
            share_data = {
                "Ks": Ks,
                "decomp_data": decomp_data,
                "sils_min": sils_min,
                "sils_mean": sils_mean,
                "sils_std": sils_std,
                "err_reg": err_reg,
                "err_mean": err_mean,
                "err_std": err_std,
            }

            # wait for everyone to be done
            comm.Barrier()

            # gather the shared data to the main node
            all_share_data = comm.gather(share_data, root=0)

            # organize the data in the main process
            if rank == 0:
                all_Ks = []
                all_decomp_data = []
                all_sils_min = []
                all_sils_mean = []
                all_sils_std = []
                all_err_reg = []
                all_err_mean = []
                all_err_std = []

                for data in all_share_data:
                    all_Ks += data["Ks"]
                    all_decomp_data += data["decomp_data"]
                    all_sils_min += data["sils_min"]
                    all_sils_mean += data["sils_mean"]
                    all_sils_std += data["sils_std"]
                    all_err_reg += data["err_reg"]
                    all_err_mean += data["err_mean"]
                    all_err_std += data["err_std"]

                all_Ks = np.array(all_Ks)
                Ks_sort_indices = np.argsort(all_Ks)
                Ks = list(all_Ks[Ks_sort_indices])
                sils_min = np.array(all_sils_min)[Ks_sort_indices]
                sils_mean = np.array(all_sils_mean)[Ks_sort_indices]
                sils_std = np.array(all_sils_std)[Ks_sort_indices]
                err_reg = np.array(all_err_reg)[Ks_sort_indices]
                err_mean = np.array(all_err_mean)[Ks_sort_indices]
                err_std = np.array(all_err_std)[Ks_sort_indices]

                if self.collect_output:
                    decomp_data = np.array(all_decomp_data)[Ks_sort_indices]

            else:
                sys.exit(0)

        if self.n_nodes == 1 or (self.n_nodes > 1 and rank == 0):

            # holds the final results
            results = {}
            total_exec_seconds = time.time() - start_time
            results["time"] = total_exec_seconds

            k_predict = 0

            # latent factors W,H for each k
            if self.collect_output:
                results["all_factors"] = decomp_data

            # final plot
            if self.save_output:
                plot_data = dict()
                plot_data["Ks"] = Ks
                plot_data["sils_min"] = sils_min
                plot_data["sils_mean"] = sils_mean
                plot_data["sils_std"] = sils_std
                plot_data["err_reg"] = err_reg
                plot_data["err_mean"] = err_mean
                plot_data["err_std"] = err_std
                plot_NMFk(
                    plot_data, k_predict, self.experiment_name, save_path  # , plot_predict=self.predict_k
                )

            if self.get_plot_data:
                plot_data = dict()
                plot_data["Ks"] = Ks
                plot_data["sils_min"] = sils_min
                plot_data["sils_mean"] = sils_mean
                plot_data["sils_std"] = sils_std
                plot_data["err_reg"] = err_reg
                plot_data["err_mean"] = err_mean
                plot_data["err_std"] = err_std
                results["plot_data"] = plot_data

            if self.elegant_verbose:
                print("===========================================")
                print("Final Error Mean= " + str(err_mean[-1]))
                print("Final Regress Error= " + str(err_reg[-1]))
                print("Final Sill Min= " + str(sils_min[-1]))
                print("Total Execution Time= " + str(round(total_exec_seconds, 4)) + " seconds")
                print("===========================================")

            return results

    def chunk_Ks(self, Ks: list, n_chunks=2) -> list:
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
