try:
    import cupy as cp
    import cupyx.scipy.sparse
except Exception:
    cp = None
    cupyx = None

from ..decompositions.rescal_fro_mu import R_update
import numpy as np

#
# Matrix operations
#
def run_nmf(Y, W, H, nmf, nmf_params, use_gpu:bool, gpuid:int):
    if use_gpu:
        with cp.cuda.Device(gpuid):
            W, H, other_results = nmf(X=Y, W=W, H=H, **nmf_params)
    else:
        W, H, other_results = nmf(X=Y, W=W, H=H, **nmf_params)

    return W, H, other_results

def run_symnmf(Y, W, nmf, nmf_params, use_gpu:bool, gpuid:int):
    if use_gpu:
        with cp.cuda.Device(gpuid):
            W, obj = nmf(Y, W=W, **nmf_params)
    else:
        W, obj = nmf(Y, W=W, **nmf_params)
    return W, obj

def run_thresholding(Y, W, H, factor_thresholding_obj_params, thresholding_function, use_gpu, gpuid):
    if use_gpu:
        with cp.cuda.Device(gpuid):
            dtype = Y.dtype
            thresh_results = thresholding_function(Y, W, H, use_gpu=use_gpu, **factor_thresholding_obj_params)
            
            if isinstance(thresh_results["wt"], np.ndarray):
                thresh_results["wt"] = cp.array(thresh_results["wt"])
            if isinstance(thresh_results["ht"], np.ndarray):
                thresh_results["ht"] = cp.array(thresh_results["ht"])

            W = (W >= thresh_results["wt"]).astype(dtype)
            H = (H >= thresh_results["ht"]).astype(dtype)
    else:
        dtype = Y.dtype
        thresh_results = thresholding_function(Y, W, H, use_gpu=use_gpu, **factor_thresholding_obj_params)
        W = (W >= thresh_results["wt"]).astype(dtype)
        H = (H >= thresh_results["ht"]).astype(dtype)
    return W, H

#
# Tensor operations
#
def run_rescal(Y, A, rescal, rescal_params, use_gpu:bool, gpuid:int):
    k = A.shape[1]
    if use_gpu:
        with cp.cuda.Device(gpuid):
            R = R_update(Y, A, [cp.random.rand(k, k) for _ in range(len(Y))])
            A, R = rescal(X=Y, A=A, R=R, **rescal_params)
    else:
        R = R_update(Y, A, [np.random.rand(k, k) for _ in range(len(Y))])
        A, R = rescal(X=Y, A=A, R=R, **rescal_params)

    return A, R