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