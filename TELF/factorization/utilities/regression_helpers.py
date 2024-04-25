try:
    import cupy as cp
except Exception:
    cp = None
    cupyx = None

from ..decompositions.nmf_fro_mu import H_update
from ..decompositions.rescal_fro_mu import R_update
from .data_host_transfer_helpers import put_X_gpu, put_tensor_X_gpu, put_tensor_R_cpu
import numpy as np

#
# Matrix regression
#
def H_regression(X, W, mask, use_gpu:bool, gpuid:int):
    if use_gpu:
        Y = put_X_gpu(X, gpuid)
        with cp.cuda.Device(gpuid):
            H_ = H_update(Y, cp.array(W), cp.random.rand(
                W.shape[1], Y.shape[1]), use_gpu=use_gpu, mask=mask)
            H = cp.asnumpy(H_)
            
        del Y, H_
        cp._default_memory_pool.free_all_blocks()
        
    else:
        H = H_update(X, W, np.random.rand(
            W.shape[1], X.shape[1]), use_gpu=use_gpu, mask=mask)
        
    return H

#
# Tensor regression
#
def R_regression(X, A, use_gpu:bool, gpuid:int):
    k = A.shape[1]
    if use_gpu:
        Y = put_tensor_X_gpu(X, gpuid)

        with cp.cuda.Device(gpuid):
            R_ = R_update(Y, cp.array(A), [cp.random.rand(k, k) for _ in range(len(Y))], use_gpu=use_gpu)
            R = put_tensor_R_cpu(R_)
            
        del Y, R_
        cp._default_memory_pool.free_all_blocks()
        
    else:
        R = R_update(X, A, [np.random.rand(k, k) for _ in range(len(X))], use_gpu=use_gpu)
        
    return R