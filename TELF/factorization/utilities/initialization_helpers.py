
from ..decompositions.utilities.nnsvd import nnsvd

import numpy as np
import scipy.sparse

def init_WH(Y, k, mask, init_type:str):

    if init_type == "nnsvd":
        if mask is not None:
            Y[mask] = 0
        W, H = nnsvd(Y, k, use_gpu=False)
            
    elif init_type == "random":
        W, H = np.random.rand(Y.shape[0], k), np.random.rand(k, Y.shape[1])
        
    return W, H

def init_W(Y, k, mask, init_type:str, seed=42):
    if init_type == "nnsvd":
        if mask is not None:
            Y[mask] = 0
        W, _ = nnsvd(Y, k, use_gpu=False)
    elif init_type == "random":
        np.random.seed(seed)
        W = 2 * np.sqrt(np.mean(Y) / k) * np.random.rand(Y.shape[0], k)
    return W

def init_A(Y, k, init_type:str):
    if init_type == "nnsvd":
        if scipy.sparse.issparse(Y[0]):
            Y_tmp = scipy.sparse.hstack((Y))
        else:
            Y_tmp = np.hstack((Y))

        A, _ = nnsvd(Y_tmp, k, use_gpu=False)
            
    elif init_type == "random":
        A = np.random.rand(Y[0].shape[0], k)
        
    return A
