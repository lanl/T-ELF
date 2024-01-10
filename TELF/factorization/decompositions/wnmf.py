"""
Implements WLRA as WNMF from the MATLAB version in https://gitlab.com/ngillis/nmfbook
"""
from .utilities.generic_utils import get_np, get_scipy
from tqdm import tqdm
import sys
def nmf(X, W, H,
        WEIGHTS = None,
        lamb = 1e-6,
        niter=1000, use_gpu=True,
        nmf_verbose=True,
        ):
    
    #
    # prepare
    #
    scipy = get_scipy(X, use_gpu=use_gpu)
    np = get_np(X, use_gpu=use_gpu)
    dtype = X.dtype

    if scipy.sparse.issparse(X):
        sys.exit("WNMFk is not designed to work with sparse matrix.")

    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    
    W = np.maximum(W.astype(dtype), eps)
    H = np.maximum(H.astype(dtype), eps)
    K = H.shape[0]
    assert K == W.shape[1]
    W, H = __scaleWH(X=X, W=W, H=H, use_gpu=use_gpu)
    
    if WEIGHTS is None:
        WEIGHTS = X.copy()
        WEIGHTS[np.where(WEIGHTS)] = 1
    
    #
    # main loop
    #
    for _ in tqdm(range(niter), disable=nmf_verbose == False):
        R = X - W@H

        # for each rank
        for curr_k in range(K):
            R += np.outer(W[:, curr_k], H[curr_k])
            Rp = R * WEIGHTS

            # W update
            W[:, curr_k] = np.dot(Rp, H[curr_k]) / (np.dot(WEIGHTS, H[curr_k]**2) + lamb)
            W[:, curr_k] = np.maximum(W[:, curr_k].astype(dtype), eps)

            # H update
            H[curr_k] = np.dot(Rp.T, W[:, curr_k]) / (np.dot(WEIGHTS.T, W[:, curr_k]**2) + lamb)
            W[:, curr_k] = np.maximum(W[:, curr_k].astype(dtype), eps)

            R -= np.outer(W[:, curr_k], H[curr_k])

        W, H = __scaleWH(X=X, W=W, H=H, use_gpu=use_gpu)

    return (W, H, {})

def __scaleWH(X, W, H, use_gpu):
    np = get_np(X, use_gpu=use_gpu)
    m, r = W.shape
    
    normW = np.sqrt(np.sum(W**2, axis=0)) + 1e-16
    normH = np.sqrt(np.sum(H**2, axis=1)) + 1e-16
    
    for k in range(r):
        W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])
        H[k] = H[k] / np.sqrt(normH[k]) * np.sqrt(normW[k])
    
    return W, H