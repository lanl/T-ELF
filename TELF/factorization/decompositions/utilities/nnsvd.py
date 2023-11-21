from .generic_utils import get_np, get_scipy
from .math_utils import fro_norm


def nnsvd(X, k, use_gpu=False):
    """
    Nonnegative SVD algorithm for NMF initialization based off of Gillis et al. in https://arxiv.org/pdf/1807.04020.pdf.

    Args:
        X (ndarray): Nonnegative m by n matrix to approximate with nnsvd.
        k (int): The desired rank of the nonnegative approximation.

    Returns:
        W (ndarray): Nonnegative m by k left factor of X.
        H (ndarray): Nonnegative k by n right factor of X.
    """
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    dtype = X.dtype
    
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    
    #
    # Truncated SVD
    #
    U, S, V = scipy.sparse.linalg.svds(X, k=k)
    V = V.T
    
    #
    # NN-SVD
    #
    UP = np.where(U > 0, U, 0)
    UN = np.where(U < 0, -U, 0)
    VP = np.where(V > 0, V, 0)
    VN = np.where(V < 0, -V, 0)

    UP_norm = np.sqrt(np.sum(np.square(UP), 0), dtype=dtype)
    UN_norm = np.sqrt(np.sum(np.square(UN), 0), dtype=dtype)
    VP_norm = np.sqrt(np.sum(np.square(VP), 0), dtype=dtype)
    VN_norm = np.sqrt(np.sum(np.square(VN), 0), dtype=dtype)

    mp = np.sqrt(UP_norm * VP_norm * S, dtype=dtype)
    mn = np.sqrt(UN_norm * VN_norm * S, dtype=dtype)

    W = np.where(mp > mn, mp * UP / (UP_norm + eps), mn * UN / (UN_norm + eps))
    H = np.where(mp > mn, mp * VP / (VP_norm + eps), mn * VN / (VN_norm + eps)).T
    Wsum = np.sum(W, 0, keepdims=True)
    H = H * Wsum.T
    W = W / np.maximum(Wsum, eps)
    return W, H
