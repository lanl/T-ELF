from .utilities.generic_utils import get_np, get_scipy, update_opts
from .utilities.math_utils import (
    kl_divergence,
    sparse_divide_product,
    nz_indices,
    nan_to_num,
)
from tqdm import tqdm
from .utilities.concensus_matrix import compute_connectivity_mat


def H_update(X, W, H, opts=None, nz_rows=None, nz_cols=None, use_gpu=True, mask=None):
    r"""
    Multiplicative update algorithm for the right factor, :math:`H`, in a nonnegative optimization with Kullback–Leibler divergence loss function.

    .. math::
       \underset{H}{\operatorname{minimize}} &= \operatorname{D}_{KL}(X, W H) = \sum_{i,j} X_{i,j} \log \frac{X_{i,j}}{(WH)_{i,j}} - X_{i,j} + (WH)_{i,j} \\
       \text{subject to} & \quad H \geq 0

    Args:
      X (ndarray, sparse matrix): Nonnegative m by n matrix to decompose.

      W (ndarray): Nonnegative m by k left factor of X.

      H (ndarray): Nonnegative k by n initialization of right factor of X.

      opts (dict), optional: Dictionary or optional arguments.

        'hist' (list): list to append the objective function to.

        'niter' (int): number of iterations.

      nz_rows (ndarray), optional: If X is sparse, nz_rows is a 1d array of the row indices when X is in csc format. Useful when calling this function multiple times with the same sparse matrix X.

      nz_cols (ndarray), optional: If X is sparse, nz_cols is a 1d array of the col indices when X is in csc format. Useful when calling this function multiple times with the same sparse matrix X.

    Returns:
      H (ndarray): Nonnegative k by n right factor of X.
    """
    if mask is not None:
        mask = mask.T
    return W_update(X.T, H.T, W.T, opts=opts, use_gpu=use_gpu, mask=None, nz_rows=nz_cols, nz_cols=nz_rows).T


def W_update(X, W, H, opts=None, nz_rows=None, nz_cols=None, use_gpu=True, mask=None):
    r"""
    Multiplicative update algorithm for the left factor, :math:`W`, in a nonnegative optimization with Kullback–Leibler divergence loss function.

    .. math::
       \underset{W}{\operatorname{minimize}} &= \operatorname{D}_{KL}(X, W H) = \sum_{i,j} X_{i,j} \log \frac{X_{i,j}}{(WH)_{i,j}} - X_{i,j} + (WH)_{i,j} \\
       \text{subject to} & \quad W \geq 0

    Args:
      X (ndarray, sparse matrix): Nonnegative m by n matrix to decompose.

      W (ndarray): Nonnegative m by k initialization of left factor of X.

      H (ndarray): Nonnegative k by n right factor of X.

      opts (dict), optional: Dictionary or optional arguments.

        'hist' (list): list to append the objective function to.

        'niter' (int): number of iterations.

      nz_rows (ndarray), optional: If X is sparse, nz_rows is a 1d array of the row indices when X is in csr format. Useful when calling this function multiple times with the same sparse matrix X.

      nz_cols (ndarray), optional: If X is sparse, nz_cols is a 1d array of the col indices when X is in csr format. Useful when calling this function multiple times with the same sparse matrix X.

    Returns:
      W (ndarray): Nonnegative m by k left factor of X.
    """
    default_opts = {"niter": 1000, "hist": None}
    opts = update_opts(default_opts, opts)
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    dtype = X.dtype
    
    if np.issubdtype(dtype, np.integer):
        eps = np.iinfo(dtype).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")

    if mask is not None:
        X[mask] = 0  # * set NaNs to zeros first, will update later

        
    W = np.maximum(W.astype(dtype), eps)
    H = H.astype(dtype)

    #Hnormalized = (H / np.maximum(np.sum(H, 1, keepdims=True), eps)).T
    H_norm = np.sum(H, axis=1, keepdims=True).T + eps
    
    if mask is not None:
        for i in range(opts["niter"]):

            if scipy.sparse.issparse(X):
                X = X.tocsr()
                X._has_canonical_format = True
                XHT = X.dot(H.T)
            else:
                XHT = X @ H.T

            W /= (W @ HHT + eps)
            W *= XHT

            if (i + 1) % 10 == 0:
                W = np.maximum(W, eps)

            Xhat = W@H
            X[mask] = Xhat[mask]  # *update NaN spots

            if opts["hist"] is not None:
                opts["hist"].append(fro_norm(X - W @ H))
    else:
        if scipy.sparse.issparse(X):
            # bug in setting has_canonical_format flag in cupy
            # https://github.com/cupy/cupy/issues/2365
            # issue is closed, but still not fixed.
            X = X.tocsr()
            X._has_canonical_format = True

        else:
            pass
        for i in range(opts["niter"]):
            if scipy.sparse.issparse(X):
                # W *= nan_to_num(sparse_divide_product(X, W, H, nz_rows, nz_cols), 1.0).dot(
                #    Hnormalized
                W *= ((sparse_divide_product(X, W, H, nz_rows, nz_cols)).dot(H.T)) / H_norm

            else:
                #W *= nan_to_num(X / (W @ H), 1.0) @ Hnormalized
                W *= ((X / (W @ H + eps)) @ H.T) / H_norm
            if (i + 1) % 10 == 0:
                W = np.maximum(W, eps)

            if opts["hist"] is not None:
                opts["hist"].append(kl_divergence(X, np.maximum(W @ H, eps)))
    return W


def nmf(X, W, H,
        niter=1000, hist=None,
        W_opts={"niter": 1, "hist": None}, H_opts={"niter": 1, "hist": None},
        use_gpu=True,
        nmf_verbose=True,
        mask=None, use_consensus_stopping=0
        ):
    r"""
    Multiplicative update algorithm for a nonnegative optimization with Kullback–Leibler divergence loss function.

    .. math::
       \underset{W,H}{\operatorname{minimize}} &= \operatorname{D}_{KL}(X, W H) = \sum_{i,j} X_{i,j} \log \frac{X_{i,j}}{(WH)_{i,j}} - X_{i,j} + (WH)_{i,j} \\
       \text{subject to} & \quad W \geq 0 \\
       & \quad H \geq 0

    Args:
      X (ndarray, sparse matrix): Nonnegative m by n matrix to decompose.

      W (ndarray): Nonnegative m by k initialization of left factor of X.

      H (ndarray): Nonnegative k by n initialization of right factor of X.

      opts (dict), optional: Dictionary or optional arguments.

        'hist' (list): list to append the objective function to.

        'niter' (int): number of iterations.

        'W_opts' (dict): options dictionary for :meth:`W_update`.

        'H_opts' (dict): options dictionary for :meth:`H_update`.

    Returns:
      W (ndarray): Nonnegative m by k left factor of X.

      H (ndarray): Nonnegative k by n right factor of X.
    """

    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    dtype = X.dtype

    # * Nans currently only works with numpy
    if mask is not None:
        X[mask] = 0  # * set NaNs to zeros first, will update later

    if np.issubdtype(dtype, np.integer):
        eps = np.iinfo(dtype).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")

    W = np.maximum(W.astype(dtype), eps)
    H = np.maximum(H.astype(dtype), eps)

    if scipy.sparse.issparse(X):
        Xcsr = X.tocsr()
        Xcsc = X.tocsc()
        H_args, W_args = {}, {}
        H_args["nz_rows"], H_args["nz_cols"] = nz_indices(Xcsc, use_gpu=use_gpu)
        W_args["nz_rows"], W_args["nz_cols"] = nz_indices(Xcsr, use_gpu=use_gpu)
    else:
        pass

    if use_consensus_stopping > 0:
        conmatold = 0
        conmat = 0
        inc = 0

    for i in tqdm(range(niter), disable=nmf_verbose == False):
        if scipy.sparse.issparse(X):
            H = H_update(Xcsc, W, H, H_opts, **H_args)
            W = W_update(Xcsr, W, H, W_opts, **W_args)
        else:
            H = H_update(X, W, H, H_opts)
            W = W_update(X, W, H, W_opts)
        if i % 10 == 0:
            H = np.maximum(H.astype(dtype), eps)
            W = np.maximum(W.astype(dtype), eps)
        if hist is not None:
            hist.append(kl_divergence(X, np.maximum(W @ H, eps)))

        if mask is not None:  # *update mask
            Xhat = W@H
            X[mask] = Xhat[mask]

        # * checking the consensus stopping
        if use_consensus_stopping > 0:
            conmat = compute_connectivity_mat(H)
            if np.sum(conmat != conmatold) == 0:
                inc += 1
            if inc >= use_consensus_stopping:
                break
            else:
                conmatold = conmat

    Wsum = np.sum(W, 0, keepdims=True)
    H = H * Wsum.T
    W = W / Wsum

    return (W, H)
