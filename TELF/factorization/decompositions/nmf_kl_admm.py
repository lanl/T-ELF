from .utilities.math_utils import kl_divergence, fro_norm
from .utilities.generic_utils import get_np, get_scipy, update_opts


def H_update(X, W, H, opts=None, use_gpu=True):
    r"""
    ADMM algorithm for the right factor, :math:`H`, in a nonnegative optimization with Kullback–Leibler divergence loss function.

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

        'rho' (double): convergence parameter.

    Returns:
      H (ndarray): Nonnegative k by n right factor of X.
    """
    return W_update(X.T, H.T, W.T, opts=opts, use_gpu=use_gpu).T


def W_update(X, W, H, opts=None, use_gpu=True):
    r"""
    ADMM algorithm for the left factor, :math:`W`, in a nonnegative optimization with Kullback–Leibler divergence loss function.

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

        'rho' (double): convergence parameter.

      nz_rows (ndarray), optional: If X is sparse, nz_rows is a 1d array of the row indices when X is in csr format. Useful when calling this function multiple times with the same sparse matrix X.

      nz_cols (ndarray), optional: If X is sparse, nz_cols is a 1d array of the col indices when X is in csr format. Useful when calling this function multiple times with the same sparse matrix X.

    Returns:
      W (ndarray): Nonnegative m by k left factor of X.
    """
    default_opts = {"niter": 1000, "hist": None, "rho": 1.0}
    opts = update_opts(default_opts, opts)
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    dtype = X.dtype
    
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    
    Y_dual = np.zeros(X.shape, dtype=dtype)
    W = W.astype(dtype)
    W_aux = W.copy()
    W_dual = np.zeros_like(W)
    H = H.astype(dtype)
    
    HHT_I = scipy.linalg.lu_factor(
        H @ H.T + np.identity(H.shape[0], dtype=dtype),
        overwrite_a=True,
        check_finite=False,
    )
    for i in range(opts["niter"]):
        C = opts["rho"] * (W_aux @ H - Y_dual) - 1
        Y = (C + np.sqrt(C ** 2 + 4 * opts["rho"] * X)) / (2 * opts["rho"])
        W = np.maximum(W_aux - W_dual, 0)
        W_aux = scipy.linalg.lu_solve(HHT_I, ((Y + Y_dual) @ H.T + W + W_dual).T).T
        Y_dual += Y - W_aux @ H
        W_dual += W - W_aux

        if opts["hist"] is not None:
            opts["hist"].append(kl_divergence(X, np.maximum(W @ H, eps)))
    return W


def nmf(X, W, H, opts=None, use_gpu=True):
    r"""
    ADMM algorithm for a nonnegative optimization with Kullback–Leibler divergence loss function.

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

        'pruned' (bool): indicator if the input matrix needs to be pruned of zeros.

        'rho' (double): convergence parameter.

    Returns:
      W (ndarray): Nonnegative m by k left factor of X.

      H (ndarray): Nonnegative k by n right factor of X.
    """
    default_opts = {"niter": 1000, "hist": None, "rho": 1.0}
    opts = update_opts(default_opts, opts)
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    dtype = X.dtype
    
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")

    W = W.astype(dtype)
    H = H.astype(dtype)
    Y_dual = np.zeros(X.shape, dtype=dtype)
    W_aux = W.astype(dtype)
    W_dual = np.zeros_like(W)
    H_aux = H.astype(dtype)
    H_dual = np.zeros_like(H)

    for i in range(opts["niter"]):
        C = opts["rho"] * (W_aux @ H_aux - Y_dual) - 1
        Y = (C + np.sqrt(C ** 2 + 4 * opts["rho"] * X)) / (2 * opts["rho"])
        W = np.maximum(W_aux - W_dual, 0)
        H = np.maximum(H_aux - H_dual, 0)
        WTW_I = scipy.linalg.lu_factor(
            W_aux.T @ W_aux + np.identity(W.shape[1], dtype=dtype),
            overwrite_a=True,
            check_finite=False,
        )
        H_aux = scipy.linalg.lu_solve(WTW_I, W_aux.T @ (Y + Y_dual) + H + H_dual)
        HHT_I = scipy.linalg.lu_factor(
            H_aux @ H_aux.T + np.identity(H.shape[0], dtype=dtype),
            overwrite_a=True,
            check_finite=False,
        )
        W_aux = scipy.linalg.lu_solve(HHT_I, ((Y + Y_dual) @ H_aux.T + W + W_dual).T).T
        Y_dual += Y - W_aux @ H_aux
        W_dual += W - W_aux
        H_dual += H - H_aux

        if opts["hist"] is not None:
            opts["hist"][0].append(kl_divergence(X, np.maximum(W @ H, eps)))
            opts["hist"][1].append(fro_norm(X - W @ H) / fro_norm(X))
    Wsum = np.sum(W, 0, keepdims=True)
    H = H * Wsum.T
    W = W / np.maximum(Wsum, eps)

    return (W, H)
