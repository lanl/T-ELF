from .utilities.generic_utils import get_np, get_scipy, update_opts
from .utilities.math_utils import fro_norm


def H_update(X, W, H, opts=None, dual=None, use_gpu=True):
    r"""
    ADMM algorithm for the right factor, :math:`H`, in a nonnegative optimization with Frobenius norm loss function.

    .. math::
       \underset{H}{\operatorname{minimize}} &= \frac{1}{2} \Vert X - W H \Vert_F^2 \\
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
    if dual is not None:
        dual = dual.T
    return W_update(X.T, H.T, W.T, opts=opts, dual=dual, use_gpu=use_gpu).T


def W_update(X, W, H, opts=None, dual=None, use_gpu=True):
    r"""
    ADMM algorithm for the left factor, :math:`W`, in a nonnegative optimization with Frobenius norm loss function.

    .. math::
       \underset{W}{\operatorname{minimize}} &= \frac{1}{2} \Vert X - W H \Vert_F^2 \\
       \text{subject to} & \quad W \geq 0

    Args:
      X (ndarray, sparse matrix): Nonnegative m by n matrix to decompose.

      W (ndarray): Nonnegative m by k initialization of left factor of X.

      H (ndarray): Nonnegative k by n right factor of X.

      opts (dict), optional: Dictionary or optional arguments.

        'hist' (list): list to append the objective function to.

        'niter' (int): number of iterations.

        'rho' (double): convergence parameter.

    Returns:
      W (ndarray): Nonnegative k by n right factor of X.
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
    if dual is None:
        dual = np.zeros_like(W)
    H = H.astype(dtype)
    HHT_rhoI = scipy.linalg.lu_factor(
        H @ H.T + opts["rho"] * np.identity(H.shape[0], dtype=dtype),
        overwrite_a=True,
        check_finite=False,
    )
    if scipy.sparse.issparse(X):
        # bug in setting has_canonical_format flag in cupy
        # https://github.com/cupy/cupy/issues/2365
        # issue is closed, but still not fixed.
        X = X.tocsr()
        X._has_canonical_format = True
        XHT = np.array(X.dot(H.T))
    else:
        XHT = X @ H.T
    for i in range(opts["niter"]):
        primal = scipy.linalg.lu_solve(
            HHT_rhoI,
            (XHT + opts["rho"] * (W - dual)).T,
            overwrite_b=True,
            check_finite=False,
        ).T
        W = np.maximum(primal + dual, 0)
        dual += primal - W

        if opts["hist"] is not None:
            opts["hist"].append(fro_norm(X - W @ H))
    return W


def nmf(X, W, H, use_gpu=True, opts=None):
    r"""
    ADMM algorithm for a nonnegative optimization with Frobenius norm loss function.

    .. math::
       \underset{W,H}{\operatorname{minimize}} &= \frac{1}{2} \Vert X - W H \Vert_F^2 \\
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
    default_opts = {
        "niter": 1000,
        "hist": None,
        "W_opts": {"niter": 1, "rho": 1.0, "hist": None},
        "H_opts": {"niter": 1, "hist": None, "rho": 1.0},
    }
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

    Wd = np.zeros_like(W)
    Hd = np.zeros_like(H)
    if scipy.sparse.issparse(X):
        Xcsc = X.T.T.tocsc()
        Xcsr = X.T.T.tocsr()
    else:
        pass

    for i in range(opts["niter"]):
        if scipy.sparse.issparse(X):
            H = H_update(Xcsc, W, H, opts["H_opts"], dual=Hd)
            W = W_update(Xcsr, W, H, opts["W_opts"], dual=Wd)
        else:
            H = H_update(X, W, H, opts["H_opts"], dual=Hd)
            W = W_update(X, W, H, opts["W_opts"], dual=Wd)

        if opts["hist"] is not None:
            opts["hist"].append(fro_norm(X - W @ H))
    Wsum = np.sum(W, 0, keepdims=True)
    H = H * Wsum.T
    W = W / np.maximum(Wsum, eps)
    return (W, H)
