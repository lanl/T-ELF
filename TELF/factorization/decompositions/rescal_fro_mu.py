from .utilities.generic_utils import get_np, get_scipy, update_opts
from .utilities.math_utils import fro_norm
from tqdm import tqdm


def A_update(X, A, R, opts=None, XT=None, use_gpu=True):
    r"""
    Multiplicative update algorithm for the A factor in a nonnegative optimization with Frobenius norm loss function.

    .. math::
       \underset{A}{\operatorname{minimize}} &= \sum \frac{1}{2} \Vert X_i - A R_i A^\top \Vert_F^2 \\
       \text{subject to} & \quad A \geq 0

    Args:
      X (list of ndarray, sparse matrix): List of nonnegative m by m matrices to decompose.

      A (ndarray): Nonnegative m by k initialization of the A factor.

      R (list of ndarray): List of nonnegative k by k matrices in the decomposition so X[k] is paired with R[k].

      opts (dict), optional: Dictionary or optional arguments.

        'hist' (list): list to append the objective function to.

        'niter' (int): number of iterations.

      XT (list of sparse matrix), optional: If X is list of sparse matrix, XT should be list of transpose matrices in csr format, XT = [x.T.tocsr() for x in X].

    Returns:
      A (ndarray): Nonnegative m by k factor in the decomposition.
    """
    np = get_np(*X)
    scipy = get_scipy(*X)
    dtype = X[0].dtype
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    default_opts = {"niter": 1000, "hist": None}
    opts = update_opts(default_opts, opts)

    A = np.maximum(A.astype(dtype), eps)
    R = [r.astype(dtype) for r in R]
    if scipy.sparse.issparse(X[0]):
        X = [x.astype(dtype).tocsr() for x in X]
        if XT is None:
            XT = [x.T.astype(dtype).tocsr() for x in X]
        # bug in setting has_canonical_format flag in cupy
        # https://github.com/cupy/cupy/issues/2365
        # issue is closed, but still not fixed.
        for x, xt in zip(X, XT):
            x._has_canonical_format = True
            xt._has_canonical_format = True
    else:
        X = [x.astype(dtype) for x in X]
    for i in range(opts["niter"]):
        ATA = A.T @ A
        num = np.zeros_like(A)
        denom = np.zeros_like(R[0])
        for j, r in enumerate(R):
            if scipy.sparse.issparse(X[j]):
                num += X[j].dot(A).dot(r.T) + XT[j].dot(A).dot(r)
            else:
                num += X[j].dot(A).dot(r.T) + X[j].T.dot(A).dot(r)
            denom += r.dot(ATA).dot(r.T) + (r.T).dot(ATA).dot(r)
        A *= num / (A @ denom+eps)

        if (i + 1) % 10 == 0:
            A = np.maximum(A, eps)

        if opts["hist"] is not None:
            opts["hist"].append(
                np.sqrt(np.sum([fro_norm(x - A @ r @ A.T) ** 2 for x, r in zip(X, R)]))
            )

    return A


def R_update(X, A, R, opts=None, use_gpu=True):
    r"""
    Multiplicative update algorithm for the R factor in a nonnegative optimization with Frobenius norm loss function.

    .. math::
       \underset{R}{\operatorname{minimize}} &= \sum \frac{1}{2} \Vert X_i - A R_i A^\top \Vert_F^2 \\
       \text{subject to} & \quad R \geq 0

    Args:
      X (list of ndarray, sparse matrix): List of nonnegative m by m matrices to decompose.

      A (ndarray): Nonnegative m by k matrix in the decomposition.

      R (list of ndarray): List of nonnegative k by k initilizations for the decomposition so X[k] is paired with R[k].

      opts (dict), optional: Dictionary or optional arguments.

        'hist' (list): list to append the objective function to.

        'niter' (int): number of iterations.

    Returns:
      R (list of ndarray): List of nonnegative k by k factors in the decomposition.
    """
    np = get_np(*X, use_gpu=use_gpu)
    scipy = get_scipy(*X, use_gpu=use_gpu)
    dtype = X[0].dtype
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    default_opts = {"niter": 1000, "hist": None}
    opts = update_opts(default_opts, opts)
    A = A.astype(dtype)
    R = [np.maximum(r.astype(dtype), eps) for r in R]

    if scipy.sparse.issparse(X[0]):
        X = [x.astype(dtype).tocsr() for x in X]
        # bug in setting has_canonical_format flag in cupy
        # https://github.com/cupy/cupy/issues/2365
        # issue is closed, but still not fixed.
        for x in X:
            x._has_canonical_format = True
    else:
        X = [x.astype(dtype) for x in X]
    ATXA = [A.T.dot(x.dot(A)) for x in X]
    ATA = A.T @ A
    for i in range(opts["niter"]):
        for j, r in enumerate(R):
            r /= np.maximum(ATA.T @ r @ ATA, eps)
            r *= ATXA[j]

        if (i + 1) % 10 == 0:
            for r in R:
                r = np.maximum(r, eps)

        if opts["hist"] is not None:
            opts["hist"].append(
                np.sqrt(np.sum([fro_norm(x - A @ r @ A.T) ** 2 for x, r in zip(X, R)]))
            )

    return R


# def rescal(X, A, R, opts=None):
def rescal(X, A, R,
           niter=1000, hist=None,
           A_opts={"niter": 1, "hist": None}, R_opts={"niter": 1, "hist": None},
           use_gpu=False,
           rescal_verbose=True
           ):
    r"""
    Multiplicative update algorithm for the R factor in a nonnegative optimization with Frobenius norm loss function.

    .. math::
       \underset{R}{\operatorname{minimize}} &= \sum \frac{1}{2} \Vert X_i - A R_i A^\top \Vert_F^2 \\
       \text{subject to} & \quad R \geq 0

    Args:
      X (list of ndarray, sparse matrix): List of nonnegative m by m matrices to decompose.

      A (ndarray): Nonnegative m by k initialization of the A factor.

      R (list of ndarray): List of nonnegative k by k initilizations for the decomposition so X[k] is paired with R[k].

      opts (dict), optional: Dictionary or optional arguments.

        'hist' (list): list to append the objective function to.

        'niter' (int): number of iterations.

    Returns:
      A (ndarray): Nonnegative m by k factor in the decomposition.

      R (list of ndarray): List of nonnegative k by k factors in the decomposition.
    """
    np = get_np(*X, use_gpu=use_gpu)
    scipy = get_scipy(*X, use_gpu=use_gpu)
    dtype = X[0].dtype
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")

    if scipy.sparse.issparse(X[0]):
        X = [x.astype(dtype).tocsr() for x in X]
        XT = [x.T.astype(dtype).tocsr() for x in X]
        # bug in setting has_canonical_format flag in cupy
        # https://github.com/cupy/cupy/issues/2365
        # issue is closed, but still not fixed.
        for x, xt in zip(X, XT):
            x._has_canonical_format = True
            xt._has_canonical_format = True
        A_args = {"XT": XT}
    else:
        X = [x.astype(dtype) for x in X]
    A = A.astype(dtype)
    R = [r.astype(dtype) for r in R]
    for i in tqdm(range(niter), disable=rescal_verbose == False):
        if scipy.sparse.issparse(X[0]):
            A = A_update(X, A, R, A_opts)
            R = R_update(X, A, R, R_opts)
        else:
            A = A_update(X, A, R, A_opts)
            R = R_update(X, A, R, R_opts)

        if hist is not None:
            hist.append(
                np.sqrt(np.sum([fro_norm(x - A @ r @ A.T) ** 2 for x, r in zip(X, R)]))
            )
    Asum = np.sum(A, 0, keepdims=True)+eps
    R = [Asum * r * Asum.T for r in R]
    A = A / Asum

    return (A, R)
