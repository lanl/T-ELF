from .utilities.generic_utils import get_np, get_scipy, update_opts
from .utilities.math_utils import fro_norm
from tqdm import tqdm


def A_update(X, A, R, opts=None, use_gpu=True):
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

    Returns:
      A (ndarray): Nonnegative m by k factor in the decomposition.
    """
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
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

    for i in range(opts["niter"]):
        ATA = A.T @ A
        num = np.zeros_like(A)
        denom = np.zeros_like(R[0])
        for j, r in enumerate(R):
            if scipy.sparse.issparse(X[j]):
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
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
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
        for x in X:
            x._has_canonical_format = True
            
    ATXA = [A.T.dot(x.dot(A)) for x in X]
    ATA = A.T @ A
    for i in range(opts["niter"]):
        for j in range(len(R)):
            R[j] /= np.maximum(ATA.T @ R[j] @ ATA, eps)
            R[j] *= ATXA[j]
        if (i + 1) % 10 == 0:
            R[j] = np.maximum(R[j], eps)

        if opts["hist"] is not None:
            opts["hist"].append(
                np.sqrt(np.sum([fro_norm(x - A @ r @ A.T) ** 2 for x, r in zip(X, R)]))
            )

    return R

def rescal(X, A, R,
           niter=1000, hist=None,
           A_opts={"niter": 1, "hist": None}, R_opts={"niter": 1, "hist": None},
           use_gpu=True,
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
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    dtype = X[0].dtype
    
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")

    A = A.astype(dtype)
    R = [r.astype(dtype) for r in R]
    
    for _ in tqdm(range(niter), disable=rescal_verbose == False):
        A = A_update(X, A, R, A_opts, use_gpu=use_gpu)
        R = R_update(X, A, R, R_opts, use_gpu=use_gpu)

        if hist is not None:
            hist.append(
                np.sqrt(np.sum([fro_norm(x - A @ r @ A.T) ** 2 for x, r in zip(X, R)]))
            )
    Asum = np.sum(A, 0, keepdims=True)+eps
    R = [Asum * r * Asum.T for r in R]
    A = A / Asum

    return (A, R)
