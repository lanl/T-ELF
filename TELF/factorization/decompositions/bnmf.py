from .utilities.generic_utils import get_np, get_scipy, update_opts
from .utilities.math_utils import fro_norm

from tqdm import tqdm
import sys

def H_update(X, W, H, alpha=0.0, opts=None, use_gpu=True):
    """
    Multiplicative update algorithm for the right factor in a nonnegative optimization with Frobenius norm loss function. ||X-WH||_F^2

    Args:
        X (ndarray, sparse matrix): Nonnegative m by n matrix to decompose.
        W (ndarray): Nonnegative m by k left factor of X.
        H (ndarray): Nonnegative k by n initialization of right factor of X.
        opts (dict), optional: Dictionary or optional arguments.
                'hist' (list): list to append the objective function to.
                'niter' (int): number of iterations.

    Returns:
        H (ndarray): Nonnegative k by n right factor of X.
    """
    return W_update(X.T, H.T, W.T, alpha=alpha, opts=opts, use_gpu=use_gpu).T

def W_update(X, W, H, alpha=0.0, opts=None, use_gpu=True):
    """
    Multiplicative update algorithm for the left factor in a nonnegative optimization with Frobenius norm loss function. ||X-WH||_F^2

    Args:
        X (ndarray, sparse matrix): Nonnegative m by n matrix to decompose.
        W (ndarray): Nonnegative m by k initialization of left factor of X.
        H (ndarray): Nonnegative k by n right factor of X.
        opts (dict), optional: Dictionary or optional arguments.
                'hist' (list): list to append the objective function to.
                'niter' (int): number of iterations.

    Returns:
        W (ndarray): Nonnegative m by k left factor of X.
    """
    default_opts = {"niter": 1000, "hist": None}
    opts = update_opts(default_opts, opts)
    np = get_np(X, use_gpu=use_gpu)
    dtype = X.dtype
    
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
        
    W = np.maximum(W.astype(dtype), eps)
    H = H.astype(dtype)
    HHT = H @ H.T

    for i in range(opts["niter"]):
        XHT = X @ H.T	
        W = (W * (XHT + 3 * alpha * W ** 2)
            / (W @ HHT + 2 * alpha * (W ** 3) + alpha * W + 1e-9)
        )
        if (i + 1) % 10 == 0:
            W = np.maximum(W, eps)

        if opts["hist"] is not None:
            opts["hist"].append(W)

    return W


def nmf(X, W, H, 
        MASK = None,
        lower_thresh=1, upper_thresh=None,
        niter = 1000, nmf_verbose=False, use_gpu=True,
        tol=None, constraint_tol=None,
        alpha_W=0.0, alpha_H=0.0,
        W_opts={"niter": 1, "hist": None},
        H_opts={"niter": 1, "hist": None},
    ):
    """
    penalized nmf with adaptive alpha
    """
    scipy = get_scipy(X, use_gpu=use_gpu)
    np = get_np(X, use_gpu=use_gpu)

    if scipy.sparse.issparse(X):
        sys.exit("BNMFk is not designed to work with sparse matrix.")

    dtype = X.dtype
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    
    W = np.maximum(W.astype(dtype), eps)
    H = np.maximum(H.astype(dtype), eps)
    X = X.copy()

    if MASK is None:
        MASK = X.copy()
        MASK[np.where(MASK)] = 1

    positive_positions = X == 1
    nan_positions = MASK == 0
    YErr = YErrold = YErrchange = np.nan

    # Mask, 1 at locations of 1 in X, and 0s everywhere else
    # useNanMask, 1 at places its NaN, and 0s everywhere else

    for i in tqdm(range(niter), disable=nmf_verbose is False):

        #* adaptive alpha
        if constraint_tol is not None:
            alpha_W = np.logspace(-4, 10, niter)[i]
            alpha_H = np.logspace(-4, 10, niter)[i]

        H = H_update(X=X, W=W, H=H, opts=H_opts, alpha=alpha_H)
        W = W_update(X=X, W=W, H=H, opts=W_opts, alpha=alpha_W)

        #* update X step
        # * update rule: only update X at ij location if (WH)_{ij}>1 and X_{ij} = 1
        # Xhat = W@H
        if upper_thresh is not None:
            X = np.where(positive_positions, np.clip(W @ H, lower_thresh, upper_thresh), 0)
        else:
            X = np.where(positive_positions, np.maximum(lower_thresh, W @ H), 0)

        #* Update Nan data
        Xhat = W@H
        X[nan_positions] = Xhat[nan_positions]

        #!check for constraint tolerance
        if constraint_tol is not None:
            if (fro_norm(H ** 2 - H, use_gpu=use_gpu) + fro_norm(W ** 2 - W, use_gpu=use_gpu)) < constraint_tol:
                print("factors converge in {} iterations".format(i + 1))
                break

        if tol is not None:
            YErrold = YErr
            YErr = fro_norm(X - W @ H, use_gpu=use_gpu) / fro_norm(X, use_gpu=use_gpu)
            YErrchange = np.abs(YErr - YErrold)
            if np.abs(YErr - YErrold) < tol:
                print("converge in {} iterations -- {}".format(i + 1, YErrchange))
                break
            
    return (W, H, {})