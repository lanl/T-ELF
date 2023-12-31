import numpy as np
import scipy
from .generic_utils import get_np, get_scipy


def uniform_product(X, epsilon, use_gpu=False, random_state=None):
    """
    Multiplies each element of X by a uniform random number in (1-epsilon, 1+epsilon).

    Args:
        X (ndarray, sparse matrix): Array of which to find a perturbation.
        epsilon (float): The perturbation amount.

    Returns:
        Y (ndarray): The perturbed matrix.
    """
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    np.random.seed(random_state)
    if scipy.sparse.issparse(X):
        Y = X.copy()
        Y.data = Y.data * (
            1 - epsilon + 2 * epsilon * np.random.rand(*Y.data.shape).astype(X.dtype)
        )
    else:
        Y = X * (1 - epsilon + 2 * epsilon * np.random.rand(*X.shape).astype(X.dtype))
    return Y


def poisson(X, use_gpu=False, random_state=None):
    """
    Resamples each element of a matrix from a Poisson distribution with the mean set by that element. Y_{i,j} = Poisson(X_{i,j})

    Args:
        X (ndarray, sparse matrix): Array of which to find a perturbation.

    Returns:
        Y (ndarray): The perturbed matrix.
    """
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    np.random.seed(random_state)
    if scipy.sparse.issparse(X):
        X = X.copy()
        X.data = np.random.poisson(X.data).astype(X.dtype)
    else:
        X = np.random.poisson(X).astype(X.dtype)
    return X
