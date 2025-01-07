import numpy as np
import scipy
from .generic_utils import get_np, get_scipy

def boolean(X, epsilon, use_gpu=False, random_state=None):
    """
    positive noise: flip 0s to 1s (additive noise), negative noise: flip 1s to 0s (subtractive noise)

    Args:
        X (ndarray, sparse matrix): Array of which to find a perturbation.
        epsilon (float): The perturbation amount.
        random_state (int): Random seed

    Returns:
        Y (ndarray): The perturbed matrix.

    """
    np = get_np(X, use_gpu=use_gpu)
    np.random.seed(random_state)

    dtype = X.dtype
    X = X.astype(bool)
    n, m = X.shape
    X = X.ravel()
    pos_noisepercent = epsilon[0]
    neg_noisepercent = epsilon[1]

    for s, p in zip([True, False], [neg_noisepercent, pos_noisepercent]):
        I = np.where(X == s)[0]
        flipidx = np.random.choice(I.size, int(p * I.size), replace=False)
        X[I[flipidx]] = ~X[I[flipidx]]

    X = X.reshape(n, m)
    return X.astype(dtype)


def uniform_product(X, epsilon, use_gpu=False, random_state=None):
    """
    Multiplies each element of X by a uniform random number in (1-epsilon, 1+epsilon).

    Args:
        X (ndarray, sparse matrix): Array of which to find a perturbation.
        epsilon (float): The perturbation amount.
        random_state (int): Random seed

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
        random_state (int): Random seed

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
