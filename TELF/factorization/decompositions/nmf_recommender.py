from .utilities.generic_utils import get_np, get_scipy, update_opts
from .utilities.math_utils import fro_norm
from tqdm import tqdm
import numpy as np

def nmf(X, W, H, 
        niter=100,
        nmf_verbose=False,
        use_gpu=True,
        biased=True,
        reg_pu=.06,
        reg_qi=.06,
        reg_bu=.02,
        reg_bi=.02,
        lr_bu=.005,
        lr_bi=.005,
        global_mean=0,
        calculate_global_mean=True
        ):


    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    dtype = X.dtype

    # correct type
    if scipy.sparse.issparse(X):
        if X.getformat() != "csr":
            X = X.tocsr()
            X._has_canonical_format = True

    # get epsilon
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")

    W = np.maximum(W.astype(dtype), eps)
    H = np.maximum(H.astype(dtype), eps)

    # number of users and items
    n_users = X.shape[0]
    n_items = X.shape[1]

    # calculate the nnz users and ratings
    if scipy.sparse.issparse(X):
        n_ratings_users = X.getnnz(axis=1)
        n_ratings_items = X.getnnz(axis=0)
    else:
        n_ratings_users = np.count_nonzero(X, axis=1)
        n_ratings_items = np.count_nonzero(X, axis=0)

    # bias for users
    bu = np.zeros(n_users, dtype)
    # bias for items
    bi = np.zeros(n_items, dtype)

    # k
    k = W.shape[1]
    assert k == H.shape[0]

    # nnz coords and entries
    rows, columns = X.nonzero()
    if scipy.sparse.issparse(X):
        entries = X.data
    else:
        entries = X[rows, columns]

    # use bias
    if not biased:
        global_mean = 0
    else:
        # Calculate mean
        if calculate_global_mean:
            global_mean = entries.mean()

        # use the pre-determined mean
        else:
            global_mean = global_mean

    for current_epoch in tqdm(range(niter), disable=not nmf_verbose, total=niter):
        
        # (re)initialize nums and denoms to zero
        user_num = np.zeros((n_users, k))
        user_denom = np.zeros((n_users, k))
        item_num = np.zeros((n_items, k))
        item_denom = np.zeros((n_items, k))

        # Compute numerators and denominators for users and items factors
        for idx, r in enumerate(entries):
            u = rows[idx]
            i = columns[idx]

            # compute current estimation and error
            dot = 0
            for f in range(k):
                dot += H[f, i] * W[u, f]
            est = global_mean + bu[u] + bi[i] + dot
            err = r - est

            # update biases
            if biased:
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])

            # compute numerators and denominators
            user_num[u, :] += H[:, i] * r
            user_denom[u, :] += H[:, i] * est
            item_num[i, :] += W[u, :] * r
            item_denom[i, :] += W[u, :] * est

        # Update user factors
        user_denom += n_ratings_users[:, np.newaxis] * reg_pu * W
        W *= user_num / user_denom  
        
        # Update item factors
        item_denom += (n_ratings_items[np.newaxis, :] * reg_qi * H).T
        H *= item_num.T / item_denom.T

        # preserve non-zero
        if (current_epoch + 1) % 10 == 0:
            H = np.maximum(H.astype(dtype), eps)
            W = np.maximum(W.astype(dtype), eps)

    return W, H, {"bi":bi, "bu":bu, "global_mean":global_mean}