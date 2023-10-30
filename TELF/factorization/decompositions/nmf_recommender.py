from .utilities.generic_utils import get_np, get_scipy, get_cupyx
from tqdm import tqdm
from collections import Counter

def RNMFk_predict(W, H, global_mean, bu, bi, u, i):
    predict = lambda W, H, global_mean, bu, bi, u, i: global_mean + bu[u] + bi[i] + H[:, i]@W[u]
    return predict(W, H, global_mean, bu, bi, u, i)

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
    cupyx, HAS_CUPY = get_cupyx(use_gpu=use_gpu)
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
    
    # nnz coords and entries
    if use_gpu and HAS_CUPY and scipy.sparse.issparse(X):
        rows, columns, entries = cupyx.scipy.sparse.find(X)
    else:
        rows, columns = X.nonzero()
        if scipy.sparse.issparse(X):
            entries = X.data
        else:
            entries = X[rows, columns]
        
    # calculate the nnz users and ratings
    if scipy.sparse.issparse(X):
        if use_gpu:
            user_counts = Counter(rows.get())
            item_counts = Counter(columns.get())
            n_ratings_users = []
            n_ratings_items = []
            
            for idx in range(X.shape[0]):
                n_ratings_users.append(user_counts[idx])
            for idx in range(X.shape[1]):
                n_ratings_items.append(item_counts[idx])
            
            n_ratings_users = np.array(n_ratings_users)
            n_ratings_items = np.array(n_ratings_items)
        else:
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
        estimations = global_mean + bu[:, np.newaxis] + bi + W@H
        errors = entries - estimations[rows, columns]

        if not use_gpu or (not HAS_CUPY):
            # update biases
            np.add.at(bu, rows, lr_bu * (errors - reg_bu * bu[rows]))
            np.add.at(bi, columns, lr_bi * (errors - reg_bi * bi[columns]))

            # compute numerators and denominators
            np.add.at(user_num, rows, (H[:, columns] * entries).T)
            np.add.at(user_denom, rows, (H[:, columns] * estimations[rows, columns]).T)

            np.add.at(item_num, columns, (W[rows].T * entries).T)
            np.add.at(item_denom, columns, (W[rows].T * estimations[rows, columns]).T)

        elif use_gpu and HAS_CUPY:
            # update biases
            cupyx.scatter_add(bu, rows, lr_bu * (errors - reg_bu * bu[rows]))
            cupyx.scatter_add(bi, columns, lr_bi * (errors - reg_bi * bi[columns]))

            # compute numerators and denominators
            cupyx.scatter_add(user_num, rows, (H[:, columns] * entries).T)
            cupyx.scatter_add(user_denom, rows, (H[:, columns] * estimations[rows, columns]).T)

            cupyx.scatter_add(item_num, columns, (W[rows].T * entries).T)
            cupyx.scatter_add(item_denom, columns, (W[rows].T * estimations[rows, columns]).T)
        
        else:
            raise Exception("Requested GPU but did not find Cupy!")

        # Update user factors
        user_denom += n_ratings_users[:, np.newaxis] * reg_pu * W
        W *= np.divide(user_num, user_denom + eps)

        # Update item factors
        item_denom += (n_ratings_items[np.newaxis, :] * reg_qi * H).T
        H *= np.divide(item_num.T, item_denom.T + eps)

        # preserve non-zero
        if (current_epoch + 1) % 10 == 0:
            H = np.maximum(H.astype(dtype), eps)
            W = np.maximum(W.astype(dtype), eps)

    return W, H, {"bi":bi, "bu":bu, "global_mean":global_mean}