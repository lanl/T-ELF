# %%
from .utilities.generic_utils import get_np, get_scipy, update_opts
from .utilities.math_utils import prune, unprune, fro_norm
import numpy as np


def H_update_ADMM(X, W, H, opts=None, dual=None, use_gpu=True):
    """
    ADMM algorithm for the right factor in a nonnegative optimization with Frobenius norm loss function. ||X-WH||_F^2

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
    return W_update_ADMM(X.T, H.T, W.T, opts=opts, dual=dual, use_gpu=use_gpu).T


def W_update_ADMM(X, W, H, opts=None, dual=None, use_gpu=True):
    """
    ADMM algorithm for the left factor in a nonnegative optimization with Frobenius norm loss function. ||X-WH||_F^2

    Args:
                    X (ndarray, sparse matrix): Nonnegative m by n matrix to decompose.
                    W (ndarray): Nonnegative m by k initialization of left factor of X.
                    H (ndarray): Nonnegative k by n right factor of X.
                    opts (dict), optional: Dictionary or optional arguments.
                            'hist' (list): list to append the objective function to.
                            'niter' (int): number of iterations.
                            'rho' (double): convergence parameter.

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
        W = np.maximum(primal + dual, 0)  # ! change this line for W constraint
        W = np.minimum(primal + dual, 1)
        dual += primal - W

        if opts["hist"] is not None:
            opts["hist"].append(fro_norm(W))
    return W


def H_update_MU(X, W, H, opts=None, use_gpu=True):
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
    return W_update_MU(X.T, H.T, W.T, opts=opts, use_gpu=use_gpu).T


def W_update_MU(X, W, H, opts=None, use_gpu=True):
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
    default_opts = {"niter": 1000, "hist": None, "alpha": 0}
    opts = update_opts(default_opts, opts)

    alpha = opts[
        "alpha"
    ]  #! this is the regularization parameter (lambda in derivation)
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
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
    if scipy.sparse.issparse(X):
        # bug in setting has_canonical_format flag in cupy
        # https://github.com/cupy/cupy/issues/2365
        # issue is closed, but still not fixed.
        X = X.tocsr()
        X._has_canonical_format = True
        XHT = X.dot(H.T)
    else:
        XHT = X @ H.T
    for i in range(opts["niter"]):
        # import pdb; pdb.set_trace() #! stop here for debugging
        W = (
            W
            * (XHT + 3 * alpha * W ** 2)
            / (W @ HHT + 2 * alpha * (W ** 3) + alpha * W + 1e-9)
        )
        if (i + 1) % 10 == 0:
            W = np.maximum(W, eps)

        if opts["hist"] is not None:
            opts["hist"].append(W)
    return W


def nmf_with_ADMM(X, W, H, lowerthres=1, upperthres=None, opts=None, use_gpu=True):
    """
    Multiplicative update algorithm for NMF with Frobenius norm loss function. ||X-WH||_F^2

    Args:
            X (ndarray, sparse matrix): Nonnegative m by n matrix to decompose.
            W (ndarray): Nonnegative m by k initialization of left factor of X.
            H (ndarray): Nonnegative k by n initialization of right factor of X.
            opts (dict), optional: Dictionary or optional arguments.
                    'hist' (list): list to append the objective function to.
                    'niter' (int): number of iterations.
                    'pruned' (bool): indicator if the input matrix needs to be pruned of zeros.
                    'W_opts' (dict): options dictionary for W update
                    'H_opts' (dict): options dictionary for H update

    Returns:
            W (ndarray): Nonnegative m by k left factor of X.
            H (ndarray): Nonnegative k by n right factor of X.
    """

    # *update algorithm
    if opts["algorithm"] == "MU":
        W_update = W_update_MU
        H_update = H_update_MU
        default_opts = {
            "niter": 1000,
            "histX": None,
            "hist": None,
            "pruned": False,
            "algorithm": "MU",
            "tol": None,
            "W_opts": {"niter": 1, "hist": None, "alpha": 0},
            "H_opts": {"niter": 1, "hist": None, "alpha": 0},
        }
    elif opts["algorithm"] == "ADMM":
        W_update = W_update_ADMM
        H_update = H_update_ADMM
        default_opts = {
            "niter": 1000,
            "histX": None,
            "hist": None,
            "pruned": False,
            "algorithm": "ADMM",
            "tol": None,
            "W_opts": {"niter": 1, "hist": None, "rho": 1.0},
            "H_opts": {"niter": 1, "hist": None, "rho": 1.0},
        }

    # print('Using {}'.format(opts['algorithm']))

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
    W = np.maximum(W.astype(dtype), eps)
    H = np.maximum(H.astype(dtype), eps)

    if upperthres is None:
        upperthres = W.shape[1]

    #! X is a boolean matrix dtype = float
    Mask = X == 1

    if opts["pruned"]:
        X, rows, cols = prune(X, use_gpu=use_gpu)
        W = W[rows, :]
        H = H[:, cols]
    if scipy.sparse.issparse(X):
        Xcsc = X.T.T.tocsc()
        Xcsr = X.T.T.tocsr()
    else:
        pass

    YErr = YErrold = YErrchange = np.nan

    for i in range(opts["niter"]):
        if scipy.sparse.issparse(X):
            H = H_update(Xcsc, W, H, opts["H_opts"])
            W = W_update(Xcsr, W, H, opts["W_opts"])
        else:
            H = H_update(X, W, H, opts["H_opts"])
            W = W_update(X, W, H, opts["W_opts"])

        #! update X step
        # * update rule: only update X at ij location if (WH)_{ij}>1 and X_{ij} = 1
        # Xhat = W@H

        if upperthres is not None:
            X = np.where(Mask, np.clip(W @ H, lowerthres, upperthres), 0)
        else:
            X = np.where(Mask, np.maximum(lowerthres, W @ H), 0)
        YErrold = YErr
        YErr = fro_norm(X - W @ H) / fro_norm(X)
        YErrchange = np.abs(YErr - YErrold)
        if opts["hist"] is not None:
            opts["hist"].append(YErr)
        if opts["histX"] is not None:
            # opts['histX'].append(fro_norm(Mask.astype(float) - W@H))
            opts["histX"].append(W @ H)

        if opts["tol"] is not None:
            if np.abs(YErr - YErrold) < opts["tol"]:
                print("converge in {} iterations -- {}".format(i + 1, YErrchange))
                break
    # * Normalize W
    # Wsum  = np.sum(W,0,keepdims=True)
    # H = H * Wsum.T
    # W = W / Wsum
    if opts["pruned"]:
        W = unprune(W, rows, 0)
        H = unprune(H, cols, 1)
    # print('id of X = {}'.format(id(X)))
    # print('id of Mask = {}'.format(id(Mask)))
    return (W, H, X)


def old_nmf(X, W, H, lowerthres=1, upperthres=None, opts=None):
    """
    Multiplicative update algorithm for NMF with Frobenius norm loss function. ||X-WH||_F^2

    Args:
            X (ndarray, sparse matrix): Nonnegative m by n matrix to decompose.
            W (ndarray): Nonnegative m by k initialization of left factor of X.
            H (ndarray): Nonnegative k by n initialization of right factor of X.
            opts (dict), optional: Dictionary or optional arguments.
                    'hist' (list): list to append the objective function to.
                    'niter' (int): number of iterations.
                    'pruned' (bool): indicator if the input matrix needs to be pruned of zeros.
                    'W_opts' (dict): options dictionary for W update
                    'H_opts' (dict): options dictionary for H update

    Returns:
            W (ndarray): Nonnegative m by k left factor of X.
            H (ndarray): Nonnegative k by n right factor of X.
    """

    # *update algorithm
    default_opts = {
        "niter": 1000,
        "histX": None,
        "hist": None,
        "pruned": False,
        "tol": None,
        "W_opts": {"niter": 1, "hist": None, "alpha": 0},
        "H_opts": {"niter": 1, "hist": None, "alpha": 0},
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
    W = np.maximum(W.astype(dtype), eps)
    H = np.maximum(H.astype(dtype), eps)
    k = W.shape[1]
    #! X is a boolean matrix dtype = float
    Mask = X == 1

    if opts["pruned"]:
        X, rows, cols = prune(X, use_gpu=use_gpu)
        W = W[rows, :]
        H = H[:, cols]
    if scipy.sparse.issparse(X):
        Xcsc = X.T.T.tocsc()
        Xcsr = X.T.T.tocsr()
    else:
        pass

    YErr = YErrold = YErrchange = np.nan

    for i in range(opts["niter"]):
        #! set adaptive alpha here

        if scipy.sparse.issparse(X):
            H = H_update_MU(Xcsc, W, H, opts["H_opts"])
            W = W_update_MU(Xcsr, W, H, opts["W_opts"])
        else:
            H = H_update_MU(X, W, H, opts["H_opts"])
            W = W_update_MU(X, W, H, opts["W_opts"])

        #! update X step
        # * update rule: only update X at ij location if (WH)_{ij}>1 and X_{ij} = 1
        # Xhat = W@H

        if upperthres is not None:
            X = np.where(Mask, np.clip(W @ H, lowerthres, upperthres), 0)
        else:
            X = np.where(Mask, np.maximum(lowerthres, W @ H), 0)
        if opts["hist"] is not None:
            opts["hist"].append(fro_norm(X - W @ H))
        if opts["histX"] is not None:
            # opts['histX'].append(fro_norm(Mask.astype(float) - W@H))
            opts["histX"].append(W @ H)

        if opts["tol"] is not None:
            YErrold = YErr
            YErr = fro_norm(X - W @ H) / fro_norm(X)
            YErrchange = np.abs(YErr - YErrold)
            if np.abs(YErr - YErrold) < opts["tol"]:
                print("converge in {} iterations -- {}".format(i + 1, YErrchange))
                break

    if opts["pruned"]:
        W = unprune(W, rows, 0)
        H = unprune(H, cols, 1)

    return (W, H, X)


def nmf(X, W, H, lowerthres=1, upperthres=None, opts=None, Mask=None):
    """
    penalized nmf with adaptive alpha
    """
    default_opts = {
        "niter": 1000,
        "histX": None,
        "hist": None,
        "pruned": True,
        "tol": None,
        "constrainttol": None,
        "W_opts": {"niter": 1, "hist": None, "alpha": 0.0},
        "H_opts": {"niter": 1, "hist": None, "alpha": 0.0},
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
    W = np.maximum(W.astype(dtype), eps)
    H = np.maximum(H.astype(dtype), eps)

    #! X is a boolean matrix dtype = float
    if Mask is None:
        Mask = X == 1

    if opts["pruned"]:
        X, rows, cols = prune(X, use_gpu=use_gpu)
        W = W[rows, :]
        H = H[:, cols]
    if scipy.sparse.issparse(X):
        Xcsc = X.T.T.tocsc()
        Xcsr = X.T.T.tocsr()
    else:
        pass

    YErr = YErrold = YErrchange = np.nan

    for i in range(opts["niter"]):

        #! adaptive alpha
        if opts["constrainttol"] is not None:
            opts["H_opts"]["alpha"] = np.logspace(-4, 10, opts["niter"])[i]
            opts["W_opts"]["alpha"] = np.logspace(-4, 10, opts["niter"])[i]

        if scipy.sparse.issparse(X):
            H = H_update_MU(Xcsc, W, H, opts["H_opts"])
            W = W_update_MU(Xcsr, W, H, opts["W_opts"])
        else:
            H = H_update_MU(X, W, H, opts["H_opts"])
            W = W_update_MU(X, W, H, opts["W_opts"])

        #! update X step
        # * update rule: only update X at ij location if (WH)_{ij}>1 and X_{ij} = 1
        # Xhat = W@H
        if upperthres is not None:
            X = np.where(Mask, np.clip(W @ H, lowerthres, upperthres), 0)
        else:
            X = np.where(Mask, np.maximum(lowerthres, W @ H), 0)

        #!check for constraint tolerance
        if opts["constrainttol"] is not None:
            if (fro_norm(H ** 2 - H) + fro_norm(W ** 2 - W)) < opts["constrainttol"]:
                print("factors converge in {} iterations".format(i + 1))
                break

        if opts["hist"] is not None:
            opts["hist"].append(fro_norm(Mask.astype(float) - W @ H))
        if opts["histX"] is not None:
            # opts['histX'].append(fro_norm(Mask.astype(float) - W@H))
            opts["histX"].append(fro_norm(X - W @ H) / fro_norm(X))

        if opts["tol"] is not None:
            YErrold = YErr
            YErr = fro_norm(X - W @ H) / fro_norm(X)
            YErrchange = np.abs(YErr - YErrold)
            if np.abs(YErr - YErrold) < opts["tol"]:
                print("converge in {} iterations -- {}".format(i + 1, YErrchange))
                break
    # * Normalize W
    # Wsum  = np.sum(W,0,keepdims=True)
    # H = H * Wsum.T
    # W = W / Wsum
    if opts["pruned"]:
        W = unprune(W, rows, 0)
        H = unprune(H, cols, 1)

    return (W, H, X)


def thres_norm(W, H):
    # try to normalize BOTH W,H by dividing to column minimum
    Wsum = np.max(W, axis=0)
    Hsum = np.max(H, axis=1)
    # Wsum = np.max(W)
    # Hsum = np.max(H)
    W = W / Wsum
    H = (H.T / Hsum).T
    D = np.diag(Wsum * Hsum)
    W = W @ np.sqrt(D)
    H = np.sqrt(D) @ H
    return W, H


def old_find_thres_WH(X, Wn, Hn, method="grid_search", output="best", npoint=100):
    # Wn, Hn are normalized using thres_norm function

    if method == "W@H":
        # npoint = 100
        k = Wn.shape[1]
        y = X.reshape(1, -1)
        x = (Wn @ Hn).reshape(1, -1)
        a = np.max(x[y == 0])
        b = np.min(x[y == 1])
        L = np.linspace(a, b, npoint)
        err = np.zeros((1, len(L)))

        # *brute force way
        for i in range(npoint):
            lambdaWH = L[i]
            lambdaW = lambdaWH / np.sqrt(k)
            lambdaH = lambdaWH / np.sqrt(k)
            Xhat = np.dot(Wn > lambdaW, Hn > lambdaH)
            err[0, i] = np.sum(X.astype(bool) ^ Xhat)
        I = np.argmin(err)
        return L[I] / np.sqrt(k), err[0, I]  # optimal threshold for W and H and error

    elif method == "grid_search":
        k = Wn.shape[1]
        Lw = np.linspace(np.min(Wn), np.max(Wn), npoint)
        Lh = np.linspace(np.min(Hn), np.max(Hn), npoint)

        # Lw = np.unique(np.concatenate(Wn.reshape(-1,1)))
        # Lh = np.unique(np.concatenate(Hn.reshape(-1,1)))

        Wpoint = len(Lw)
        Hpoint = len(Lh)
        err = np.zeros((len(Lw), len(Lh)))

        for i in range(Wpoint):
            lambdaW = Lw[i]
            for j in range(Hpoint):
                lambdaH = Lh[j]
                Xhat = np.dot(Wn >= lambdaW, Hn >= lambdaH)
                err[i, j] = np.sum(X.astype(bool) ^ Xhat)

        [iw, ih] = np.unravel_index(err.argmin(), err.shape)
        if output == "best":
            return np.min(err), [Lw[iw], Lh[ih]]
        elif output == "grid":
            return err, Lw, Lh

    elif method == "WH":
        Lthres = np.unique(np.concatenate((Wn.reshape(-1, 1), Hn.reshape(-1, 1))))
        Lthres.sort()
        err = np.zeros((1, len(Lthres)))

        # *brute force way
        for i in range(len(Lthres)):
            t = Lthres[i]
            Xhat = np.dot(Wn > t, Hn > t)
            err[0, i] = np.sum(X.astype(bool) ^ Xhat)
        I = np.argmin(err)
        return err[0, I], Lthres[I]


def find_thres_WH(X, Wn, Hn, output="best", npoint=None):
    # Wn, Hn are normalized using thres_norm function
    np = get_np(X)
    if npoint is None:
        Lw = np.unique(np.concatenate(Wn.reshape(-1, 1)))
        Lh = np.unique(np.concatenate(Hn.reshape(-1, 1)))
    else:
        Lw = np.linspace(np.min(Wn), np.max(Wn), npoint)
        Lh = np.linspace(np.min(Hn), np.max(Hn), npoint)

    Wpoint = len(Lw)
    Hpoint = len(Lh)
    err = np.zeros((len(Lw), len(Lh)))

    for i in range(Wpoint):
        lambdaW = Lw[i]
        for j in range(Hpoint):
            lambdaH = Lh[j]
            Xhat = np.dot(Wn >= lambdaW, Hn >= lambdaH)
            err[i, j] = np.mean(X.astype(bool) ^ Xhat)

    if output == "best":
        [iw, ih] = np.unravel_index(err.argmin(), err.shape)
        return {"error":err[iw, ih], "Lw":Lw[iw], "Lh":Lh[ih]}
    elif output == "grid":
        return {"error":err, "Lw":Lw, "Lh":Lh}


def coord_desc_thresh(X, W, H, wt=None, ht=None, max_iter=100):
    np = get_np(X)
    k = W.shape[1]
    if wt == None:
        wt = np.array([0.5 for _ in range(k)])
    if ht == None:
        ht = np.array([0.5 for _ in range(k)])
    Wrange = [np.unique(W[:, i]) for i in range(k)]
    Hrange = [np.unique(H[i, :]) for i in range(k)]
    err = 1

    for j in range(max_iter):
        old_err = err
        for i in range(k):
            np.random.shuffle(Wrange[i])
            wrange = Wrange[i]
            Wt = W >= wt[None, :]
            Ht = H >= ht[:, None]
            cache = (
                Wt[:, [j for j in range(k) if j != i]]
                @ Ht[[j for j in range(k) if j != i], :]
            )
            this_component = (W[None, :, i] >= wrange[:, None])[:, :, None] @ Ht[[i], :]
            Ys = np.logical_or(cache, this_component)
            errors = np.sum(np.logical_xor(X, Ys), axis=(1, 2)) / X.size
            wt[i] = wrange[np.argmin(errors)]

            # * update H
            np.random.shuffle(Hrange[i])
            hrange = Hrange[i]
            Ht = H >= ht[:, None]
            Wt = W >= wt[None, :]
            cache = (
                Wt[:, [j for j in range(k) if j != i]]
                @ Ht[[j for j in range(k) if j != i], :]
            )
            this_component = Wt[:, [i]] @ (H[None, i, :] >= hrange[:, None])[:, None, :]
            Ys = np.logical_or(cache, this_component)
            errors = np.sum(np.logical_xor(X, Ys), axis=(1, 2)) / X.size
            ht[i] = hrange[np.argmin(errors)]

        # * error after one sweep
        err = np.min(errors)
        # * stopping criteria
        if err == old_err:
            # print('converged in num iter = {}, err = {}'.format(j, np.min(errors)))
            pass
    return err, wt, ht


def coord_desc_thresh_onefactor(X, W, H, max_iter=100):
    # * only threshold W given H. Use transpose to threshold H
    np = get_np(X)
    k = W.shape[1]
    wt = np.array([0.5 for _ in range(k)])
    Wrange = [np.unique(W[:, i]) for i in range(k)]
    err = 1

    for _ in range(max_iter):
        old_err = err
        for i in range(k):
            np.random.shuffle(Wrange[i])
            wrange = Wrange[i]
            Wt = W >= wt[None, :]
            Ht = H.astype(bool)
            cache = (
                Wt[:, [j for j in range(k) if j != i]]
                @ Ht[[j for j in range(k) if j != i], :]
            )
            this_component = (W[None, :, i] >= wrange[:, None])[:, :, None] @ Ht[[i], :]
            Ys = np.logical_or(cache, this_component)
            errors = np.sum(np.logical_xor(X, Ys), axis=(1, 2)) / X.size
            wt[i] = wrange[np.argmin(errors)]

        # * error after one sweep
        err = np.min(errors)
    return err, wt


def roc_W_H(X, W, H):
    Lthres = np.unique(np.concatenate((W.reshape(-1, 1), H.reshape(-1, 1))))
    Lthres.sort()
    fpr = np.zeros((len(Lthres), 1))
    tpr = np.zeros((len(Lthres), 1))
    for j in range(len(Lthres)):
        t = Lthres[j]
        Xhat = np.dot(W > t, H > t)
        Xhat = Xhat.astype(float)
        # * compute TP, TN, FP, FN
        tp = np.sum((X + Xhat) == 2)
        tn = np.sum((X + Xhat) == 0)
        fp = np.sum((X - Xhat) == -1)
        fn = np.sum((X - Xhat) == 1)
        fpr[j] = fp / (fp + tn)
        tpr[j] = tp / (tp + fn)
    return fpr, tpr, Lthres
