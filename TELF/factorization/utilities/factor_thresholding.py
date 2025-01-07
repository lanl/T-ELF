from ..decompositions.utilities.generic_utils import get_np
from sklearn.cluster import KMeans


def kmeans_thresh(X, W, H, use_gpu=False):
    """
    Threshold W and H using K-means clustering to minimize error between X and W @ H.
    """
    if use_gpu:
        use_gpu = False
        W = W.get()
        H = H.get()
        
    np = get_np(X, use_gpu=use_gpu)
    k = W.shape[1]
    ht = np.zeros(k)
    wt = np.zeros(k)

    for i in range(k):
        # Threshold H[i, :] using K-means clustering
        kmeans_H = KMeans(n_clusters=2, random_state=0)
        H_i = H[i, :].reshape(-1, 1)
        kmeans_H.fit(H_i)
        centers_H = kmeans_H.cluster_centers_.flatten()
        threshold_H = np.mean(centers_H)
        ht[i] = threshold_H

        # Threshold W[:, i] using K-means clustering
        kmeans_W = KMeans(n_clusters=2, random_state=0)
        W_i = W[:, i].reshape(-1, 1)
        kmeans_W.fit(W_i)
        centers_W = kmeans_W.cluster_centers_.flatten()
        threshold_W = np.mean(centers_W)
        wt[i] = threshold_W

    return {"error": [], "wt": wt[None, :], "ht": ht[:, None]}


def otsu_thresh(X, W, H, use_gpu=False):
    """
    Applies Otsu's method to find optimal thresholds for W and H.

    Parameters:
    - X: Binary matrix (observed data).
    - W: Non-negative matrix factor (n_samples x n_components).
    - H: Non-negative matrix factor (n_components x n_features).
    - use_gpu: Boolean flag to use GPU-accelerated computations.

    Returns:
    - Dictionary containing the error, wt (thresholds for W), and ht (thresholds for H).
    """
    np = get_np(X, use_gpu=use_gpu)

    dtype = X.dtype
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")

    k = W.shape[1]
    wt = np.zeros(k)
    ht = np.zeros(k)

    # Apply Otsu's method to each component
    for i in range(k):
        # Otsu's method for W[:, i]
        wt[i] = _otsu(W[:, i], np, eps)

        # Otsu's method for H[i, :]
        ht[i] = _otsu(H[i, :], np, eps)

    return {"error": [], "wt": wt[None, :], "ht": ht[:, None]}

def _otsu(values, np, eps):
    """
    Computes Otsu's threshold for a 1D array of values.

    Parameters:
    - values: 1D array of values.
    - np_module: Numpy module (could be GPU-accelerated).

    Returns:
    - Optimal threshold computed using Otsu's method.
    """
    
    # Flatten the values to ensure 1D
    values = values.flatten()

    # Compute histogram and bin edges
    hist, bin_edges = np.histogram(values, bins=256)

    # Normalize histogram
    hist = hist.astype(float) / hist.sum()

    # Compute cumulative sums
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * bin_edges[:-1])

    # Total mean
    total_mean = cumulative_mean[-1]

    # Between-class variance
    numerator = (total_mean * cumulative_sum - cumulative_mean) ** 2
    denominator = cumulative_sum * (1 - cumulative_sum)
    # Avoid division by zero
    denominator = np.where(denominator == 0, eps, denominator)
    variance = numerator / denominator

    # Find the threshold that maximizes variance
    idx = np.nanargmax(variance)
    optimal_threshold = bin_edges[idx]

    return optimal_threshold


def find_thres_WH(X, W, H, npoint=None, use_gpu=False):
    np = get_np(X, use_gpu=use_gpu)

    if npoint is None:
        Lw = np.unique(np.concatenate(W.reshape(-1, 1)))
        Lh = np.unique(np.concatenate(H.reshape(-1, 1)))
    else:
        Lw = np.linspace(np.min(W), np.max(W), npoint)
        Lh = np.linspace(np.min(H), np.max(H), npoint)

    Wpoint = len(Lw)
    Hpoint = len(Lh)
    err = np.zeros((len(Lw), len(Lh)))

    for i in range(Wpoint):
        lambdaW = Lw[i]
        for j in range(Hpoint):
            lambdaH = Lh[j]
            Xhat = np.dot(W >= lambdaW, H >= lambdaH)
            err[i, j] = np.mean(X.astype(bool) ^ Xhat)

    [iw, ih] = np.unravel_index(err.argmin(), err.shape)
    return {"error":err[iw, ih], "wt":Lw[iw], "ht":Lh[ih]}


def coord_desc_thresh(X, W, H, wt=None, ht=None, max_iter=100, use_gpu=False):
    """
    """

    np = get_np(X, use_gpu=use_gpu)
    k = W.shape[1]
    if wt == None:
        wt = np.array([0.5 for _ in range(k)])
    if ht == None:
        ht = np.array([0.5 for _ in range(k)])
        
    Wrange = [np.unique(W[:, i]) for i in range(k)]
    Hrange = [np.unique(H[i, :]) for i in range(k)]
    err = 1
    Xsize = X.size

    for _ in range(max_iter):
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
            errors = np.sum(np.logical_xor(X, Ys), axis=(1, 2)) / Xsize
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
            errors = np.sum(np.logical_xor(X, Ys), axis=(1, 2)) / Xsize
            ht[i] = hrange[np.argmin(errors)]

        # * error after one sweep
        err = np.min(errors)

    return {"error":err, "wt":wt[None,:], "ht":ht[:,None]}


def coord_desc_thresh_onefactor(X, W, H, max_iter=100, use_gpu=False):
    """
    only threshold W given H. Use transpose to threshold H
    """
    np = get_np(X, use_gpu=use_gpu)
    k = W.shape[1]
    wt = np.array([0.5 for _ in range(k)])
    Wrange = [np.unique(W[:, i]) for i in range(k)]
    err = 1
    Ht = H.astype(bool)

    for _ in range(max_iter):
        for i in range(k):
            np.random.shuffle(Wrange[i])
            wrange = Wrange[i]
            Wt = W >= wt[None, :]
            
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

    return {"error":err, "wt":wt[:, None]}


def otsu_thresh_onefactor(X, W, H, use_gpu=False):
    """
    Applies Otsu's method to find optimal thresholds for W.

    Parameters:
    - X: Binary matrix (observed data).
    - W: Non-negative matrix factor (n_samples x n_components).
    - H: Non-negative matrix factor (n_components x n_features).
    - use_gpu: Boolean flag to use GPU-accelerated computations.

    Returns:
    - Dictionary containing the error, wt (thresholds for W), and ht (thresholds for H).
    """
    np = get_np(X, use_gpu=use_gpu)

    dtype = X.dtype
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")

    k = W.shape[1]
    wt = np.zeros(k)

    # Apply Otsu's method to each component
    for i in range(k):
        # Otsu's method for W[:, i]
        wt[i] = _otsu(W[:, i], np, eps)

    return {"error": [], "wt": wt[None, :]}

def kmeans_thresh_onefactor(X, W, H, use_gpu=False):
    """
    Threshold H using K-means clustering to minimize error between X and W @ H.
    """
    np = get_np(X, use_gpu=use_gpu)
    k = W.shape[1]
    ht = np.zeros(k)

    for i in range(k):
        # Apply K-means clustering with 2 clusters to H[i, :]
        kmeans = KMeans(n_clusters=2, random_state=0)
        H_i = H[i, :].reshape(-1, 1)
        kmeans.fit(H_i)
        centers = kmeans.cluster_centers_.flatten()
        threshold = np.mean(centers)
        ht[i] = threshold
    
    return {"error": [], "wt": ht[:, None]}
