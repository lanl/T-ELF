from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples
from .generic_utils import get_np

def silhouettes(W_all, use_gpu=False):
    """
    Computes the cosine distances silhouettes of a clustering of vectors.

    Args:
        W_all (ndarray): Order three tensor of clustered vectors of shape m by k by p, where m is the ambient dimension of the vectors, k is the number of vectors in each group, and p is the number of groups of vectors.

    Returns:
        sils (ndarray): The k by p array of silhouettes where sils[i,j] is the silhouette measure for the vector W_all[:,i,j].
    """
    np = get_np(W_all, use_gpu=use_gpu)
    dtype = W_all[0].dtype
    
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    
    N, k, n_pert = W_all.shape
    W_all = W_all / np.sqrt(np.sum(W_all ** 2, axis=0))
    W_flat = W_all.reshape(N, k * n_pert)
    W_all2 = (W_flat.T @ W_flat).reshape(k, n_pert, k, n_pert)
    distances = np.arccos(np.clip(W_all2, -1.0, 1.0))
    (N, K, n_perts) = W_all.shape
    if K == 1:
        sils = np.ones((K, n_perts))
    else:
        a = np.zeros((K, n_perts))
        b = np.zeros((K, n_perts))
        for k in range(K):
            for n in range(n_perts):
                a[k, n] = 1 / (n_perts - 1) * np.sum(distances[k, n, k, :])
                tmp = np.sum(distances[k, n, :, :], axis=1)
                tmp[k] = np.inf
                b[k, n] = 1 / n_perts * np.min(tmp)
        
        a = np.maximum(a, eps)
        b = np.maximum(b, eps)
        sils = (b - a) / np.maximum(a, b)
    return sils

def silhouettes_with_distance(W_all, distance="hamming", use_gpu=False):
    # compute the distance matrix, and pass it to sklearn silhouettes samples
    np = get_np(W_all, use_gpu=use_gpu)

    N, k, n_pert = W_all.shape
    if k == 1:
        return np.ones((k, n_pert))

    W_flat = W_all.reshape(N, k * n_pert, order="F")
    label = list(range(k)) * n_pert
    # * compute distance matrix
    dist = np.zeros((k * n_pert, k * n_pert))
    if (distance == "FN") or (distance == "FP"):
        for i in np.arange(k * n_pert):
            for j in np.arange(k * n_pert):
                dist[i, j] = np.mean(
                    np.logical_and(W_flat[:, i] == 1, W_flat[:, j] == 0)
                )
        if distance == "FP":
            dist = dist.T
    elif distance == "hamming":
        dist = cdist(W_flat.T, W_flat.T, metric=distance)
    else:
        raise Exception("Unknown distance metric!")
    
    # now pass dist to silhouette
    S = silhouette_samples(dist, labels=label, metric="precomputed")
    return np.reshape(S, [k, n_pert], order="F")
