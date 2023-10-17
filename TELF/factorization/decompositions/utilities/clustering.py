from .generic_utils import get_np, get_scipy
import warnings


def custom_k_means(W_all, centroids=None, max_iters=100):
    """
    Greedy algorithm to approximate a quadratic assignment problem to cluster vectors. Given p groups of k vectors, construct k clusters, each cluster containing a single vector from each of the p groups. This clustering approximation uses cos distances and mean centroids.

    Args:
        W_all (ndarray): Order three tensor of shape m by k by p, where m is the ambient dimension of the vectors, k is the number of vectors in each group, and p is the number of groups of vectors.
        centroids (ndarray), optional: The m by k initialization of the centroids of the clusters. None corresponds to using the first slice, W_all[:,:,0], as the initial centroids. Defaults to None.
        max_iters (int), optional: The maximum number of iterations of the algorithm. If a stable point has been been reached in max_iters iterations, then a warning is given. Defaults to 100.

    Returns:
        centroids (ndarray): The m by k centroids of the clusters.
        W_all (ndarray): Clustered organization of the vectors. W_all[:,i,:] is all p, m dimensional vectors in the ith cluster.
    """
    np = get_np(W_all)
    dtype = W_all.dtype

    (N, K, n_perts) = W_all.shape
    if centroids is None:
        centroids = W_all[:, :, 0]
    centroids = centroids / np.sqrt(np.sum(centroids ** 2, axis=0), dtype=dtype)
    W_all = W_all / np.sqrt(np.sum(W_all ** 2, axis=0), dtype=dtype)
    
    iteration = 0
    while iteration < max_iters:
        
        should_break = True
        for perturbation in range(n_perts):
            dist = centroids.T @ W_all[:, :, perturbation]
            permutation = [i for i in range(K)]
            for k in range(K):
                r, c = np.unravel_index(np.argmax(dist), dist.shape)
                r = int(r)
                c = int(c)
                permutation[r] = c
                dist[r, :] = -1
                dist[:, c] = -1
            W_all[:, :, perturbation] = W_all[:, permutation, perturbation]
            if permutation != [i for i in range(K)]:
                should_break = False
        centroids = np.mean(W_all, axis=2)
        centroids = centroids / np.sqrt(np.sum(centroids ** 2, axis=0), dtype=dtype)
        
        iteration += 1
        if iteration == (max_iters - 1):
            max_iters = 1000
        
        if should_break:
            break

    if iteration == max_iters - 1:
        warnings.warn("Did not converge in " + str(max_iters) + " iterations.")
    return (centroids / np.sum(centroids, axis=0), W_all / np.sum(W_all, axis=0))


def silhouettes(W_all):
    """
    Computes the cosine distances silhouettes of a clustering of vectors.

    Args:
        W_all (ndarray): Order three tensor of clustered vectors of shape m by k by p, where m is the ambient dimension of the vectors, k is the number of vectors in each group, and p is the number of groups of vectors.

    Returns:
        sils (ndarray): The k by p array of silhouettes where sils[i,j] is the silhouette measure for the vector W_all[:,i,j].
    """
    
    np = get_np(W_all)
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
