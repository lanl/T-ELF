from .generic_utils import get_np, get_scipy
import warnings
from scipy.spatial.distance import cdist

def custom_bool_clustering(W_all, centroids=None, max_iters=100, distance="hamming", use_gpu=False):
    """
        options for distance: ' false negative', 'false positive', 'distance from cdist
        change this function to use different distance, and use different centroids
    """
    np = get_np(W_all, use_gpu=use_gpu)
    dtype = W_all.dtype
    (N, K, n_perts) = W_all.shape
    if centroids is None:
        centroids = W_all[:, :, 0]

    for iteration in range(max_iters):
        should_break = True
        for perturbation in range(n_perts):
            #! distance step
            dist = _compute_distance(
                centroids, W_all[:, :, perturbation], np=np, distance=distance
            )
            permutation = [i for i in range(K)]
            for _ in range(K):
                r, c = np.unravel_index(np.argmin(dist), dist.shape)
                r = int(r)
                c = int(c)
                permutation[r] = c
                dist[r, :] = 100
                dist[:, c] = 100
            W_all[:, :, perturbation] = W_all[:, permutation, perturbation]
            if permutation != [i for i in range(K)]:
                should_break = False
        #! centroid step
        centroids = _compute_Bool_centroids(W_all, np=np, distance=distance)

        if should_break:
            break

    return (centroids, W_all)


def custom_k_means(W_all, centroids=None, max_iters=100, use_gpu=False):
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
    np = get_np(W_all, use_gpu=use_gpu)
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


def _compute_distance(W1, W2, np, distance="hamming"):
    k = W1.shape[1]
    dist = np.empty((k, k))  # store the distance
    if distance == "FN":
        for i in range(k):
            for j in range(k):
                dist[i, j] = np.mean(np.logical_and(W1[:, i] == 1, W2[:, j] == 0))
    elif distance == "FP":
        for i in range(k):
            for j in range(k):
                dist[i, j] = np.mean(np.logical_and(W1[:, i] == 0, W2[:, j] == 1))
    elif distance == "hamming":
        dist = cdist(W1.T, W2.T, metric=distance)
    else:
        raise Exception("Unknown clustering distance!")

    return dist

def _compute_Bool_centroids(W_all, np, distance="hamming", centroidfunc=None):

    k = W_all.shape[1]
    if centroidfunc is None:
        if distance == "FN":
            centroids = np.logical_and.reduce(W_all, axis=2)
        elif distance == "FP":
            centroids = np.logical_or.reduce(W_all, axis=2)
        elif distance == "hamming":
            centroids = np.median(W_all, axis=2)
        else:
            raise Exception("Unknown clustering distance!")
    else:
        centroids = centroidfunc(W_all, axis=2)
    return centroids
