from .generic_utils import get_np

from scipy.spatial.distance import cdist
import numpy as np


def custom_bool_clustering(W_all, centroids=None, max_iters=100, distance="hamming"):
    # * options for distance: ' false negative', 'false positive', 'distance from cdist
    # ? change this function to use different distance, and use different centroids
    np = get_np(W_all)
    dtype = W_all.dtype
    (N, K, n_perts) = W_all.shape
    if centroids is None:
        centroids = W_all[:, :, 0]

    for iteration in range(max_iters):
        should_break = True
        for perturbation in range(n_perts):
            #! distance step
            dist = _compute_distance(
                centroids, W_all[:, :, perturbation], distance=distance
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
        centroids = _compute_Bool_centroids(W_all, distance=distance)

        if should_break:
            break

    return (centroids, W_all)


def _compute_distance(W1, W2, distance="hamming"):
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
    else:  # use cdist
        dist = cdist(W1.T, W2.T, metric=distance)
    return dist


def _compute_Bool_centroids(W_all, distance="hamming", centroidfunc=None):

    k = W_all.shape[1]
    if centroidfunc is None:
        if distance == "FN":
            centroids = np.logical_and.reduce(W_all, axis=2)
        elif distance == "FP":
            centroids = np.logical_or.reduce(W_all, axis=2)
        else:
            centroids = np.median(W_all, axis=2)
    else:
        centroids = centroidfunc(W_all, axis=2)
    return centroids
