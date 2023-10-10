from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples
import numpy as np


def custom_silhouettes(W_all, distance="hamming"):
    # compute the distance matrix, and pass it to sklearn silhouettes samples
    N, k, n_pert = W_all.shape
    if k == 1:
        return np.ones((k, n_pert))

    W_flat = W_all.reshape(N, k * n_pert, order="F")
    label = list(range(k)) * n_pert
    # * compute distance matrix
    dist = np.zeros((k * n_pert, k * n_pert))
    if (distance == "FN") | (distance == "FP"):
        for i in np.arange(k * n_pert):
            for j in np.arange(k * n_pert):
                dist[i, j] = np.mean(
                    np.logical_and(W_flat[:, i] == 1, W_flat[:, j] == 0)
                )
        if distance == "FP":
            dist = dist.T
    else:
        dist = cdist(W_flat.T, W_flat.T, metric=distance)
    # now pass dist to silhouette
    S = silhouette_samples(dist, labels=label, metric="precomputed")
    return np.reshape(S, [k, n_pert], order="F")
