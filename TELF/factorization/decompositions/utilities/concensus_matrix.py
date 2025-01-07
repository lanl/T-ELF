import numpy as np
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, cophenet, leaves_list, optimal_leaf_ordering
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from .math_utils import unprune

def compute_consensus_matrix(H_all, pruned=False, pruned_cols=None):

    # * H_all's shape: k x m x n_perturb
    k, m, npert = H_all.shape
    consensus_mat = 0
    # * loop through all perturb and add connectivity matrix to consensus_mat
    for i in range(npert):
        h = H_all[:, :, i]

        if pruned:
            if pruned_cols is None:
                raise Exception("Attempted to calculate consensus matrix on a pruned matrix without passing perturb_cols!")
                
            h = unprune(h, pruned_cols, 1)

        connect_mat = compute_connectivity_mat(h)
        consensus_mat += connect_mat

    consensus_mat /= npert  # * take average

    return consensus_mat


def compute_connectivity_mat(h):
    #* need to add a step to normalize each row of h here
    hsum = np.sum(h,axis=1,keepdims=True)
    h = h/hsum
    # * h will have shape k x m
    m = h.shape[1]
    index = np.argmax(h, axis=0)
    mat1 = np.tile(index, (m, 1))
    mat2 = np.tile(index[:, None], (1, m))
    connect_mat = (mat1 == mat2)*1  # * multiply by 1 to convert to number
    return connect_mat.astype(float)


def reorder_con_mat(C, k, return_index=False, method='HC'):
    # * method option: 'spectral' and 'HC' - Hierachy Clustering
    cophenetic_coeff = np.nan
    m = C.shape[0]
    if method == 'spectral':
        # * C is a consensus matrix
        clust = SpectralClustering(n_clusters=k, affinity='precomputed').fit(C)
    elif method == 'HC':
        # clust = AgglomerativeClustering(n_clusters=k,affinity='precomputed',linkage='average', compute_distances=True).fit(2-C)
        # * compute cophenetic coefficient
        Cdist = []
        for i in range(m-1):
            for j in np.arange(i+1, m):
                Cdist.append(C[i, j])
        Cdist = np.array(Cdist)
        Z = linkage(2-Cdist, method='average')
        Codist = cophenet(Z)
        cophenetic_coeff = np.corrcoef(2-Cdist, Codist)[0, 1]

    # sample_index = np.argsort(clust.labels_)
    sample_index = leaves_list(Z)

    # * reorder the concensus matrix
    C_reordered = C[sample_index, :]
    C_reordered = C_reordered[:, sample_index]

    if return_index:
        return C_reordered, cophenetic_coeff, sample_index, Z
    else:
        return C_reordered, cophenetic_coeff


if __name__ == "__main__":
    # * testing the HC averge linkage
    import matplotlib.pyplot as plt
    k, m = 3, 10
    W = np.random.rand(k, m)
    W[0, :3] += 10
    W[1, 3:7] += 10
    W[2, 7:] += 10
    C = np.abs(np.corrcoef(W.T))
    # plt.figure()
    # C_fig = plt.imshow(C, cmap='hot')
    # plt.colorbar(C_fig)

    I0 = np.arange(m)
    np.random.shuffle(I0)
    print(I0)

    C = C[:, I0]
    C = C[I0, :]

    plt.figure()
    C_fig = plt.imshow(C, cmap='hot')
    plt.colorbar(C_fig)

    # clustering

    C_reordered, cophe_eff = reorder_con_mat(C.astype(float), k)
    # clust = AgglomerativeClustering(n_clusters=k,affinity='precomputed',linkage='average', compute_distances=True).fit(2-C)
    # #* compute cophenetic coefficient
    # Cdist = []
    # for i in range(m-1):
    #   for j in np.arange(i+1,m):
    #     Cdist.append(C[i,j])
    # Cdist = np.array(Cdist)
    # Z = linkage(2-Cdist, method='average')
    # sample_index = leaves_list(Z)
    # C_reordered = C[sample_index,:]
    # C_reordered = C_reordered[:,sample_index]
    # Codist = cophenet(Z)
    # cophenetic_coeff = np.corrcoef(2-Cdist,Codist)[0,1]
    print(cophe_eff)

    # * test the cophenetic
    if 0:
        C = np.block([[np.ones((7, 7)), np.zeros((7, 3))], [np.zeros((3, 7)), np.ones((3, 3))]])

        Cdist = []
        for i in range(m-1):
            for j in np.arange(i+1, m):
                Cdist.append(C[i, j])
        Cdist = np.array(Cdist)
        Z = linkage(2-Cdist, method='average')
        Codist = cophenet(Z)
        np.corrcoef(Cdist, Codist)[0, 1]
        Codist
        Cdist

    plt.figure()
    plt.imshow(C_reordered)
