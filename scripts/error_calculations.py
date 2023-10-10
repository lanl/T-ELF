from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import numpy as np

def permute_compute_err(W1,W2):
    cost = pairwise_distances(W1.T,W2.T)
    row_ind, col_ind = linear_sum_assignment(cost)
    W2 = W2[:,col_ind]
    error = np.linalg.norm(W1-W2)/np.linalg.norm(W1)
    return error, W2, col_ind, row_ind

def err(X, Xhat):
    error = np.linalg.norm(X-Xhat)/np.linalg.norm(X)
    return error

def normalize_W(W):
    Wsum = np.sum(W,axis=0,keepdims=True)
    W = W/Wsum
    return W

def columnwise_corr_coff(W1, W2):
    k = W1.shape[1]
    W1norm = normalize_W(W1)
    Werr, W_final_permuted, col_ind, row_ind = permute_compute_err(W1norm, W2)
    corrs = []
    for ii in range(k):
        corrs.append(np.corrcoef(W1norm[:,ii], W_final_permuted[:,ii], rowvar=False)[:,0][1])
        
    return corrs