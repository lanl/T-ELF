'''
File:   similarity_matrix.py
Author: Ryan Barron, Nick Solovyev
Desc:   Helper functions for computing similarity matrix for SymNMFk
'''
import math
import scipy.sparse
from scipy.linalg import fractional_matrix_power
from .generic_utils import get_np, get_scipy


def gaussian_similarity(a, b, sigma=1, use_gpu=False):
    """
    Similarity metric between a and b, >= 0.

    Parameters
    ----------
    a: np.array
        Node containing a row of the data.
    b: np.array
        Node containing a row of the data.
    sigma: float
        Controls globally the width of the neighborhood of each node
    use_gpu: bool
        Flag that determines whether GPU should be used
        
    Returns
    -------
    float
        gaussian similarity score of a and b
    """
    np = get_np(a, b, use_gpu=use_gpu)
    
    distance_squared = np.sum((a - b) ** 2)
    similarity = np.exp(-distance_squared /  (2 * sigma**2))
    return similarity


def build_degree_matrix(Xq, sigma, use_gpu=False):
    """
    D matrix for normalized cut.
    di = ∑(from j=1 to m) similarity(xi, xj ).

    Parameters
    ----------
    Xq: np.array
        Perturbed input matrix
    sigma: float
        Controls globally the width of the neighborhood of each node
    use_gpu: bool
        Flag that determines whether GPU should be used
          
    Returns
    -------
    D : np.array
        m x m diagonal matrix
    """
    np = get_np(Xq, use_gpu=use_gpu)
    
    diags = []
    m = len(Xq)
    
    for i in range(m):
        di = 0
        for j in range(m):
            di += gaussian_similarity(Xq[i], Xq[j], sigma)
        diags.append(di)

    D = np.diag(diags)
    
    return D


def build_similarity_matrix_helper(Xq, sigma, use_gpu=False):
    """
    Builds similarity matrix.

    Parameters
    ----------
    Xq : np.array
        Perturbed input matrix
    sigma : float
        Controls globally the width of the neighborhood of each node
    use_gpu: bool
        Flag that determines whether GPU should be used
        
    Returns
    -------
    S : np.array
        m x m unscaled simlarity matrix
    """
    np = get_np(Xq, use_gpu=use_gpu)
    
    m = len(Xq)
    S = np.zeros((m,m), float)
  
    for i in range(m):
        for j in range(m):
            S[i][j] = gaussian_similarity(Xq[i], Xq[j], sigma)
    return S


def get_pth_nearest_neighbor(S, index: int, p:int = 7, use_gpu=False):
    """
    Gets the pth index of decreasing similarities.

    Parameters
    ----------
    S : np.array
        m x m unlocalized simlarity matrix
    index : int 
        Which row to find neighbors for
    p : int
        How close the localizing neighbor is to the current node.
    use_gpu: bool
        Flag that determines whether GPU should be used
        
    Returns
    -------
    pth_index : int
        pth neighbor's index closest to current node
    """
    np = get_np(S, use_gpu=use_gpu)
    
    ranked_indices = np.argsort(-S[index, :])
    assert len(ranked_indices) > p
    
    pth_index = ranked_indices[p]
    return ranked_indices[p]


def build_similarity_matrix(Xq, sigma: int, pth_nearest: int = 7, use_gpu=False):
    """
    Gets the pth index of decreasing similarities.

    Parameters
    ----------
    Xq : np.array
        Perturbed input matrix
    sigma : float
        Controls globally the width of the neighborhood of each node
    pth_nearest : int
        How close the localizing neighbor is to the current node.
    use_gpu: bool
        Flag that determines whether GPU should be used
        
    Returns
    -------
    A : int
        Locally scaled similarity matrix
    """
    np = get_np(Xq, use_gpu=use_gpu)
    
    m = len(Xq)    
    S = build_similarity_matrix_helper(Xq, sigma)
    for i in range(m):
        # Part of local scaling -- need to get 7th nearest neighbor
        pi = get_pth_nearest_neighbor(S, index = i, p = pth_nearest)
        sigma_i = S[i][pi]
       
        for j in range(m):
            # Part of local scaling -- need to get 7th nearest neighbor
            pj = get_pth_nearest_neighbor(S, index = j, p = pth_nearest)
            sigma_j = S[j][pj]
            
            # Local scaling
            numerator = -(S[i][j])
            denominator = sigma_i * sigma_j
            
            S[i][j] = math.exp( numerator / denominator)

    D = build_degree_matrix(Xq, sigma)
    
    # normalized cut
    D_cut = fractional_matrix_power(D, -0.5)
    innerA = np.matmul(D_cut, S)
    A = np.matmul(innerA, D_cut) # == (D^(-1/2))S(D^(-1/2))
    return A


def get_connectivity_matrix(IDq, use_gpu=False):
    """
    Binary connectivity matrices are built based on IDq
        
    Input:
    Parameters:
    -----------
    IDq: np.ndarray
        Vector of cluster ids shaped m x 1
    use_gpu: bool
        Flag that determines whether GPU should be used
        
    Returns:
    --------
    Bq: np.ndarray
        Binary connectivity matrix of shape  m x m
    """
    np = get_np(IDq, use_gpu=use_gpu)
    IDq_tile = np.tile(IDq, (len(IDq), 1))
    Bq = IDq_tile.T == IDq
    return Bq.astype("int32")


def dist2(X, C, use_gpu=False):
    """ 
    Calculates squared distance between two sets of points.

    Parameters:
    -----------
    X: np.ndarray, scipy.sparse.csr_matrix
        First matrix of vectors with shape (m, n)
    C: np.ndarray
        Second matrix of vectors with shape (l, n)
    use_gpu: bool
        Flag that determines whether GPU should be used

    Returns:
    --------
    np.ndarray
        Resultant matrix with shape (m, l) where the i,j entry  is the squared distance 
        from the ith row of X to the jth row of C.
    """
    np = get_np(X, C, use_gpu=use_gpu)
    scipy = get_scipy(X, C, use_gpu=use_gpu)
    
    assert X.shape[1] == C.shape[1], 'Data dimension does not match dimension of centres!'

    # If input is sparse
    if scipy.sparse.issparse(X):
        X = X.toarray()
    if scipy.sparse.issparse(C):
        C = C.toarray()
    
    # compute squared norms of each row in X and C
    X_tmp = np.sum(X**2, axis=1, keepdims=True)
    C_tmp = np.sum(C**2, axis=1, keepdims=True).T

    # compute squared distances
    return X_tmp + C_tmp - 2 * np.dot(X, C.T)


def scale_dist3(D, nn=7, use_gpu=False):
    """
    Returns a self-tuned affinity matrix A based on the distance matrix D.
    
    Parameters:
    -----------
    D: np.ndarray
        Distance matrix.
    nn: int
        Number of nearest neighbors. Default is 7.
    use_gpu: bool
        Flag that determines whether GPU should be used
        
    Returns:
    --------
    A: np.ndarray
        The affinity matrix.
    """
    np = get_np(D, use_gpu=use_gpu)
    
    n = D.shape[0]
    if nn > n - 1:
        nn = n - 1

    sorted_indices = np.argsort(D, axis=0)
    ls = np.sqrt(D[sorted_indices[nn], np.arange(D.shape[1])])

    A = np.exp(-D / np.outer(ls, ls))
    np.fill_diagonal(A, 0)
    return A
