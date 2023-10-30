from .generic_utils import get_np, get_scipy
import itertools


def get_pac(C, use_gpu=False):
    """
    TODO: Duc to fix Pac so mathematically it is correct

    Removes zero rows and columns from a matrix

    Parameters
    ----------
    C : ndarray, dense matrix
        3D consensus matrix where dimensions are (num. of k, N, N)

    Returns
    -------
    cdf: 1d np.array
        PAC calculation
        
    """
    np = get_np(C, use_gpu=use_gpu)
    
    C_shape = np.shape(C)
    kmax, N = C_shape[0], C_shape[1]
    c_index = np.linspace(start=0, stop=1, num=10)
    cdf = np.zeros((kmax, len(c_index)))

    for k in range(kmax):
        M = np.squeeze(C[k,:,:])
        for cval in range(0, len(c_index)):
            sum_ = 0
            for j in range(0, M.shape[1]):
                i = 1
                while i < j:
                    sum_ += np.double(M[i,j] <= c_index[cval])
                    i += 1

            cdf[k, cval] = sum_ / (N*(N-1)*0.5)
    
    if use_gpu:
        cdf = np.asarray(cdf)

    pac = abs(cdf[:,2]-cdf[:,-2])
    pac = pac / np.max(pac)
    return pac


def prune(X, use_gpu=False):
    """
    Removes zero rows and columns from a matrix

    Parameters
    ----------
    X : ndarray, sparse matrix
        Matrix to prune
    use_gpu: Boolean
        Flag for whether decomposition is being performed on GPU or not. 

    Returns
    -------
    Y : scipy.sparse._csr.csr_matrix
        Pruned matrix
    rows: ndarray 
        Boolean array for all rows; True if non-zero row, else False
    cols: ndarray 
        Boolean array for all cols; True if non-zero col, else False
    """
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)  
    if scipy.sparse.issparse(X):
        sparse_format = X.getformat()
        
        # cupyx does not have getnnz(axis) support as of v11.0.0
        # if using gpu, need to convert to scipy, prune the matrix,
        # and then convert back to cupyx
        if use_gpu:
            ss = get_scipy(X, use_gpu=use_gpu)
            
            try:
                X = X.get()  # cupy.sparse --> scipy.sparse
            except Exception as e:
                pass
        else:
            ss = scipy
        
        if not isinstance(X, ss.sparse._csr.csr_matrix):
            X = X.tocsr()
        
        # get boolean array corresponding to if a given row contains non-zero values
        rows = X.getnnz(1) > 0
        Y = X[rows]  # drop the zero rows
        Y = Y.tocsc()

        # get boolean array corresponding to if a given col contains non-zero values
        cols = Y.getnnz(0) > 0
        Y = Y[:, cols]  # drop the zero cols
        if use_gpu:
            Y = scipy.sparse.csc_matrix(Y)
            cols = np.asarray(cols)
            rows = np.asarray(rows)
            
        return Y.asformat(sparse_format), rows, cols
    
    else:
        zero_inds = []
        for i in range(X.ndim):
            zero_inds.append(
                np.sum(X != 0, axis=tuple([j for j in range(X.ndim) if j != i]))
            )
        Y = X[np.ix_(*[np.nonzero(zi)[0] for zi in zero_inds])]
        rows, cols = [zi != 0 for zi in zero_inds]
        return Y, rows, cols


def unprune(A, indices, axis, use_gpu=False):
    np = get_np(A, use_gpu=use_gpu)
    if axis == 0:
        B = np.zeros((len(indices), A.shape[1]), dtype=A.dtype)
        B[indices, :] = A
    elif axis == 1:
        B = np.zeros((A.shape[0], len(indices)), dtype=A.dtype)
        B[:, indices] = A
    return B


def kl_divergence(X, Y):
    np = get_np(X)
    scipy = get_scipy(X)
    
    dtype = X.dtype
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    
    if scipy.sparse.issparse(X):
        div = np.nansum(X.multiply(np.log(X / (Y + eps) + eps)) - X + Y)
    else:
        div = np.nansum(X * np.log(X / (Y + eps) + eps) - X + Y)
    return div

def norm_X(X):
    import scipy
    import numpy as np
    if scipy.sparse.issparse(X):
        normX = np.linalg.norm(X.data)
    else:
        normX = np.linalg.norm(X)
    return normX

def relative_trinmf_error(X, W, S, H):
    r"""
    input:
        X (sparse array, ndarray): shape $m \times n$ array or sparse array.
        W (ndarray): shape $m \times kw$ left factor of X.
        S (ndarray): shape $kw \times kk$ middle factor of X.
        H (ndarray): shape $kh \times n$ right factor of X.
    output:
        rel_err (double): the relative error $||X-WSH||_F/||X||_F$.
    """
    np = get_np(X, use_gpu=False)
    scipy = get_scipy(X, use_gpu=False)
    dtype = X.dtype

    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    
    return fro_norm(X-W@S@H) / fro_norm(X)

def relative_error_rescal(X,A, R, normX=None):
    r"""
    input:
      X (sparse array, ndarray): shape $m \times n$ array or sparse array.
      W (ndarray): shape $m \times k$ left factor of X.
      H (ndarray): shape $k \times n$ right factor of X.
      normX, optional (double): Optional argument if you already know the norm of X.
    output:
      rel_err (double): the relative error $||X-WH||_F/||X||_F$.
    """
    import numpy as np

    error = np.linalg.norm([relative_error(X[i],A,R[i]@A.T)*norm_X(X[i]) for i in range(len(X))]) \
            /np.linalg.norm([norm_X(X[i]) for i in range(len(X))])
    return error


def relative_error(X, W, H, normX=None):
    r"""
    input:
      X (sparse array, ndarray): shape $m \times n$ array or sparse array.
      W (ndarray): shape $m \times k$ left factor of X.
      H (ndarray): shape $k \times n$ right factor of X.
      normX, optional (double): Optional argument if you already know the norm of X. 
    output:
      rel_err (double): the relative error $||X-WH||_F/||X||_F$.
    """
    np = get_np(X, use_gpu=False)
    scipy = get_scipy(X, use_gpu=False)
    
    dtype = X.dtype
    
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    
    if not scipy.sparse.issparse(X):
        return fro_norm(X-W@H) / fro_norm(X)

    if normX is None:
        if scipy.sparse.issparse(X):
            normX = np.linalg.norm(X.data)
        else:
            normX = np.linalg.norm(X)
    
    normX = normX + eps
    W=np.maximum(W,eps)
    H=np.maximum(H,eps)
    
    WTWHHT = np.trace((W.T@W)@(H@H.T))
    WTXHT = np.trace(W.T@np.array(X.dot(H.T)))
    
    rel_err = np.sqrt(normX**2 + WTWHHT -2*WTXHT) / normX
    return rel_err


def fro_norm(X, use_gpu=False):
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    if scipy.sparse.issparse(X):
        X = X.data
    return np.sqrt(np.nansum(np.square(X)), dtype=X.dtype)


def nz_indices(X, use_gpu=False):
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    dtype = X.dtype
    
    if np.issubdtype(dtype, np.integer):
        eps = np.finfo(float).eps
    elif np.issubdtype(dtype, np.floating):
        eps = np.finfo(dtype).eps
    else:
        raise Exception("Unknown data type!")
    
    if scipy.sparse.issparse(X):
        Y = X.copy()
        Y._has_canonical_format = X.has_canonical_format
        if scipy.sparse.isspmatrix_csr(X):
            try:
                nz_rows = np.repeat(
                    np.arange(X.shape[0], dtype=X.indices.dtype), np.diff(X.indptr)
                )
            except:
                nz_rows = np.repeat(
                    np.arange(X.shape[0], dtype=X.indices.dtype),
                    [int(a) for a in np.diff(X.indptr)],
                )
            nz_cols = X.indices
        elif scipy.sparse.isspmatrix_csc(X):
            try:
                nz_cols = np.repeat(
                    np.arange(X.shape[1], dtype=X.indices.dtype), np.diff(X.indptr)
                )
            except:
                nz_cols = np.repeat(
                    np.arange(X.shape[1], dtype=X.indices.dtype),
                    [int(a) for a in np.diff(X.indptr)],
                )
            nz_rows = X.indices
            
        elif scipy.sparse.isspmatrix_coo(X):
            nz_rows = X.row
            nz_cols = X.col
            
        else:
            raise Exception("Unknown type:", type(X))
            
    else:
        raise Exception("Not known sparse type:", type(X))
        
    return nz_rows, nz_cols


def masked_nmf(X,W,H,mask,itr=1000):
    for i in range(itr):
        X = np.where(mask,X,W@H)
        W = W_update(X,W,H,{'niter':1})
        X = np.where(mask,X,W@H)
        H = H_update(X,W,H,{'niter':1})   
    return W,H     
        

def sparse_divide_product(X, A, B, nz_rows=None, nz_cols=None, use_gpu=False):
    """
    Efficiently computes X/(A@B).
    """
    np = get_np(X, use_gpu=use_gpu)
    scipy = get_scipy(X, use_gpu=use_gpu)
    dtype = X.dtype
    A = A.astype(dtype)
    B = B.astype(dtype)

    if nz_rows is None or nz_cols is None:
        nz_rows, nz_cols = nz_indices(X, use_gpu=use_gpu)

    Y = X.copy()
    Y._has_canonical_format = X.has_canonical_format
    Y.data /= np.sum(A[nz_rows, :].T * B[:, nz_cols], 0)
    return Y

def sparse_dot_product(X, A, B, nz_rows=None, nz_cols=None):
    """
    Efficiently computes (A@B).
    """
    np = get_np(X, use_gpu=False)
    scipy = get_scipy(X, use_gpu=False)
    dtype = X.dtype
    A = A.astype(dtype)
    B = B.astype(dtype)

    if nz_rows is None or nz_cols is None:
        nz_rows, nz_cols = nz_indices(X)

    Y = X.copy()
    Y._has_canonical_format = X.has_canonical_format
    Y.data = np.sum(A[nz_rows, :].T * B[:, nz_cols], 0)
    return Y

def nan_to_num(X, num, copy=False):
    """
    Replaces nan in X with num
    """
    np = get_np(X, use_gpu=False)
    scipy = get_scipy(X, use_gpu=False)

    if copy:
        X = X.copy()

    if scipy.sparse.issparse(X):
        X.data[np.isnan(X.data)] = num
    else:
        X[np.isnan(X)] = num
    return X


def bary_coords(H):
    """
    Computes generalized Barycentric coordinates from an activation matrix.

    Args:
        H (ndarray): Nonnegative k by n activation matrix

    Returns:
        x (ndarray): Generalized Barycentric x coordinate.
        y (ndarray): Generalized Barycentric y coordinate.
    """
    np = get_np(H, use_gpu=False)
    H = H / np.sum(H, 0, keepdims=True)
    theta = np.linspace(0, 2 * np.pi, H.shape[0], endpoint=False) + np.pi / 2

    x = np.cos(theta)
    y = np.sin(theta)

    v = np.vstack([x, y])

    bary = H.T @ v.T
    return np.arctan2(bary[:, 0], bary[:, 1]), np.sqrt(np.sum(bary ** 2, axis=1))
