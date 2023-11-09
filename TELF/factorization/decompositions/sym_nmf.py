'''
File:    symm_nmf.py
Author:  Nick Solovyev
Date:    09/20/2023
Desc:    Newton-like optimization algorithm for Symmetric NMF (SymNMF)
         This code has been adapted from the work of of Da Kuang, Chris Ding, 
         and Haesun Park. The original method was described in "Symmetric 
         Nonnegative Matrix Factorization for Graph Clustering" and was featured in
         The 12th SIAM International Conference on Data Mining (SDM '12), pp. 106--117.
'''
import sys
from joblib import Parallel, delayed
from .utilities.generic_utils import get_np, get_scipy

    
def chol(A, use_gpu=False):
    """
    Compute the Cholesky factorization of a matrix A.
    
    This function mimics the behavior of Matlab's chol function. If A is symmetric and positive definite, 
    the factorization is completed and the function returns R and flag=0. If A is not symmetric positive 
    definite, the function returns a partial factorization up to the pivot position where the factorization 
    failed, along with a flag indicating the index of the failed pivot.
    
    Parameters:
    -----------
    A: np.ndarray 
        A symmetric matrix. Note that this function does not validate A.
    use_gpu: bool
        Flag that determines whether GPU should be used. Default is False.
        
    Returns:
    --------
    R: np.ndarray
        If the factorization is successful then R is an upper triangular matrix of shape (n, n) such that A = R'*R.
        Otherwise, R is a partial upper triangular matrix up to the pivot position where the factorization failed.
    flag: int
        A status indicator where:
          - If flag = 0 then the input matrix is symmetric positive definite and the factorization was successful.
          - If flag is not zero, then the input matrix is not symmetric positive definite and flag is an integer 
            indicating the index of the pivot position where the factorization failed
        
    Example:
    --------
    >>> A = np.array([[1,  1,  1,  1,  1],
                      [1,  2,  3,  4,  5],
                      [1,  3,  6, 10, 15],
                      [1,  4, 10, 20, 35],
                      [1,  5, 15, 35, 69]])
    >>> R, flag = chol(A)
    >>> print(R)
    >>> [[1. 1. 1. 1.]
         [0. 1. 2. 3.]
         [0. 0. 1. 3.]
         [0. 0. 0. 1.]]
    >>> print(flag)
    >>> 5
    """
    np = get_np(A, use_gpu=use_gpu)
    
    n = A.shape[0]
    R = np.zeros((n, n))
    flag = 0
    
    for i in range(n):
        s = A[i, i] - np.sum(R[:i, i]**2)
        
        if s <= 0:
            flag = i + 1
            break
            
        R[i, i] = np.sqrt(s)
        
        if i < n - 1:
            for j in range(i + 1, n):
                R[i, j] = (A[i, j] - np.dot(R[:i, i], R[:i, j])) / R[i, i]
    
    if flag == 0:
        return R, 0
    else:
        return R[:flag-1, :flag-1], flag

    
def sym_nmf_objective(A, W, use_gpu=False):
    """
    Compute the Symmetric non-Negative Matrix Factorization (SymNMF) objective value.
    The symNMF objective is given by:
        || A - W * W' ||_F^2
    Where || . ||_F denotes the Frobenius norm. This objective seeks a non-negative
    factorization of the matrix A into two identical matrices W, such that A is 
    approximated by W * W'. 

    Parameters:
    -----------
    A: np.ndarray
        A symmetric matrix of shape (m, m)
    W: np.ndarray
        The current estimate of the factor matrix of shape (m, k) where k is the rank.
    use_gpu: bool
        Flag that determines whether GPU should be used. Default is False.

    Returns:
    --------
    float: 
        The value of the SymNMF objective function for the given A and W.
    """
    np = get_np(A, W, use_gpu=use_gpu)
    
    left_term = W.T @ W
    return np.linalg.norm(A, 'fro')**2 - 2 * np.trace(W.T @ (A @ W)) + np.trace(left_term @ left_term)


def compute_gradient(A, W):
    """
    Compute the gradient of the objective function for SymNMF. The gradient provides the direction 
    and rate of change of the objective function at the current point (given by W).
    
    Parameters:
    -----------
    A: np.ndarray
        A symmetric matrix of shape (m, m)
    W: np.ndarray
        The current estimate of the factor matrix of shape (m, k) where k is the rank.

    
    Returns:
    --------
    ndarray: 
        Gradient of the SymNMF objective function with respect to W.
    
    Notes:
    """
    return 4 * (W @ (W.T @ W) - A @ W)


def hessian_blkdiag(temp, W, idx, projnorm_idx, use_gpu=False):
    """
    This function calculates the Hessian matrix based on a subset of rows from `A` and `W` matrices.
    The subset of rows is determined by non-zero elements of column `idx` in the `projnorm_idx` matrix.

    Parameters:
    -----------
    temp: np.ndarray
        A symmetric matrix of shape (m, m)
    W: np.ndarray
        The current estimate of the factor matrix of shape (m, k) where k is the rank.
    idx: int
        An index representing the column of interest in both `W` and `projnorm_idx`.
    projnorm_idx: np.ndarray
        A matrix used to find a subset of rows based on its non-zero elements in the specified column `idx`.
    use_gpu: bool
        Flag that determines whether GPU should be used. Default is False.

    Returns:
    --------
    np.ndarray: 
        The modified Hessian matrix of size corresponding to the number of non-zero elements 
        in the column `idx` of `projnorm_idx`.
    """
    np = get_np(temp, W, projnorm_idx, use_gpu=use_gpu)
    
    m, _ = W.shape
    
    # find subset of rows based on non-zero elements in column idx of projnorm_idx
    subset = np.where(projnorm_idx[:, idx] != 0)[0]
    W_idx = W[subset, idx].reshape(-1, 1) 
    eye0 = (W[:, idx].T @ W[:, idx]) * np.eye(m)  # create scaled identity matrix

    # compute the Hessian matrix
    He = 4 * (temp[np.ix_(subset, subset)] + W_idx @ W_idx.T + eye0[np.ix_(subset, subset)])
    return He


def compute_newton_step(T, W, gradW, projnorm_idx, projnorm_idx_prev, R, p, n_jobs=1, use_gpu=False):
    """
    Compute the Newton step for SymNMF. This function iterates over each column of W. For columns 
    where the projected gradient index has changed, the Hessian is recomputed.
    
    Parameters:
    -----------
    T: np.ndarray 
        Temporary matrix formed from the target matrix and W (typically W @ W.T - A).
    W: np.ndarray
        The current approximation of the factor matrix in SymNMF.
    gradW: np.ndarray
        The gradient of the objective with respect to W.
    projnorm_idx: np.ndarray
        The indices of the elements in W where the gradient is projected.
    projnorm_idx_prev: np.ndarray
        The previous state of the projected gradient indices.
    R: np.ndarray
        Cholesky factor of the Hessian matrix.
    p: list 
        A list to record decisions related to whether the Newton step or the gradient direction should be 
        taken for updating W.
    n_jobs: 
        Number of parallel jobs. Default is 1
    use_gpu: bool
        Flag that determines whether GPU should be used. Default is False.
    
    Returns:
    --------
    step: np.ndarray
        The computed Newton step for W.
    """
    np = get_np(T, W, gradW, projnorm_idx, projnorm_idx_prev, R, use_gpu=use_gpu)
    
    m, Ks = W.shape
    step = np.zeros((m, Ks))
    hessian = [None] * Ks

    def compute_newton_step_helper(k):
        nonlocal hessian  # modify hessian in the scope of compute_newton_step()
        step_k = np.zeros(m)
        if np.any(projnorm_idx_prev[:, k] != projnorm_idx[:, k]):
            hessian[k] = hessian_blkdiag(T, W, k, projnorm_idx, use_gpu=use_gpu)
            R[k], p[k] = chol(hessian[k], use_gpu=use_gpu)  # R and p are modified in place

        if p[k] > 0:
            return gradW[:, k]
        else:
            step_temp = np.linalg.solve(R[k].T, gradW[projnorm_idx[:, k], k])
            step_temp = np.linalg.solve(R[k], step_temp)     
            step_part = np.zeros(m)
            step_part[projnorm_idx[:, k]] = step_temp
            
            eps = np.finfo(float).eps
            step_part[(step_part > -eps) & (W[:, k] <= eps)] = 0
      
            if np.sum(gradW[:, k] * step_part) / (np.linalg.norm(gradW[:, k], 2) * np.linalg.norm(step_part, 2)) <= eps: 
                p[k] = 1
                return gradW[:, k]
            else:
                return step_part
            
    if n_jobs == 1:
        for k in range(Ks):
            step[:, k] = compute_newton_step_helper(k)
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(compute_newton_step_helper)(k) for k in range(Ks))
        for k, result in enumerate(results):
            step[:, k] = result
    return step


def armijo_update(A, W, gradW, step, sigma, beta, obj=None, use_gpu=False):
    """
    Perform an Armijo line search to update W for SymNMF.
    
    This function performs an Armijo line search to find a suitable step size for the Newton method update 
    used for SymNMF. The objective value is checked in each iteration, and the step size is decreased by a 
    factor of beta until the Armijo condition is satisfied.

    Parameters:
    -----------
    A: np.ndarray
        A symmetric matrix of shape (m, m)
    W: np.ndarray
        The current estimate of the factor matrix of shape (m, k) where k is the rank.
    gradW: np.ndarray
        The gradient of the objective with respect to W.
    step: np.ndarray
        The Newton step direction.
    sigma: float
        The Armijo condition parameter, typically in (0, 1). A smaller value results in a larger step size.
    beta: float
        The step size decrease factor, typically in (0, 1). Determines the factor by which step size is reduced in each iteration.
    obj: float (optional)
        The current objective value. If not provided, it will be computed.
    use_gpu: bool
        Flag that determines whether GPU should be used. Default is False.

    Returns:
    --------
    Wn: np.ndarray
        The updated approximation of the factor matrix after Armijo line search.
    newobj: 
        The new objective value after the update.
    """
    np = get_np(A, W, gradW, step, use_gpu=use_gpu)
    
    alpha_newton = 1
    Wn = W - alpha_newton * step
    Wn[Wn < 0] = 0

    # comput starting objective value if not provided
    original_obj = sym_nmf_objective(A, W, use_gpu=use_gpu) if obj is None else obj

    # precompute difference between W and Wn
    delta_W = Wn - W
    gradient_product = np.sum(gradW * delta_W)

    # get objective value for Wn
    new_obj = sym_nmf_objective(A, Wn, use_gpu=use_gpu)

    while new_obj - original_obj > sigma * gradient_product:
        alpha_newton *= beta
        Wn = W - alpha_newton * step
        Wn[Wn < 0] = 0
        delta_W = Wn - W
        gradient_product = np.sum(gradW * delta_W)
        new_obj = sym_nmf_objective(A, Wn, use_gpu=use_gpu)
    return Wn, new_obj


def sym_nmf_newt(A, W, n_iters=100, n_jobs=1, use_gpu=True, tol=1e-3, sigma=0.1, beta=0.1, 
                 use_consensus_stopping=True, debug=0):
    """
    Perform Symmetric Non-Negative Matrix Factorization (SymNMF) using a Newton-like algorithm.

    Parameters:
    -----------
    A: np.ndarray
        Input symmetric matrix to be factorized.
    W: np.ndarray
        Initial matrix for factorization.
    n_iters: int (optional)
        Maximum number of iterations for the algorithm. Default is 1000.
    use_gpu: bool (optional): 
        Flag to indicate if GPU should be used for computations. Default is True.
    tol: float (optional):
        Tolerance threshold for gradient norm as a stopping criterion. Default is 1e-4.
    sigma: float (optional)
        Parameter for the Armijo line search. Default is 0.1.
    beta: float (optional)
        Step size reduction factor for the Armijo line search. Default is 0.1.
    use_consensus_stopping: bool (optional)
        If True, the algorithm stops when the gradient norm is below the tolerance times initial
        gradient norm. Default is True.
    debug: int, bool
        Level of verbosity in debug print for sym_nfm_newt(). Debug print is disabled when debug in (0, False).
        Otherwise verbosity increases as integer value of debug increases. Current max value == 2. Default is 0.

    Returns:
    --------
    W: np.ndarray
        The optimized (m,k) SymNMF factor where m is the number of samples in A and k is the pre-determined
        rank
    obj: float
        The final objective value.
    """
    np = get_np(A, W, use_gpu=use_gpu)
    debug = int(debug)
    
    n, m = A.shape
    assert n == m, 'A is not a a symmetric matrix!'
    
    m, k = W.shape
    assert m == n, 'W Initialization does not match A shape!'
    
    projnorm_idx = np.zeros((m, k), dtype=bool)
    R = [None] * k
    p = np.zeros(k)
    obj = sym_nmf_objective(A, W, use_gpu=use_gpu)
    initgrad = np.linalg.norm(compute_gradient(A, W), 'fro')
    if debug:
        print(f'init grad norm: {initgrad}', file=sys.stderr)
    
    for i in range(n_iters):
        gradW = compute_gradient(A, W)
        projnorm_idx_prev = projnorm_idx.copy()
        projnorm_idx = (gradW <= np.finfo(float).eps) | (W > np.finfo(float).eps)
        projnorm = np.linalg.norm(gradW[projnorm_idx], 2)  # 2-norm
        if (projnorm < tol * initgrad) and use_consensus_stopping:
            if debug:
                print(f'early termination after {i} iterations.', end=' ', file=sys.stderr)
            break
        elif debug > 1:
            print(f'iter {i}: grad norm {projnorm}', file=sys.stderr)
            
        if (i + 1) % 100 == 0:
            p = np.ones(k)
        
        temp = W @ W.T - A
        step = compute_newton_step(temp, W, gradW, projnorm_idx, projnorm_idx_prev, R, p, use_gpu=use_gpu)
        W, obj = armijo_update(A, W, gradW, step, sigma, beta, obj=None, use_gpu=use_gpu)

    if debug:
        print(f'final grad norm: {projnorm}', file=sys.stderr)
    return W, obj
