try:
    import cupy as cp
    import cupyx.scipy.sparse
except Exception:
    cp = None
    cupyx = None

import scipy.sparse

#
# Matrix operations
#
def put_X_gpu(X, gpuid:int):
    with cp.cuda.Device(gpuid):
        if scipy.sparse.issparse(X):
            Y = cupyx.scipy.sparse.csr_matrix(
                (cp.array(X.data), cp.array(X.indices), cp.array(X.indptr)),
                shape=X.shape,
                dtype=X.dtype,
            )
        else:
            Y = cp.array(X)
    return Y

def put_A_gpu(A, gpuid:int):
    with cp.cuda.Device(gpuid):
        A = cp.array(A)
        
    return A

def put_A_cpu(A):
    A = cp.asnumpy(A)
    return A

#
# Dictionary operations
#
def put_other_results_cpu(other_results):
    other_results_cpu = {}
    for key, value in other_results.items():
        other_results_cpu[key] = cp.asnumpy(value)
    del other_results
    return other_results_cpu

# Tensor operations
def put_tensor_X_gpu(X, gpuid:int):
    with cp.cuda.Device(gpuid):
        if scipy.sparse.issparse(X[0]):
            Y = [cupyx.scipy.sparse.csr_matrix((cp.array(X1.data), cp.array(X1.indices), cp.array(X1.indptr)),
                                                   shape=X1.shape, dtype=X1.dtype) for X1 in X]
        else:
            Y = [cp.array(X1) for X1 in X]
    return Y

def put_tensor_R_cpu(R):
    R = [cp.asnumpy(h_) for h_ in R]
    return R
