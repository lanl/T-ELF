import numpy as np
import warnings
from scipy.stats import norm
from scipy import sparse
import sparse as ss

def __generate_Gaussian_factor(n,kw,factor=1/8):
    xval = np.linspace(0, 1, n)
    meanval = np.linspace(0, 1, kw+2)
    meanval = meanval[1:-1]
    stdval = factor*(meanval[1]-meanval[0])
    W = np.empty((n,kw))
    for i in range(kw):
        W[:,i] = norm.pdf(xval,loc=meanval[i], scale=stdval) + 0.1
    return W

def gen_trinmf_data(shape=(10,20), kwkh=(3,2), factor_wh=(0.5, 1), factor_S=5, dtype='float32', random_state=None):
    kw,kh = kwkh[0], kwkh[1]
    if random_state is not None:
        np.random.seed(random_state)
    W = __generate_Gaussian_factor(shape[0], kw, factor=factor_wh[0])
    S = np.random.rand(kw, kh)
    H = np.random.rand(kh,shape[1])
    l = int(np.round(kwkh[0]/kwkh[1]))
    S = np.random.rand(kwkh[0],kwkh[1])
    for i in range(kwkh[1]-1):
        S[i*l:(i+1)*l,i] = factor_S+np.random.rand(l)
    S[l*(kwkh[1]-1):,kwkh[1]-1] = factor_S+np.random.rand(kwkh[0]-l*(kwkh[1]-1))
    X = W@S@H
    
    return {"X":X.astype(dtype), "W":W.astype(dtype), "S":S.astype(dtype), "H":H.astype(dtype)}
    
def gen_data(R, shape=(10,20,30),dtype='float32', random_state=42,gen=None):
    """"""
    # sets the seed
    np.random.seed(random_state)

    try:
        if gen=='rescal':
            print('Generating Rescal dataset with shape=',shape,' and rank=',R)
            A = np.random.rand(shape[0],R).astype(dtype)
            R = [np.random.rand(R,R).astype(dtype) for _ in range(shape[-1])]
            X = [A@r@A.T for r in R]
            return {"X": X, "factors":{'A':A,'R':R}}
    except:
        pass

    # creates 3 different random matrices NxR
    factors = [np.random.rand(N,R).astype(dtype) for N in shape]

    # prod and sum of matrices to get the ABC
    contraction = ','.join([chr(i+66)+chr(65) for i in range(len(factors))]) + '->' + ''.join([chr(i+66) for i in range(len(factors))])

    if len(shape) == 3:
        X = np.einsum('Ak,Bk,Ck->ABC', *factors)
    elif len(shape) == 2:
        X = np.einsum('Ak,Bk->AB', *factors)

    if np.sum(np.minimum(X.shape,R)) < 2*R+X.ndim-1:
        warnings.warn("Kruskal's theorem probably won't apply, may not have a unique nCPD.")

    return {"X":X, "factors":factors}

def gen_data_sparse(shape=(10, 20), dtype='float32', density=0.5, gen=None):
    try:
        if gen == 'rescal':
            local_m = shape[-1]
            local_n = shape[0]
            nnz = int(np.prod(shape) * density)

            X = [sparse.coo_matrix(
                (np.random.rand(nnz).astype(dtype),
                 (np.random.randint(0, local_n, size=nnz), np.random.randint(0, local_n, size=nnz))),
                shape=(local_n, local_n)).tocsr() for _ in range(local_m)]
            return {"X": X}
    except:
        pass
    localn_row = shape[0]
    localn_col = shape[1]
    nnz = int(np.prod(shape) * density)
    if len(shape) == 2:
        X = sparse.coo_matrix((np.random.rand(nnz).astype(dtype), (
        np.random.randint(0, localn_row, size=nnz), np.random.randint(0, localn_col, size=nnz))),
                              shape=(localn_row, localn_col)).tocsr()
    elif len(shape) == 3:
        X = ss.COO((
            np.random.randint(0, shape[0], size=nnz), np.random.randint(0, shape[1], size=nnz),
            np.random.randint(0, shape[2], size=nnz)), np.random.rand(nnz).astype(dtype),
            shape=shape)

    return {"X": X}
