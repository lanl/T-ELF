from TELF.factorization.decompositions.utilities.generic_utils import get_np
from TELF.factorization.decompositions import nmf_fro_mu
import time
import numpy as np
import scipy
import cupy as cp
import cupyx

m, k, n = 10000, 20, 10000
density = 0.01
niter = 10000
dtype = np.float32

X_sp = scipy.sparse.random(m, n, density=density).astype(dtype)
X = np.array(X_sp.todense())
X_gpu = cp.array(X)
X_gpu_sp = cupyx.scipy.sparse.csr_matrix(X_gpu)

for x in [X, X_sp, X_gpu, X_gpu_sp]:
    xp = get_np(x)
    W = xp.random.rand(m, k)
    H = xp.random.rand(k, n)
    start_time = time.time()
    nmf_fro_mu.nmf(x, W, H, {'niter': niter})
    end_time = time.time()
    print('for x of type ' + str(type(x)) + ' the time was ' +
          str(np.round(end_time-start_time, 2)))
