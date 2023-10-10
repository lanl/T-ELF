from TELF.factorization.decompositions.utilities.generic_utils import get_np
from TELF.factorization.decompositions import rescal_fro_mu
import time
import numpy as np
import scipy
import cupy as cp
import cupyx

n, k, t = 2500, 4, 16
density = 0.01
niter = 10
dtype = np.float32

X_sp = [scipy.sparse.random(n, n, density=density).astype(dtype)
        for _ in range(t)]
X = [np.array(x.todense()) for x in X_sp]
X_gpu = [cp.array(x) for x in X]
X_gpu_sp = [cupyx.scipy.sparse.csr_matrix(x) for x in X_gpu]

outs = []
for x in [X, X_sp, X_gpu, X_gpu_sp]:
    xp = get_np(*x)
    A = xp.random.rand(n, k)
    R = [xp.random.rand(k, k) for _ in range(t)]
    start_time = time.time()
    outs.append(rescal_fro_mu.rescal(x, A, R, {'niter': niter}))
    end_time = time.time()
    print('for x of type ' +
          str(type(x[0])) + ' the time was ' + str(np.round(end_time-start_time, 2)))
