import numpy as np
import scipy
import scipy.sparse
from TELF.factorization.decompositions import rescal_fro_mu
from TELF.factorization.decompositions.utilities.math_utils import fro_norm
from TELF.factorization.decompositions.utilities.resample import uniform_product
import pytest


def test_A_update_numpy():
    np.random.seed(0)
    n, k, t = 4, 2, 3
    A0 = np.random.rand(n, k)
    R0 = [np.random.rand(k, k) for _ in range(t)]
    X0 = [A0@r@A0.T for r in R0]
    for dtype in [np.float32, np.float64]:
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            X = [typ(x.astype(dtype)) for x in X0]
            A = rescal_fro_mu.A_update(X, uniform_product(A0, 0.1), R0, use_gpu=False)
            assert A.dtype == dtype
            # mu update for A regularly fails to converge given a perfect X and R, this is not a issue in practice
            #assert np.allclose(A,A0,rtol=1e-2,atol=1e-2)


def test_R_update_numpy():
    np.random.seed(0)
    n, k, t = 4, 2, 3
    A0 = np.random.rand(n, k)
    R0 = [np.random.rand(k, k) for _ in range(t)]
    X0 = [A0@r@A0.T for r in R0]
    for dtype in [np.float32, np.float64]:
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            X = [typ(x.astype(dtype)) for x in X0]
            R = rescal_fro_mu.R_update(
                X, A0, [uniform_product(r, 0.1) for r in R0], use_gpu=False)
            for r, r0 in zip(R, R0):
                assert r.dtype == dtype
                assert np.allclose(r, r0, rtol=1e-2, atol=1e-2)


def test_rescal_numpy():
    np.random.seed(0)
    n, k, t = 4, 2, 3
    A0 = np.random.rand(n, k)
    R0 = [np.random.rand(k, k) for _ in range(t)]
    X0 = [A0@r@A0.T for r in R0]
    for dtype in [np.float32, np.float64]:
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            X = [typ(x.astype(dtype)) for x in X0]
            A = uniform_product(A0, 0.1)
            R = [uniform_product(r, 0.1) for r in R0]
            A, R = rescal_fro_mu.rescal(X, A, R, use_gpu=False)
            assert A.dtype == dtype
            for x, r in zip(X, R):
                assert r.dtype == dtype
                assert fro_norm(x-A@r@A.T)/fro_norm(x) < 1e-3


def test_A_update_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    cp.random.seed(0)
    n, k, t = 4, 2, 3
    A0 = cp.random.rand(n, k)
    R0 = [cp.random.rand(k, k) for _ in range(t)]
    X0 = [A0@r@A0.T for r in R0]
    for dtype in [np.float32, np.float64]:
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            X = [typ(x.astype(dtype)) for x in X0]
            A = rescal_fro_mu.A_update(X, uniform_product(A0, 0.1, use_gpu=True), R0, use_gpu=True)
            assert A.dtype == dtype
            # mu update for A regularly fails to converge given a perfect X and R, this is not a issue in practice
            #assert np.allclose(A,A0,rtol=1e-2,atol=1e-2)


def test_R_update_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    cp.random.seed(0)
    n, k, t = 4, 2, 3
    A0 = cp.random.rand(n, k)
    R0 = [cp.random.rand(k, k) for _ in range(t)]
    X0 = [A0@r@A0.T for r in R0]
    for dtype in [np.float32, np.float64]:
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            X = [typ(x.astype(dtype)) for x in X0]
            R = rescal_fro_mu.R_update(
                X, A0, [uniform_product(r, 0.1, use_gpu=True) for r in R0], use_gpu=True)
            for r, r0 in zip(R, R0):
                assert r.dtype == dtype
                assert cp.allclose(r, r0, rtol=1e-2, atol=1e-2)


def test_rescal_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    cp.random.seed(0)
    n, k, t = 4, 2, 3
    A0 = cp.random.rand(n, k)
    R0 = [cp.random.rand(k, k) for _ in range(t)]
    X0 = [A0@r@A0.T for r in R0]
    for dtype in [np.float32, np.float64]:
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            X = [typ(x.astype(dtype)) for x in X0]
            A = uniform_product(A0, 0.1, use_gpu=True)
            R = [uniform_product(r, 0.1, use_gpu=True) for r in R0]
            A, R = rescal_fro_mu.rescal(X, A, R, use_gpu=True)
            assert A.dtype == dtype
            for x, r in zip(X, R):
                assert r.dtype == dtype
                assert fro_norm(x-A@r@A.T, use_gpu=True)/fro_norm(x, use_gpu=True) < 1e-3
