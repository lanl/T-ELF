import numpy as np
import scipy
import scipy.sparse
from TELF.factorization.decompositions import nmf_fro_mu
from TELF.factorization.decompositions.utilities.math_utils import fro_norm
from TELF.factorization.decompositions.utilities.resample import uniform_product
import pytest


def test_H_update_numpy():
    np.random.seed(0)
    m, k, n = 3, 2, 4
    W0 = np.random.rand(m, k)
    H0 = np.random.rand(k, n)
    X0 = W0@H0
    for dtype in [np.float32, np.float64]:
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]:
            X = typ(X0.astype(dtype))
            H = nmf_fro_mu.H_update(X, W0, uniform_product(H0, 0.1), use_gpu=False)
            assert H.dtype == dtype
            assert np.allclose(H, H0, rtol=1e-3, atol=1e-3)


def test_W_update_numpy():
    np.random.seed(0)
    m, k, n = 3, 2, 4
    W0 = np.random.rand(m, k)
    H0 = np.random.rand(k, n)
    X0 = W0@H0
    for dtype in [np.float32, np.float64]:
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix, scipy.sparse.coo_matrix]:
            X = typ(X0.astype(dtype))
            W = nmf_fro_mu.W_update(X, uniform_product(W0, 0.1), H0, use_gpu=False)
            assert W.dtype == dtype
            assert np.allclose(W, W0, rtol=1e-3, atol=1e-3)


def test_nmf_numpy():
    np.random.seed(0)
    m, k, n = 3, 2, 4
    W0 = np.random.rand(m, k)
    H0 = np.random.rand(k, n)
    X0 = W0@H0
    for dtype in [np.float32, np.float64]:
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            X = typ(X0.astype(dtype))
            W, H, {} = nmf_fro_mu.nmf(X, uniform_product(
                W0, 0.1), uniform_product(H0, 0.1), use_gpu=False)
            assert W.dtype == dtype
            assert H.dtype == dtype
            assert fro_norm(X-W@H)/fro_norm(X) < 1e-5


def test_H_update_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    cp.random.seed(0)
    m, k, n = 3, 2, 4
    W0 = cp.random.rand(m, k)
    H0 = cp.random.rand(k, n)
    X0 = W0@H0
    for dtype in [np.float32, np.float64]:
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            X = typ(X0.astype(dtype))
            H = nmf_fro_mu.H_update(X, W0, uniform_product(H0, 0.1), use_gpu=True)
            assert H.dtype == dtype
            assert cp.allclose(H, H0, rtol=1e-3, atol=1e-3)


def test_W_update_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    cp.random.seed(0)
    m, k, n = 3, 2, 4
    W0 = cp.random.rand(m, k)
    H0 = cp.random.rand(k, n)
    X0 = W0@H0
    for dtype in [np.float32, np.float64]:
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            X = typ(X0.astype(dtype))
            W = nmf_fro_mu.W_update(X, uniform_product(W0, 0.1), H0, use_gpu=True)
            assert W.dtype == dtype
            assert cp.allclose(W, W0, rtol=1e-3, atol=1e-3)


def test_nmf_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    cp.random.seed(0)
    m, k, n = 3, 2, 4
    W0 = cp.random.rand(m, k)
    H0 = cp.random.rand(k, n)
    X0 = W0@H0
    for dtype in [np.float32, np.float64]:
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            X = typ(X0.astype(dtype))
            W, H = nmf_fro_mu.nmf(X, uniform_product(
                W0, 0.1), uniform_product(H0, 0.1), use_gpu=True)
            assert W.dtype == dtype
            assert H.dtype == dtype
            assert fro_norm(X-W@H)/fro_norm(X) < 1e-5
