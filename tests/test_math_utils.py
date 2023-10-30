import numpy as np
import scipy
import scipy.sparse
from TELF.factorization.decompositions.utilities import math_utils
import time
import pytest


def test_prune_numpy():
    for dtype in [np.float32, np.float64]:
        A0 = np.array([[1, 0, 0, 3], [0, 0, 0, 0], [1, 0, 2, 4]], dtype=dtype)
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            A = typ(A0)
            B, rows, cols = math_utils.prune(A)
            assert np.all(np.equal(cols, np.array([True, False, True, True])))
            assert np.all(np.equal(rows, np.array([True, False, True])))


def test_unprune_numpy():
    for dtype in [np.float32, np.float64]:
        A = np.array([[1, 0, 0, 3], [0, 0, 0, 0], [1, 0, 2, 4]], dtype=dtype)
        B, rows, cols = math_utils.prune(A)
        Arecon = math_utils.unprune(math_utils.unprune(B, rows, 0), cols, 1)
        assert np.allclose(A, Arecon)
        assert A.dtype == Arecon.dtype
        assert type(B) == type(A)


def test_fro_norm_numpy():
    for dtype in [np.float32, np.float64]:
        A0 = np.array([[1, 2], [3, 4]], dtype=dtype)
        norm0 = math_utils.fro_norm(A0)
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            A = typ(A0)
            norm = math_utils.fro_norm(A)
            assert np.dtype(type(norm)) == np.dtype(dtype)


def test_kl_divergence_numpy():
    for dtype in [np.float32, np.float64]:
        X0 = np.array([[1, 2], [3, 4]], dtype=dtype)
        Y = np.array([[2, 3], [4, 5]], dtype=dtype)
        div0 = math_utils.kl_divergence(X0, Y)
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            X = typ(X0)
            div = math_utils.kl_divergence(X, Y)
            assert np.dtype(type(div)) == np.dtype(dtype)


def test_sparse_divide_product_numpy():
    np.random.seed(0)
    m, k, n = 10, 3, 9
    for dtype in [np.float32, np.float64]:
        W0 = np.random.rand(m, k).astype(dtype)
        H0 = np.random.rand(k, n).astype(dtype)
        X0 = W0@H0
        for typ in [scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            X = typ(X0)
            ans = math_utils.sparse_divide_product(X, W0, H0)
            assert type(ans) == type(X)
            if scipy.sparse.issparse(ans):
                assert np.allclose(np.array(ans.todense()),
                                   np.ones((m, n), dtype=dtype))
            else:
                assert np.allclose(ans, np.ones((m, n), dtype=dtype))


def test_prune_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    for dtype in [np.float32, np.float64]:
        A0 = cp.array([[1, 0, 0, 3], [0, 0, 0, 0], [1, 0, 2, 4]], dtype=dtype)
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            A = typ(A0)
            B, rows, cols = math_utils.prune(A, use_gpu=True)
            assert cp.all(cp.equal(cols, cp.array([True, False, True, True])))
            assert cp.all(cp.equal(rows, cp.array([True, False, True])))


def test_unprune_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    for dtype in [np.float32, np.float64]:
        A = cp.array([[1, 0, 0, 3], [0, 0, 0, 0], [1, 0, 2, 4]], dtype=dtype)
        B, rows, cols = math_utils.prune(A, use_gpu=True)
        Arecon = math_utils.unprune(math_utils.unprune(B, rows, 0, use_gpu=True), cols, 1, use_gpu=True)
        assert np.allclose(A, Arecon)
        assert A.dtype == Arecon.dtype
        assert type(B) == type(A)


def test_fro_norm_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    for dtype in [np.float32, np.float64]:
        A0 = cp.array([[1, 2], [3, 4]], dtype=dtype)
        norm0 = math_utils.fro_norm(A0)
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            A = typ(A0)
            norm = math_utils.fro_norm(A, use_gpu=True)
            assert cp.dtype(norm.dtype) == cp.dtype(dtype)


def test_kl_divergence_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    for dtype in [np.float32, np.float64]:
        X0 = np.array([[1, 2], [3, 4]], dtype=dtype)
        Y = np.array([[2, 3], [4, 5]], dtype=dtype)
        div0 = math_utils.kl_divergence(X0, Y)
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            X = typ(X0)
            div = math_utils.kl_divergence(X, Y)
            assert np.dtype(type(div)) == np.dtype(dtype)


def test_sparse_divide_product_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    np.random.seed(0)
    m, k, n = 10, 3, 9
    for dtype in [np.float32, np.float64]:
        W0 = cp.random.rand(m, k).astype(dtype)
        H0 = cp.random.rand(k, n).astype(dtype)
        X0 = W0@H0
        for typ in [cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            X = typ(X0)
            ans = math_utils.sparse_divide_product(X, W0, H0, use_gpu=True)
            assert type(ans) == type(X)
            if cupyx.scipy.sparse.issparse(ans):
                assert cp.allclose(cp.array(ans.todense()),
                                   np.ones((m, n), dtype=dtype))
            else:
                assert cp.allclose(ans, cp.ones((m, n), dtype=dtype))
