import numpy as np
import scipy
import scipy.sparse
from TELF.factorization.decompositions.utilities.resample import (uniform_product, poisson)
import pytest


def test_uniform_product_numpy():
    for dtype in [np.float32, np.float64]:
        A0 = np.array([[1, 2], [3, 4]], dtype=dtype)
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            A = typ(A0)
            B = uniform_product(A, .1)
            assert isinstance(B, type(A))
            assert A.dtype == B.dtype


def test_poisson_numpy():
    for dtype in [np.float32, np.float64]:
        A0 = np.array([[1, 2], [3, 4]], dtype=dtype)
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            A = typ(A0)
            B = poisson(A)
            assert isinstance(B, type(A))
            assert A.dtype == B.dtype

# @pytest.mark.skipif(sys.version_info < (3,3),
#                     reason="requires python3.3")


def test_uniform_product_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    for dtype in [np.float32, np.float64]:
        A0 = cp.array([[1, 2], [3, 4]], dtype=dtype)
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            A = typ(A0)
            B = uniform_product(A, .1, use_gpu=True)
            assert isinstance(B, type(A))
            assert A.dtype == B.dtype


def test_poisson_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    for dtype in [np.float32, np.float64]:
        A0 = cp.array([[1, 2], [3, 4]], dtype=dtype)
        for typ in [cp.array, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.csr_matrix]:
            A = typ(A0)
            B = poisson(A, use_gpu=True)
            assert isinstance(B, type(A))
            assert A.dtype == B.dtype
