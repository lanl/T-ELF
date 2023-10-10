import numpy as np
import scipy
import scipy.sparse
from TELF.factorization.decompositions.utilities import nnsvd
import pytest


def test_nnsvd_numpy():
    for dtype in [np.float32, np.float64]:
        X = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=dtype)
        for typ in [np.array, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]:
            W, H = nnsvd.nnsvd(typ(X), 2)
            print(W)
            assert dtype == W.dtype == H.dtype


def test_nnsvd_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    for dtype in [np.float32, np.float64]:
        X = cp.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=dtype)
        for typ in [cp.array]:
            W, H = nnsvd.nnsvd(typ(X), 2)
            print(W)
            assert dtype == W.dtype == H.dtype
