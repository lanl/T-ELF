import numpy as np
import scipy
import scipy.sparse
from TELF.pre_processing.Beaver import tenmat
import pytest

def test_unfold_fold_numpy():
    X1 = np.random.rand(3,4,5,6)
    for dtype in [np.float32, np.float64]:
        X = np.array(X1, dtype=dtype)
        unfold_x = [tenmat.unfold(X,axes) for axes in range(len(X.shape))]
        fold_x = [tenmat.fold(unfold_x[axes],axes,X.shape) for axes in range(len(X.shape)) ]
        for f in fold_x:
            assert np.all(f == X)

def test_unfold_fold_sparse():
    import sparse
    X = sparse.random((3,4,5,6),density=.25)
    unfold_x = [tenmat.unfold(X,axes) for axes in range(len(X.shape))]
    fold_x = [tenmat.fold(unfold_x[axes],axes,X.shape) for axes in range(len(X.shape)) ]
    for f in fold_x:
        assert np.all(f == X)