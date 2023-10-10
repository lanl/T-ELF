import numpy as np
import scipy
import scipy.sparse
from TELF.factorization.decompositions.utilities import clustering
import pytest


def test_custom_k_means_numpy():
    for dtype in [np.float32, np.float64]:
        W = np.random.rand(5, 3).astype(dtype)
        W = W/np.sum(W, axis=0)
        W_all = np.stack((W, W[:, [0, 1, 2]], W[:, [1, 0, 2]], W[:, [
                         1, 2, 0]], W[:, [2, 0, 1]], W[:, [2, 1, 0]]), axis=2)
        W_cent, W_clust = clustering.custom_k_means(W_all)
        assert W_cent.dtype == W_clust.dtype == dtype
        assert np.allclose(W_cent, W)

# cupy median not implemented yet, https://github.com/cupy/cupy/issues/2305
# def test_custom_k_means_cupy():
    # cp = pytest.importorskip("cupy")
    # cupyx = pytest.importorskip("cupyx")
#     for dtype in [np.float32, np.float64]:
#         W = cp.random.rand(5,3).astype(dtype)
#         W = W/np.sum(W,axis=0)
#         W_all = cp.stack((W, W[:,[0,1,2]], W[:,[1,0,2]], W[:,[1,2,0]], W[:,[2,0,1]], W[:,[2,1,0]]),axis=2)
#         W_cent, W_clust = clustering.custom_k_means(W_all)
#         assert W_cent.dtype == W_clust.dtype == dtype
#         print(W_cent)
#         print(W)
#         assert cp.allclose(W_cent,W)


def test_silhouettes_numpy():
    for dtype in [np.float32, np.float64]:
        W_clust = np.arange(60).reshape(4, 3, 5)
        sil_ans = np.array([[0.60103616, 0.67218012, 0.65374111, 0.48149359, 0.00397548],
                            [0.30287937, 0.63102511, 0.65766421,
                                0.48852792, 0.0183531],
                            [0.29249252, 0.62571947, 0.73744142, 0.73927311, 0.67945464]])
        sils = clustering.silhouettes(W_clust)
        assert np.allclose(sils, sil_ans)


def test_silhouettes_cupy():
    cp = pytest.importorskip("cupy")
    cupyx = pytest.importorskip("cupyx")
    for dtype in [np.float32, np.float64]:
        W_clust = cp.arange(60).reshape(4, 3, 5)
        sil_ans = cp.array([[0.60103616, 0.67218012, 0.65374111, 0.48149359, 0.00397548],
                            [0.30287937, 0.63102511, 0.65766421,
                                0.48852792, 0.0183531],
                            [0.29249252, 0.62571947, 0.73744142, 0.73927311, 0.67945464]])
        sils = clustering.silhouettes(W_clust)
        assert cp.allclose(sils, sil_ans)
