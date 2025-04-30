#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tenmat.py creates a matricized tensor.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.\n
[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.\n
[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.\n
[4] Chi, E.C. and Kolda, T.G., 2012. On tensors, sparsity, and nonnegative factorizations. SIAM Journal on Matrix Analysis and Applications, 33(4), pp.1272-1299.

@author: Maksim Ekin Eren
"""
import sparse
import numpy as np
import scipy.sparse

def unfold(X, mode):
    """
    Create a matricized tensor. i.e. Tensor unfolding along a mode.
    Parameters
    ----------
    X : array
        N-dim array to be unfoled.
    mode : int
        Dimension number to unfold on.

    Returns
    -------
    X : np.ndarray
        Matriced version of the sparse tensor in as dense matrix.

    """
    rdims = [mode]
    tmp = [True] * len(X.shape)
    tmp[rdims[0]] = False
    cdims = np.where(tmp)[0]
    order = rdims + list(cdims)
    X_t = X.transpose(order)
    x = np.prod([X.shape[i] for i in rdims])
    y = np.prod([X.shape[i] for i in cdims])
    return X_t.reshape([x, y])

def fold(X, axis, shape):
    """
    Create a tensor from matrix.
    Parameters
    ----------
    X : ndarray/sparse array
        an unfolded array
    axis : int
        Dimension number to fold on.
    shape : target tensor shape
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(axis)
    full_shape.insert(0, mode_dim)
    if scipy.sparse.issparse(X):
        X = sparse.COO.from_scipy_sparse(X.tocoo())
        return sparse.moveaxis(sparse.COO.reshape(X, full_shape), 0, axis)
    else:
        return np.moveaxis(np.reshape(X, full_shape), 0, axis)

def move_axis(X, source=None,target=None):
    """
        Create a matricized tensor.
        Parameters
        ----------
        X : ndarray/sparse array
            A tensor as a Numpy/Sparse Array
        axis : int
            Dimension number to unfold on.
        """
    import sparse
    import numpy as np
    if source is None:
        source = range(len(X.shape))
    if target is None:
        target = range(len(X.shape))
    X_t = sparse.moveaxis(X,source,target)
    return X_t
