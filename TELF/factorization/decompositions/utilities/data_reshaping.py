def unfold(X, axis=0):
    """
        Create a matricized tensor.
        Parameters
        ----------
        X : ndarray/sparse array
            A tensor as a Numpy/Sparse Array
        axis : int
            Dimension number to unfold on.
        """
    import numpy as np
    rdims = [axis]
    tmp = [True] * len(X.shape)
    tmp[rdims[0]] = False
    cdims = np.where(tmp)[0]
    order = rdims + list(cdims)
    X_t = X.transpose(order)
    x = np.prod([X.shape[i] for i in rdims])
    y = np.prod([X.shape[i] for i in cdims])
    return X_t.reshape([x, y])


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
    import numpy as np
    from scipy import sparse
    full_shape = list(shape)
    mode_dim = full_shape.pop(axis)
    full_shape.insert(0, mode_dim)
    if sparse.issparse(X):
        import sparse
        X = sparse.COO.from_scipy_sparse(X.tocoo())
        return sparse.moveaxis(sparse.COO.reshape(X, full_shape), 0, axis)
    else:
        return np.moveaxis(np.reshape(X, full_shape), 0, axis)
