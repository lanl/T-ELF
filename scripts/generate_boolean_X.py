import numpy as np

def gen_data(R, shape=[10,20], density=0.5):
    assert len(shape) == 2
    r = R
    while r<=R:
        thresh = np.sqrt(1 - (1-density)**(1/float(R)))
        W = np.random.rand(shape[0], R) <= thresh
        H = np.random.rand(R, shape[1]) <= thresh
        X = W@H
        X = X.astype(float)
        r = np.linalg.matrix_rank(X)
    return {"X":X, "W":W, "H":H}
