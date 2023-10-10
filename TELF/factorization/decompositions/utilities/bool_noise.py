import numpy as np


def add_Bool_noise(X, noisepercent):
    X = X.astype(bool)
    n, m = X.shape
    Xr = X.ravel()
    flipidx = np.random.choice(n * m, int(noisepercent * n * m), replace=False)
    Xr[flipidx] = ~Xr[flipidx]
    X = Xr.reshape(n, m)
    return X.astype(float)


def add_Bool_posneg_noise(X, noisepercent):
    # * positive noise: flip 0s to 1s (additive noise), negative noise: flip 1s to 0s (subtractive noise)
    X = X.astype(bool)
    n, m = X.shape
    X = X.ravel()
    pos_noisepercent = noisepercent[0]
    neg_noisepercent = noisepercent[1]

    for s, p in zip([True, False], [neg_noisepercent, pos_noisepercent]):
        I = np.where(X == s)[0]
        flipidx = np.random.choice(I.size, int(p * I.size), replace=False)
        X[I[flipidx]] = ~X[I[flipidx]]

    X = X.reshape(n, m)
    return X.astype(float)
