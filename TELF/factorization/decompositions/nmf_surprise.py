from .utilities.generic_utils import get_np, get_scipy, update_opts
from .utilities.math_utils import fro_norm
from tqdm import tqdm
import numpy as np


class SNMF():

    def __init__(
            self,
            n_iters=50,
            biased=True,
            reg_pu=.06,
            reg_qi=.06,
            reg_bu=.02,
            reg_bi=.02,
            lr_bu=.005,
            lr_bi=.005,
            verbose=False,
            global_mean=0,
            use_gpu=True,
            calculate_global_mean=True
    ):

        self.n_iters = n_iters
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.calculate_global_mean = calculate_global_mean
        self.global_mean = global_mean

    def fit(self, X, W, H, k):

        # get np, scipy, and data type
        np = get_np(X, use_gpu=self.use_gpu)
        scipy = get_scipy(X, use_gpu=self.use_gpu)
        dtype = X.dtype

        # correct type
        if scipy.sparse.issparse(X):
            if X.getformat() != "csr":
                X = X.tocsr()
                X._has_canonical_format = True

        # get epsilon
        if np.issubdtype(dtype, np.integer):
            eps = np.finfo(float).eps
        elif np.issubdtype(dtype, np.floating):
            eps = np.finfo(dtype).eps
        else:
            raise Exception("Unknown data type!")

        W = np.maximum(W.astype(dtype), eps)
        H = np.maximum(H.astype(dtype), eps)
        
        if self.pruned:
            X, rows_p, cols_p = prune(X, use_gpu=self.use_gpu)
            W = W[rows_p, :]
            H = H[:, cols_p]
        # number of users and items
        n_users = X.shape[0]
        n_items = X.shape[1]

        # bias for users
        bu = np.zeros(n_users, dtype)
        # bias for items
        bi = np.zeros(n_items, dtype)

        # nnz coords and entries
        rows, columns = X.nonzero()
        if scipy.sparse.issparse(X):
            entries = X.data
        else:
            entries = X[rows, columns]

        # use bias
        if not self.biased:
            global_mean = 0
        else:
            # Calculate mean
            if self.calculate_global_mean:
                global_mean = entries.mean()

            # use the pre-determined mean
            else:
                global_mean = self.global_mean

        for current_epoch in tqdm(range(self.n_iters), disable=not self.verbose, total=self.n_iters):

            # (re)initialize nums and denoms to zero
            user_num = np.zeros((n_users, k))
            user_denom = np.zeros((n_users, k))
            item_num = np.zeros((n_items, k))
            item_denom = np.zeros((n_items, k))

            # Compute numerators and denominators for users and items factors
            for idx, r in enumerate(entries):
                u = rows[idx]
                i = columns[idx]

                # compute current estimation and error
                dot = 0
                for f in range(k):
                    dot += H[f, i] * W[u, f]
                est = global_mean + bu[u] + bi[i] + dot
                err = r - est

                # update biases
                if self.biased:
                    bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                    bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])

                # compute numerators and denominators
                for f in range(k):
                    user_num[u, f] += H[f, i] * r
                    user_denom[u, f] += H[f, i] * est
                    item_num[i, f] += W[u, f] * r
                    item_denom[i, f] += W[u, f] * est

            # Update user factors
            for u in range(n_users):

                if scipy.sparse.issparse(X):
                    n_ratings = X[u].nnz
                else:
                    n_ratings = len(X[u].nonzero()[0])

                for f in range(k):
                    user_denom[u, f] += n_ratings * self.reg_pu * W[u, f]
                    W[u, f] *= user_num[u, f] / user_denom[u, f]

            # Update item factors
            for i in range(n_items):

                if scipy.sparse.issparse(X):
                    n_ratings = X[:, i].nnz
                else:
                    n_ratings = len(X[:, i].nonzero()[0])

                for f in range(k):
                    item_denom[i, f] += n_ratings * self.reg_qi * H[f, i]
                    H[f, i] *= item_num[i, f] / item_denom[i, f]

            # preserve non-zero
            if (current_epoch + 1) % 10 == 0:
                H = np.maximum(H.astype(dtype), eps)
                W = np.maximum(W.astype(dtype), eps)

        self.W = W
        self.H = H
        self.bi = bi
        self.bu = bu

        return W, H, bi, bu, global_mean

    def predict(self, u, i):

        if self.biased:
            est = self.global_mean
            est += self.bu[u]
            est += self.bi[i]
            est += np.dot(self.H[:, i], self.W[u])

        else:
            est = np.dot(self.H[:, i], self.W[u])

        return est
