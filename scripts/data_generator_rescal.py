
#r: Manish Bhattarai, Erik Skau
import argparse
import os

import numpy as np
from mpi4py import MPI


def parser():
    r"""
    Reads the input arguments from the user and parses the parameters to the data generator module.
    """
    parser = argparse.ArgumentParser(description='Data generator arguments')
    parser.add_argument('--p_r', type=int, help='Now of row processors')
    parser.add_argument('--p_c', type=int, help='Now of column processors')
    parser.add_argument('--m', type=int, help='Global m')
    parser.add_argument('--n', type=int, help='Global n')
    parser.add_argument('--k', type=int, help='factors')
    parser.add_argument('--fpath', default='../Data/tmp/', type=str, help='data path to store(eg: tmp/)')
    args = parser.parse_args()
    return args


class data_generator():
    r"""
    Generates synthetic data in distributed manner where each MPI process generates a chunk from the data parallelly.
    The W matrix is generated with gaussian distribution whereas the H matrix is random.

    Parameters
    ----------
        args : class
            Class which comprises following attributes
        fpath : str
            Directory path of file to be stored
        p_r : int
            Count of row processor in the cartesian grid
        p_c  : int
            Count of column processor in the cartesian grid
        m : int
            row dimension of the data
        n : int
            Column dimension of the data
        k : int
            Feature count

    """
    def __init__(self, args):

        self.rank = args.rank
        self.pgrid = [args.p_r, args.p_c]
        self.shape = [args.m, args.n]
        self.p_r = args.p_r
        self.p_c = args.p_c
        self.m = args.m
        self.n = args.n
        self.fpath = args.fpath
        self.k = args.k
        # self.factor = k

    def gauss_matrix_generator(self, n, k):
        r"""
        Construct a matrix of dimensions n by k where the ith column is a Gaussian kernel corresponding to approximately N(i*n/k, 0.01*n^2)

        Parameters
        ----------
          n : int
            the ambient space dimension
          k :int
            the latent space diemnsion


        Returns
        ----------
          W : ndarray
             A matrix with Gaussian kernel columns of size n x k.
        """

        offset = n / k / 2 - 0.5
        noverk = n / k
        coeff = -k / (.01 * n ** 2)
        return lambda i, j: np.exp(coeff * (i - (j * noverk + offset)) ** 2)

    def determine_block_index_range_asymm(self):
        '''Determines the start and end indices for the Data block for each rank'''
        chunk_ind = np.unravel_index(self.rank, self.pgrid)
        start_inds = [i * (n // k) + min(i, n % k) for n, k, i in zip(self.shape, self.pgrid, chunk_ind)]
        end_inds = [(i + 1) * (n // k) + min((i + 1), n % k) - 1 for n, k, i in zip(self.shape, self.pgrid, chunk_ind)]
        return start_inds, end_inds

    def determine_block_shape_asymm(self):
        '''Determines the shape for the Data block for each rank'''
        start_inds, end_inds = self.determine_block_index_range_asymm()
        return [(j - i + 1) for (i, j) in zip(start_inds, end_inds)]

    def random_matrix_generator(self, k, n, seed):
        '''Generator for random matric with given seed'''
        np.random.seed(seed)
        return np.random.rand(n,k,k)

    def dist_fromfunction(self, func, shape, pgrid, *args, unravel_index=np.unravel_index, **kwargs):
        """
        produces X_{i,j} = func(i,j) in a distributed manner, so that each processor has an array_split section of X according to the grid.
        """
        grid_index = unravel_index()
        block_shape = [(n // k) + (i < (n % k)) * 1 for n, k, i in zip(shape, pgrid, grid_index)]
        start_index = [i * (n // k) + min(i, n % k) for n, k, i in zip(shape, pgrid, grid_index)]
        return np.fromfunction(lambda *x: func(*[a + b for a, b in zip(x, start_index)]), block_shape, *args, **kwargs)

    def unravel_column(self):
        '''finds the column rank for 2d grid'''

        def wrapper(*args, **kwargs):
            row, col = np.unravel_index(self.rank, self.pgrid)
            return (row, col // self.pgrid[1])

        return wrapper

    def unravel_row(self):  # ,ind, shape):
        '''finds the row rank for 2d grid'''
        row, col = np.unravel_index(self.rank, self.pgrid)
        return (row // self.pgrid[0], col)

    def create_folder_dir(self, fpath):
        '''Create a folder if doesn't exist'''
        try:
            os.mkdir(fpath)
        except:
            pass

    def generate_factors_data(self):
        """Generates the chunk of factors W,H and data X for each MPI process"""
        A_gen1 = self.dist_fromfunction(self.gauss_matrix_generator(self.n, self.k), (self.n, self.k), (self.p_r, 1),
                                       unravel_index=self.unravel_column())
        A_gen2 = self.dist_fromfunction(self.gauss_matrix_generator(self.k, self.n), (self.k, self.n), (self.p_r, 1),
                                       unravel_index=self.unravel_column())
        R_gen = self.random_matrix_generator(self.k,self.m,1) # self.determine_block_shape_asymm()[1],
                                             #self.unravel_row()[1])
        X_gen = np.stack(([A_gen1@R_gen[i]@A_gen1.T for i in range(self.m)]))#np.einsum('ik,kkj,kl->jil', A_gen1, R_gen,A_gen2) #A_gen @ R_gen @ A_gen.T
        print('For rank=', self.rank, ' dimensions of A,R and X are ', A_gen1.shape, R_gen.shape, X_gen.shape)
        return [A_gen1,A_gen2], R_gen, X_gen

    def fit(self):
        '''generates and save factors'''
        A_gen, R_gen, X_gen = self.generate_factors_data()
        self.create_folder_dir(self.fpath)
        self.create_folder_dir(self.fpath + 'A_factors')
        self.create_folder_dir(self.fpath + 'R_factors')
        self.create_folder_dir(self.fpath + 'X')
        np.save(self.fpath + 'A_factors/A_' + str(self.rank), A_gen)
        np.save(self.fpath + 'R_factors/R_' + str(self.rank), R_gen)
        np.save(self.fpath + 'X/X_' + str(self.rank), X_gen)
        print('File successfully created and saved')


if __name__ == '__main__':
    main_comm = MPI.COMM_WORLD
    args = parser()
    args.rank = main_comm.rank
    data_gen = data_generator(args)
    data_gen.fit()

