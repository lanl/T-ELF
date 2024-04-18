import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

from TELF.factorization.HNMFk import HNMFk
import sys; sys.path.append("../../../scripts/")
from generate_X import gen_data,gen_data_sparse
import numpy as np

def main():

    # generate data
    X = gen_data(R=4, shape=[500, 500])["X"]
    Ks = np.arange(1, 10, 1)
    perts = 2
    iters = 1000
    eps = 0.015
    init = "nnsvd"
    save_path = "HNMFk_results_path"
    name = "example_HNMFk"

    nmfk_params = {
        "n_perturbs":perts,
        "n_iters":iters,
        "epsilon":eps,
        "n_jobs":-1,
        "init":init, 
        "use_gpu":False,
        "save_path":save_path, 
        "predict_k_method":"sill",
        "verbose":False,
        "nmf_verbose":False,
        "transpose":False,
        "sill_thresh":0.8,
        "pruned":True,
        'nmf_method':'nmf_fro_mu',
        "calculate_error":False,
        "use_consensus_stopping":0,
        "calculate_pac":False,
        "consensus_mat":False,
        "perturb_type":"uniform",
        "perturb_multiprocessing":False,
        "perturb_verbose":False,
        "simple_plot":True
    }

    hnmfk_params = {
        # number of nodes to use
        "n_nodes":2,
        # we can specify nmfk parameters for each depth, or use same for all depth
        # below will use the same nmfk parameters for all depths
        # when using for each depth, append to the list 
        # for example, [nmfk_params0, nmfk_params1, nmfk_params2] for depth of 2
        "nmfk_params": [nmfk_params], 
        # where to perform clustering, can be W or H
        # if W, row of X should be samples
        # if H, columns of X should be samples
        "cluster_on":"H",
        # how deep to go in each topic after root node
        # if -1, it goes until samples cannot be seperated further
        "depth":2,
        # stopping criteria for num of samples
        "sample_thresh":5,
        # if K2=True, decomposition is done only for k=2 instead of 
        # finding and predicting the number of stable latent features
        "K2":False,
        # after first nmfk, when selecting Ks search range, minimum k to start
        "Ks_deep_min":1,
        # After first nmfk, when selecting Ks search range, maximum k to try.
        # When None, maximum k will be same as k selected for parent node.
        "Ks_deep_max": 20,
        # after first nmfk, when selecting Ks search range, k step size
        "Ks_deep_step":1,
        # where to save
        "experiment_name":name
    }

    model = HNMFk(**hnmfk_params)
    model.fit(X, Ks, from_checkpoint=True, save_checkpoint=True)
    all_nodes = model.traverse_nodes()
    print(len(all_nodes))

if __name__ == "__main__":
    main()