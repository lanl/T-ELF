import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

from TELF.factorization import NMFk
import sys; sys.path.append("../../../scripts/")
from generate_X import gen_data

def main():

    params = {
        "n_perturbs":30,
        "n_iters":200,
        "epsilon":0.015,
        "n_jobs":-1,
        "n_nodes":2,
        "init":"nnsvd", 
        "use_gpu":False,
        "save_path":"../../results/", 
        "save_output":True,
        "collect_output":True,
        "predict_k_method":"sill",
        "verbose":True,
        "nmf_verbose":False,
        "transpose":False,
        "sill_thresh":.8,
        "pruned":True,
        'nmf_method':'nmf_fro_mu', # nmf_fro_mu, nmf_recommender
        "calculate_error":True,
        "predict_k":True,
        "use_consensus_stopping":0,
        "calculate_pac":True,
        "consensus_mat":True,
        "perturb_type":"uniform",
        "perturb_multiprocessing":False,
        "perturb_verbose":False,
        "simple_plot":True,
        "k_search_method":"bst_post",
        "H_sill_thresh":0.1
    }
    Ks = range(1,21,1)
    name = "Example_HPC_NMFk"
    note = "This is an example run of NMFk"

    model = NMFk(**params)
    X = gen_data(R=4, shape=[1000, 2000])["X"]
    results = model.fit(X, Ks, name, note)

if __name__ == "__main__":
    main()