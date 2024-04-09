import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

from TELF.factorization import NMFk
import sys; sys.path.append("../../../T-ELF/scripts/")
from generate_X import gen_data
import numpy as np

def main():
    X = gen_data(R=4, shape=[1000, 2000])["X"]
    WEIGHTS = np.random.randint(0,2, X.shape)

    params = {
        "n_perturbs":12,
        "n_iters":100,
        "n_nodes":2,
        "epsilon":0.015,
        "n_jobs":1,
        "init":"nnsvd", 
        "use_gpu":False,
        "save_path":"../../results/", 
        "save_output":True,
        "collect_output":True,
        "predict_k_method":"sill",
        "verbose":True,
        "nmf_verbose":False,
        "transpose":False,
        "sill_thresh":0.8,
        "pruned":True,
        'nmf_method':'wnmf',
        "calculate_error":True,
        "predict_k":True,
        "use_consensus_stopping":0,
        "calculate_pac":True,
        "consensus_mat":True,
        "perturb_type":"uniform",
        "perturb_multiprocessing":False,
        "perturb_verbose":False,
        "simple_plot":True,
        "nmf_obj_params":{
            "WEIGHTS":WEIGHTS,
            "lamb":0.1,
            }
    }

    Ks = range(1,11,1)
    name = "Example_HPC_WNMFk"
    note = "This is an example run of WNMFk"

    model = NMFk(**params)
    results = model.fit(X, range(1,10,1), name, note)

if __name__ == "__main__":
    main()