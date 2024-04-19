import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

from TELF.factorization import NMFk
import sys; sys.path.append("../../../scripts/")
from generate_X import gen_data,gen_data_sparse

def main():

    params = {
        "n_perturbs":12,
        "n_nodes":2,
        "n_iters":110,
        "epsilon":0.015,
        "n_jobs":-1,
        "init":"nnsvd", 
        "use_gpu":True,
        "save_path":"results/", 
        "save_output":True,
        "collect_output":False,
        "predict_k":True,
        "predict_k_method":"sill",
        "verbose":True,
        "nmf_verbose":False,
        "transpose":False,
        "sill_thresh":0.9,
        "pruned":True,
        'nmf_method':'nmf_fro_mu',
        "calculate_error":True,
        "predict_k":True,
        "use_consensus_stopping":0,
        "calculate_pac":False,
        "perturb_type":"uniform"
    }
    Ks = range(1,11,1)
    name = "Example_HPC_NMFk"
    note = "This is an example run of NMFk"

    model = NMFk(**params)
    Xsp = gen_data_sparse(shape=[100, 200], density=0.01)["X"]
    X = gen_data(R=4, shape=[1000, 2000])["X"]
    results = model.fit(X, range(1,10,1), name, note)

if __name__ == "__main__":
    main()