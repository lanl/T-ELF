import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

from TELF.factorization import RESCALk
import sys; sys.path.append(os.path.join("..", "..", "..", "T-ELF","scripts"))
from generate_X import gen_data,gen_data_sparse

def main():
    params = {
        "n_perturbs": 12,
        "n_iters": 110,
        "epsilon": 0.015,
        "n_jobs": -1,
        "n_nodes":2,
        "init": "nnsvd", 
        "use_gpu": True,
        "save_path": os.path.join("..", "..", "results"), 
        "save_output": True,
        "verbose": True,
        "pruned":False,
        "rescal_verbose": False,
        "calculate_error":False,
        "verbose":True,
        "rescal_func":None,
        "rescal_obj_params":{},
        "simple_plot":True,
        "rescal_method": 'rescal_fro_mu',
        "get_plot_data":True,
        "perturb_type":"uniform",
        "perturb_multiprocessing":False,
        "perturb_verbose":False,
        "joblib_backend":"multiprocessing",
    }

    model = RESCALk(**params)
    Xsp = [gen_data_sparse(shape=[500, 500], density=0.01)["X"] for _ in range(8)]
    X = gen_data(R=4, shape=[500, 500, 8], gen='rescal')["X"]


    Ks = range(1, 7, 1)
    name = "RESCALk_HPC"
    note = "This is an example run of RESCALk"
    results = model.fit(X, Ks, name, note)

if __name__ == "__main__":
    main()