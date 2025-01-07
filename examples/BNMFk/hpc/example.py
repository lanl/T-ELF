import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import TELF
print(TELF.__version__)
from TELF.factorization import NMFk
import numpy as np
import sys; sys.path.append(os.path.join("..", "..", "..", "scripts"))
from sample_with_mask import sample_matrix
from sklearn.metrics import root_mean_squared_error

def get_rmse_score(Xtrue, Xtilda, coords):
    
    y1 = Xtrue[coords]
    y2 = Xtilda[coords]

    rmse = root_mean_squared_error(y1, y2)
    return rmse

def main():
    X = np.load(os.path.join("..", "..", "..", "data", "dog.npz"))["X"]
    X, Xtrain, MASK, removed_coords= sample_matrix(X, sample_ratio=0.2, random_state=42, stratify=True)

    params = {
        "n_perturbs":12,
        "n_iters":100,
        "epsilon":0.015, # or (0.015, 0.015) which it does automatically
        "n_jobs":-1,
        "init":"nnsvd",
        "n_nodes":2,

        "use_gpu":True,
        "verbose":True,
        "nmf_verbose":False,
        "perturb_verbose":False,
        "perturb_multiprocessing":False,
        "simple_plot":True,

        "save_path":os.path.join("..", "..", "..", "results"), 
        "save_output":True,
        "collect_output":True,

        "transpose":False,
        "calculate_error":True,
        "predict_k":True,
        "predict_k_method":"sill",
        "sill_thresh":0.75,
        "H_sill_thresh":0.1,
        "k_search_method":"bst_pre",

        "use_consensus_stopping":0,
        "calculate_pac":True,
        "consensus_mat":True,

        "nmf_method":"bnmf",
        "perturb_type":"boolean",

        "nmf_obj_params":{
            "MASK":MASK,
            "lower_thresh":1,
            "upper_thresh":None,
            "tol":None,
            "constraint_tol":None,
            "alpha_W":0.0,
            "alpha_H":0.0,
        },

        "factor_thresholding":"otsu_thresh", # "coord_desc_thresh" or "WH_thresh"
        "factor_thresholding_obj_params":{
        },

        "clustering_method":"bool", # "kmeans" or "bool" or "boolean"
        "clustering_obj_params":{
            #"distance":"hamming",
            "max_iters":100
        }
    }
    Ks = range(1,11,1)
    name = "Example_HPC_BNMFk"
    note = "This is an example run of BNMFk"

    model = NMFk(**params)
    results = model.fit(Xtrain, Ks, name, note)
    print("k_predict", results["k_predict"])
    Xtilda = results["W"] @ results["H"]
    print("RMSE =", get_rmse_score(X, Xtilda, MASK==0))

if __name__ == "__main__":
    main()