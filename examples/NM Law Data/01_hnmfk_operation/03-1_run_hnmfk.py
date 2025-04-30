import os
import sys
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as ss

# limit multi-threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# From https://github.com/lanl/T-ELF, need to install 
from TELF.factorization.HNMFk import HNMFk
from TELF.pre_processing.Beaver import Beaver

class CustomSemanticCallback:
    """
    Example semantic callback that demonstrates how to regenerate 
    a text matrix (sparse) for deeper levels of hierarchical factorization.
    
    Args:
        df (pd.DataFrame): Main dataframe containing documents or text data.
        target_column (str): Column name in df that contains text data.
        vocabulary (list): List of vocabulary terms.
        matrix_type (str): Type of matrix to build ('tfidf', 'bow', etc.).
    """
    def __init__(self,
                 df: pd.DataFrame,
                 target_column="text_column",
                 vocabulary=None,
                 matrix_type="tfidf"):
        self.df = df
        self.target_column = target_column
        self.vocabulary = vocabulary or []
        self.matrix_type = matrix_type

    def __call__(self, original_indices: np.ndarray):
        """
        Given a set of document indices, build and return a new sparse matrix
        for those documents. This function is meant to be invoked at deeper
        levels of hierarchical factorization.
        """
        current_df = self.df.iloc[original_indices].copy()
        
        try:
            num_docs = current_df.shape[0]
            num_terms = len(self.vocabulary)
            csr_matrix = ss.csr_matrix((num_docs, num_terms))
            return csr_matrix.T.tocsr(), {'vocab': self.vocabulary}
        
        except Exception as e:
            print(f"Error creating matrix for subset: {e}")
            return ss.csr_matrix([[1]]), {'stop_reason': "Could not build matrix"}

def main():
    """
    Main script demonstrating how to load data, build a callback,
    set hierarchical NMF parameters, and run the factorization.
    """
    # -------------------------------------------------------------------------
    # ------------------------- DATA PATH DEFINITIONS -------------------------
    vocab_path = "path_to_vocab.txt"
    data_path = "path_to_your_data.csv"  
    matrix_path = "path_to_your_sparse_matrix.npz"
    # -------------------------------------------------------------------------

    df = pd.read_csv(data_path)  
    X = ss.load_npz(matrix_path)
    
    # If your matrix needs transposing for correct shape:
    X = X.T.tocsr()
    # List of possible ranks for NMF or hierarchical factorization
    Ks = np.arange(4, 40, 1)

    nmfk_params = {
        "k_search_method": "bst_pre",
        "sill_thresh": 0.7,
        "H_sill_thresh": 0.05,
        "n_perturbs": 128,
        "n_iters": 2000,
        "epsilon": 0.025,
        "n_jobs": -1,
        "init": "nnsvd",
        "use_gpu": True,
        "save_path": "./HNMFk_decomp_output", 
        "predict_k_method": "WH_sill",
        "predict_k": True,
        "verbose": True,
        "nmf_verbose": False,
        "transpose": False,
        "pruned": True,
        "nmf_method": "nmf_fro_mu",
        "calculate_error": False,
        "use_consensus_stopping": 0,
        "calculate_pac": False,
        "consensus_mat": False,
        "perturb_type": "uniform",
        "perturb_multiprocessing": False,
        "perturb_verbose": False,
        "simple_plot": True
    }
    
    # Load vocabulary if relevant
    with open(vocab_path, "r") as f:
        vocabulary = [w.strip() for w in f]

    custom_callback = CustomSemanticCallback(
        df=df,
        target_column="text_column",  # adjust to whatever your DF's text column is
        vocabulary=vocabulary,
        matrix_type="tfidf"
    )

    hnmfk_params = {
        "n_nodes": 1,
        "nmfk_params": [nmfk_params], 
        "generate_X_callback": custom_callback,
        "cluster_on": "H",
        "depth": 2,
        "sample_thresh": 100,
        "K2": False,
        "Ks_deep_min": 1,
        "Ks_deep_max": 30,
        "Ks_deep_step": 1,
        "experiment_name": "Example_HNMFk"
    }

    model = HNMFk(**hnmfk_params)

    model.fit(X, Ks, from_checkpoint=True, save_checkpoint=True)
    # Save the model
    with open('./Example_HNMFk_model.pkl', 'wb') as output_file:
        pickle.dump(model, output_file)
    print("HNMFk pipeline complete (placeholder).")

if __name__ == "__main__":
    main()
