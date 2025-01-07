"""
Run this example with: sbatch job_distributed-multiprocessing-cooccurance.sh
"""
import sys
import pandas as pd
import numpy as np
from TELF.pre_processing import Beaver
import os 

def main(argv):

    # number of nodes and jobs within the node that will be used
    n_nodes = int(argv[0])
    n_jobs = int(argv[1])

    # load data
    df = pd.read_csv(os.path.join("..", "..", "..", "data", "sample.csv"))
    df.references.replace(np.nan, '', regex=True, inplace=True)

    
    
    # build matrix
    beaver = Beaver()
    settings = {
        "dataset":df,
        "target_column":"clean_abstract",
        "options":{"min_df": 5, "max_df": 0.5},
        "matrix_type":"tfidf",
        "save_path":None}
    X, vocabulary = beaver.documents_words(**settings)
    
    # co-occurance
    settings = {
    "dataset":df,
    "target_column":"clean_abstract",
    "cooccurrence_settings":{
        "n_jobs":n_jobs,
        "n_nodes":n_nodes,
        "window_size": 200, 
        "vocabulary":vocabulary
    },
    "sppmi_settings":{},
    "save_path":None
    }

    CO_OCCURRENCE, SPPMI = beaver.cooccurrence_matrix(**settings)
    print(CO_OCCURRENCE)
    print(SPPMI)

if __name__ == "__main__":
    main(sys.argv[1:])
