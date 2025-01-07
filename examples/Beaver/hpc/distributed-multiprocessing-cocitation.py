"""
Run this example with: sbatch job_distributed-multiprocessing-cocitation.sh
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

    # build tensor
    beaver = Beaver()

    # clean
    settings = {
        "dataset":df,
        "target_columns":("author_ids", "year", "eid", "references"),
        "split_authors_with":";",
        "split_references_with":";",
        "save_path":None,
        "verbose":True,
        "n_jobs":n_jobs,
        "n_nodes":n_nodes
    }

    X, author_ids, years = beaver.cocitation_tensor(**settings)
    print(X)

if __name__ == "__main__":
    main(sys.argv[1:])
