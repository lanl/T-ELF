"""
Run this example with: sbatch job_distributed-multiprocessing-coauthor.sh
"""
import sys
import pandas as pd
from TELF.pre_processing import Beaver

def main(argv):

    # number of nodes and jobs within the node that will be used
    n_nodes = int(argv[0])
    n_jobs = int(argv[1])

    # load data
    df = pd.read_csv("../../../data/sample.csv")

    # build tensor
    beaver = Beaver()

    # clean
    settings = {
        "dataset":df,
        "target_columns":("author_ids", "year"),
        "split_authors_with":";",
        "save_path":None,
        "verbose":True,
        "n_jobs":n_jobs,
        "n_nodes":n_nodes
    }

    X, author_ids, years = beaver.coauthor_tensor(**settings)
    print(X)

if __name__ == "__main__":
    main(sys.argv[1:])
