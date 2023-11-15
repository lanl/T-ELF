"""
Run this example with: sbatch 01-multi_node_parallel_job.sh
"""
import os
import sys
import pickle
import pathlib

from TELF.pre_processing import Vulture


DATA_FILE = 'documents.p'
DATA_DIR = os.path.join('..', '..', 'data')
DATA_DIR = pathlib.Path(DATA_DIR).resolve()

RESULTS_FILE = 'clean_documents.p'
RESULTS_DIR = 'results'
RESULTS_DIR = pathlib.Path(RESULTS_DIR).resolve()


def main(n_jobs, n_nodes, verbose):

    #
    # load data
    #
    documents = pickle.load(open(os.path.join(DATA_DIR, DATA_FILE), 'rb'))

    # clean
    settings = {
        "n_jobs": n_jobs,
        "n_nodes": n_nodes,
        "verbose": verbose,
    }
    vulture = Vulture(**settings)
    vulture.clean(documents, save_path=os.path.join(RESULTS_DIR, RESULTS_FILE))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Vulture cleaning example script')
    parser.add_argument('--n_jobs', type=int, help='Number of jobs to run', required=True)
    parser.add_argument('--n_nodes', type=int, help='Number of nodes to use', required=True)
    parser.add_argument('--verbose', type=int, help='Verbose mode: 0 for False, >=1 for True', default=1)
    
    args = parser.parse_args()
    main(args.n_jobs, args.n_nodes, args.verbose)