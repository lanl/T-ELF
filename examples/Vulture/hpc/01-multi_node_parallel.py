"""
Run this example with: sbatch 01-multi_node_parallel_job.sh
"""
from TELF.pre_processing import Vulture
import sys
import pickle


def main(argv):

    # number of nodes and jobs within the node that will be used
    n_nodes = int(argv[0])
    n_jobs = int(argv[1])

    #
    # load data and stopwords
    #
    documents = pickle.load(open("../../../data/documents.p", "rb"))
    file = open("../../../data/stop_words.txt", "r")
    stop_words = file.read().split("\n")
    file.close()

    # clean
    settings = {
        "n_jobs": n_jobs,
        "n_nodes": n_nodes,
        "verbose": 1,
        "min_characters": 1,
        "min_unique_characters": 2,
        "lemmatize": True
    }
    vulture = Vulture(**settings)
    vulture.clean(documents, stop_words, filename="clean_example")


if __name__ == "__main__":
    main(sys.argv[1:])
