"""
Run this example with: sbatch 02-distributed_parallel.p
"""
from Vulture import Vulture
import sys
from glob import glob
import pickle


def main(argv):

    # number of nodes and jobs within the node that will be used
    n_nodes = int(argv[0])
    n_jobs = int(argv[1])

    #
    # stopwords
    #
    file = open("../../../data/stop_words.txt", "r")
    stop_words = file.read().split("\n")
    file.close()

    #
    # files to clean
    #
    files = glob("../../../data/multiple_files/*.p")

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
    vulture.distributed_clean(files, stop_words, filename="clean_example")


if __name__ == "__main__":
    main(sys.argv[1:])
