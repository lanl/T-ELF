- **Single-node parallel document pre-processing:** Clean documents in parallel on a single node, or a computer. This example also includes looking at top n words or n-grams.
    - Notebook: [00-single_node_parallel.ipynb](00-single_node_parallel.ipynb)

- **Multi-node parallel document pre-processing:** Each node loads the entire corpus, and works on certain segment of documents in parallel within each node.
    - Code: [01-multi_node_parallel.py](hpc/01-multi_node_parallel.py)
    - Run Example With: ```sbatch 01-multi_node_parallel_job.sh``` 

- **Distributed parallel document pre-processing:** Each node loads certain file(s). Each file(s) contains documents which are cleaned in parallel within the node.
    - Code: [02-distributed_parallel.py](hpc/02-distributed_parallel.py)
    - Run Example With: ```sbatch 02-distributed_parallel_job.sh``` 