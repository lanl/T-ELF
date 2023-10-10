#!/bin/bash

# Submit this script with: sbatch 01-multi_node_parallel_job.sh
# Example running pre-processing on 2 nodes, 32 cores in each node.

#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH -p shared-gpu
#SBATCH -C gpu_count:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# LOAD MODULEFILES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load openmpi
module load miniconda3

# Activate the env
source activate TELF
echo $CONDA_DEFAULT_ENV

#Run the code here
export n_jobs=$SLURM_CPUS_PER_TASK
mpirun -n 2 python 01-multi_node_parallel.py 2 $n_jobs