#!/bin/bash
# Submit this with sbatch run.sh

#SBATCH --time 00:30:00
#SBATCH --nodes=2
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=120

export PMIX_MCA_gds=hash
source activate TELF
echo $CONDA_DEFAULT_ENV
export n_jobs=$SLURM_CPUS_PER_TASK
export PMIX_MCA_gds=hash


mpirun -n 2 python example.py 
