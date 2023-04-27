#!/bin/bash
#SBATCH --nodes=1           # run on the same node
#SBATCH --ntasks=1          # run one time
#SBATCH --cpus-per-task=8   # with 8 threads
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=0
#SBATCH --time=0-00:40
#SBATCH --job-name="nlp"
#SBATCH --mem=20G
#SBATCH --output=logs/nlp-%J.out
#SBATCH --error=logs/nlp-%J.err

# build container
srun singularity build containers/nmt.sif Singularity.def
