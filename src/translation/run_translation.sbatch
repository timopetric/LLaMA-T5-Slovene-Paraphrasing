#!/bin/bash
#SBATCH --job-name=translate       # NAME OF THE JOB
#SBATCH --output=logs/nlp-%a.out   # LOG TO OUTPUT FILE (%a IS THE TASK ID IN ARRAY)  ! logs dir must exist !
#SBATCH --error=logs/nlp-%a.err    # LOG TO ERROR FILE  (%a IS THE TASK ID IN ARRAY)  ! logs dir must exist !
#SBATCH --partition=gpu            # SELECT THE PARTITION (FOR GPU USE gpu)
#SBATCH --time=0-00:30             # TIME LIMIT (D-HH:MM:SS) (GPU partition MAX 4D)
#SBATCH --gpus=1                   # SELECT 1 GPU per array task
#SBATCH --cpus-per-task=8         # SELECT 8 CPUs for each array task
#SBATCH --mem=16G                  # SELECT 16 GB OF RAM for each array task
#SBATCH --nodes=1                  # SELECT 1 NODE
#SBATCH --array=0                  # MAKE 6 TASKS (0-5) THAT WILL BE RUN IN PARALLEL AND WILL BE AVAILABLE AS $SLURM_ARRAY_TASK_ID

singularity exec \
    --nv \
    /d/hpc/projects/FRI/tp1859/nlp_project8/containers/nmt.sif \
    python run_translation.py
    # python read_sentences.py
