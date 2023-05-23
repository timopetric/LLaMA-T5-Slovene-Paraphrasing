#!/bin/bash
#SBATCH --job-name=nlp             # NAME OF THE JOB
#SBATCH --output=logs/nlp-%a.out   # LOG TO OUTPUT FILE (%a IS THE TASK ID IN ARRAY)  ! logs dir must exist !
#SBATCH --error=logs/nlp-%a.err    # LOG TO ERROR FILE  (%a IS THE TASK ID IN ARRAY)  ! logs dir must exist !
#SBATCH --partition=gpu            # SELECT THE PARTITION (FOR GPU USE gpu)
#SBATCH --time=4-00:00             # TIME LIMIT (D-HH:MM:SS) (GPU partition MAX 4D)
#SBATCH --gpus=1                   # SELECT 1 GPU per array task
#SBATCH --cpus-per-task=12         # SELECT 12 CPUs for each array task
#SBATCH --mem=30G                  # SELECT 30 GB OF RAM for each array task
#SBATCH --nodes=1                  # SELECT 1 NODE
#SBATCH --array=0

##SBATCH --mail-type=ALL
##SBATCH --mail-user=TODO@student.uni-lj.si

# #############################################
# # Run vicuna 13B model for paraphrasing
# # 30 GB ram needed for 13B (only needed at init, before the model is loaded to GPU)
# # only 1 GPU needed for 7B and 13B, because the weights are 16bit and fit onto one 32GB GPU (V100)
# # Tested using 7B and 13B vicuna. 13B uses 25 GiB, 7B uses 12 GiB of GPU memory.
# #############################################
srun \
    singularity exec \
        --nv \
        containers/hf.sif \
        python3 cli.py \
            --num-gpus 1 \
            --model-path model_hf_vicuna \
            --corpus-name europarl \
            --file-in data/europarl-orig-all.out
            # --debug # --max-gpu-memory 10GiB
