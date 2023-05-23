#!/bin/bash
#SBATCH --nodes=1           # run on the same node
#SBATCH --ntasks=1          # run one time
#SBATCH --cpus-per-task=12  # with 12 threads
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --time=2-00:00
#SBATCH --job-name="nlp"
#SBATCH --mem=32G
#SBATCH --output=logs/nlp-%J.out
#SBATCH --error=logs/nlp-%J.err

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: sbatch run.sbatch batch_size learning_rate gradient_accumulation_steps"
  exit 1
fi

batch_size=$1
learning_rate=$2
gradient_accumulation_steps=$3

echo "SLURM batch size set:                  $batch_size"
echo "SLURM learning rate set:               $learning_rate"
echo "SLURM gradient accumulation steps set: $gradient_accumulation_steps"

# export NLP8_ENV_MODEL_CHECKPOINT_TO_FINETUNE
export NLP8_ENV_BATCH_SIZE="$batch_size"
export NLP8_ENV_LEARING_RATE="$learning_rate"
# export NLP8_ENV_LEARING_RATE="5e-5"
export NLP8_ENV_GRADIENT_ACCUMULATION_STEPS="$gradient_accumulation_steps"
# export NLP8_ENV_ADD_END_TOKEN=
export NLP8_ENV_REVERSE_INPUT_OUTPUT=0

export TRANSFORMERS_CACHE="/d/hpc/projects/FRI/tp1859/nlp_project8/tmp/transformers_cache"
export HF_HOME="/d/hpc/projects/FRI/tp1859/nlp_project8/tmp/transformers_cache_models"

srun singularity exec --nv /d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif python3 train.py
