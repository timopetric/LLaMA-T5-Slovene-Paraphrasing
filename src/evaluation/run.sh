#!/bin/bash

export TRANSFORMERS_CACHE="/d/hpc/projects/FRI/tp1859/nlp_project8/tmp/transformers_cache"
export HF_HOME="/d/hpc/projects/FRI/tp1859/nlp_project8/tmp/transformers_cache_models"

DATASET_PATH="/d/hpc/home/tp1859/nlp/opus/europarl-llama"
DATASET_ORIG_SENTS_FILE="europarl-orig-sl.out"
DATASET_TRAN_SENTS_FILE="europarl-llamapara-sl.out"
echo "Evaluating on $DATASET_PATH"
srun \
    -N1 \
    -n1 \
    --reservation=fri-vr \
    --cpus-per-task=8 \
    --partition=gpu \
    --gpus=1 \
    --time=2-00:00 \
    --job-name="nlp" \
    --mem=64G \
    singularity exec --nv /d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif python3 evaluate.py \
        --dataset-path "$DATASET_PATH" \
        --orig-sents-file "$DATASET_ORIG_SENTS_FILE" \
        --tran-sents-file "$DATASET_TRAN_SENTS_FILE" \
    && cp "$DATASET_PATH/parascores.out" parascores-llama.out \
    && echo "Done."

DATASET_PATH="/d/hpc/home/tp1859/nlp/opus/euparl600k_ensl"
DATASET_ORIG_SENTS_FILE="europarl-orig-sl-all.out"
DATASET_TRAN_SENTS_FILE="europarl-tran-all.out"
echo "Evaluating on $DATASET_PATH"
srun \
    -N1 \
    -n1 \
    --reservation=fri-vr \
    --cpus-per-task=8 \
    --partition=gpu \
    --gpus=1 \
    --time=2-00:00 \
    --job-name="nlp" \
    --mem=64G \
    singularity exec --nv /d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif python3 evaluate.py \
        --dataset-path "$DATASET_PATH" \
        --orig-sents-file "$DATASET_ORIG_SENTS_FILE" \
        --tran-sents-file "$DATASET_TRAN_SENTS_FILE" \
    && cp "$DATASET_PATH/parascores.out" parascores-euparl-tran.out \
    && echo "Done."

echo "Both parascores.out files should be in the current directory."
echo -e "\t parascores-llama.out"
echo -e "\t parascores-euparl-tran.out"
