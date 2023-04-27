#!/bin/bash

export NVIDIA_VISIBLE_DEVICES=all
# export NVIDIA_DRIVER_CAPABILITIES=all
export CUDA_VISIBLE_DEVICES=0

srun \
    --partition=gpu --gpus=1 \
    -n1 -N1 -c8 \
    --preserve-env --pty \
    singularity exec \
        --nv \
        ./containers/nmt.sif \
        python run_translation.py
        # python read_sentences.py
