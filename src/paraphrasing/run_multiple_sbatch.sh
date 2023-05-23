#!/bin/bash

# if $1 is "actuallyrun" then run the sbatch commands, otherwise just print params
actuallyrun=$1
echo "(add actuallyrun to run the sbatch commands)"

counter=0
for batch_size in 4 8; do
    for learning_rate in 3e-4 3e-5; do
        # for gradient_accumulation_steps in 4 8; do
        for gradient_accumulation_steps in 16; do
            echo "Running run.sbatch with params: batch_size: $batch_size, learning_rate: $learning_rate, gradient_accumulation_steps: $gradient_accumulation_steps"
            counter=$((counter+1))

            if [ "$actuallyrun" = "actuallyrun" ]; then
                sbatch run.sbatch $batch_size $learning_rate $gradient_accumulation_steps
            fi
        done
    done
done

echo "Total number of jobs: $counter"
