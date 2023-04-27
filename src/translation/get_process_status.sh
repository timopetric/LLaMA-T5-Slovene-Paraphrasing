#!/bin/bash

jobid="$1"

echo "##############################"
echo "nlp_translation.err"
tail nlp_translation.err 

echo "##############################"
echo "nlp_translation.out"
tail nlp_translation.out

echo "##############################"
echo "User $USER running jobs:"
squeue --me

echo "##############################"
echo "jon status:"
squeue -h -j $jobid -o %T
echo "job has been running for this long:"
squeue -h -j $jobid -o %M

echo "number of lines in translated.out:"
wc -l translated.out
echo "of 623490"

# calculate percentage between number of lines in translated.out and 623490 to 3 decimal places 
echo "translated.out is $(echo "scale=3; $(wc -l translated.out | cut -d' ' -f1) / 623490 * 100" | bc) % complete"

# from percentage of translated sentences, calculate how many hours are left
echo "translated.out has $(echo "scale=3; 24 * (1 - $(echo "scale=3; $(wc -l translated.out | cut -d' ' -f1) / 623490" | bc))" | bc) hours left"

echo "job still has this much SLURM time allocated:"
squeue -h -j $jobid -o %L
