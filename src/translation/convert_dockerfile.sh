#!/bin/bash

# a file to convert the dockerfile to singularity file
# not needed - use Singularity.def image that is provided

module load Anaconda3

pip install spython -y

git clone https://github.com/clarinsi/Slovene_NMT.git /tmp/Slovene_NMT_tmp_dir

spython recipe /tmp/Slovene_NMT_tmp_dir/Dockerfile &> Singularity.def

rm -rf /tmp/Slovene_NMT_tmp_dir