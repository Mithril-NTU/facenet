#!/bin/bash

cur_dir=`pwd`
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${cur_dir}/src
export CUDA_HOME="/usr/local/cuda-9.0"
