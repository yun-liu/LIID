#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling stnm kernels by nvcc..."
nvcc -c -o iou_kernel.cu.o iou_kernel.cu -x cu -Xcompiler -fPIC

cd ../
python3 build.py
