#!/bin/bash -l
#SBATCH --job-name=nnp_cuda
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output nnp_cuda-job_%j.out
#SBATCH --error nnp_cuda-job_%j.err
#SBATCH --partition=gpu-v100
vpkg_require gcc
vpkg_require cuda

make clean
make

srun ./nnp train
srun ./nnp predict
