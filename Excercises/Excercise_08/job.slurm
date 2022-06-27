#!/usr/bin/env bash

#SBATCH --job-name=mlp_distributed
#SBATCH --output=mlp_distributed_%j.out
#SBATCH -p long
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00

module load pytorch/arm22/1.10

OMP_NUM_THREADS=1 mpiexec -n 1 --bind-to socket --report-bindings python mlp_fashion_mnist.py