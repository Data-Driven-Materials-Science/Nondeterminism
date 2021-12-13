#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:00:00
#SBATCH --gpus=1
python3 noConfig.py

./gpua.out
