#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --mem=4000
#SBATCH --time=01:00:00
#SBATCH --output=job.out

nvidia-smi