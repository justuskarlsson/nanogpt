#!/bin/bash

#SBATCH --job-name=llm_job
#SBATCH --output=/proj/cvl/users/x_juska/slurm_logs/llm.out
#SBATCH --error=/proj/cvl/users/x_juska/slurm_logs/llm.err
#SBATCH --time=14:00:00
#SBATCH --gpus=4
#SBATCH --constraint=thin
#SBATCH --mem=300000
#SBATCH --cpus-per-task=32

sh run.sh
