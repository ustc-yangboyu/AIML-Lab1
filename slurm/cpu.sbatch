#!/bin/bash
#SBATCH --job-name=lab1_$NAME
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

set -euo pipefail

cd ~/YangBoyu/Course/AIML/Lab1

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

.venv/bin/python -u scripts/$SCRIPT
