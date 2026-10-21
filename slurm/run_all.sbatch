#!/bin/bash
#SBATCH --job-name=lab1_pipeline
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

set -euo pipefail

export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12

cd ~/YangBoyu/Course/AIML/Lab1

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PY=.venv/bin/python

# Q4 层扫 → Q5 读其全局最优层与同口径 λ2;Q8/Q9 为 truth 侧(相互独立);
# Q6/Q7 经 load_world_submission 读 Q5/Q9 产物
$PY -u scripts/Q4.py
$PY -u scripts/Q5.py
$PY -u scripts/Q9.py
$PY -u scripts/Q8.py
$PY -u scripts/Q6.py
$PY -u scripts/Q7.py
