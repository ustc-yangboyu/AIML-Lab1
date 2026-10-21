#!/bin/bash
#SBATCH --job-name=lab1_extract
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

set -euo pipefail

export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12

cd ~/YangBoyu/Course/AIML/Lab1

PY=.venv/bin/python

# world base/extra 全 36 层(供 Q4/Q5 全层扫描);truth 全层已在 data/truth/layers_all
LAYERS=$(seq -s' ' 0 35)
for CSV in data/world_base/origin/train.csv data/world_base/origin/valid.csv \
           data/world_extra/origin/train.csv data/world_extra/origin/valid.csv; do
    $PY scripts/extract_all_layers.py --input "$CSV" \
        --output-dir "$(dirname "$CSV")/../processed" --layers $LAYERS --token-selection country
done
