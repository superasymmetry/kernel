#!/bin/bash
#SBATCH --job-name=engine-verify
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/oscar/scratch/szeng26/logs/engine-verify-%j.out

set -euo pipefail
cd /oscar/scratch/szeng26/kernel/engine

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/oscar/scratch/szeng26/kernel/.venv/bin/python verify_engine.py
