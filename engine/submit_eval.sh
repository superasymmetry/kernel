#!/bin/bash
#SBATCH --job-name=engine-bench
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/oscar/scratch/szeng26/logs/engine-bench-%j.out

set -euo pipefail
cd /oscar/scratch/szeng26/kernel/engine
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/oscar/scratch/szeng26/kernel/.venv/bin/python

echo "=== correctness: optimized vs baseline ==="
$PY check_engine.py

echo; echo "=== baseline throughput ==="
$PY -c "
import benchmark, engine_baseline
benchmark.Engine = engine_baseline.Engine
benchmark.main()
"

echo; echo "=== optimized throughput ==="
$PY benchmark.py
