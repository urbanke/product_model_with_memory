#!/bin/bash
#SBATCH --job-name=pooled
#SBATCH --mail-user=ruediger@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=1-00:00:00
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --partition=slurm-cluster
#SBATCH --constraint=epyc7302
#SBATCH --output=output/pooled_lags_v4096/slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p output/pooled_lags_v4096
PY=""
for cand in "$HOME/anaconda3/envs/pmm/bin/python" \
            "$HOME/miniforge3/envs/pmm/bin/python" \
            /usr/bin/python3 \
            "$HOME/anaconda3/bin/python"; do
  if [ -x "$cand" ] && "$cand" -c 'import sys; assert sys.version_info>=(3,7); import numpy, scipy' 2>/dev/null; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || { echo "ERROR: no usable python"; exit 1; }
echo "using python: $PY"
srun "$PY" scripts/pooled_lag_experiment.py \
    --corpus data/text8 --top-k 4095 --lags 1,2,3,4,6,8 \
    --checkpoints 32 --jobs "$SLURM_CPUS_PER_TASK" \
    --out output/pooled_lags_v4096
