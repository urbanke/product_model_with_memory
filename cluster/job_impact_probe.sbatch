#!/bin/bash
#SBATCH --job-name=impact
#SBATCH --mail-user=ruediger@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=6:00:00
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --partition=slurm-cluster
#SBATCH --output=output/impact_probe/slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p output/impact_probe
PY=""
for cand in "$HOME/anaconda3/envs/pmm/bin/python" \
            /usr/bin/python3 \
            "$HOME/anaconda3/bin/python"; do
  if [ -x "$cand" ] && "$cand" -c 'import sys; assert sys.version_info>=(3,7); import numpy, scipy' 2>/dev/null; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || { echo "ERROR: no usable python"; exit 1; }
echo "using python: $PY"
srun "$PY" scripts/impact_probe.py \
    --corpus data/text8 --out output/impact_probe
