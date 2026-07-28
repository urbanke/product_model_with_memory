#!/bin/bash
#SBATCH --job-name=ct-16k
#SBATCH --mail-user=ruediger@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --partition=slurm-cluster
#SBATCH --constraint=epyc7302   # node15: 64 cores, 124G
#SBATCH --output=output/ctree_v16384_d2/slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p output/ctree_v16384_d2
# pick the first usable python (>=3.7 with numpy+scipy)
PY=""
for cand in "$HOME/anaconda3/envs/pmm/bin/python" \
            "$HOME/miniforge3/envs/pmm/bin/python" \
            /usr/bin/python3 \
            "$HOME/anaconda3/bin/python"; do
  if [ -x "$cand" ] && "$cand" -c 'import sys; assert sys.version_info>=(3,7); import numpy, scipy' 2>/dev/null; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || { echo "ERROR: no python >=3.7 with numpy/scipy found; see cluster/README.md"; exit 1; }
echo "using python: $PY"
srun "$PY" scripts/context_tree_experiment.py \
    --corpus data/text8 --top-k 16383 --depth 2 \
    --jobs "$SLURM_CPUS_PER_TASK" \
    --out output/ctree_v16384_d2
