#!/bin/bash
#SBATCH --job-name=sf-mid
#SBATCH --mail-user=ruediger@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=1-00:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --partition=slurm-cluster
#SBATCH --constraint=epyc7302   # node15: 64 cores, 124G
#SBATCH --output=output/state_family_fullvocab_mid/slurm-%j.out

SLURM_CPUS_PER_TASK="32"
SLURM_SUBMIT_DIR="$HOME/Projects/product_model_with_memory/"
cd "$SLURM_SUBMIT_DIR"

mkdir -p output/state_family_fullvocab_mid
# pick the first usable python (>=3.7 with numpy+scipy)
PY=""
#for cand in "$HOME/anaconda3/envs/pmm/bin/python" \
#            "$HOME/miniforge3/envs/pmm/bin/python" \
#            /usr/bin/python3 \
#            "$HOME/anaconda3/bin/python"; do
#  if [ -x "$cand" ] && "$cand" -c 'import sys; assert sys.version_info>=(3,7); import numpy, scipy' 2>/dev/null; then
#    PY="$cand"; break
#  fi
#done
PY="/home/urbanke/anaconda3/bin/python"
#[ -n "$PY" ] || { echo "ERROR: no python >=3.7 with numpy/scipy found; see cluster/README.md"; exit 1; }
echo "using python: $PY"
srun "$PY" scripts/state_family_experiment.py \
    --corpus data/text8 --top-k 300000 \
    --m-grid 1024,4096,16384,65536 \
	 --jobs "$SLURM_CPUS_PER_TASK" \
    --out output/state_family_fullvocab_mid
 
