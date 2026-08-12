#!/bin/bash
# Resumable local runner for the frozen ten-point text8 Wave 1.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PMM_UNIVERSAL_TABLES=${PMM_UNIVERSAL_TABLES:-tables/anchors_prod}
export PMM_PHI_LADDER_EVERY=${PMM_PHI_LADDER_EVERY:-1}
export PMM_PHI_LADDER_DEGREE=${PMM_PHI_LADDER_DEGREE:-11}
export PMM_PHI_SADDLE_MIN_L=${PMM_PHI_SADDLE_MIN_L:-54}

root=output/memory2_text8_wave1_20260810
exec .venv/bin/python3 -u scripts/run_memory2_triplet_campaign_local.py \
  --plan "$root/plan.json" \
  --root "$root" \
  --ids output/streams/bpe_text8 \
  --jobs "${PMM_JOBS:-12}" \
  --members-per-batch "${PMM_MEMBERS_PER_BATCH:-4}"
