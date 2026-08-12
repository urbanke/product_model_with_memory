#!/bin/bash
# Resumable local runner for the frozen power-of-two refinement campaign.
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

root=output/memory2_power2_refinement_20260810
plan="$root/plan.json"
if [[ ! -s "$plan" ]]; then
  .venv/bin/python3 scripts/plan_memory2_power2_refinement.py --out "$plan"
fi

exec .venv/bin/python3 -u scripts/run_memory2_triplet_campaign_local.py \
  --plan "$plan" \
  --root "$root" \
  --jobs "${PMM_JOBS:-12}" \
  --members-per-batch "${PMM_MEMBERS_PER_BATCH:-4}"
