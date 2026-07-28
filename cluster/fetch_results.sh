#!/bin/bash
# Pull experiment results from the cluster to this Mac.
# Fetches only the small files (results.json, slurm logs) — never the
# multi-GB table caches. Run from anywhere:  bash cluster/fetch_results.sh
set -e
REMOTE_HOST="urbanke@lth.epfl.ch"
cd "$(dirname "$0")/.."

# locate the repo on the cluster (first match wins)
REMOTE_DIR=$(ssh "$REMOTE_HOST" '
  for d in ~/product_model_with_memory ~/Projects/product_model_with_memory ~/pmm; do
    if [ -d "$d/output" ]; then echo "$d"; break; fi
  done')
if [ -z "$REMOTE_DIR" ]; then
  echo "Could not find the repo (with an output/ dir) on $REMOTE_HOST."
  echo "Looked in: ~/product_model_with_memory, ~/Projects/product_model_with_memory, ~/pmm"
  exit 1
fi
echo "fetching from $REMOTE_HOST:$REMOTE_DIR/output/"

rsync -av \
    --include='*/' \
    --include='results.json' \
    --include='slurm-*.out' \
    --exclude='*' \
    --prune-empty-dirs \
    "$REMOTE_HOST:$REMOTE_DIR/output/"  output/
echo
echo "Result files now present:"
find output -name results.json -exec ls -l {} \;
