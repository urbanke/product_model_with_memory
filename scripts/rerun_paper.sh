#!/usr/bin/env bash
# Re-run every paper experiment on the designed anchor store.
#
# Why this exists as a script rather than a list to paste: the point of
# the sweep is to find out WHICH numbers move, so one run failing must
# not abort the other twenty.  Each command is timed, logged separately,
# and failures are recorded and stepped over.  A summary at the end says
# what ran, what it cost, and what broke.
#
#   bash scripts/rerun_paper.sh              # everything, cheap first
#   bash scripts/rerun_paper.sh stage_a      # just Stage A
#   bash scripts/rerun_paper.sh --dry-run    # print the plan only
#
# Ordering is cheapest-first so a systematic problem shows up in the
# first ten minutes rather than after the enwik9 runs.

set -u -o pipefail

JOBS="${JOBS:-12}"
LOGDIR="${LOGDIR:-output/v3_logs}"
TABLES="${TABLES:-tables/anchors_prod}"
DRY=0
WHICH="all"
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    stage_a|stage_b|all) WHICH="$a" ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done

# The store configuration.  Exported here rather than left to the shell
# so the sweep cannot silently run against the cache -- which would
# quietly reproduce the published numbers and prove nothing.
export PMM_UNIVERSAL_TABLES="$TABLES"
export PMM_PHI_LADDER_EVERY=1
export PMM_PHI_LADDER_DEGREE=11
export PMM_PHI_SADDLE_MIN_L=54

if [ ! -f "$TABLES/anchors.json" ]; then
  echo "ERROR: $TABLES is not a designed anchor store (no anchors.json)."
  echo "Build it with scripts/build_anchor_store.py first."
  exit 1
fi

mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
[ -f "$SUMMARY" ] || printf 'name\tstatus\tseconds\tlog\n' > "$SUMMARY"

run () {                      # run <name> <command...>
  local name="$1"; shift
  local log="$LOGDIR/$name.log"
  if [ "$DRY" = "1" ]; then printf '  %-24s %s\n' "$name" "$*"; return; fi
  printf '[%s] %-24s ' "$(date +%H:%M:%S)" "$name"
  local t0=$SECONDS
  if "$@" > "$log" 2>&1; then
    local dt=$((SECONDS - t0))
    printf 'ok    %6ss\n' "$dt"
    printf '%s\tok\t%s\t%s\n' "$name" "$dt" "$log" >> "$SUMMARY"
  else
    local dt=$((SECONDS - t0))
    printf 'FAIL  %6ss  -> %s\n' "$dt" "$log"
    printf '%s\tFAIL\t%s\t%s\n' "$name" "$dt" "$log" >> "$SUMMARY"
    tail -3 "$log" | sed 's/^/      /'
  fi
}

echo "store:   $TABLES   (ladder every=1 degree=11, expansion at L>=54)"
echo "logs:    $LOGDIR"
echo "jobs:    $JOBS"
echo

if [ "$WHICH" = "all" ] || [ "$WHICH" = "stage_a" ]; then
echo "=== Stage A: the model runs (paper/main.tex) ==="

run state256 \
  python scripts/state_family_experiment.py --corpus data/text8 --top-k 255 \
    --m-grid 0,1,2,4,8,16,32,64,128,256 --out output/v3_state256 --jobs "$JOBS"

run spelling \
  python scripts/spelling_experiment.py --corpus data/text8 \
    --out output/v3_spelling --jobs "$JOBS"

run kt1024 \
  python scripts/context_tree_experiment.py --corpus data/text8 --top-k 1023 \
    --depth 2 --leaf-model kt --out output/v3_kt1024 --jobs "$JOBS"
run ct1024 \
  python scripts/context_tree_experiment.py --corpus data/text8 --top-k 1023 \
    --depth 2 --leaf-model layered --out output/v3_ct1024 --jobs "$JOBS"
run kt4096 \
  python scripts/context_tree_experiment.py --corpus data/text8 --top-k 4095 \
    --depth 2 --leaf-model kt --out output/v3_kt4096 --jobs "$JOBS"
run ct4096 \
  python scripts/context_tree_experiment.py --corpus data/text8 --top-k 4095 \
    --depth 2 --leaf-model layered --out output/v3_ct4096 --jobs "$JOBS"

run state4096 \
  python scripts/state_family_experiment.py --corpus data/text8 --top-k 4095 \
    --m-grid 0,64,256,1024,2048,4096 --out output/v3_state4096 --jobs "$JOBS"

run ct16384 \
  python scripts/context_tree_experiment.py --corpus data/text8 --top-k 16383 \
    --depth 2 --out output/v3_ct16384 --jobs "$JOBS"

run state_fullvocab \
  python scripts/state_family_experiment.py --corpus data/text8 --top-k 300000 \
    --m-grid 0,1024,4096,16384,65536,300000 --out output/v3_state_fullvocab \
    --jobs "$JOBS"

run unigram_full \
  python scripts/unigram_experiment.py --corpus data/text8 --d 262144 \
    --checkpoints 10000,100000,1000000,all --out output/v3_unigram_full \
    --jobs "$JOBS"

run ctree_fullvocab \
  python scripts/context_tree_experiment.py --corpus data/text8 --top-k 300000 \
    --depth 2 --out output/v3_ctree_fullvocab --jobs "$JOBS"

run pooled_v1024 \
  python scripts/pooled_lag_experiment.py --corpus data/text8 --top-k 1023 \
    --lags 1,2,3,4,6,8 --checkpoints 32 --expert-model layered \
    --out output/v3_pooled_v1024 --jobs "$JOBS"
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "stage_b" ]; then
echo
echo "=== Stage B: the representation runs (paper/compress.tex) ==="

run byte_text8 \
  python scripts/byte_baseline.py --file data/text8 --out output/v3_byte_text8 --jobs "$JOBS"
run byte_enwik8 \
  python scripts/byte_baseline.py --file data/enwik8 --out output/v3_byte_enwik8 --jobs "$JOBS"

run tok_text8 \
  python scripts/token_baseline.py --file data/text8 --numbers intern --case conditioned \
    --out output/v3_tok_text8 --jobs "$JOBS"
run tok_enwik8_ic \
  python scripts/token_baseline.py --file data/enwik8 --numbers intern --case conditioned \
    --out output/v3_tok_enwik8_ic --jobs "$JOBS"
run tok_enwik8_if \
  python scripts/token_baseline.py --file data/enwik8 --numbers intern --case folded \
    --out output/v3_tok_enwik8_if --jobs "$JOBS"
run tok_enwik8_cc \
  python scripts/token_baseline.py --file data/enwik8 --numbers compositional --case conditioned \
    --out output/v3_tok_enwik8_cc --jobs "$JOBS"
run tok_enwik8_cf \
  python scripts/token_baseline.py --file data/enwik8 --numbers compositional --case folded \
    --out output/v3_tok_enwik8_cf --jobs "$JOBS"

run llm_text8 \
  python scripts/llm_token_baseline.py --file data/text8 --encoding cl100k_base \
    --vocab-dir vocab_cache --out output/v3_llm_text8 --jobs "$JOBS"
run llm_enwik8 \
  python scripts/llm_token_baseline.py --file data/enwik8 --encoding cl100k_base \
    --vocab-dir vocab_cache --out output/v3_llm_enwik8 --jobs "$JOBS"

echo
echo "--- enwik9: 10x the rest, deliberately last ---"

run byte_enwik9 \
  python scripts/byte_baseline.py --file data/enwik9 --out output/v3_byte_enwik9 --jobs "$JOBS"
run tok_enwik9 \
  python scripts/token_baseline.py --file data/enwik9 --numbers intern --case conditioned \
    --out output/v3_tok_enwik9 --jobs "$JOBS"
run llm_enwik9 \
  python scripts/llm_token_baseline.py --file data/enwik9 --encoding cl100k_base \
    --vocab-dir vocab_cache --out output/v3_llm_enwik9 --jobs "$JOBS"
fi

if [ "$DRY" = "1" ]; then exit 0; fi

echo
echo "=== summary ==="
column -t -s "$(printf '\t')" "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
nfail=$(awk -F'\t' '$2=="FAIL"' "$SUMMARY" | wc -l | tr -d ' ')
echo
echo "$nfail failed; logs in $LOGDIR"
