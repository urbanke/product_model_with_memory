#!/bin/bash
# DIAGNOSTIC ONLY: cross-tokenizer comparison, not a production paper pipeline.
# Outputs involving `ours`, bytes, or words must never feed production runs.
# Strict first-order Markov on every representation, in order, from
# scratch.  Rebuilds the streams first: the ones made before 31 July
# picked the wrong case-model scheme for `fixed_bits` on text8 (the
# conditioned model loses there), which put that cell 0.0055 bpc high.
#
#     bash scripts/run_first_order.sh
#
# Safe to re-run: every step overwrites its own output.  Stops at the
# first failure rather than carrying on with a broken input.  Each step
# logs to output/logs/, so a killed run leaves evidence.

set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-12}
mkdir -p output/logs

step () {            # step <name> <command...>
  local name=$1; shift
  echo ""
  echo "=== $name  ($(date +%H:%M:%S))"
  if "$@" > "output/logs/$name.log" 2>&1; then
    echo "    ok   ($(tail -1 "output/logs/$name.log" | cut -c1-90))"
  else
    echo "    FAILED --- see output/logs/$name.log"; tail -20 "output/logs/$name.log"; exit 1
  fi
}

topk () {            # the full alphabet minus one, so nothing is capped
  python - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1] + "/stream.json"))["alphabet"] - 1)
PY
}

echo "############ streams"
for f in text8 enwik8; do
  step "stream_bytes_$f" python scripts/make_stream.py --representation bytes \
      --diagnostic-only --file "data/$f" --out "output/streams/bytes_$f"
  step "stream_bpe_$f" python scripts/make_stream.py --representation bpe \
      --file "data/$f" --vocab-dir vocab_cache --charge-vocabulary 776019 \
      --out "output/streams/bpe_$f"
done
step "stream_ours_text8" python scripts/make_stream.py --representation ours \
    --diagnostic-only --file data/text8 --aux-results output/tok_text8/results.json \
    --out output/streams/ours_text8
step "stream_ours_enwik8" python scripts/make_stream.py --representation ours \
    --diagnostic-only --file data/enwik8 --aux-results output/tok_enwik8_ic/results.json \
    --out output/streams/ours_enwik8

echo ""
echo "############ first order --- state is the previous symbol, M = full"
for f in text8 enwik8; do
  for r in bytes bpe ours; do
    S="output/streams/${r}_${f}"
    K=$(topk "$S")
    step "fo_${r}_${f}" python scripts/state_family_experiment.py \
        --ids "$S" --top-k "$K" --m-grid "0,$((K + 1))" \
        --out "output/fo_${r}_${f}" --jobs "$JOBS"
  done
done

echo ""
echo "############ summary"
python - <<'PY'
import json, pathlib
ref = {("bytes", "text8"): 4.1235, ("ours", "text8"): 2.2483,
       ("bpe", "text8"): 2.1716, ("bytes", "enwik8"): 5.0802,
       ("ours", "enwik8"): 3.1282, ("bpe", "enwik8"): 2.9552}
print(f"{'cell':18} {'memoryless':>11} {'Table 3':>9} {'first order':>12} "
      f"{'gain':>8} {'states':>10} {'sec':>6}")
for f in ("text8", "enwik8"):
    for r in ("bytes", "ours", "bpe"):
        p = pathlib.Path(f"output/fo_{r}_{f}/results.json")
        if not p.exists():
            print(f"{r+'/'+f:18} {'missing':>11}"); continue
        d = json.loads(p.read_text())
        m = d["member_bits_per_character"]; k = sorted(m, key=int)
        a, b = m[k[0]], m[k[-1]]
        t = ref.get((r, f))
        flag = "" if t is None or abs(a - t) < 5e-4 else "   <-- CHECK"
        print(f"{r+'/'+f:18} {a:11.4f} {t if t else 0:9.4f} {b:12.4f} "
              f"{a-b:8.4f} {d['member_states_observed'][k[-1]]:10,} "
              f"{d['seconds']:6.0f}{flag}")
PY
echo ""
echo "done.  The 'Table 3' column is the memoryless number this cell must"
echo "reproduce at M = 0; any CHECK flag means the two disagree."
