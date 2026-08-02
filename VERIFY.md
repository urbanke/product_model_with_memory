# Verification reruns (paper results on the corrected tables)

Purpose: recompute every number in the paper on the NEW certified
universal table store, (a) to verify the published results, (b) to
measure the speedups, and (c) to locate the current bottlenecks for a
second round of efficiency work.

## What changed underneath

All experiments now read moment values from the permanent certified
store (`tables/universal_v2`, created automatically; location
overridable with the environment variable `PMM_UNIVERSAL_TABLES`).
The old per-run caches are no longer used unless you force them with
`PMM_TABLES_SOURCE=cache`.  Nothing in the experiment commands
changes.  Prediction tables now go through the shared-integrand
family evaluator (same numbers, ~k-fold fewer grid operations per
row).

Expectations for the numbers (updated 2 Aug 2026; the earlier
1e-3..1e-2 bpc tolerance was for the v1->v2 correction and is far too
loose for this round): the evaluator+ladder system is measured to
under 1e-4 bits per profile on every instrument case
(`scripts/compare_evaluators.py`, worst 5.6e-6 bits), so a v4 rerun
shifting a published number by MORE than ~1e-4 bpc means one of the
known defects reached it --- the pre-fix evaluator's Laplace
curvature (HANDOVER.md, 2 Aug evening entry) or universal_v2's
series-era columns at large r --- and the v4 value, not the published
one, is the corrected number.  Confirmed so far: state256 unchanged
(3.7198465 vs published 3.719846); ctree_fullvocab corrected
10.0919 -> 10.0926.

Two rules for any verification run: export the four PMM_* store
variables explicitly (a stale environment silently reruns the cache
and proves nothing), and set PMM_BUILD_EXACT=1 for anything that can
build columns, including check_store.py --- without it the contour
reference can take the same series branch as the bug it is checking.

Timing: every script prints its phases with elapsed seconds
("tables k/n (Xs)" = column building, "depth L/Lmax (Xs)" =
evaluation) and writes total `seconds` into its results.json --- these
lines ARE the bottleneck measurement; please keep the logs.

The FIRST run pays for building new table columns (the current
bottleneck; ~0.4 s per column per core, thousands of columns for a
V=256 run).  Every later run reuses them.  Use all cores via
`--jobs`.

## The runs (all on the laptop, from the repo root; use --jobs = your cores)

Each command is one published table/paragraph.  Order: cheap first.

1. Memoryless + first-order, V=256 (results chain start, 1.8870 bpc;
   the m=0 member IS the memoryless code, so this one run covers both):

       python scripts/state_family_experiment.py --corpus data/text8 --top-k 255 \
           --m-grid 0,1,2,4,8,16,32,64,128,256 --out output/v2_state256 --jobs 12

2. Spelling / full-fidelity accounting (Phase 0):

       python scripts/spelling_experiment.py --corpus data/text8 --out output/v2_spelling --jobs 12

3. KT ablation (tab:kt-ablation is at V=1024 and V=4096; four runs):

       python scripts/context_tree_experiment.py --corpus data/text8 --top-k 1023 --depth 2 --leaf-model kt --out output/v2_kt1024 --jobs 12
       python scripts/context_tree_experiment.py --corpus data/text8 --top-k 1023 --depth 2 --leaf-model layered --out output/v2_ct1024 --jobs 12
       python scripts/context_tree_experiment.py --corpus data/text8 --top-k 4095 --depth 2 --leaf-model kt --out output/v2_kt4096 --jobs 12
       python scripts/context_tree_experiment.py --corpus data/text8 --top-k 4095 --depth 2 --leaf-model layered --out output/v2_ct4096 --jobs 12

4. State family V=4096 (the V=4096 row of tab:state-family):

       python scripts/state_family_experiment.py --corpus data/text8 --top-k 4095 \
           --m-grid 0,64,256,1024,2048,4096 --out output/v2_state4096 --jobs 12

4b. Interior-optimum intermediate-M (tab:interior-m, M*=4096, 9.7826;
    FULL vocabulary -- run together with 6/7, they share table columns):

       python scripts/state_family_experiment.py --corpus data/text8 --top-k 300000 \
           --m-grid 0,1024,4096,16384,65536,300000 --out output/v2_state_fullvocab \
           --jobs 12

5. Context tree V=16384 (tab:context-tree-large):

       python scripts/context_tree_experiment.py --corpus data/text8 --top-k 16383 --depth 2 --out output/v2_ct16384 --jobs 12

6. Full-vocabulary unigram (unparked: table build is now fast enough):

       python scripts/unigram_experiment.py --corpus data/text8 --d 262144 \
           --checkpoints 10000,100000,1000000,all --out output/v2_unigram_full --jobs 12

7. Full-vocabulary context tree, depth 2 (unparked: counting now needs
   ~3 GB instead of ~200 GB):

       python scripts/context_tree_experiment.py --corpus data/text8 --top-k 300000 --depth 2 \
           --out output/v2_ctree_fullvocab --jobs 12

8. Pooled lags V=1024 (the killed cluster run; the long one, overnight):

       python scripts/pooled_lag_experiment.py --corpus data/text8 --top-k 1023 \
           --lags 1,2,3,4,6,8 --checkpoints 32 --expert-model layered \
           --out output/v2_pooled_v1024 --jobs 12

Run everything ONE AT A TIME: the runs share the growing table store,
and concurrent writers could corrupt a level file.

Comparison: each results.json against the corresponding number in
paper/main.tex (the tally 1.8870 -> 1.7503 -> 1.6979 bpc and the
tables named above).  I (Claude) do the comparison and update the
paper.  There is nothing to transfer: the results land in this
project folder and I read them directly --- just say when a run is
done.

## Stage B: the representation runs (three benchmarks x three regimes)

These are the runs behind Section "Results" of `paper/compress.tex`,
which is organised by ACCOUNTING REGIME, not by tokenizer.

**(a) Bytes --- DONE.**  No representation cost; a self-contained,
comparable entry.  text8 4.1235, enwik8 5.0802, enwik9 5.1565
bits/character; redundancy over the empirical order-0 entropy at most
2e-5 bits/character at all three scales.

       python scripts/byte_baseline.py --file data/text8  --out output/byte_text8  --jobs 12
       python scripts/byte_baseline.py --file data/enwik8 --out output/byte_enwik8 --jobs 12
       python scripts/byte_baseline.py --file data/enwik9 --out output/byte_enwik9 --jobs 12

**(b) Our adaptive tokenizer** (TOKENIZER.md v3) --- DONE at 10^8.
enwik8: intern+conditioned **3.1282** (winner), intern+folded 3.1413,
compositional+conditioned 3.1902, compositional+folded 3.2032.
text8: 2.2483.  Only enwik9 remains, with the winning setting.

       python scripts/token_baseline.py --file data/enwik8 --numbers intern        --case conditioned --out output/tok_enwik8_ic --jobs 12
       python scripts/token_baseline.py --file data/enwik8 --numbers intern        --case folded      --out output/tok_enwik8_if --jobs 12
       python scripts/token_baseline.py --file data/enwik8 --numbers compositional --case conditioned --out output/tok_enwik8_cc --jobs 12
       python scripts/token_baseline.py --file data/enwik8 --numbers compositional --case folded      --out output/tok_enwik8_cf --jobs 12

       python scripts/token_baseline.py --file data/text8  --numbers intern --case conditioned --out output/tok_text8 --jobs 12

Then, once the four have been read (enwik9, ~10x, add
`--skip-round-trip` only if the decode pass is the thing holding it
up; the round trip has already been verified on enwik8 and in the
unit tests):

       python scripts/token_baseline.py --file data/enwik9 --numbers WINNER --case WINNER --out output/tok_enwik9 --jobs 12

Every stream cost is printed and written separately, so the spelling
stream --- the price of adaptivity --- can be read off directly.  The
case stream is reported both independently and conditioned on the
current token; the difference is the number TOKENIZER.md argues about.

**(c) A standard LLM tokenizer** --- external, data-derived
vocabulary, so it is reported twice (charged / not charged) and the
not-charged line is NOT a benchmark entry.  DONE at 10^8 with
`cl100k_base`: text8 2.1716 charged / 2.1095 free; enwik8 2.9552
charged / 2.8931 free.  The charge is the zipped vocabulary file,
776,019 bytes = 0.0621 bpc at 10^8 and 0.0062 at 10^9.  enwik9
remains:

       python scripts/llm_token_baseline.py --file data/enwik9 --encoding cl100k_base --vocab-dir vocab_cache --out output/llm_enwik9 --jobs 12

NETWORK NOTE.  tiktoken ships the tokenizer code but not the
vocabulary, which it downloads from
`openaipublic.blob.core.windows.net` on first use --- a host EPFL
blocks.  The file was fetched over a phone tether into `vocab_cache/`,
named by the SHA-1 of its URL, which is exactly how tiktoken addresses
its cache; `--vocab-dir vocab_cache` points at it and no run touches
the network again.  `--show-vocab-urls` prints the plan for any other
encoding.

## Bottleneck reading (for efficiency round two)

From the printed phase lines, note per run: (a) seconds in "tables"
(column building, T1), (b) seconds per "depth" line (evaluation,
T2/T3), (c) for pooled runs, seconds per "checkpoint" (T3+T6/T7).
Round-two candidates, in order, are in paper/complexity.tex
(implementation-status section): sparse product normalization (T7),
remembered peaks (T2), integrand updates (T3), native kernels for the
column builder, suffix-array counting (T4).
