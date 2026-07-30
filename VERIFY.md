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

Expectations for the numbers: small shifts are EXPECTED and are the
point --- the old tables were measurably wrong at counts above ~360
(far-left region, tens of nats per moment value) and by ~5e-3 bits
per heavy profile at depth even at counts ~300.  The new values are
certified to ~1e-7 nats worst case.  Shifts should be at the
1e-3..1e-2 bpc scale at most; anything larger deserves a look.

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

## Bottleneck reading (for efficiency round two)

From the printed phase lines, note per run: (a) seconds in "tables"
(column building, T1), (b) seconds per "depth" line (evaluation,
T2/T3), (c) for pooled runs, seconds per "checkpoint" (T3+T6/T7).
Round-two candidates, in order, are in paper/complexity.tex
(implementation-status section): sparse product normalization (T7),
remembered peaks (T2), integrand updates (T3), native kernels for the
column builder, suffix-array counting (T4).
