# HANDOVER — product_model_with_memory

Running record of goals, decisions, and project state. Companion project to
`product_model`. Update this file as the project evolves.

## Goal

Extend the layered product-simplex predictor of the paper *"A Layered Simplex
Architecture for Large Alphabets"* (July 2026) from exchangeable (memoryless)
sources to **sources with memory**. The past is summarized in a state `s`,
and the scheme predicts from `(state, next token)` pairs:

- Simplest instance: first-order Markov, state = previous token.
- Later: more sophisticated state maps (the past compressed into a richer state).

## Background (where we start from)

The memoryless scheme: draw L iid uniform points on the d-simplex, multiply
coordinatewise, renormalize; depth L is the only structural parameter
(L = 1 is Laplace add-one, L ≈ c·ln d gives sparse heavy-tailed draws), and
averaging the mixtures over all depths yields a single coherent Bayesian
predictor with exactly computable regret. On the King James Bible
(N = 915,849 word-and-punctuation tokens, d = 100,000 vocabulary, 13,550
distinct types) the depth-averaged predictor was the best method tested at
every prefix length (0.095 bits/token redundancy on the full text), with the
depth posterior essentially a point mass drifting from L = 21 to L = 15.

Memory results so far (paper, Section 5.4): an order-one context partition
(states = M most frequent tokens + one backoff state), each state carrying its
own depth-averaged predictor, with hierarchical averaging over the split size.
Code drops from 8.572 to 7.100 bits/token on the KJV; order-one KT codes at
9.78, *worse* than the memoryless layered model — on large alphabets the
choice of marginal prior matters more than the memory it feeds.

**Failed attempt to learn from:** treat sliding-window pairs of the KJV as
symbols of a joint alphabet and run the scheme on pairs, then derive
conditionals. With only ~9.2·10^5 pairs against a joint alphabet of
13,550² ≈ 1.8·10^8, the joint estimate is data-starved (the pair stream
never stops "discovering new symbols", which is exactly what the scheme's
regret pays for). Lessons: (1) much more data is needed; (2) plain joint
estimation shares nothing across related states — structure sharing is the
open opportunity.

## Dataset ladder

We want corpora between the KJV (~0.9M tokens) and full Wikipedia scale,
that others have used, so results are comparable:

| Corpus | Size | Notes |
|---|---|---|
| KJV (paper) | 915,849 tokens, 13,550 types | our baseline numbers exist |
| WikiText-2 | ~2M tokens, ~33k vocab | quick iteration; LM literature |
| **text8** | 17,005,207 tokens, 253,854 types (100 MB) | **current corpus**; clean lowercase a–z + space; used by compression *and* embedding/LM communities |
| WikiText-103 | ~103M tokens, ~268k vocab | word-level benchmark with a large published-perplexity literature (perplexity = 2^(bits/token)) |
| enwik8 | 10^8 bytes raw wiki XML | Hutter Prize / LTCB standard, results in bits/char |
| enwik9 | 10^9 bytes | the "Wikipedia dataset" for compression; same source as enwik8 |

Sources:
- Large Text Compression Benchmark (enwik8/9, text8, hundreds of results): https://www.mattmahoney.net/dc/text.html
- Mahoney data page (downloads): https://mattmahoney.net/dc/
- Hutter Prize: https://en.wikipedia.org/wiki/Hutter_Prize
- WikiText-103 overview: https://www.emergentmind.com/topics/wikitext-103-benchmark
- WikiText-103 published results (SOTA table): https://hyper.ai/en/sota/tasks/language-modelling/benchmark/language-modelling-on-wikitext-103
- text8 mirror used by the download script (gensim-data GitHub release): https://github.com/piskvorky/gensim-data/releases/tag/text8

Comparison cultures: compression papers quote bits/char on enwik8/9; the LM
community quotes perplexity on WikiText. text8 is used in both worlds.

## text8 baseline statistics (computed 2026-07-25)

Canonical file: 100,000,000 bytes, md5 `3bea1919949baf155f99411df5fada7e`,
charset a–z + space only.

- Tokens: 17,005,207; distinct types: 253,854
  (≥2 occurrences: 135,335; ≥10: 47,134; ≥100: 11,815)
- Sliding-window bigrams: 17,005,206; distinct: 4,146,848
  (≥2: 1,168,729; ≥10: 173,467)
- Empirical unigram entropy: 10.856 bits/token
- Empirical conditional entropy H(next | prev): 7.286 bits/token
  → in-sample, first-order memory is worth ~3.57 bits/token

Contrast with the KJV pair experiment: here we have ~4.1 samples per observed
bigram (vs ~1 there), and the observed bigram count (4.1M) is far below the
full joint alphabet 253,854² ≈ 6.4·10^10 — sparse, but no longer hopeless.

## Repo / environment state

- Layout mirrors `product_model`: `src/product_model_with_memory/`, `tests/`,
  `scripts/`, `docs/`, `output/`; setuptools src-layout, pytest, ruff.
- Local env: `python3 -m venv .venv && source .venv/bin/activate && python -m pip install -e ".[dev]"`.
- Remote: `git@github.com:urbanke/product_model_with_memory.git`, branch `main`.
- Data policy: corpora live in `data/` (gitignored); `scripts/get_text8.py`
  downloads and verifies text8 reproducibly.

## Next steps

- Re-verify Stage A under the corrected evaluator (out-dirs `output/v4_*`,
  compare against PUBLISHED, not v3): `ctree_fullvocab` and `pooled_v1024`
  first --- they are the two that moved. `state256` already confirmed
  (3.7198465 vs published 3.719846).
- Then unblock Stage B (`bash scripts/rerun_paper.sh stage_b`).
- Freeze the reference store: `chmod -R a-w tables/probe_exact`.
- Open correctness items: right-series certificate (unexplained, live in
  read path at 1e-10); kernel curvature arithmetic (bypassed, not removed);
  NARROW-peak refinement (unimplemented, none observed yet).

## Log

- 2026-07-25: Project scaffolded (mirrors product_model). Dataset ladder
  decided; text8 chosen as first corpus, downloaded and verified in the
  Claude cloud session; baseline token/bigram statistics computed (above);
  `scripts/get_text8.py` added; `data/` gitignored.

### Log update (2026-07-25, afternoon)

- Ported the Appendix-B numerics from `product_model` into this package
  (`layered.py`), with additions: `fast_tables.py` (batched, disk-cached,
  optionally parallel moment-table builder, validated to 1e-9 against the
  reference); `log_q_lambda_scan` (global-peak evaluation of the outer
  integral, which is multimodal for deep layers with heavy counts: finds
  all significant peaks, refines each by bracketed saddle solve,
  log-sum-exp combination, exact analytic left tail); adaptive grid with
  left edge u_min = min(-70, -(L*ln(r_max+1)) - 40) and a coarse far-left
  segment - the heavy-count/deep-layer regime (analogue of the paper's
  Appendix B.2). Without the adaptive edge the tables fabricate log q > 0;
  a regression test pins this and the codelength API refuses log q > 0.
- Memory-frugal full-corpus path: tables stream to the disk cache
  (materialize=False) and evaluation loads one depth at a time via mmap
  (`depth_averaged_codelength_profiles`). First full-KJV attempt in the
  cloud had died holding ~57 GB of tables in 8 GB of RAM; fixed.
- Test suite: 13 tests, all passing (pytest, ~3 min).
- text8 smoke run (cloud, d = 2^18, L_max = 59):
  n=1e4: H=9.275, PS avg=11.463, redundancy=2.187, mode L=15;
  n=3e4: 9.736 / 11.183 / 1.446 / L=13;
  n=1e5: 10.109 / 10.939 / 0.830 / L=12.
  (output/unigram_text8_smoke/results.json)
- KJV corpus: Gutenberg eBook #30 (paper used #10). Preprocessing (strip
  header/footer, verse markers NN:NNN:NNN, Book headings; tokens =
  [A-Za-z]+ or single punctuation) gives 917,150 tokens / 13,542 types vs
  the paper's 915,849 / 13,550 (0.15%, edition difference). Full validation
  run (d=1e5, L_max=54, n=10k/30k/100k/300k/full) IN PROGRESS in the cloud.
  Table 10 targets: redundancy 1.052/0.598/0.306/0.168/0.095, posterior
  mode L 21/21/21/19/15.

### KJV validation: PASSED (2026-07-25, evening)

Full run of scripts/unigram_experiment.py on the eBook-#30 KJV corpus
(d = 1e5, L_max = 54, single cloud core, ~55 min) vs paper Table 10:

  n         red. ours / paper    mode L ours / paper
  10,000       1.019 / 1.052          22 / 21
  30,000       0.584 / 0.598          21 / 21
  100,000      0.302 / 0.306          21 / 21
  300,000      0.167 / 0.168          19 / 19
  full         0.095 / 0.095          15 / 15

Full-text redundancy matches to the third decimal; small-n deviations are
consistent with the 0.15% edition difference (at n=1e4 the corpora differ
most: 1,108 vs 1,161 observed types). Posterior depth trajectory 21-22 -> 15
reproduced. The ported pipeline is considered VALIDATED; results in
output/unigram_kjv/results.json, comparison table now filled in
paper/main.tex (Table 1), text8 smoke numbers in Table 2.

Next: (a) run text8 at larger n locally (scripts/unigram_experiment.py
--jobs <cores>; full corpus feasible, table cache ~15-20 GB), then (b) the
sliding-window pair experiment: joint distribution over (prev, next) via
the same machinery, conditionals from the joint, compare log-loss to the
empirical conditional entropy (7.286 bits/token on full text8).

### text8 unigram baseline: COMPLETE (2026-07-25, evening)

Full run on Ruediger's Mac (20 cores, 41 min; d = 2^18, L_max = 59, depth-
averaged over all L). Redundancy (bits/token) and posterior-mode depth:

  n=1e4: 2.187 (L*=15)   n=3e4: 1.446 (13)   n=1e5: 0.830 (12)
  n=3e5: 0.492 (10)      n=1e6: 0.262 (8)    n=3e6: 0.134 (6)
  n=1e7: 0.058 (4)       full 17,005,207: 0.039 (L*=4)

First three checkpoints reproduce the cloud smoke run exactly. Key
observation: the selected depth falls to L=4 (~0.3 ln d) at full corpus -
text8 reveals 97% of its alphabet (253,854 / 262,144), so the target is
not sparse relative to d and the posterior moves to the shallow end,
unlike the KJV (13.5% of vocab used; depth settles at 15 ~ 1.3 ln d).
The complexity-spectrum tilt shows up on real text by varying n alone.
Results: output/unigram_text8/results.json; Table 2 of paper/main.tex
filled. Table cache (output/unigram_text8/cache, ~80 GB) can be deleted,
or kept to make additional checkpoints nearly free.

NEXT: the pair experiment - joint (prev, next) distribution over the pair
alphabet with the same depth-averaged machinery, conditionals from the
joint, log-loss vs the empirical conditional entropy H(next|prev) = 7.286
bits/token. On full text8: 17,005,206 pairs, 4,146,848 distinct - the
data-starved regime the KJV pair attempt died in, now with ~4.1 samples
per observed pair.

### Pair experiment implemented (2026-07-25, night)

Plug-in scheme per Ruediger's decision: joint pair distribution estimated
from the WHOLE corpus (in-sample; honest sequential variants deferred, see
discussion of per-state 5.4 scheme / causal joint-conditional scheme with
stale-refresh trick - to revisit).

- Vocabulary ladder: top-K tokens + <unk>; pair alphabet d = (K+1)^2.
  Measured on full text8: count classes stay ~2-4k at every K, so ALL rungs
  up to K=16,383 are laptop-sized (cache 55-96 GB, build ~40 min @20 cores).
  Reduced-corpus targets (full text8): K=255: H_u=4.423, H(n|p)=3.709;
  K=1023: 6.063/4.979; K=4095: 8.051/6.338; K=16383: 9.702/7.154.
- src/product_model_with_memory/pairs.py: vocabulary reduction, pair
  profiles, depth-averaged posterior-mean predictive per count class
  (saddle-rho ratio: E[theta_i|data] = rho(L,c,u*)/N at the scan's peaks,
  normalizes exactly at saddle order; validated <2% vs exact q-ratios,
  exact add-one at L=1), plug-in conditional log-loss from per-row count
  histograms. scripts/pair_experiment.py drives it.
- tests/test_pairs.py: 4 tests, all passing (suite now 17).
- Cloud demo K=63 (d_pair=4096, L_max=39) running on full text8;
  targets H_u=3.224, H(next|prev)=2.733.

### Program section added to paper (2026-07-26)

paper/main.tex Section 5 "Program: averaging over a family of state maps"
now records the guiding design principle agreed with Ruediger: treat
memory exactly as depth is treated - a nested family of state maps
sigma_1 <= ... <= sigma_M of increasing memory, a prior over the family,
one coherent mixture Q = sum_m pi_m Q^(sigma_m); the data selects the
effective memory through the posterior over m at a price <= (log2 M)/n
bits/token. Instances: top-K ladder (current), context trees with layered
leaves (CTW upgrade), successor-distribution clustering (poor man's
information bottleneck), learned state maps (architecture-as-prior).
Orthogonal axis: joint-coupled vs share-nothing estimation at fixed
states; comparison isolates the value of coupling.

### First pair results: at the ceiling (2026-07-26)

Full text8, plug-in conditional log-loss vs empirical H(next|prev):
  K=63  (cloud,  51 min, 2 cores): loss 2.7334, gap 0.000013, mode L=7
  K=255 (laptop, 25 min, 20 cores): loss 3.7098, gap 0.00055, mode L=6
Normalization diagnostics 4e-10 / 5e-7. Dense regime (4150 resp. 260
samples per possible pair): the joint mixture recovers the conditional
essentially exactly. Data-starved regime starts at K>=1023. Table 3 of
paper/main.tex updated. Next runs: K=1023, then 4095, then 16383.

### State-family experiment implemented (2026-07-26)

The proof of concept for "memory as a depth-like axis": nested family
sigma_M (state = last token if among top M, else backoff; M=0 memoryless),
per-member HONEST codelength = product over states of per-state
depth-averaged mixtures (5.4 construction from the validated port),
uniform prior over the M grid, log-sum-exp family mixture + posterior
over M. All members share one moment-table cache. Checks: interior
optimum M*, posterior concentration, family tracks best member within
log2|grid|/n. src/state_family.py, scripts/state_family_experiment.py,
tests/test_state_family.py (4 tests; suite 21). Paper: new subsection
5.1 with scaling sequence (V=64 cloud -> V=256 laptop -> V=1024/4096
overnight -> full vocab + richer families on cluster) and Table 4
skeleton. Cloud sanity run (V=64, n=1e6) in progress.

### K=1023 pair result (2026-07-26)

Laptop, 37 min: plug-in conditional loss 4.9878 vs H(next|prev)=4.979,
gap 0.0085 bits/token (0.17% of the conditional entropy) with 71% of the
2^20 pair alphabet unseen; posterior point mass at L=8; norm err 2e-11.
Gap trend across rungs: 1.3e-5 -> 5.5e-4 -> 8.5e-3 (K=63/255/1023),
roughly a decade per rung. Table 3 updated. NOTE: run reused the k255
output dir; results moved to output/pairs_text8_k1023/, K=255 json
restored. Next: K=4095.

### K=4095 pair result + state-family sanity run (2026-07-26)

K=4095 (laptop, 41 min): plug-in loss 6.3951 vs H=6.338, gap 0.0575
bits/token (0.9% of ceiling) with 93% of the 2^24 pair alphabet unseen;
posterior point mass L=10; norm err 6e-12. Gap ladder now 1.3e-5 ->
5.5e-4 -> 8.5e-3 -> 5.7e-2: ~decade per rung, mild slowdown at the
sparse end. K=16383 running (last rung).

State-family sanity run (cloud, V=64, n=1e6, 11 min): member curve falls
monotonically 3.2573 (M=0) -> 2.7644 (M=64); posterior mass 1.0 on M=64;
family = best member to 4 decimals; ceiling H(next|prev)=2.753. At this
small V / large n even full memory is cheap, so the optimum sits at the
boundary - the interior optimum should appear as V grows (V=256 laptop
run next, then V=1024/4096). Machinery verified end to end.

### Pair ladder COMPLETE (2026-07-26)

K=16383 (laptop, 43 min): loss 7.3104 vs H=7.154, gap 0.1562 bits/token
(2.2% of ceiling) with 99% of the 2^28 pair alphabet unseen, 6.6 samples
per observed pair; posterior point mass L=15; norm err 5e-13.

Full gap ladder (K=63/255/1023/4095/16383):
  1.3e-5 / 5.5e-4 / 8.5e-3 / 5.7e-2 / 1.56e-1 bits/token
  growth factors x42 / x15 / x6.8 / x2.7 - strong deceleration.
Depth modes 7/6/8/10/15 (growing with sparsity - spectrum tilt).
KEY TAKEAWAY: the joint-mixture plug-in degrades gracefully deep into
the data-starved regime; extrapolating the decelerating factors, even
the full 253,854-vocabulary (d~6.4e10) may concede only a fraction of a
bit - the regime that produced the original failed KJV attempt looks
tractable with this estimator. Table 3 complete in paper/main.tex.
Next: state-family V=256 run (command issued), then joint-vs-share-
nothing comparison at identical states, then V=1024/4096 family runs.

### State-family V=256 result (2026-07-26)

Full text8, honest per-state codes, 22 min laptop. Member curve monotone:
4.4234 (M=0) -> 3.7198 (M=256); posterior mass 1.0 on M=256; family =
best to 4 decimals. Boundary optimum again (65k pairs vs 17M samples:
full memory still cheap). Quantitative: honest code captures 98.6% of
the available memory value (0.704 of 0.714 bits/token); price of honesty
vs plug-in ceiling = 3.7198 - 3.7093 = 0.0105 bits/token (~19x the
plug-in gap). Table 4 rows 1-2 filled. Next: V=1024 family run (interior
optimum expected when V^2 outruns n), then V=4096.

### State-family V=1024 result (2026-07-26)

Full text8, 42 min laptop. STILL a boundary optimum: member curve monotone
6.0636 (M=0) -> 5.0564 (M=1024), posterior mass 1.0 on M=1024, family =
best. Even with 71% of the pair alphabet unseen, 1024 share-nothing
states beat every coarsening at n=17M - the layered per-state prior
prices rare contexts gently. Honest code captures 92.9% of available
memory value; price of honesty vs plug-in ceiling: 5.0564 - 4.9793 =
0.0771 bits/token. Table 4 row 3 filled. V=4096 running; interior
optimum (if any at this n) expected where fine states have ~tens of
samples. Discussion queued: richer state maps beyond one previous token.

### Order-two product-state family implemented (2026-07-26)

States (b_M1(prev), b_M2(prev-prev)) with asymmetric resolution; grid
over (M1,M2) contains first-order family as M2=0 slice, memoryless as
(0,0); all members code x_3..x_n -> mixable. Global profile dedup: 205k
states of a 10-member grid on full text8 collapse to 26,969 unique
profiles. src/product_family.py, scripts/product_family_experiment.py,
tests/test_product_family.py (3 tests, incl. order-2 synthetic source
where (4,4) crushes (4,0); (M,0) slice matches first-order family to
1e-6). Suite: 24 tests. Paper: order-two paragraph added to Section 5.1.
Readout to watch: posterior over (M1,M2) = how resolution splits across
distances into the past.

### State-family V=4096 result (2026-07-26)

Full text8, 64 min laptop. Boundary optimum a THIRD time: monotone
8.0523 (M=0) -> 6.7132 (M=4096), posterior 1.0 on the finest member.
Capture fraction of available memory value now falling hard: 98.6% ->
92.9% -> 78.2% (V=256/1024/4096); price of honesty vs plug-in ceiling
0.0105 -> 0.0771 -> 0.3756 bits/token. Yet every refinement still pays:
for last-token states at n=17M the effective-memory frontier lies beyond
the vocabulary. Table 4 complete. The family must grow along the order-2
axis for the data to be forced to choose -> product-family run is next
(command issued; Table 5 skeleton ready).

### Outlook section + context-tree scheme implemented (2026-07-26)

Paper Section 7 "Outlook: generating state maps, not designing them"
records the four schemes agreed with Ruediger: (1) recursive tree growth
= context trees with layered leaves (IMPLEMENTED); (2) random recursive
automata / reservoir states; (3) learned coarsenings (approximate causal
states, honesty restored two-part or causally); (4) composed random
coarsenings - the layered construction applied to memory itself,
composition depth R as the memory analogue of emission depth L,
candidate for DERIVING memory scaling laws. Ultimate goal recorded:
architectures-as-priors with recurrence, two nested layered
constructions (L and R), jointly adapting.

Scheme 1 code: src/context_tree.py (half-half growth prior over all
prunings to depth D, per-node layered mixtures, log-domain beta
recursion, MAP-pruning diagnostic with leaves-by-depth histogram,
fixed-depth baselines), scripts/context_tree_experiment.py,
tests/test_context_tree.py (4 tests incl. exact brute-force enumeration
match; suite 28). Sizing on full text8, V=1024, D=2: 308,101 contexts,
32,479 unique profiles -> overnight-able laptop run.

### FIRST INTERIOR OPTIMUM: order-two family result (2026-07-26)

Full text8, V=1024, 42 min laptop, 26,969 unique profiles. Along the
(1024, M2) arm the codelength dips 5.0564 -> 5.0395 -> 5.0138 -> 5.0104
(M2*=64, posterior 1.0) -> rises to 5.0419 at M2=256. The data chose its
effective memory in the interior for the first time: a resolution-64
glimpse of two-back is worth 0.046 bits/token on top of full one-back;
finer costs more than it reveals. Constant-budget split question
answered: (1024,4) > (256,16) > (64,64) at ~4k states - resolution
belongs to the most recent token first. Table 5 filled. Next-size run:
V=4096 product family (command issued). Then: context-tree experiment
(implemented, ready).

## 2026-07-26 — Product family V=4096: snap-back to the first-order boundary

`output/product_family_v4096/results.json` (3,338 s, 63,657 unique profiles).
Members (bits/token): (0,0)=8.0523, (256,0)=7.0742, (1024,0)=6.8776,
**(4096,0)=6.7132 (best, posterior 1.0)**, (4096,16)=6.7855, (4096,64)=6.8592,
(4096,256)=6.9601, (4096,1024)=7.1486, (1024,64)=6.8777, (1024,256)=6.9317.
Family = 6.7132. Targets: H_unigram=8.0508, H(next|prev)=6.3376.

Finding: unlike V=1024 (interior optimum M2*=64, worth 0.046 bits), at
V=4096 the optimum snaps back to the boundary (4096,0) — every uniform
allocation of two-back resolution hurts, and (1024,64) now merely ties
(1024,0). Effective-memory law is non-monotone at fixed n. The weakness is
that uniform product states spend two-back resolution identically in every
context. Added as `tab:product-family-4096` in paper/main.tex.

Next run (agreed direction): context-tree experiment — adaptive per-context
depth allocation, precisely the fix for the uniform-grid failure:
  rm -rf output/product_family_v4096/cache
  caffeinate -i python scripts/context_tree_experiment.py --corpus data/text8 \
    --top-k 1023 --depth 2 --jobs 20 --out output/ctree_v1024_d2
Expect ~1-1.5 h on 20 cores (308,101 contexts of depth <=2, ~32k unique
profiles). fixed_depth[1] in the output is the direct benchmark against the
first-order pair family at V=1024; family_bits <= min fixed depth + prior.
MAP leaves-by-depth histogram is the effective-memory read-out.

## 2026-07-26 — Paper: context-tree experiment section added

New Section 5.2 "Adaptive depth: the context-tree family"
(label sec:context-tree) in paper/main.tex, ahead of the run's results:
motivation from the V=4096 snap-back (uniformity is the defect), family =
prunings of the depth-D suffix tree, 1/2-1/2 recursive growth prior, honest
share-nothing per-leaf layered mixtures, all members code x_{D+1}..x_n,
beta recursion (eq:ctw-beta) = CTW with layered mixtures replacing KT,
dedup figures, and the three read-outs (fixed-depth baselines anchored to
Tables 4/5, family-vs-best-fixed margin = value of adaptive depth, MAP
leaves-by-depth histogram). Results of the running V=1024/D=2 job go into
this section when done; follow-ups noted: V=4096 D=2 (does adaptivity undo
the snap-back?) and D=3. Outlook scheme 1 now points to this section.

## 2026-07-26 — Context-tree run V=1024 D=2: three headline results

`output/ctree_v1024_d2/results.json` (1,876 s = 31 min on 20 cores;
308,101 contexts -> 32,479 unique profiles; L_max=33; n_coded=17,005,205).
Fixed depths: d0=6.0636, d1=5.0564 (= product (1024,0), conventions
validated), d2=5.1309 (complete depth-2 LOSES to depth 1 by 0.074).
**Family = 4.9507**; MAP = 4.9507 (agree to 4e-9/token; posterior ~0.96
on the single MAP pruning).

Findings (now Table tab:context-tree in the paper):
1. Adaptive depth converts a losing axis into the largest memory gain so
   far: beats best fixed depth by 0.106, best uniform product (1024,64)
   by 0.060 bits/token — with FEWER states (27,473 leaves vs 47,687).
2. Honest code CROSSES the first-order plug-in ceiling:
   4.9507 < H(next|prev)=4.9793 — first predictor in the study to do so.
3. Effective memory is sparse: MAP splits only 57 of 1,024 depth-1
   contexts (967 stay leaves; 26,506 depth-2 leaves). Depth pays after
   ~1 token in 20, worthless elsewhere — the allocation no uniform
   (M1,M2) grid can express.

Next runs (in paper as planned follow-ups):
  rm -rf output/ctree_v1024_d2/cache
  caffeinate -i python scripts/context_tree_experiment.py --corpus data/text8 \
    --top-k 4095 --depth 2 --jobs 20 --out output/ctree_v4096_d2
(V=4096 D=2: does adaptivity undo the snap-back? L_max=39; expect roughly
2-4 h.) Then D=3 at V=1024.

## 2026-07-26 — Outlook scheme 5: pooled lag experts (combining predictions)

Ruediger's proposal: use pairwise conditionals p_delta(x_t | x_{t-delta})
for several lags delta, combine the per-lag estimates, recurse. Recorded
as scheme 5 in the Outlook of paper/main.tex, with the derivation
hierarchy: (a) Bayesian mixture over delta = selection, posterior =
distance profile of memory value (Chow-Liu tree degenerates to this);
(b) naive-Bayes logarithmic pool q ∝ p(x) prod p_delta(x|y)/p(x), i.e.
log q = log prior + sum of PMIs — accumulates evidence, fails by double
counting redundant lags; (c) maxent pairwise MRF as the principled fix,
weighted pool q ∝ p^(1-sum w) prod p_delta^w as its cheap surrogate,
weights LEARNED per gating context (prior over w, mixed online). Dyadic
recursion (pool pools over distance bands) reaches 2^R at cost linear in
R. Boundaries recorded: honesty requires per-step predictives (deferred
sequential machinery); pairwise info cannot see synergy (XOR sources) —
pooling and state refinement are complements (additive vs synergistic
part of the past); 57 split contexts of the ctree = where synergy lives.
Bridge to attention noted. Cost: additive (Delta*V states), not V^Delta.

Next implementation candidate: lag family sigma_delta, delta=1..8, via
existing state-family machinery (two-line change) — gives the distance
profile any pooling scheme feeds on. Not yet implemented.

Still pending run: ctree V=4096 D=2 (does adaptivity undo the snap-back?)
  rm -rf output/ctree_v1024_d2/cache
  caffeinate -i python scripts/context_tree_experiment.py --corpus data/text8 \
    --top-k 4095 --depth 2 --jobs 20 --out output/ctree_v4096_d2

## 2026-07-26 — Lag family implemented (scheme 5, level 0); ctree V=4096 running

User started: context_tree_experiment --top-k 4095 --depth 2
  -> output/ctree_v4096_d2 (does adaptivity undo the snap-back?).

New code (synced, 3 tests passing):
- src/product_model_with_memory/lag_family.py — sigma_delta: s_t =
  x_{t-delta} at full vocab resolution; delta=0 = memoryless anchor; all
  members code x_{delta_max+1}..x_n so they are mixable; global profile
  dedup across members; uniform mixture + posterior over delta (the
  distance profile of memory value).
- tests/test_lag_family.py — partition; deltas=[1] == state-family M=V
  member within 1e-6; lag-2 decisive on the order-2 toy source
  (posterior > 0.999), family within log2(4)/n of best.
- scripts/lag_family_experiment.py — CLI: --deltas 0,1,2,3,4,6,8 etc.

Suggested run AFTER ctree_v4096_d2 finishes (clean its cache first):
  rm -rf output/ctree_v4096_d2/cache
  caffeinate -i python scripts/lag_family_experiment.py --corpus data/text8 \
    --top-k 1023 --deltas 0,1,2,3,4,6,8 --jobs 20 --out output/lag_family_v1024
Cost ~ a few members' worth of unique profiles at V=1024 (distant lags
dedup heavily toward unigram-like rows); expect well under an hour.
Paper: experiment subsection + results table to be added when it runs.

## 2026-07-27 — Context tree V=4096 D=2: adaptivity undoes the snap-back

`output/ctree_v4096_d2/results.json` (2,750 s = 46 min; 1,177,038
contexts -> 59,002 unique profiles; L_max=39). Fixed: d0=8.0523,
d1=6.7132 (= product (4096,0), conventions validated), d2=7.4231.
**Family = 6.6807**, MAP = 6.6807 (posterior ~0.93 on MAP pruning).
MAP splits 24 of 4,096 depth-1 contexts -> 31,626 depth-2 leaves
(35,698 leaves total).

Reading (now Table tab:context-tree-4096): the snap-back of the uniform
product grid was a failure of the FAMILY, not absence of structure —
order-two memory is still worth 0.032 bits/token at V=4096, hidden in
24 contexts. Law refined: at fixed n, growing V makes effective memory
CONCENTRATE (gain 0.106->0.032, split contexts 57->24, ceiling no
longer crossed: 6.6807 > H(next|prev)=6.3376), not switch off.

Next run (lag family, scheme 5 level 0 — code already synced):
  rm -rf output/ctree_v4096_d2/cache
  caffeinate -i python scripts/lag_family_experiment.py --corpus data/text8 \
    --top-k 1023 --deltas 0,1,2,3,4,6,8 --jobs 20 --out output/lag_family_v1024
Remaining ctree follow-up: D=3 at V=1024 (note: depth-3 contexts will be
several million; check unique-profile count/feasibility before running).

## 2026-07-27 — Lag family V=1024: the distance profile measured

`output/lag_family_v1024/results.json` (2,799 s = 47 min; 6,145 unique
profiles — no dedup benefit at full resolution, every row distinct).
Member bits/token (v(delta) = gain over memoryless 6.0636):
  d=1: 5.0564 (v=1.0072; = first-order finest member, conventions OK)
  d=2: 5.6962 (0.3675)   d=3: 5.9046 (0.1590)   d=4: 5.9902 (0.0735)
  d=6: 6.0293 (0.0343)   d=8: 6.0418 (0.0218)
Family = 5.0564, posterior 1.0 on d=1 (mixture selects, as designed).

Now Section 5.3 / Table tab:lag-family in paper. Readings:
1. Distance profile decays ~x2.7, x2.3, x2.2 per step early, flattening
   to ~x1.25/step at lags 6-8 — fat, power-law-like tail; lag 8 still
   carries 0.022 bits, honestly learned. Learning costs identical across
   members (1024 states each), so v(delta) ≈ pairwise information.
2. Pooling target: lags 2-8 hold 0.656 bits/token of pairwise value that
   SELECTION cannot touch (posterior 1.0 on d=1).
3. First redundancy estimate: lag-2 pairwise value 0.3675 vs ctree gain
   0.106 from combining lags 1+2 via states -> most of lag-2 info is
   redundant given lag 1; exactly the double counting the naive pool
   commits and learned gate weights must discount.
Scheme 5 in outlook now cross-references Section 5.3.

Open experiment directions (no run queued yet — discuss first):
- ctree D=3 at V=1024 (check depth-3 context count first)
- scheme 5 level (b/c): pooled predictions — needs per-step sequential
  machinery (the deferred honest-sequential door)
- lag family at V=4096, or with more lags (16, 32) to pin the tail law

## 2026-07-27 — Plot script for the distance profile; V=4096 lag run queued

User is running:
  caffeinate -i python scripts/lag_family_experiment.py --corpus data/text8 \
    --top-k 4095 --deltas 0,1,2,3,4,6,8,12,16,24,32 --jobs 20 \
    --out output/lag_family_v4096
(optionally followed by the V=1024 tail run, deltas up to 64,
 --out output/lag_family_v1024_tail).

New: scripts/plot_lag_profile.py — plots v(delta)=bits(0)-bits(delta) on
log-log axes for one or more results.json inputs (path:label), fits BOTH
a power law v=c*delta^-alpha and an exponential v=c*rho^delta by least
squares in log v over delta >= --fit-from (default 2), draws the better
fit solid / the other dotted, prints parameters and RMS residuals, and
writes PDF + PNG. Usage:
  python scripts/plot_lag_profile.py \
    --inputs output/lag_family_v1024/results.json:V=1024 \
             output/lag_family_v4096/results.json:V=4096 \
    --out paper/lag_profile.pdf
Preview on the existing V=1024 run (lags 1-8): power law clearly beats
exponential (alpha=2.07, rms 0.08 vs 0.27) — consistent with the
power-law long-range correlation literature; extended lags will pin the
tail. Figure to be added to Section 5.3 when the new runs land.

## 2026-07-27 — Decay law + Figure 1 added to Section 5.3

paper/lag_profile.pdf (generated by scripts/plot_lag_profile.py from the
V=1024 run) is now Figure fig:lag-profile, with a "The decay law"
paragraph: power law v(delta) ~ 1.48 * delta^-2.07 (RMS 0.08 in log v)
clearly beats exponential (rho=0.63, RMS 0.27); fits over delta>=2.
Structural consequence recorded: under a power law the value beyond a
cutoff Delta falls only polynomially (~Delta^-(alpha-1)), so reaching far
back stays worthwhile — the case for the dyadic pooling recursion.
Extended runs (V=4096 to lag 32, running; optional V=1024 tail to lag 64)
will be added as extra curves to the same figure and will pin the tail
exponent + V-dependence (conjecture: alpha is a property of the text,
V only scales the prefactor).

Also this session: agreed design for the sequential pooled evaluator
(scheme 5 level b/c): per-step log-linear pool of causal lag experts with
checkpointed predictive tables (staleness = honest, causality per block),
weight grid mixed at sequence level, gating later; loss computed by
streaming n * Delta * V einsum in chunks; first run V=256 full corpus,
lags {1,2,3,4}. Validation: brute-force tiny-alphabet test, one-hot
weight -> lag-1 member consistency (gap = staleness price), anchors vs
lag mixture / ctree / 0.656 additive target. Not yet implemented.

## 2026-07-27 — Lag family V=4096 to lag 32: offset power law, exponent invariant

`output/lag_family_v4096/results.json` (5,079 s = 85 min; 40,961
profiles). Posterior 1.0 on d=1; d=1 member 6.7132 = (4096,0) ✓.
NET profile v(d)=bits(0)-bits(d) crosses ZERO at d~4: distant lags cost
more to learn (4,096 honest conditionals vs 1) than they pay. The tail
level measures the learning-cost offset directly: 3-param fit
v = c*d^-alpha - L (linear LS, scipy, d>=2) gives
  V=4096: alpha=2.02+-0.08, c=2.17, L=0.1248+-0.0045, rms 0.007
  V=1024 (same method): alpha=2.17, c=1.66, L~0, rms 0.005
Exponential+offset loses 2x at both scales. HEADLINE: exponent is a
property of the text (~2.0-2.2 at both V); V enters via prefactor
(more info at every lag) and offset (price of honesty). Corrected
profile I(d)=v+L; d=12,16 points sit ABOVE the fit — far tail flatter
than alpha=2, but beyond d~20 the offset uncertainty drowns the signal
at this V. Paper: new Table tab:lag-family-4096, rewritten "decay law"
paragraph, updated Figure (both corrected profiles, offset fits);
design-point paragraph amended (v = NET value, = information only up
to common offset L vs the 1-state anchor).
plot_lag_profile.py: new --offset-fit flag (3-param power and exp fits,
plots I=v+L). Demo results.json files for both runs mirrored in cloud
output/ for regeneration.

RUNNING on laptop: V=1024 tail run (lags to 64, output/lag_family_v1024_tail)
— the instrument for the asymptotic exponent (L is ~20x smaller there).
Its curve goes into the same figure; then revisit total-memory-gain
estimate (planning numbers currently: naive 0.86 bits beyond lag 1 at
alpha~2.07; harvest ~0.25-0.3 at 29% survival; tail mass very sensitive
to alpha).

## 2026-07-27 — V=1024 tail run (lags to 64): the decay law has TWO regimes

`output/lag_family_v1024_tail/results.json` (3,767 s = 63 min; 10,241
profiles). Net profile crosses zero at d~24; L_1024 = 0.0153+-0.0052
(vs 0.1368+-0.0067 at V=4096 — 9x for 4x the states).

HEADLINE: a single power law does NOT survive the tail. Both corrected
profiles I(d)=v(d)+L break at d~4 into two regimes:
  short (2<=d<=4): alpha_s = 2.10 (V=1024) / 2.02 (V=4096)  — syntax
  tail  (d>=4):    alpha_t = 1.06+-0.19  /  1.34+-0.20      — topic
Tail rms 0.004 both scales; exp+offset loses ~1.8x. Regime structure
invariant across V; V moves prefactors + offset only. alpha_t sits near
the DIVERGENCE BOUNDARY alpha=1: cumulative tail info grows ~ log(hor.)
— individually worthless lags (net), unbounded in aggregate. Recorded
in paper: this value-shape CANNOT be harvested by adding states per lag;
it is exactly the regime for pooled predictions (scheme 5) — one shared
state space, marginal cost of reach = a weight, not a model. The pooled
evaluator measures what fraction survives redundancy.

Paper changes: Table tab:lag-family-tail (new); tab:lag-family-4096 I
column + caption updated to tail-fit L=0.1368; "The decay law: two
regimes" paragraph replaces the single-law paragraph; figure regenerated
(broken power law, merged V=1024 series, open circles = below noise
floor). plot_lag_profile.py: new --break-at flag (two-regime fit);
merged input output/lag_family_v1024_merged.json built by combining the
two V=1024 runs (deltas 3,6 re-anchored from the first run).

Lag experiments now complete at both scales. Next big step (agreed
design in earlier entry): the sequential pooled evaluator — first run
V=256 full corpus, lags {1,2,3,4}, checkpointed causal experts, weight
grid mixed at sequence level. Not yet implemented.

## 2026-07-27 — New Section 6: external comparison points, units, rules

paper/main.tex: new \section{External comparison points: units, rules,
and trajectory} (label sec:comparison) before the Outlook. Contents:
- Units made precise: all our numbers = bits/token, in-sample, on
  top-K+UNK reduced streams; conversion 100e6 chars / 17,005,207 tokens
  = 5.8806 chars/token; bpc-equivalent = bits/token / 5.8806, valid as
  ORIENTATION only (UNK identities unpaid: 32.5% of tokens at V=1024,
  18.0% at V=4096, ~8.1-8.4 chars each). text8 = tokens joined by single
  spaces, so token code + UNK spelling model = full-fidelity code.
- Rules, concretely: LTCB (rank = compressed size + zipped decompressor,
  no resource caps, single pass — OUR protocol; text8 table on Mahoney's
  test-data page: gzip 2.64 bpc, bzip2 2.11, ppmd 1.60, paq8h 1.40;
  cmix ~1.17 on rawer enwik8). Hutter Prize (enwik9 only, total incl.
  decompressor, record 114,156,155 B ≈ 0.913 bpc, 2023; <10GB RAM,
  <50h, min 1% improvement, ~5000 EUR/percent). Neural leaderboard
  (90/5/5 split, test bpc 1.04-1.19; training uncharged — NOT a
  codelength; fair meeting point = our bits on final 5% given first 95%).
- Where we stand: ctree V=1024 = 0.84 bpc-equiv (flattered), V=4096 =
  1.14; naive per-occurrence spelling (+3.0-4.5 bits/token at V=4096) →
  first full-fidelity ≈ 1.65-1.90 bpc (between bzip2 and ppmd).
- How we compete (three levers): (1) spell each distinct type ONCE
  (253,854 types, 2,237,382 chars → 0.26-0.40 bits/token amortized at
  2-3 b/char; PPM escape mechanism done honestly; enables full-vocab
  token model); (2) context tree = the alpha_s≈2 syntax regime;
  (3) pooled lags = the alpha_t≈1 topic regime — the exact structure
  separating ppmd (1.60) from CM programs (1.40→1.1) which reach it via
  hand-built skip contexts + ad-hoc mixing; scheme 5 is the principled
  counterpart. Trajectory: entry 1.4-1.6 (beat ppmd) → 1.1-1.3 (pooling
  matures) → <1.0 needs neural-grade modeling.
Sources checked live: mattmahoney.net/dc/textdata.html (text8 table),
prize.hutter1.net (rules + record), nlpprogress.com (neural bpc).

## 2026-07-27 — Section 6 rewritten in plain factual style (user request)

Renamed to "Comparison with published results on text8". Changes per
Ruediger's instructions: all editorializing removed ("orientation
device", "honestly", "sober", "flattered", "home turf", "trajectory"
rhetoric); everything stated as fact. The UNK issue is now explicit:
our codelength transmits the REDUCED sequence — the decoder learns
that a rare word occurred at a position but not which one; the original
file cannot be reconstructed from what we transmit; published figures
are for byte-for-byte reconstruction; comparison requires additionally
encoding rare-word spellings. Same content otherwise: units (5.8806
chars/token), the three sets of published numbers with their exact
measurement rules, our converted numbers with the per-occurrence
spelling estimate (1.65-1.90 bpc), and Prospects with the three
changes and their estimated sizes (spell-once 0.26-0.40 bits/token ->
est. 1.4-1.6 bpc; deeper trees/larger V; pooled lags for the CM range
1.1-1.6; explicit statement that <1.0 bpc is not within reach of the
present constructions).

## 2026-07-27 — Full-paper style revision (plain English, facts only)

Per Ruediger: whole write-up revised to the style of the rewritten
Section 6. Rules applied throughout, no numbers or technical content
changed:
- Editorial/evaluative words removed (remarkably, decisively, notably,
  markedly, hopeless, natural next, indicts, worth having, etc.).
- Metaphors replaced: "snap-back" -> "the optimum returns to the
  boundary"; "glimpse" -> "resolution"; "harvest" -> "capture"/"use";
  "ladder/rung" -> "sequence/row"; "read-out" -> "reported quantity";
  "anchor" -> "memoryless member" (in the lag family); "data-starved" ->
  "sparsely observed"; "explodes" -> "grows multiplicatively";
  "warm-up" -> "special case"; "complexity dial" -> "complexity
  parameter"; "Completing the picture" -> plain heading.
- "honest" (as a code qualifier) -> "sequential" everywhere; "price of
  honesty" -> "cost of sequential estimation" / "excess over the plug-in
  value"; "honesty must be restored" -> "the code must be made
  sequential again". Zero occurrences of "honest" remain.
- Claims restated as facts: ceiling-crossing now stated as "4.9507 is
  below H(x'|x)=4.9793, a lower bound for every predictor conditioning
  only on the previous token"; syntax/topic reading of the two regimes
  explicitly marked "a plausible interpretation, which we do not test
  here"; tail consequence stated via the growth of the cumulative sum.
- Fixed a real error: Outlook said "Four schemes" while listing five.
Compiles cleanly (17 pages).

## 2026-07-27 — Section 1: empirical-entropy floor made precise

Expanded the empirical-entropy paragraph per Ruediger's question: H_n
minimizes the per-token cost over constant assignments (minimizer =
relative frequencies), but is NOT the length of a code — the decoder
would need the same probabilities and cannot compute them from data it
has not yet received (the encoder can). A decodable scheme closes the
gap either by transmitting the frequencies first (two-part code, pays
their description cost) or by using at step t only probabilities
computable from x^{t-1} (sequential code, pays through early
predictions from few observations). All codelengths in the paper are
sequential; redundancy = the cost of closing this gap. This also gives
"sequential" its explicit definition in Section 1, which the rest of
the paper now relies on.

## 2026-07-27 — Section 1: model definition spelled out in three steps

Per Ruediger: the opening of Section 1 now defines the model explicitly
instead of in one sentence. Three labelled steps:
1. The prior (new eq:product-prior): theta_i = prod_l theta_i^(l) /
   normalization, L iid uniform simplex points; law = pi_L; L=1 uniform,
   larger L concentrates on sparse low-entropy vectors.
2. The mixture over sequences (new eq:mixture): data are sequences of
   length N drawn iid from theta given theta; Q_N^(L)(x^N) = int
   prod_t theta_{x_t} pi_L(dtheta), normalized over all d^N sequences;
   conditionals = posterior means E[theta_i | x^t]; L=1 gives exactly
   add-one (m_i+1)/(t+d); exchangeability (depends on counts only)
   stated here.
3. The average over depths (existing eq:depth-average).
Compiles cleanly (18 pages).

## 2026-07-28 — GAME PLAN agreed + Phase 0 executed: first real bpc number

Game plan (now the "Roadmap" paragraph in Section 6 of the paper):
- Phase 0 (DONE): full-fidelity accounting. Escape mechanism =
  spell each distinct word once at first occurrence; conversion
  bpc = (X_token_bits + 0.202)/5.8806.
- Phase 1 (cluster, code exists): state-family + ctree at V=16k/65k/full
  vocab; KT-leaf ablation; D=3. Targets: X=9.21 beats ppmd 1.60;
  X=8.03 beats paq8h 1.40. First-order needs learning cost <1.9 (ppmd)
  / <0.75 (paq8h) over in-sample H(x'|x)=7.286.
- Phase 2 (code then compute): sequential pooled evaluator
  (checkpointed expert tables, per-step normalization, weight grids,
  gating, doubling bands) + test suite; measures how much of the
  alpha_t~1 tail survives lag overlap.
- Phase 3: assembled single code; report whole-file bpc + last-5%|95%.

Phase 0 measurements (scripts/spelling_experiment.py, run in cloud,
output/spelling_text8/results.json, 272 s):
- Spelling stream: 253,854 words, 2,237,382 chars, 27 symbols,
  first-occurrence order. Char context-tree family (depth<=3):
  3.4312 bits/char (fixed depths 4.284/3.836/3.596/3.454).
  Total 7.68 Mbit = 0.4514 bits/token. Higher than the 2-3 b/char
  estimate: rare words are names/foreign words + char model pays its
  own learning cost on 2.2M chars.
- Index-assignment correction (fixed-alphabet token code pays for a
  SPECIFIC unseen symbol at each first occurrence; only "new symbol"
  event is needed; symmetric unseen mass => saving log2(u)):
  4.24 Mbit = 0.2496 bits/token.
- Net full-fidelity surcharge = 0.2018 bits/token.
- **Full-fidelity memoryless = (10.895 - 0.2496 + 0.4514)/5.8806
  = 1.8870 bits/char** — first number that is the length of a code for
  the file. Below bzip2 2.11, above ppmd 1.60, zero memory.
Paper: new "The spelling cost, measured" paragraph; "Prospects"
rewritten around the conversion formula and X targets; "Roadmap"
paragraph added. 19 pages, compiles cleanly.

NOTE (cloud disk): old table caches were deleted to free space
(output/*/cache in cloud). The spelling run's cache remains in cloud.

Next up (Phase 2 build starts now unless redirected): sequential pooled
evaluator + tests; Phase 1 cluster prep (distributable table cache,
checkpointable runs) folded in.

## 2026-07-28 — KT-leaf ablation implemented; laptop run queue

New: context_tree.py gains kt_log2_q(profile, V) (closed-form
Dirichlet-1/2) and leaf_model="layered"|"kt" in
context_tree_codelengths; script flag --leaf-model kt. KT needs NO
moment tables — runs in minutes. Tests added (closed form ==
telescoping sequential product to 1e-12; KT tree finds order-2 on toy
source); 2 new tests pass + existing ones.

LAPTOP RUN QUEUE (in order):
1. KT ablation, both scales (~5 min each, no cache):
   python scripts/context_tree_experiment.py --corpus data/text8 \
     --top-k 1023 --depth 2 --leaf-model kt --out output/ctree_v1024_d2_kt
   python scripts/context_tree_experiment.py --corpus data/text8 \
     --top-k 4095 --depth 2 --leaf-model kt --out output/ctree_v4096_d2_kt
   Comparison rows: layered family was 4.9507 (V=1024) / 6.6807 (V=4096).
2. FULL-VOCABULARY first-order (the ppmd test; overnight, est. 3-7 h):
   caffeinate -i python scripts/state_family_experiment.py \
     --corpus data/text8 --top-k 300000 --m-grid 0,253854 --jobs 20 \
     --out output/state_family_fullvocab
   --top-k above the type count => UNK unused, V=253,854; M=V member =
   true first-order full-vocab code; M=0 = memoryless cross-check
   (expect ~10.89, small diff vs Sec 3's d=2^18 is the alphabet size).
   WATCH the "unique profiles" line early: if >500k, abort (cluster
   job instead). Disk: table cache ~10-20 GB expected.
   Payoff: X for first order; bpc = (X+0.202)/5.8806; X=9.21 beats
   ppmd, X=8.03 beats paq8h.
3. V=16,384 ctree D=2 (est. 2-3 h), after #2's verdict.

## 2026-07-28 — KT ablation V=1024 D=2: layered leaves worth 0.134 bits/token

`output/ctree_v1024_d2_kt/results.json` (29 s — closed form, no tables).
KT: d0=6.0636 (= layered to 4 decimals), d1=5.1123 (+0.056),
d2=5.6164 (+0.486), family=5.0849 (+0.134 vs layered 4.9507).
KT MAP splits only 4 contexts (2,038 depth-2 leaves) vs layered's 57
(26,506): the leaf prior determines how much memory the data can
afford. KT family is worse than the LAYERED FIXED depth-1 code. Now
Table tab:kt-ablation in Section 5.2. Companion-paper comparison
(layered vs KT on sparse targets) reproduced inside a memory model.
Pending: KT run at V=4096 (same command, --top-k 4095) and the
full-vocab first-order overnight run (queue item 2).

## 2026-07-28 — KT ablation V=4096; full-vocab run vetted (26k profiles)

KT V=4096 D=2 (`output/ctree_v4096_d2_kt`, 40 s): d0=8.0524, d1=7.0871
(+0.374), d2=8.6236 (+1.200), family=7.0873 (+0.407 vs layered 6.6807).
KT MAP splits ZERO contexts (MAP = complete depth-1): with KT leaves
second-order memory is worth nothing at V=4096, where layered recovers
0.032 from 24 contexts. Gap grows with V: family +0.134 -> +0.407.
Table tab:kt-ablation now two-scale.

Full-vocab first-order run vetted from the cloud (user killed his run
early for unrelated reasons; the older state_family script does not
print the "unique profiles" line — that was my error in the
instructions): measured directly on text8: 253,854 states, 4,146,848
distinct pairs, **26,164 unique profiles**, 1,661 distinct r values
(first-order member alone). Small! Table build was ~4,779 orders (~25-30
min, resumes from cache on restart); evaluation est. ~45-60 min at
L_max=59. Total well under 2 h on 20 cores. SAFE to restart same
command; cache in output/state_family_fullvocab/cache persists.

## 2026-07-28 — Full-vocab first-order: X=10.0909, full fidelity 1.7503 bpc

`output/state_family_fullvocab/results.json` (106 min, 26,164 unique
profiles). M=0: 10.8942 (matches Sec 3's 10.8952 up to alphabet size);
M=V=253,854: **10.0909 bits/token**, posterior 1.0 on M=V. Full
fidelity: (10.0909+0.2018)/5.8806 = **1.7503 bpc** — first
full-fidelity number WITH memory. Tally: 1.8870 memoryless -> 1.7503
first order; 0.15 bpc to ppmd.
KEY DIAGNOSIS: sequential capture of in-sample memory value falls to
22.5% (0.8033 of 3.5700); learning cost 2.8046 bits/token — the
complete first-order map misses BOTH thresholds (1.9 ppmd / 0.75
paq8h). Complete map = blunt instrument: every word buys its own row.
Paper: new subsubsection "The endpoints at the full vocabulary"
(sec:state-family-endpoints) in Sec 5; measured paragraph + Prospects
updated (burden now on selective structures). 20 pages, compiles.

NEXT LAPTOP RUN (cheap — tables cached; est. 1-2 h): intermediate M at
full vocab; first plausible INTERIOR optimum of the first-order family:
  caffeinate -i python scripts/state_family_experiment.py \
    --corpus data/text8 --top-k 300000 \
    --m-grid 1024,4096,16384,65536 --jobs 20 \
    --cache-dir output/state_family_fullvocab/cache \
    --out output/state_family_fullvocab_mid
Then: V=16,384 ctree D=2, and (cluster or patient laptop) ctree D=2 at
full vocab — the tree can decline rows that can't pay.

## 2026-07-28 — Lab SLURM cluster: plan + job files

Lab cluster (ipgdoc.epfl.ch/doku.php?id=slurm-dummies): partitions
slurm-cluster (use this) / slurm-gpu / slurm-ws (<=1h); mandatory
sbatch directives incl. mail-user; RAM up to 250G; no walltime max.
Only nodes 15/16/17 currently on; specs TBD (user will send).

STRATEGY: phase one needs NO code changes — pending runs are all
single-node-sized; existing --jobs parallelism; one run per node,
three in parallel. Multi-node sharding (--shard/--merge) only when a
single run outgrows a node (D=3 / full-vocab D=2 if slow) — not yet
implemented.

New `cluster/` dir: README.md (rsync repo+text8 to cluster, venv via
env_setup.sh, submit, rsync results back into output/ so Claude can
read them), env_setup.sh, and three sbatch jobs:
  job_state_family_mid.sbatch   (intermediate-M full vocab — also fine on laptop)
  job_ctree_v16384_d2.sbatch    (V=16,384 D=2)
  job_ctree_fullvocab_d2.sbatch (full-vocab D=2 — the tree that can
                                 decline unaffordable rows; big, start last)
Placeholders to edit: --mail-user, -c cores, --mem once node specs known.

OPEN QUESTIONS for user: (1) node 15/16/17 specs (cores/RAM); (2)
python3 version on nodes + internet for pip (else wheel transfer path
in README); (3) is home NFS-shared across nodes + quota (table caches
can be tens of GB — scratch location if tight); (4) head node
hostname for rsync.

## 2026-07-28 — Cluster jobs sized for real nodes

Node specs received: node15 = 64c/124G (epyc7302, the workhorse);
node16/17 = 20c but SLURM RealMemory=7,644M only (physical ~64G —
config quirk, flag to admin; until fixed, jobs there need --mem<=6G).
GPU node iscpc88 is DOWN; not needed. Head: ssh lth.epfl.ch (submit
only); home shared across nodes. Jobs exceeding declared time/mem are
KILLED — limits set generously.

Job files finalized (all --constraint=epyc7302 -> node15):
  sf-mid:  32c/48G/1d  \  co-run on node15 (32+32c, 96G)
  ct-16k:  32c/48G/2d  /
  ct-full: 64c/120G/7d — whole node, starts when others drain.
Mail: ruediger@gmail.com. README rewritten with lth.epfl.ch workflow,
node table, memory quirk, co-scheduling note. Submit all three at
once; SLURM orders them. Workflow: rsync repo+text8 to lth:~/pmm,
bash cluster/env_setup.sh, sbatch cluster/job_*.sbatch; results rsync
back into output/ on the Mac for Claude to read.

Remaining unknown: python3 version on nodes + pip internet access
(offline wheel path documented in README if needed).

## 2026-07-28 — Admin instructions: anaconda workflow

Admin (damir): connect ONLY to lth.epfl.ch; plain rsync/scp for file
transfer; install python via `/usr/local/bin/install-anaconda3` on lth
(one-time, then logout/login). Anaconda includes numpy+scipy — the
pip/internet concern is gone. Updated: job files now use
$HOME/anaconda3/bin/python (fallback python3), no venv;
env_setup.sh is now just a version check; README rewritten
accordingly. Cluster onboarding steps: rsync repo+text8 ->
install-anaconda3 -> bash cluster/env_setup.sh (check) ->
sbatch the three jobs.

## 2026-07-28 — Cluster anaconda is python 3.6.4 (too old); workaround in place

User's env check on lth: python 3.6.4 / numpy 1.14 / scipy 1.0 —
the cluster's install-anaconda3 ships Anaconda ~5.1 (2018). Our code
needs >=3.7 (from __future__ import annotations). Fix: create modern
env inside it: `conda create -y -n pmm python=3.11 numpy scipy`.
Job files now auto-select the first usable python among:
anaconda3/envs/pmm -> miniforge3/envs/pmm -> /usr/bin/python3 (RHEL9
=> 3.9, usable if numpy present) -> anaconda base; hard error if none.
env_setup.sh now reports all candidates with USABLE/not-usable lines.
Plan B if conda create has no network: Miniforge downloaded on Mac,
scp'd over. Awaiting user's conda create result.

## 2026-07-28 — Git-based cluster deployment; text8.gz bundled in repo

Switched cluster deployment from rsync to git (existing repo
git@github.com:urbanke/product_model_with_memory.git). IMPORTANT:
git must NOT be run via the device bridge (cannot unlink lock files;
a probe left a stale .git/index.lock the user removed). User runs
git on Mac/cluster himself.
Data in git: raw text8 (100MB) sits at GitHub's file cap, so the repo
carries data/text8.gz (~32MB) instead; .gitignore now data/* with
!data/text8.gz; get_text8.py prefers (1) existing data/text8,
(2) bundled data/text8.gz decompressed offline, (3) download.
User creates the gz on Mac once: gzip -k data/text8.
Cluster onboarding after clone: python scripts/get_text8.py, then
sbatch the three cluster/job_*.sbatch. GitHub auth from lth via
ssh key (Settings -> SSH keys) or repo deploy key.

## 2026-07-28 — Cluster job 2160 failed: missing PYTHONPATH; fixed

First cluster submission failed at import (line 24,
product_model_with_memory not installed in cluster anaconda). Fix:
job files now `export PYTHONPATH="$SLURM_SUBMIT_DIR/src..."`.
Cluster clone lives at ~/product_model_with_memory (not ~/pmm);
README paths updated. Loop: user commits+pushes on Mac, pulls on
lth, resubmits.

## 2026-07-28 — Three cluster jobs RUNNING; fetch_results.sh added

sf-mid + ct-16k + ct-full submitted successfully and running on
node15 after the PYTHONPATH fix. Result retrieval:
`bash cluster/fetch_results.sh` on the Mac pulls results.json +
slurm-*.out from lth (never the caches). Outputs are self-describing
(dir name + params inside results.json) — after fetching, the user
just tells Claude, who scans output/ for new results.json.
Expected order: sf-mid (intermediate-M full vocab — interior optimum
question) and ct-16k first, ct-full (whole node, 7d limit) after.

## 2026-07-28 — fetch_results.sh v2: auto-discovers remote repo

User's cluster username is urbanke (not the Mac's ruediger); repo on
lth is NOT at ~/product_model_with_memory (first fetch failed with
"No such file or directory"). Script now hardcodes
REMOTE_HOST=urbanke@lth.epfl.ch and discovers the repo dir by ssh,
trying ~/product_model_with_memory, ~/Projects/product_model_with_memory,
~/pmm (first with an output/ dir wins), then rsyncs results.json +
slurm-*.out only.

## 2026-07-28 — Evaluation phase parallelized (was single-core)

User caught ct-full using ONE core in evaluation: jobs was only wired
to table building; the per-level profile loop ran serially in the
parent. Fix in codelength.py: depth_averaged_codelength_profiles now
splits each level's profiles into one chunk per worker (sorted by
size, striped, for balance); workers open the already-built cache in
an initializer and mmap only the current level; default mp start
method (fork on Linux, spawn on macOS — matches build_tables_fast).
Serial path kept for jobs=1 / tiny profile sets. Verified bit-for-bit
identical to serial (60 profiles, d=1024, L_max=14) + regression test
tests/test_parallel_eval.py; existing tests pass.
RESTART ADVICE for ct-full (job 2165): worth it — scancel, pull,
resubmit; tables are cached, profile-building (~45 min, inherently
serial pass) re-runs, then eval uses all 64 cores (~5h serial eval
becomes minutes-to-tens-of-minutes). sf-mid/ct-16k nearly done; let
them finish serial.
KNOWN REMAINING SERIAL PHASES: corpus profile building (Python dict
pass), and the "orders built" table build parallelizes already.

## 2026-07-28 — Parallel eval v2: NFS-safe shared-memory design

v1 stalled on the cluster (job 2167: CPULoad ~8, no progress at 13
min). Root cause FOUND IN CODE, not assumed: each worker's initializer
called build_tables_fast, whose `missing` check np.load()s ALL ~4,936
per-r table files IN FULL — several GB per worker x 64 workers over
NFS. v2: workers do NO file I/O at all; the parent reads each level
once (the pattern job 2165 proved works on this NFS) and shares rows
via multiprocessing.shared_memory; workers attach untracked
(track=False on py>=3.13, resource_tracker.unregister fallback) so
shutdown is clean. VERIFIED locally: bit-identical to serial (60
profiles, d=1024, L=14), no tracker warnings, 20 tests pass.
NOT YET VERIFIED: actual behavior on the cluster NFS — check
observables after resubmit: (a) time between "evaluation: depth"
lines (serial was ~190 s/depth; expect a few s/depth; >30 s/depth =
investigate), (b) CPULoad during eval near worker count.
NOTE from user (process): be methodical, verify rather than assume —
applies esp. to environment-dependent performance claims.

## 2026-07-28 — Measured table geometry; header-only cache scan

COMPUTED (not assumed): full-vocab D=2 cache is G=36,354 grid points
-> 1.44 GB PER LEVEL, 85 GB total. Consequence found in code: the
warm-start completeness check (_load_cached full np.load per r) reads
all 85 GB in the parent before evaluation -> the silent 5-15 min
window after the contexts line in job 2173. Fix: _cached_levels()
header-only scan via mmap (truncated files detected at mmap creation
-> rebuilt); missing-check now header cost. Tests:
tests/test_cache_scan.py (warm scan <5s + truncation detection +
agreement with full load); 23 tests pass total.
Expected timeline for resubmitted ct-full: contexts (88s) -> scan
(seconds) -> depth 5 line within ~2-4 min -> ~10-30 s/level after
(1.4 GB NFS read + eval per level). FAILURE CRITERION: >60 s/level
sustained, or no depth line 10 min after contexts.

## 2026-07-28 — Three cluster results in. NEW HEADLINE: 1.6979 bpc

1. sf-mid (34 min): INTERIOR OPTIMUM. Full first-order curve over
   M in {0,1024,4096,16384,65536,253854} =
   {10.8942, 9.8316, 9.7826, 9.8682, 10.0060, 10.0909}.
   M*=4096, posterior 1.0. Pooling rare words into ONE backoff row
   beats the complete map by 0.3083 b/t. X=9.7826 ->
   (X+0.2018)/5.8806 = **1.6979 bpc**. Gap to ppmd: 0.098 bpc
   (X target 9.2072; 0.575 b/t to go).
2. ct-full (29 min): tree at full vocab: d2 catastrophic (13.0817),
   family 10.0919 ~ complete d1 (slightly above: split prior cost);
   DEPTH IS THE WRONG AXIS at full vocab. Cross-check: ctree d1 vs
   state-family M=V agree to 5e-7 (independent programs).
3. ct-16k (43 min): family 8.2383 (beats fixed d1 by 0.0135, 49
   splits), capture 57.6% (trend 92.9 -> 78.2 -> 57.6).
Paper: tab:context-tree-large, tab:interior-m, subsubsections added;
Sec 6 tally + Prospects updated; Roadmap Phase 1 marked largely done.
21 pages, compiles.
NEXT (direct product of these measurements): context tree over the
POOLED alphabet (b_{M*} applied to history: depth on top of ~4k
affordable contexts, full-vocab emissions) + multi-tier backoff
(several pooled tiers, not one). Then Phase 2 (pooled lag evaluator).
Parallel eval + header-scan now proven on cluster (21 s/depth vs 190).

## 2026-07-28 — Phase 2 v1 built: pooled lags, BOTH rules (mixture + tempered product)

User decision: implement both pooling rules, let the data decide.
New module src/product_model_with_memory/pooled_lags.py:
- experts: memoryless + one per lag; rows = counts smoothed toward
  checkpointed unigram (alpha=1); tables frozen at C checkpoints
  (valid sequential code; staleness measurable via C).
- mixture rule p = sum lambda_e p_e (latent-switch story); grids:
  lambda ~ powerlaw(a) over lags x share s x memoryless m + one-hots.
- tempered product p ~ q0 prod (p_d/q0)^beta_d normalized per step
  (conditional-independence story); beta = b*(1+i)^-c grids + one-hots.
- one grand uniform family over all members; posterior reported.
- perf: per-chunk gather cache + ThreadPoolExecutor over beta members
  (numpy releases GIL); step_chunk auto-capped for memory.
Tests (tests/test_pooled_lags.py, 5, all pass): exact brute-force
agreement BOTH rules; onehot-mix == onehot-prod identity; staleness
decreases with C; switching source: pooling beats best single; family
tracks best member.
SMOKE (2M tokens, V=256, lags 1,2,4, C=8; output/pooled_smoke):
mem 4.7613, lag1 4.1419, best product 4.1259 (b=0.66,c=2 — strongly
tempered => lags NOT cond. independent), best mixture 4.0821 (a=3,
s=1) — mixture wins round 1, both beat best single expert.
CALIBRATION: 2-core cloud 233 s for that smoke; projected V=4096
full-corpus 6-lag both-rules: ~17 h single-thread-equivalent ->
cluster/job_pooled_v4096.sbatch (32 threads node15) est. 1-2 h —
PROJECTION, not yet measured on cluster.
User to run: commit/push, pull on lth, sbatch cluster/job_pooled_v4096.sbatch.
NOT yet: gates (context-dependent weights), layered checkpoints
(currently simple smoothing), full-vocab pooled emissions. Paper: not
yet updated with pooled-lag machinery (write section after V=4096 run).

## 2026-07-28 — Pooled-lag V=4096 result: SUMS WIN; pooling worth 0.51 b/t

output/pooled_lags_v4096/results.json (68 min, node14, 32 checkpoints,
lags 1,2,3,4,6,8, 114 members). Key members (bits/token):
mem 8.1727 | lag1 alone 7.3593 | best product untempered (beta=1)
12.4894 (!) | best tempered product (b=0.66,c=2) 7.1702 | BEST: sum
a=2,s=0.8,m=0.02 -> 6.8491, posterior 1.0.
Findings: (1) pooling gain = 0.5102 over best single lag — first
measurement of tail-after-overlap, substantial; (2) sums beat tempered
products by 0.32; RAW INDEPENDENCE CATASTROPHIC (4 bits worse than no
memory): lags are heavily redundant witnesses; (3) winning weights:
72% prev word, 13% two-back, 6% three-back, 9% tail+mem — decay ~
delta^-2, faster than I(delta)~1/delta (redundancy squares it).
CAVEAT (stated in paper): experts here are crude checkpointed count
tables — lag1 alone 7.3593 vs layered 6.7132 (gap 0.6461). Absolute
numbers NOT comparable to layered tables; measurement = pooling value
+ rule choice. NEXT (Phase 2 v2): refresh expert tables from LAYERED
predictive (if the 0.51 gain transfers onto 6.7132 -> ~6.2-6.4, below
the in-sample first-order ceiling 6.3376 becomes possible); then
gates; then doubling bands to extend horizon.
Paper: new subsection sec:pooled-lags + tab:pooled-lags (plain-English
style per user request); Roadmap Phase 2 updated. 22 pages, compiles.

## 2026-07-28 — Pilot REMOVED from paper (user decision); layered pooled experiment built

User was right to object: the count-table experts were a silent design
substitution — estimator choices are HIS from now on. Pilot section
removed from paper (21 pages, compiles; roadmap reverted to agreed
design). Pilot results remain in output/pooled_lags_v4096/ for
reference only.

AGREED EXPERIMENT now implemented: expert_model="layered" in
pooled_lags.py — per-row predictive = q_avg(profile+e_x)/q_avg(profile)
via per-level memoized evaluations (one per distinct count value per
row + unseen; saturated-row edge case handled); rows renormalized
(absorbs ~5e-5 moment-table integration error). Both rules + grids
unchanged. Tests (tests/test_pooled_lags_layered.py, 4 pass):
row normalization; TELESCOPING IDENTITY (per-step-refreshed memoryless
expert == core-machinery codelength of unigram profile, to 1e-4/token
= integration accuracy); onehot mix==prod; layered beats counts
expert on sparse Markov source.

COST, measured not guessed (exact workload count on text8 + timed
sample calibration): V=1024/C=32: 5.86M layered evals ~ 1-3 days
node14. V=4096/C=8: 2.98M ~ 1.5-3 days. V=4096/C=32: 12.9M ~ week+.
Cause: rows re-evaluated at every refresh as counts grow.
Job files (both -w node14, 200G, 64c, 4d limit):
  cluster/job_pooled_layered_v1024.sbatch  (recommended first)
  cluster/job_pooled_layered_v4096.sbatch  (C=8)
Old cluster/job_pooled_v4096.sbatch (counts pilot) superseded — user
may git rm it. AWAITING user's choice of configuration.

## 2026-07-28 — paper/complexity.tex created: computational anatomy

User request: technical inventory of WHAT quantities we compute (not
run retrospectives), as basis for finding radically faster methods.
New standalone doc paper/complexity.tex (+pdf, 5 pages):
Q1 the atom: q_L(profile) via G-point quadrature of product of moment
  tables; cost O(L*G*ktilde) per profile. ROUTES: (1) SHARED-INTEGRAND
  predictives: all row ratios from ONE integrand Psi_L(u) -> per-row
  O(G ktilde^2)->O(G ktilde), exact, ~100x on rich rows — TOP LEVER
  for pooled runs; (2) incremental integrand updates across refreshes;
  (3) saddle-point (Laplace) for large profiles, checkable error;
  (4) certified level truncation (L_eff~5-10).
Q2 moment tables phi_{L,r}(u): build O(R L G) parallel/cached;
  routes: shared cache, level-major float32 files, large-r asymptotics.
Q3 counts/rows/profiles: hash counting O(P) MEMORY wall (200GB @ D=2
  full vocab); routes: external sort-merge, SUFFIX-ARRAY counting
  (unlocks D>=3), fixed-width arrays.
Q4 aggregation: trivial.
Q5a refresh multiplier: routes = Q1.1 + Q1.2 + GROWTH-TRIGGERED
  refreshes (refresh row when total grew by (1+eps): O(log n)
  refreshes/row instead of C; still valid sequential code).
Q5b product-rule normalizer: dense O(nVDK) — the only bad exponent;
  route: EXACT sparse+background decomposition (predictive = const
  background + sparse exceptions) -> Z_t in O(|union support| D).
Scaling scenario (10x corpus, V~1e6) + priority list in doc.
Meanwhile: pooled_layered_v1024 job running on node14 (cp3 ~20 min).

## 2026-07-29 — complexity.tex v2: fixed CS-style task template

Restructured per user request into 8 tasks, each with fields: Task /
Formula / Input-Output / Algorithm / Cost & scaling / Speedup routes
([exact] vs [approx] marked):
T1 profile probability (atom, quadrature) — saddle point, level
   truncation, adaptive grids.
T2 moment tables — shared cache, level-major float32, large-r
   asymptotics.
T3 predictive table of a row (ratios) — SHARED INTEGRAND (O(G k~^2)
   -> O(G k~), exact, top lever), incremental integrand.
T4 counting/dedup — external sort, suffix arrays, fixed-width arrays.
T5 aggregation — trivial.
T6 additive pooling pass — O(n D K), fine.
T7 multiplicative pooling pass — the V factor; EXACT sparse+background
   normalizer via power sums of q0: O(V D) -> O(|S_t| D) per step.
T8 refresh scheduling — growth-triggered refreshes: C -> O(log n)
   per row, past-measurable => valid code.
Scaling summary + priority list at end. 5 pages, compiles.

## 2026-07-29 — complexity.tex v3: proper definitions (user review round 2)

User: phi undefined, u scalar-or-vector unclear, tables must precede
atom, T2 incomprehensible. Fixed by reading layered.py/fast_tables.py
and writing the actual math: new Section 0 defines the layered prior
(W_x = prod_{l<=L} E_{x,l}, p = W/sum W; uniform over L<=L_max =
ceil(2 c* ln d), c*~2.365), profiles, q_L(m) = E[prod p_x^{m_x}],
the Gamma-identity integral representation with
phi^{(L)}_r(t) = E[Y^r e^{-tY}], Y = prod of L unit exponentials,
u = ln t SCALAR, log-integrand psi_L(u) (Eq. psi), boundary facts
(phi^{(1)} closed form; t->0 limit Gamma(r+1)^L; L=1 =
Dirichlet-multinomial). Order now: T1 tables (recursion
phi^(l)_r(t) = E[E^r phi^(l-1)_r(tE)], Gauss-Laguerre order Q=96,
cost O(|R| L G Q)), T2 atom (algorithm now honestly described:
grid sweep -> multimodal peak finding -> bracketed saddle + Laplace +
closed-form left tail), T3 predictive (shared-integrand formula
written out), T4-T8 as before with all inputs defined. New T2 route:
warm-started peaks / local grids (G -> ~100, exact, fallback sweep).
7 pages, compiles. Corrections vs v2: algorithm was scan+Laplace not
plain quadrature; build cost includes Q.

## 2026-07-29 — complexity.tex v4: plain-English rewrite (user review round 3)

User: "how often each next-symbol followed a given context" is not
proper English and unclear; write plainly and mean it; also why a
Section 0. Full rewrite: Definitions is now numbered Section 1 with
\label{sec:defs}; every term defined in complete sentences before use
(data/symbols; states via rule sigma with the three concrete examples;
count rows via m_x(s) = #{t : s_t = s and x_t = x} explained in
words; profiles with worked example (0,3,0,1,1)->(1,1,3), ktilde and
multiplicities on the example; estimator; q_L as expectation with
exchangeability explained; Gamma-identity derivation; phi defined in
words; u scalar stated; psi; the two boundary facts). All T1-T8 prose
rewritten in plain sentences (no invented compounds); speedup ideas
now in full sentences ("Remember where the peaks are", "Compute all
the ratios from one integrand", "Update the integrand instead of
rebuilding it", "Refresh on growth", "Count on disk", suffix array
explained in one sentence). 9 pages, compiles clean.
Also earlier: paper/*.aux|log|out|toc now gitignored by pattern
(user hit committed complexity.aux etc.).

## 2026-07-29 — complexity.tex v5 (user review round 4)

User: position range of t unstated; count-row set comprehension had
no range for t; (discovered in fixing: "we write s for nonzero
entries" collided with s = state). Fixes: States paragraph now says
t in {1..n}, notes the rule is undefined at the first positions,
defines the starting position t_0 (e.g. t_0 = delta+1 for lag rule),
and states all models in one experiment code the same t range.
Count rows: explicit t in {t_0..n} inside the set; symbol collision
removed (k = nonzero entries, N = row total, consistent with symbol
table); added worked example (the row of the word "the" under the
previous-symbol rule). T6 sum now t = t_0..n with t_0 = 1+max lag.
9 pages, compiles.

## 2026-07-29 — complexity.tex v6 (user review round 5: formal definitions)

User: t_0 appeared in m_x(s) without being part of any notation; s
ambiguous (element of undefined state set vs the function s_t); s_t's
dependence on the sequence unstated. Rewrite: alphabet A introduced
as a set (|A| = d); memory rule = (S, sigma) with S the finite state
set, three concrete rules (a)(b)(c) each with its S; s_t :=
sigma(x_1..x_{t-1}) with explicit statement that it depends on
positions 1..t-1 only and NOT on x_t (prediction must not peek);
t_0 = first position where the rule is defined, with values for
rules (b) and (c), coded range t_0..n stated BEFORE m_x(s) uses it;
m_x(s) defined for s in S with the range inside the set-builder;
worked example (row of state "the" under rule (a)). Symbol table
gains A and (S, t_0) rows. 9 pages, compiles.

## 2026-07-29 — complexity.tex v7 (user review round 6: the model)

User: estimator paragraph incomprehensible — estimates what? where is
the averaging? where is the whole length-n sequence? Rewrote as three
fields: (1) "The probability model": given known vectors p(s) per
state, symbols are independent draws from p(s_t); probability of the
CODED SEQUENCE factorizes over states (eq:factorize); vectors
unknown -> drawn independently per state from a prior and AVERAGED
(first averaging, eq:corpusprob); total length = -log2 of it = sum
over states of per-row terms; explicit remark that the model assigns
a probability rather than estimating a quantity, with the word
"estimator" justified via the T3 predictive. (2) "The prior": Pi_L
via product-of-exponentials weights; Pi = uniform average of Pi_L
over L<=L_max (second averaging, explicit); each state's vector ~ Pi
independently. (3) "The central quantity": q_L as expectation under
Pi_L, exchangeability => profile-only, q_avg, and the displayed total
codelength sum_s -log2 q_avg(profile(m(s))). Remaining "estimator's"
usages renamed to "model's". 10 pages, compiles.

## 2026-07-29 — Mellin prototype built; DISCOVERED: production tables
## inaccurate for large counts (r >~ 600) in the far-left u region

Prototype (user-approved, hooks-first) for the universal-table design:
src/product_model_with_memory/mellin.py — three independent methods:
(A) production recursion+Laguerre tables, (B) Mellin-Barnes contour
integral + certified small-t series (self-certifying dispatch,
certificate = first-omitted-term/total < 1e-12), (C) closed-form
saddle approx (order 1/2 with F''''/F''' correction).
Tests (tests/test_mellin.py, 5 pass): contour==L=1 closed form (1e-8),
contour==t->0 limit, saddle error decreasing in r, log-convexity in
r, derivative identity phi_{r+1} = -d phi_r/dt.
Contour pitfalls found+fixed: pole regime (small t) = catastrophic
cancellation -> series; width/sampling regime split.

MEASUREMENT (scripts/mellin_prototype.py, output/mellin_proto):
- saddle order-2 vs reference: median 1e-5..4e-3 nats falling with r;
  by L: worse for small L (L=2 median 1.3e-3), excellent for L>=16;
  by u: worst at far-left (covered by series anyway) and right edge.
- residual (ref - saddle2) interpolation on geometric r-ladder:
  median 1.7e-7, p90 1.5e-4 nats -> correction-grid design viable.
- *** TABLES vs reference: errors grow with r: max |err| 2.19e2 nats
  at r=610, 1.15e3 at r=1000, 1.8e4 at r=4000; confined to the
  far-left/plateau u region (u <= -5), worst at SMALL L (L=2: flat
  -6000 nats plateau at r=4000); cause identified: fixed-order (Q=96)
  Gauss-Laguerre cannot represent v^r integrand once r >> largest
  node (~350); error compounds per level. Also ubiquitous ~1-4 nats
  errors at right edge u>=20 (likely harmless: integrand negligible
  there). Test-harness artifacts ruled out (production grid spec used;
  uniform-grid and u_min artifacts were separately found and fixed).
IMPACT: UNKNOWN YET. Heavy rows (counts >600: frequent words) read
these regions via the multimodal far-left peaks of psi. NEXT STEP
(before any conclusions about published numbers): impact probe —
recompute q_avg for real heavy profiles (e.g. biggest V=1024
first-order rows; d=2^18 unigram profile) with mellin-corrected
columns, compare vs production values, translate to bits/token.
FIX PATH: build large-r columns from series/saddle/contour instead of
Laguerre recursion == exactly the universal-table hybrid design.

## 2026-07-29 — Impact probe built; needs cluster (cloud box = 8GB, OOM)

Probe design (scripts/impact_probe.py): three real heavy text8
profiles at V=256, production l_max=26, production grid (26,200 pts,
[-452, 35]) — memoryless profile (includes r up to ~5M via <unk>),
row of the top state, row of rank-100 state. |R|=1,602 needed count
values (1,489 >= R_SWITCH=500). Side A: production recursion tables +
production scan. Side B: SAME tables with all columns r>=500 replaced
by certified mellin columns (mellin_columns_batch: vectorized
bisection saddle order-2 + certified series; spot-checked vs contour
at <=1.3e-3 nats). Same scan, same grid: delta isolates the table
error. Reports per profile: bits A, bits B, delta, delta/n
(bits/token-equivalent), per-level deltas.
Cloud attempts OOM-killed (dict alone 8.7GB, box has 8GB; two
restructures: side-A-first + in-place mutation still short).
=> cluster job: cluster/job_impact_probe.sbatch (40G, 4c, any node,
~30-60 min). mellin.py gained log_phi_column + mellin_columns_batch
lives in the probe script.
READ THE RESULT: delta_bits per profile; if |delta|/17M << 1e-4
bits/token summed over plausible heavy-row counts -> published
numbers safe; else identify affected runs. Note deltas expected
NEGATIVE-or-positive: corrected q larger where far-left peaks were
suppressed -> bits_B < bits_A (delta<0) means production OVERSTATED
codelengths.

## 2026-07-29 — IMPACT PROBE VERDICT: published numbers STAND

Probe ran on user's laptop (96 min, single-core — mea culpa, no
--jobs). Results (output/impact_probe/results.json):
  memoryless (75.2M bits): delta +0.73 bits  = +4.3e-8 b/t
  row:<unk>  (33.7M bits): delta +0.26 bits  = +1.5e-8 b/t
  row:while  (44k bits):   delta -0.0003 bits
=> table error (100s-10,000s nats in far-left columns at r>~600) is
NEGLIGIBLE in results: the corrupted region is where these profiles'
integrals carry no mass (dominant peaks elsewhere; exact left tail
covers the extreme left). NO CORRECTION RERUNS NEEDED; all published
numbers valid at reporting precision.
Fix remains necessary (other regimes could weight that region; the
universal-table design builds on certified values), but it is now
hygiene + efficiency work, not damage control. User killed the big
cluster run earlier; plan = efficiency pause (hybrid builder, shared
integrand, refresh-on-growth, sparse normalizer), then restart
workflows once, on the faster foundation.

## 2026-07-29 — UNIVERSAL TABLE STORE v1 implemented (core done,
## integration into the evaluator PENDING)

User approved: permanent universal table, per-level files, grow on
demand ("check if exists at computation start; build if missing").
src/product_model_with_memory/universal_tables.py:
- Design constants: H=0.02 uniform master u-grid, U_MAX=35, L_MAX=70,
  R_SPLIT=256, columns start at series boundary (tau_log margin -8);
  location: PMM_UNIVERSAL_TABLES env var or ./tables/universal_v1
  (tables/ gitignored — artifact, not source).
- Hybrid builders BY MEASURED REGIME: r<=256 classical recursion
  (its clean range — r=512 was already 118 nats off at far left, so
  split moved 512->256 after certification caught it); r>256
  certified series + order-2 saddle (mellin_saddle_column).
- log_phi(L, r, u): series on the fly left of stored column,
  interpolation inside; L=1 closed form never stored.
- certify(): random spot checks vs contour on cert domain (u<=10),
  appended to manifest. Current: median 1.2e-6, max 1.3e-2 nats
  (fringes documented in manifest accuracy note).
- Tests tests/test_universal_tables.py (4, pass): grow/persist/
  reopen; series handoff; both regimes vs contour; certification.
DEAD END DOCUMENTED: exact_column_recursion (quantile-trapezoid
generalized recursion) in mellin.py FAILS at deep L + large t (its
quadrature span misses the peak reshaped by the shifted phi factor;
established by bound/typical-path argument vs contour at
(r=34,L=32,u=10): recursion -495 vs correct -259). Do NOT use it as
a builder; kept for reference. v1 accuracy target relaxed 1e-6 ->
certified ~1e-2 worst / 1e-6 median (paper-precision safe per impact
probe); path to 1e-6 everywhere = v2 (right-pole series, order-3
saddle).
NEXT (first thing next session): integrate — 
depth_averaged_codelength_profiles(universal=...) building per-level
ProductMomentTables adapters from the store (dict-like lazy columns),
regression test vs legacy path on small d, THEN experiments start
using ensure_universal_tables() and per-run caches die. After that:
T3 shared integrand, T8 refresh-on-growth, T7 sparse normalizer.

## 2026-07-29 (evening): universal table v2 --- accuracy upgrade
Files updated: src/product_model_with_memory/mellin.py (new: log_phi_right_series,
exact_log_phi_column; pole-aware contour sampling), universal_tables.py (v2: exact
builder for ALL r, per-column certified series boundary, degree-7 interpolation,
full-axis off-grid certification), tests updated/extended; test_pooled_lags now
explicitly requests the counts pilot estimator (pre-existing failure fix).
Measured on a 6-level x 16-count store (r up to 1e6), 900 random off-grid checks
vs the independent contour reference: median 7e-14 nats, max 2.4e-7 nats (the max
is the float64 representation floor at values of magnitude ~1e9 nats, not a
method error). v1 was: median 1.2e-6, max 1.3e-2.
Old v1 table dirs are incompatible; default path is now tables/universal_v2
(a v2 store builds itself on first use).

## 2026-07-30 (overnight): speedups, integration, verification prep
All experiments now read the permanent certified store by default
(tables_source="universal"; legacy cache via PMM_TABLES_SOURCE=cache).
New: shared-integrand family evaluator (complexity T3(1); verified to
reproduce member-by-member results exactly) wired into all prediction
tables; parallel universal builds (build_columns, jobs); strided column
builds with 8th-difference roughness control + spot-check fallback
(2-4x per core); pairs.py and single-profile codelength converted too.
complexity.tex rewritten (T1 = certified store; status section;
language discussion); VERIFY.md = runbook for the paper verification
reruns with timing readout. Whole test suite passes. Store-wide
certification: median 4.5e-13, max 1.0e-6 nats (one filled column in a
zero-weight region; builder tightened after). Legacy-table finding:
destructive errors from r~360 (not 600), ~5e-3 bits per heavy profile
at depth even at count ~300 -> reruns will quantify effect on paper.

## 2026-07-30 (day): T4 counting rewritten (sorted packed keys)
New module counting.py: (context, next-symbol) windows packed into
fixed-width integers, sorted; every context row is a contiguous run.
context_tree.py rewired onto it (array-based beta recursion).
Verified: equals the old hash counting exactly (synthetic + 1M-token
text8 slice, every depth); full text8 at FULL vocabulary depth 2:
4,400,703 contexts, 70,090 profiles (identical to the cluster run),
25 s, 2.6 GB peak (old: ~200 GB). Memory blocker for the parked runs
is gone; their remaining cost is table-column building (next: native
kernel / T7). New tests in tests/test_counting.py.

## 2026-07-30 (later): T7 sparse normalization implemented
pooled_lags: sparse per-row predictive tables (observed symbols +
one shared unseen value), sparse product-rule normalizer via power
sums T_gamma, sparse mixture gathers; eval_mode="auto" selects it
for V > 4096. Verified equal to the dense path to 1e-14 bits/token
on every grid member (incl. saturated rows via finite reference
value). New permanent test in test_pooled_lags_layered. Also earlier
today: shared-line quadrature in mellin.py (2-3x per column, exact),
measured on his laptop as ~1.25x on the DEEP-column tail (mixed-size
comparison). complexity.tex updated throughout (T1/T4/T7 status +
enwik9 plan: both prerequisites now done).

## 2026-07-30 (evening): right-series vectorized -- the real T1 fix
Profiling on deep columns (the tail of the V=256 build) showed the
point-by-point large-t series loop at 76-99% of build time; the
earlier contour saving had targeted the wrong part for that mix.
right_series_column vectorizes it; deep columns 0.5s -> 0.02-0.14s;
measured on the laptop: throughput ~20 -> ~250 columns/s (12 jobs).
Vector==scalar to 7e-10 with identical accept/reject; full battery
passes; fresh-store certify median 4e-13, max 1.5e-8. complexity.tex
updated (T1 numbers + enwik9 table step now under 2 node-hours; the
native kernel is no longer needed for T1). Laptop is 15 physical
cores / 24 GB; runbook set to --jobs 12.

## 2026-07-30 (night): parallel level fill in evaluation phase
Bug found by Ruediger watching cores: during evaluation the PARENT
serially read+interpolated every column per level while workers sat
idle (introduced with the universal integration; the old cache fill
was a cheap memcpy). Workers now fill their stripes of the shared
matrix (_fill_level_chunk; store opened read-only per worker).
Parallel==serial test passes. Runs 1+2 verified vs paper: state256
table identical to all printed digits (3.7198/3.7198/4.4234) and the
spelling constant 0.2018 confirmed (0.201849).

## 2026-07-30 (late): verification campaign + roadmap rewrite
Verified against the paper, ALL printed digits: state family V=256 &
V=4096, spelling block (0.202 constant), unigram full vocab (10.895
-> 1.8870 chain start), context trees V=16384 & full vocab (incl.
4,400,703/70,090 and MAP splits=49). Runbook errors found+fixed along
the way: runs 3/4 were mislabeled (ablation is at V=1024/4096; the
interior-M table is FULL vocab) -> runs 3b and 4b added. Parallel
fixes during campaign: worker-side level fill (twice: many-profile
and few-profile paths); pooled script print throttling. Run 8
(pooled v1024, the killed cluster run) started overnight - no
published target, produces the missing Phase-2 number. Roadmap in
main.tex rewritten: phases 0-2 status, Stage A (pooled-context trees
depth 3, multi-tier backoff, sequential best) targeting ppmd 1.60,
Stage B enwik8 (fidelity/tokenizer), Stage C enwik9 (one node);
sub-1.1 declared out of scope. Still open: 4b, 3b (x4), unigram_b,
run 8 result.

## 2026-07-31: refinement speedup (profiled) + pooled-run resume
Run 8 projection was 2-2.5 days; user killed it. Profiler (not
theory) located the cost: 81% in peak-refinement derivative calls
(~130k scalar full-grid interps per family). Fix: _local_derivative
-- per-bracket table slices, one vectorized interp over all parts.
Family eval 700ms -> 98ms (7x); agrees with generic to 2e-12;
independent references (wide-grid, Monte Carlo, bimodal) all pass.
A windowed-sweep attempt was REVERTED to off-by-default (missed a
far-left peak; no gain -- wrong bottleneck). pooled_lag_codelengths
gains resume_path: per-checkpoint state+memo on disk; kill-and-
resume verified EXACT vs uninterrupted. Script passes resume dir +
throttled printing. Run 8 to be restarted.

## 2026-07-31: cluster job 2177 failed - store integrity
Symptom: RuntimeError "left tail only" (converged=False) from a
worker at L=6. Reproduced the exact profile in the cloud with a
healthy store: base AND all 59 family members converge (saddle at
u=3.24). Laptop with locally-built store passed the same profiles
(now at checkpoint 7/32). Conclusion: the cluster store (rsync-ed,
likely while the job started) served truncated/garbage columns ->
monotone integrand -> no interior peak. Hardening added: index-vs-
data size check at level load, short/non-finite column read raises,
workers open the store READ-ONLY (a missing column now raises
instead of concurrent-appending and corrupting the file), and
verify_store() for checking a transferred store in one line.

## 2026-07-31: batched evaluation (step 1 of the GPU-shaped rewrite)
Profiled first: after the morning derivative fix, 41% of family time
was in the CURVATURE evaluation (_rho_prime scalar lookups) which I
had left on the old path, plus per-peak slice stacking and O(k)
multiplicity recounts. Fixes, all exact: local evaluator now serves
derivative+curvature+psi from one set of slices; ProductMomentTables
carries an optional CONTIGUOUS (R x G) matrix + row_of map and the
scan gathers from it; row indices/limits cached per profile;
augmented multiplicities derived in O(1). Measured: 40 families /
2100 members at L=21: 6.96s -> 2.47s -> 1.44s (4.8x), results
BITWISE identical to the independent plain scan (0.00e+00). Full
battery passes. The contiguous layout is the prerequisite for the
A100/H100 port (fp64 needed; Apple GPUs are fp32-only, so the Mac
GPU cannot be used).

## 2026-07-31 (cont): profiled the REAL refresh -- provisioning won
Measured per level (40 busy V=256 rows, 609 columns, run grid 20k):
  column reads 0.05s | interpolation 2.1s | far-left SERIES ~3.8s
  => provisioning 7.9s vs family evaluation 0.63s (ratio 12:1).
So after the morning batching, the dominant cost was preparing each
level, not evaluating rows. Two exact fixes: (1) series_column now
exits once every point s next term is below e^-45 (far left needs 1-2
terms, it was always running 60); (2) new log_phi_matrix serves many
columns onto one grid sharing the degree-7 stencil weights (all
columns lie on the same master grid, differing by an integer offset)
with a per-point fallback at clamped edges. Net: 5.97s -> 1.73s
(3.4x) for the same 609 columns; agreement 2.2e-11 nats (fp
reassociation, ~1e-16 relative). Full battery passes.

## 2026-08-01 morning: POOLED-LAG RUN COMPLETED ON THE CLUSTER
Job 2179 (node14, 64 cores, fresh store built by the job itself):
COMPLETED, wall clock 5h25m, CPU efficiency 63%, peak RAM 125 GB of
200 GB. This is the run that was killed on the cluster days ago for
its 120 GB cache and could never finish -> Phase-2 number is in.
NOTE it ran the code as of the pull BEFORE the 31 July afternoon
speedups (batched evaluation, provisioning). Laptop, running WITH
those speedups, was at checkpoint 25/32 after 8.4h (resumed at 15).
Cluster/laptop comparison is therefore machine-dominated, not
code-dominated: 64 cores finish the whole thing in ~5h while 12
cores need ~14h more for the remaining 7 checkpoints. Lesson: these
sequential runs belong on the cluster; the laptop is right for the
short static experiments.
DECISION: laptop run killed as redundant (same corpus/settings as
the completed cluster run); an end-to-end old-vs-new code check is
cheaper as a 4-checkpoint run when wanted.
STILL OPEN: ablation runs 3b (V=1024/4096 x kt/layered), 4b
(full-vocab interior-M, target 9.7826), unigram extra checkpoints.
NEXT BIG ITEM: eliminate the grid (T2(1)) -- see complexity.tex,
the sweep exists only to LOCATE peaks; predicting+certifying peak
positions turns O(G*k) into O(k) per profile.

## 2026-08-01: pooled-lag result written up
main.tex: new subsection "Pooling the lags: how much of the tail
survives" with tab:pooled-lags (memoryless 6.6392 | lag2 5.8967 |
lag3 6.1009 | lag8 6.2095 | lag1 5.2512 | best product 5.2300 |
best mixture = family 5.1874) and three findings: tail worth 0.064
b/token over lag1 (1.2%, an order of magnitude less than the 0.31
that CONTEXT pooling bought); data selects the ADDITIVE rule
(product trails 0.043, posterior mass 1.0 on one mixture member --
neighbouring lags too redundant, multiplication double-counts);
optimum sits ON the grid boundary (a=3.0 steepest decay, m=0.02) so
a wider decay grid is the cheap follow-up. Roadmap Phase 2 marked
done with its number. compress.tex results chain updated and flags
that the combination rule needs its own treatment -- Ruediger has a
more general scheme in mind (additive/multiplicative as special
cases); TO BE DEVELOPED, one thing at a time.

## 2026-08-01: byte-level baselines on all three corpora; paper cleanup
Measured (laptop, 51/121/206 s -- one profile each):
  text8  1e8 B, 27 values: H0 4.123527, layered 4.123534, red 7e-6
  enwik8 1e8 B, 205 values: H0 5.080140, layered 5.080163, red 2e-5
  enwik9 1e9 B, 206 values: H0 5.156490, layered 5.156493, red 3e-6
Memoryless byte coding of enwik9 = 644.6 MB vs 107.3 MB for the best
published program -- the floor, as intended. Methodological value:
redundancy ~1e-5 at all three scales shows the per-state prior costs
nothing in the DENSE regime, so later gains are attributable to
memory, not to the prior. Token vs byte on text8: 1.8870 bpc vs
4.1235 bpc -- the token representation is worth 2.24 bits/char
before any memory is spent; that is the argument for tokens.
main.tex: King James validation section (sec 2) REMOVED (it was
scaffolding); section 3 rewritten as "The three benchmark corpora:
memoryless baselines" with tab:byte-baseline, the token-level
paragraph, the n-sweep compressed to one parenthetical sentence
(2.187 -> 0.039, L* 15 -> 4), and a placeholder for the enwik8/9
token rows pending the tokenizer. Dangling ref in tab:pairs caption
repaired. 22 pages, compiles clean.

## 2026-08-01: tokenizer implemented (TOKENIZER.md v3)
src/product_model_with_memory/tokenizer.py: exactly invertible
segmentation (letter runs / digit runs / single bytes) with both
switches (numbers=intern|compositional, case=conditioned|folded).
Round trip verified on 10 hand-picked inputs incl. all 256 byte
values, UTF-8 and random bytes, x4 combos, plus text8 3MB x4.
Decoder is framing-self-sufficient (test withholds the encoder
vocabulary and still decodes). tests/test_tokenizer.py: 6 tests.
scripts/token_baseline.py: memoryless cost of every stream, with
the case stream reported BOTH independently and conditioned on the
current token (the number the spec argues about).
scripts/llm_token_baseline.py (NEW, Ruediger s suggestion): same
measurement over a standard LLM tokenizer via tiktoken -- one
stream, fixed vocabulary, no escape/spelling/case machinery, and
directly comparable to LLM perplexity work. Flagged in output and
in results.json as NOT a benchmark entry: the BPE vocabulary is
external and data-derived (would have to be shipped and counted,
~0.5-1 MB) and may have seen Wikipedia.

---

## Parked: one first-order state cannot serve a stream of two languages
*(31 July 2026 — measured, not yet acted on. `scripts/context_probe.py`,
outputs in `output/streams/*/context_probe.json`.)*

The question was whether first-order memory means anything on our
tokenizer, since 63.3% of its symbols on enwik8 are single punctuation
bytes, so a word's predecessor is usually a space or a bracket.

**It does not, for words.** Predicting a word from the previous *token*
buys 1.07 bits; from the previous *word* it buys 3.91.  A factor of
nearly four is lost on exactly the symbols that carry meaning.

**But changing the state map to skip delimiters barely helps overall.**
Splitting the conditional entropy by what is being predicted, in bits
per symbol of the stream:

| predicted | state = prev token | state = prev word | change |
|---|---|---|---|
| words and numbers (36.7%) | 3.725 | 2.684 | saves 1.041 |
| punctuation, and whether the next symbol is punctuation | 1.385 | 2.290 | loses 0.905 |
| total | 5.109 | 4.974 | saves 0.136 |

0.136 bits/symbol is 0.057 bits/character.  The gain on 37% of the
symbols is nearly cancelled by the loss on the other 63%, because
markup is a *local sequential* pattern — after `<` comes `/`, after a
tag name comes `>`, after a word comes a space — and the previous token
is exactly the right state for it, while the previous word is not.

**The conclusion to come back to:** the stream is two interleaved
languages, markup and prose, and no single first-order state serves
both.  Neither map is right.  Candidates, in the spirit of the project
(generate a family, let the posterior choose): both maps as members of
one family; a state carrying both the previous token and the previous
content token; or a context tree over this stream, which would discover
the distinction by itself.  Not decided — parked deliberately so that
the first-order section reports the plain model first.

Per-character first-order upper bounds from the same probe (plug-in, so
optimistic, and most optimistic where the state space is largest —
trust the bytes row, which is essentially unbiased):

| representation | memoryless | order-1 gain |
|---|---|---|
| bytes | 5.0802 | 1.194 |
| ours, previous token | 3.1282 | 0.805 |
| ours, previous word | 3.1282 | 0.861 |
| LLM tokenizer, charged | 2.9552 | 1.242 |

Our representation extracts *less* from one symbol of memory than
either alternative, per character, under either state map.  Same cause
as the memoryless gap: both the alphabet and the memory budget are
spent on markup, which BPE swallowed into its subwords.

## Where the time goes on the subword stream (31 July)

Three measurements on `output/streams/bpe_enwik8`, d = 100,277,
l_max = 54.  They settle the level-truncation question and re-order the
optimisation work.

**The current level truncation never fires here.**  `level_window_probe`
walked every level of 4,000 distinct first-order profiles with
`PMM_NO_TRUNCATE=1` and counted what each rule would evaluate:

| rule | level evaluations | of a full sweep |
|---|---|---|
| full sweep | 216,000 | — |
| one-sided (`_LevelWindow`, DROP=80, PATIENCE=3) | 212,204 | 98.2% |
| two-sided (ideal contiguous window around the mode) | 172,586 | 79.9% |

`one_sided` is 54, the whole range, for every profile in the sample.
The 1.52× wall-clock win recorded for level truncation is a **bytes-only**
result: at d = 256 the level curves fall fast enough to trip an 80-bit
drop, and on sparse subword profiles they do not.  The ideal two-sided
window buys 1.23×, matching the 1.22× measured on text8 bytes.  Two
independent workloads, same answer; the earlier prediction of 2–3× at
large d was wrong.

The reason the two agree: per level, the dominant cost is the O(G) scan
for local maxima over the 30,608-point grid, which does not depend on
the profile at all.  Weighting levels by k̃ moves the two-sided ratio
only from 1.23× to 1.35×.

**Only a handful of levels carry the mass, but only for heavy profiles.**
`peak_atlas --ids`, same stream, 2M tokens:

| profile | N | k̃ | mode level | levels holding all but 1e-12 |
|---|---|---|---|---|
| memoryless | 2,000,000 | 803 | 6 | 1 |
| row0 | 54,369 | 153 | 17 | 5 |
| row9 | 24,821 | 79 | 7 | 2 |
| row99 | 2,082 | 26 | 13 | 8 |
| row999 | 199 | 8 | 18 | 45 |
| row9999 | 18 | 3 | 22 | 52 |

out of 53, contiguous in every case at every tolerance.  The narrow
windows sit on the profiles with large k̃, but since per-level cost is
nearly flat in k̃ the population result above still governs.  Selecting
profiles by frequency rank over-weights the heavy ones; sampling
uniformly over distinct profiles over-weights the light ones.  The two
probes bracket the truth and agree on the conclusion.

**The integrand is unimodal on this stream.**  All 318 profile-levels had
exactly one significant peak; zero appearances and zero disappearances
along L.  The second far-left peak that motivated the multimodal scan
is a heavy-count phenomenon and does not occur here.  The peak drifts
at most 0.53 in u between levels (13 grid steps) and, across 57,134
family members, by 7.2e-8 at the median and 4.7e-4 at the 99th
percentile — a hundredth of a grid step.  A 30,608-point sweep is
locating something that barely moves.  `_scan_sparse` (`PMM_SCAN`) is
the existing path for this and is the largest untaken win.

Log-concavity in L holds in 95.4% of 306 checks, worst second
difference 35.5, so an outward-from-the-mode walk cannot be given a
rigorous tail bound from concavity; it would need the same empirical
drop-and-patience guard the current rule uses.

**Time split of the production path** (`state_family_experiment`,
1M-token prefix, jobs=1, under cProfile, 138.0 s of which 25.5 s was a
one-time build of 347 missing columns):

| part | seconds | share of steady state |
|---|---|---|
| `log_q_lambda_scan` | 72.0 | 64% |
|   — bracketed solve, derivative, curvature | 40.6 | 36% |
|   — grid integrand assembly | 31.4 | 28% |
| `log_phi_matrix` | 39.0 | 35% |
|   — the 8-point stencil | 18.3 | 16% |
|   — `series_column` | 10.4 | 9% |
|   — `_read_column` | 5.8 | 5% |

cProfile charges per call and the call counts are lopsided (3.3M
`_phis`, 2.7M `derivative`, 16.4M ufunc reductions against 53 calls to
`log_phi_matrix`), so the scan's share is inflated and provisioning's is
not.  Provisioning is close to a fixed charge per sweep — it amortises
over the ~5,750 profiles sharing each level — while the scan scales with
the corpus, so the scan's share grows with size.

Free in that profile, no compiled code and no numerical risk, because
memoisation returns the identical object: `partition_multiplicities` is
called 931,796 times for 304,826 scans, recomputing the same profile's
multiplicities at every level (9.2 s); `_read_column` re-reads columns
across levels (32,453 calls, 5.8 s); `series_column` recomputes the same
left-of-column values (32,976 calls, 10.4 s).  About 20% of steady
state.

**Compiled kernels.**  A C prototype of the 8-point stencil ran 10.4×
faster than the numpy version (13.4 ms → 1.31 ms per 120-column level)
and is **bit-identical** — but only after two fixes, both of which
change results silently.  `gcc -O3 -march=native` contracts `a*b+c` into
a fused multiply-add, which rounds once instead of twice; `-ffp-contract=off`
restores exactness and cost nothing measurable.  And numpy's
`.sum(axis=1)` reduces eight elements as a pairwise tree, so a
sequential loop disagrees; matching `((a+b)+(c+d))+((e+f)+(g+h))` fixes
it.  Numba's analogue of the first is `fastmath`, off by default.

Exactness is available for the stencil because it is only multiplies and
adds.  It is not available for the scan kernel: numpy's SIMD `exp` and
libm's `exp` differ by about one ulp, and scipy's `brentq` cannot be
called from compiled code, so Brent must be rewritten and will converge
to a slightly different root.  One ulp will not flip the 80-bit
truncation threshold, but the curvature sign test `curv < 0` is discrete
and is exactly the test that failed on 218 of 219 members in the Newton
continuation experiment.  Acceptance must therefore be equality of
`log2 q` against the current path over every distinct profile of text8
and enwik8, not a tolerance.

Numba tracks numpy's ABI and lags it, so `pip install numba` may try to
downgrade numpy; check before committing to that route.

### What the speed work actually bought (1 August)

Measured on `state_family_experiment --ids output/streams/bpe_enwik8
--top-k 100276 --m-grid 0,100277 --n 1000000 --jobs 12`, the same
command throughout:

| state | wall | CPU |
|---|---|---|
| before | 17.2 s | 144 s |
| + memoised multiplicities, cached level handles | 17.2 s | 144 s |
| + windowed scan (`PMM_SCAN`, now the default) | 14.7 s | 115 s |
| + compiled saddle solve | 13.8 s | 98 s |
| + compiled stencil | 10.9 s | 66 s |

**1.58× wall, 2.19× CPU.**  The CPU figure is the one that governs a
big machine; the wall figure is now limited by parallel efficiency, not
by the kernels: 66 s of CPU over 10.9 s of wall on twelve cores is 6.0×,
so half the machine is idle.  About 4 s of that is the serial head of
the run (stream load, state counting, column provisioning in the
parent) before the first depth is reported.

The compiled path is NOT bit-identical: up to 1.0e-9 bits on sparse
subword profiles and 1.5e-4 bits on heavy byte profiles, both about
1.1e-9 relative.  All of it comes from the saddle solve --- the stencil
kernel is exact --- and the two causes are `np.dot` dispatching to BLAS,
whose summation order C cannot reproduce, and numpy's SIMD `exp`
differing from libm's by up to one ulp above length eight.  On a
bits-per-character figure quoted to four decimals this sits eleven
orders below the last digit.  `PMM_KERNEL=0` reproduces the old numbers
exactly and `PMM_SCAN=full` restores the full sweep, so both halves
remain cross-checkable.

Next, in order of measured size: the parallel efficiency (ctypes
releases the GIL around every foreign call, so a thread pool would now
work and would stop copying the level matrix once per worker);
`series_column`, 9% of steady state recomputing values a denser store
would hold; and the serial head of the run.

### Final state of the speed work (1 August)

`state_family_experiment --ids output/streams/bpe_enwik8 --top-k 100276
--m-grid 0,100277 --jobs 12`, the paper's enwik8 first-order cell:

| | wall | CPU |
|---|---|---|
| before | 103 s | — |
| after | 42.3 s | 321 s |

**2.4×**, with 7.9534 bits per token and 2.1135 bits per character
unchanged.  What produced it, in order of size:

1. the windowed scan on the plain path (`PMM_SCAN`, now default) --- the
   grid exists only to locate a peak that the atlas shows is unique and
   drifts thirteen grid steps between levels;
2. the compiled stencil in `log_phi_matrix` (10.4x on that kernel);
3. the compiled saddle solve (17-31x on the solve in isolation);
4. memoised `partition_multiplicities` and cached level file handles;
5. skipping the certified series far left of a stored column, where
   the value is exactly its analytic limit (fill 16.0 s -> 10.8 s);
6. one shared-memory block for the whole run instead of 54 (39 GiB of
   fresh anonymous pages down to 1.5 GiB, `sys` from 31 s to 18 s).

Three changes were made against plausible-sounding theories and all
three did nothing measurable: shipping the profiles once instead of per
level, finer chunking for dynamic load balancing (reverted --- it cost
7% CPU), and the shared-block reuse above (kept, since it is a strict
reduction, but it did not move the wall clock).  The scaling fit
(wall = S + P/j from the 4- and 6-job runs) gives S = 33 s and P = 215 s
and predicts the 12-job time to within 2%: the run is now about
two-thirds serial at twelve cores, so further parallel work has little
left to win.  node14's 64 slower cores would land near 36 s.

**The bug worth remembering.**  The compiled stencil shipped broken for
several hours.  Inside `_interp_leftovers` a local array named `vals`
shadowed the parameter holding the stored column, so the edge branch
interpolated the series values instead of the column --- wrong by up to
622 nats at about 1% of query points.  Every end-to-end check passed
throughout, printing 2.1135 each time, because the profiles in those
runs never read the affected points.  A benchmark that agrees is not
evidence that a numerical kernel is correct.  `tests/test_interp_kernel.py`
now compares the two paths value by value (2.9 million values across
several levels and query grids, exact equality required) and asserts
that all three regions of the query grid --- stencil interior, series
tail, clamped window --- are actually exercised, since the defect lived
in the handoff between them.

The saddle kernel had the treatment from the start, and its only real
problem was a genuine ambiguity rather than a mistake: far out in the
left region the curvature is numerically zero, so `curv < 0` is decided
by rounding, and a kernel rejection there dropped a profile's only peak
and raised "left tail only".  `_solve_peak` now re-decides every
rejection in Python, so the compiled path can accelerate an accepted
peak but never remove one.  Residual disagreement with the Python path
is 1.0e-9 bits on sparse subword profiles and 1.5e-4 on heavy byte
profiles, about 1.1e-9 relative, from BLAS reduction order and numpy's
SIMD `exp`; `PMM_KERNEL=0`, `PMM_SCAN=full` and `PMM_INTERP_KERNEL=0`
each restore the older path for cross-checking.

The far-left series shortcut is now on by default
(`SERIES_TAIL_NATS = 40`, `PMM_SERIES_TAIL=inf` to restore the full
series).  It looked broken when first tried, but that was the shadowing
bug corrupting the same code path: with the bug fixed it moves no value
at all across 2.9 million comparisons, and it is worth 48.9 s -> 42.3 s
on the enwik8 cell.

## Coarsening the state: the family does not help (1 August)

The M-sweep over state maps sigma_M, run on all three subword streams
with the admissible (vocabulary-id) ordering.  Bits per character,
vocabulary counted:

| states kept | text8 | enwik8 | enwik9 |
|---|---|---|---|
| 0 | 2.1716 | 2.9552 | 2.9775 |
| 1,024 | 1.9805 | 2.5686 | 2.4670 |
| 8,192 | 1.8698 | 2.3311 | 2.1686 |
| 32,768 | 1.8318 | 2.1666 | 1.9323 |
| all | 1.8298 | 2.1135 | 1.8371 |

Monotone in M on every file, posterior entirely on the full model,
mixture equal to the best member.  **Coarsening never pays.**  The
prediction that an interior optimum would appear was wrong, and the
reason is that the family offers only one alternative to a state's own
counts --- pooling it with every other unpromoted symbol --- which
destroys nearly everything the state carries, while giving a rare
symbol its own state costs the layered estimator very little.

The last step is the informative one: 65,536 states to all 71,161 gains
0.019 bits per token on enwik8 and 0.010 on text8, against tenths of a
bit for every earlier doubling.  First order sits at the point where
extra state stops earning its cost.  That reverses the argument for
running this before pairs: order one is NOT limited by having too many
states, so order two is not obviously hopeless --- but it would add
states in exactly the region where marginal value is lowest, so the
prediction is that it fails on enwik8 and may succeed on enwik9.

**Admissibility.**  The family as originally coded ranked symbols by
their frequency in the file being compressed, which the decoder cannot
reproduce; transmitting the ranking costs ~1.5 Mbit at d = 100,277,
0.19 bits/character on enwik8, more than the whole first-order gain.
Member M now keeps the M smallest vocabulary ids, which the tokenizer
fixes and `fixed_bits` already pays for.  At M = 0 and M = full the two
rules coincide, so no published number moved (verified: 11.2165 and
7.9534 under both).  `tests/test_state_order.py` pins this down.

## Where the learning cost actually is (1 August)

`scripts/state_redundancy.py`: per state, model bits minus n_s H_s,
bucketed by n_s, with the Miller-Madow correction applied to the
plug-in target.  Bits per token:

| file | model | H corrected | excess | obs/state |
|---|---|---|---|---|
| text8 | 9.0983 | 7.3715 | 1.7268 | 543 |
| enwik8 | 7.9534 | 6.4895 | 1.4639 | 362 |
| enwik9 | 6.6902 | 6.2118 | 0.4784 | 3,367 |

Share of the total excess by observations in the state:

| observations | text8 | enwik8 | enwik9 |
|---|---|---|---|
| 1-29 | 3.6% | 7.5% | 1.1% |
| 30-99 | 10.1% | 14.9% | 2.6% |
| 100-999 | 41.3% | 41.8% | 25.6% |
| 1,000-9,999 | 33.2% | 22.9% | 42.3% |
| 10,000+ | 11.8% | 12.9% | 28.5% |

Two findings.  The excess falls sharply with data per state --- enwik8
against enwik9 is nine times the data and a third of the excess, at
fixed vocabulary and nearly fixed state count.  And the excess is NOT
in the near-empty states: a state seen once costs 16.614 bits per
token, exactly log2(100,277), the model correctly paying the uniform
price when it knows nothing, but such states hold too few tokens to
matter.  Two thirds sits in states with 100 to 10,000 observations.

**The direction this points.**  A state with a thousand observations
over a hundred thousand symbols still pays, in effect, to say which
symbols occur, and pays separately from every other state, although the
marginal already answers most of the question.  What is missing is a
graded way for a state to borrow the marginal while keeping its own
evidence --- a change to the construction (a hierarchical prior, per
state weights concentrated around the global ones) rather than another
family over existing pieces.  Note that a per-state independent choice
between "own counts" and "backoff" does NOT factor cleanly, because
whether a state joins the pool changes the pool's own profile.

Paper: new Section "Coarsening the state" after "Memory of order one",
with Tables tab:coarsen, tab:excess, tab:excess-where, and
Appendix A on the Miller-Madow correction.

## 2026-08-02 --- Morning handover: STOP-WORK notice (superseded the same evening; see next entry)

Kept verbatim for the record; its section 0 conclusion (ladder interpolation too coarse) was measured wrong that evening. The bug fixes in its section 2 stand.

### 0. STOP — the system is NOT ready

**The ladder is not accurate enough. Do not put any of this sweep into
the paper, and do not run Stage B until the spacing question is redone.**

MEASURED, on `tables/anchors_prod` (8.3% spacing, degree 11), the
interpolation error in nats:

| level | max error (nats) |
|---|---|
| 5 | 5.6e-05 |
| 10 | 1.2e-04 |
| 20 | 2.1e-04 |
| 33 | 1.5e-03 |

MEASURED, the codelength consequence (`scripts/compare_evaluators.py`,
exact columns vs the ladder, same profiles, no corpus):

| counts | l_max 6 | l_max 33 |
|---|---|---|
| <= 5 | 2e-13 | 3e-10 |
| 300 | — | 9.8e-05 |
| 1,000 | — | 9.6e-04 |
| 10,000 | — | **0.52 bits** |

Counts up to 255 are inside the dense floor and served exactly, hence
1e-10. Everything above it is interpolated, and the error grows fast
with r. At `l_max=33` the expansion never fires (cutoff 54), so this is
**purely the ladder**.

Consequences already visible in the sweep: `pooled_v1024` moved by
**+0.0387 bits/token** (5.1874 -> 5.2261, and the best member changed),
`ctree_fullvocab` by +8.5e-04. The cache they were compared against is
**clean** — 200 samples at each of L=5,10,20,33, zero bad — so these are
the new evaluator's error, not a correction of the old numbers.

#### How the spacing decision went wrong

The sweep that chose 8.3% ran on `anchors_f005`, whose r_max was 1.06e6,
and reported ~5e-7 nats. `anchors_prod` has r_max = 2e8, so its test
targets sit near 1e8 where the residual `ln phi - L*lgamma(r+1)` is far
larger, and the same *relative* spacing gives a far larger *absolute*
error. The measurement did not transfer to the store that was built, and
nobody re-ran it after building. **Re-measure on the store you intend to
ship, never on a proxy.**

#### What has to happen before this is usable

1. Derive the accuracy requirement from the CODELENGTH, not from nats.
   `compare_evaluators.py` is the right instrument: it needs the delta
   below ~1e-4 bits on every profile the experiments actually produce.
2. Re-choose the spacing against that, on the real store, at the r and
   l_max the experiments use — not on a proxy store with different r_max.
3. Consider raising the dense floor instead of tightening the spacing.
   Counts inside the floor are exact, and most counts in a corpus are
   small; the question is where the crossover is.
4. Only then re-run the sweep.

Everything below this section was written before this was measured.
Sections 1 and 4 in particular state that the system reproduces the
published numbers; that holds only for the nine Stage A runs whose
counts stay small.

---

State of the moment-table work: what the system is now, what was fixed,
what is measured, what is guessed, and what is left.

Throughout, **MEASURED** marks a number someone actually observed and
**INFERRED** marks a claim that has not been checked. That distinction is
the main lesson of the session that produced this file: several hours
were lost to explanations offered with the confidence of measurements.

---

### 1. What the evaluator is now

Moment values `ln phi_r^(L)(e^u)` come from three places, with **no
exact stored column read at any level**:

| region | served by |
|---|---|
| `L >= 54` | order-2 saddlepoint expansion + certified series (`log_phi_column`) |
| `L < 54`, r an anchor | the stored column, bit-identical |
| `L < 54`, r between anchors | degree-11 barycentric interpolation in `ln(r+1)` across the 12 nearest anchors |

The store is `tables/anchors_prod`: a **designed** grid, 52 levels
(2..53), 475 columns per level, 2.8 GB.

Grid per level (`scripts/build_anchor_store.py`):

- every integer `0..255` (the dense floor)
- then `r_k = round(1.083071^k)` up to `r_max = 2e8`
- then 8 further anchors above `r_max` (the pad)
- plus 40 non-anchor **targets**, built as interpolation test points and
  recorded separately in `anchors.json`; they are never used as anchors

Configuration (four variables, all four required):

```
PMM_UNIVERSAL_TABLES=tables/anchors_prod
PMM_PHI_LADDER_EVERY=1        # decimation of the store's own grid
PMM_PHI_LADDER_DEGREE=11
PMM_PHI_SADDLE_MIN_L=54
```

`scripts/rerun_paper.sh` exports these itself. Run it from a shell with
no `PMM_*` set: if the variables are inherited, a stale one silently
points the run at the cache and it reproduces the published numbers
while proving nothing.

#### Cost, MEASURED

- complete system vs exact columns on `bpe_text8`: **+2.172e-05
  bits/token = 1.829827 bits/char**, against a published 1.8298. The
  digit holds.
- store: **2.8 GB** against the **86.6 GB** cache.
- Stage A model runs: **26.5x faster** in total (6297 s -> 238 s);
  `state256` alone 1998 s -> 28 s.

#### Why the cutoff is 54 and not 46

At cutoff 46 the delta is +2.444e-04 bits/token = 1.829853 bits/char,
which **rounds to 1.8299** — the published digit moves. Extending the
store from level 45 to 53 costs ~0.4 GB and takes the delta back to
1.829827. MEASURED.

---

### 2. Bugs fixed this session

Each is a real defect that existed before this work, with the evidence
that identified it.

**Concurrent writers corrupted the store silently.** `_append_column`
recorded each column's offset from `level["size"]`, an in-memory
counter, while appending in `"ab"` mode. Two processes writing one level
each kept their own counter, so both wrote real bytes and both recorded
plausible offsets, but each one's offsets were short by whatever the
other had written. A column read through a wrong offset is finite,
smooth, correctly sized real data from elsewhere in the file — every
existing guard passed it. Fixed: offsets come from the file under an
exclusive lock, the index is merged rather than clobbered, and **every
column now carries a CRC verified on read**.

**The right-pole series certificate understates its error by ~8e6.**
MEASURED at `L=10, r=1045889, u=14.52`: the series reports `cert=4.5e-11`
and is wrong by **3.7e-4 nats**; contour and saddle agree with each other
to 1.4e-5 and both disagree with it. It affected **90 of 1920** anchor
columns at L=10, all at large r. `PMM_BUILD_EXACT=1` disables the branch
in both the builder and `log_phi_contour` (the reference must not be the
method under test). **Use it for every store build.**

The certificate itself is still **unexplained** — presumably it bounds
truncation of the series it sums and misses another term. It remains in
the read path at a 1e-10 threshold. *This is an open correctness
question, see §5.*

**The ladder was broken four separate ways**, each found by a five-minute
run rather than a test: `log_phi` had the hook but `log_phi_matrix` (the
hot path) did not; provisioning demanded columns for levels the expansion
serves; decimation deleted the small-r anchors, producing `log2 q = +1321
bits`; and `r=0` fell below the grid and was silently **extrapolated**,
producing `log2 q = +91`. All fixed, and extrapolation outside the anchor
span now raises instead of returning a diverging Lagrange value.

**A designed store could grow.** Pointing an ordinary run at the anchor
store began rebuilding the whole cache inside it; the only symptom was an
apparent hang. A store with an `anchors.json` is now **sealed**: appends
raise, and a missing column raises naming what was asked for. The
builder also refuses to build into an existing store, which otherwise
silently did nothing (its columns exist, so none are appended) while
reporting success.

**`pooled_lag_codelengths` was quadratic in memo size.** Every checkpoint
did `known_before = set(builder.memo)` — a fresh set of every key, **five
million** of them — then filtered the whole memo against it to find the
new entries. Two O(total memo) passes per checkpoint, on an append-only
memo where the new entries are simply everything past the previous
length. The allocation spike also kept triggering full generation-2
collections over a five-million-object heap: a sampled profile showed
**91% of the process inside `gc_collect_main`**, and gen2 passes took
**38 s each**. Fixed with an `islice` from the recorded offset, plus
`gc.freeze()` per checkpoint.

MEASURED, before and after: checkpoints ran 78–2866 s, wildly
oscillating; checkpoint 28 after the fix took **193 s** (177 s work +
15 s save) with gen2 at 0.01–0.05 s.

Note this job never benefited from the 26x speedup: its log shows no
`tables:` lines at all, so it spends no time in the phase that was
optimised.

---

### 3. Tooling

| script | what it is for |
|---|---|
| `build_anchor_store.py` | build a designed store; dry-run by default, `--go` to build |
| `check_store.py` | verify columns against contour integration; `--self-test` forges known damage and confirms detection |
| `smoke_ladder.py` | **seconds-long** check of the complete system on the real code path; run before spending a real run |
| `ladder_accuracy.py` | interpolation error vs spacing and degree, anchors verified before use |
| `phi_sensitivity.py` | end-to-end cost of the ladder and the expansion, in bits |
| `rerun_paper.sh` | the whole paper sweep, logged, failures stepped over |

`smoke_ladder.py` exists because five consecutive five-minute runs were
each spent discovering one bug. It checks: the grid starts at r=0, the
dense floor survives decimation, r=0/1/2/3 are bit-identical, non-anchors
interpolate within tolerance, `log_phi_matrix` agrees with `log_phi`, and
a real codelength comes out with `log2 q <= 0`. **Run it after any change
to the ladder, the grid, or provisioning.**

`check_store.py --self-test` forges damage twice, once with checksums and
once with them stripped, because a store written before checksums existed
has no CRC to fail and detection there rests entirely on the numerics.

---

### 4. What has been run, and what has not

#### Stage A — model runs (`paper/main.tex`)

All 12 complete except `pooled_v1024`, which was still running when this
was written.

MEASURED, v2 (old cache) vs v3 (complete system):

| run | v2 | v3 | delta |
|---|---|---|---|
| state256 | 3.719846 | 3.719848 | +1.3e-06 |
| state4096 | 6.713180 | 6.713180 | +9.1e-08 |
| ct16384 | 8.238249 | 8.238249 | +8.9e-09 |
| spelling (bpc) | 3.431218 | 3.431218 | +1.3e-08 |
| **ctree_fullvocab** | 10.091905 | 10.092753 | **+8.5e-04** |

`state_fullvocab` gives 9.7826, matching the published figure exactly.

**`ctree_fullvocab` is the one unresolved number.** The shift is ~1.4e-4
bits/char, about 40x the measured approximation ceiling (2.2e-5
bits/token), and it is the full-vocabulary run — the largest r values in
the sweep, exactly where the right-series certificate fails. INFERRED
that this is the bug having been in the published number, with the new
value correct. **Not established**: the ceiling was measured at
V=100,277 and this run is V=300,000 with larger counts, so the ceiling
here could genuinely be higher. Settling it means a `phi_sensitivity`
run against a matching stream.

#### Stage B — representation runs (`paper/compress.tex`)

**None of these has been run.** `bash scripts/rerun_paper.sh stage_b`

- `byte_baseline` on text8, enwik8, enwik9
- `token_baseline` on text8, and enwik8 in four settings
  (intern/compositional x conditioned/folded)
- `llm_token_baseline` (cl100k_base) on text8, enwik8
- then enwik9 for all three — 10x the rest, deliberately last

Published values to compare against: bytes 4.1235 / 5.0802 / 5.1565;
our tokenizer 2.2483 (text8) and 3.1282 (enwik8, intern+conditioned
winning); cl100k 2.1716 charged / 2.1095 free (text8), 2.9552 / 2.8931
(enwik8). The enwik9 tokenizer runs were listed as pending rather than
published, so they are new work, not verification.

All eight experiment scripts read the moment store, so everything in the
paper is in scope.

#### Attribution rule

A shift below ~1e-4 bits/char is the new evaluator and expected. Anything
larger means the right-series bug reached that published number — stop
and look rather than continuing the sweep.

---

### 5. Open questions

**The right-series certificate.** Why does it report 4.5e-11 when the
error is 3.7e-4? Until that is understood the branch cannot be trusted,
and it is still live in the read path at a 1e-10 threshold — only the
*builder* is bypassed by `PMM_BUILD_EXACT`. This is the most important
open item, because it is a correctness question, not a performance one.

**`ctree_fullvocab`.** See §4.

**The dense floor is unmeasured.** 256 was chosen by reasoning — small r
is where the counts are and where `ln phi` is steepest in `ln(r+1)` — not
by measurement. It is 256 of the 475 columns per level, so it now
dominates the store size; the spacing does not. Whether it can be 64, or
needs to be 1024, is a `ladder_accuracy.py` sweep nobody has run.

**The pad is unmeasured too.** 8 anchors above `r_max`, sized so degree
11 has six on each side. Padding by too little silently degrades accuracy
near the largest counts by two orders (MEASURED: 5e-9 -> 2.6e-6 nats).

**Spacing has margin nobody has spent.** MEASURED: degree 11 is flat from
0.5% to 4% spacing (~5e-7 nats worst case) and the end-to-end ladder cost
is below 1e-6 bits/token even at 8.3%. The current store is 8.3%. Going
coarser is possible but the floor dominates, so it saves little.

**Degree 7 is unstable.** It shows sporadic blowups two to three orders
above its own median at the same setting (7.5e-5 at L=15/2%, 2.6e-4 at
L=22/4%) where degree 11 has none. Do not lower the degree without
re-measuring.

**Threads, not processes.** `pooled_lag_codelengths` parallelises with
`ThreadPoolExecutor`, so `--jobs 12` buys much less than in the other
experiments except where numpy releases the GIL. Read from the code, not
measured.

---

### 6. Loose ends

- `tables/anchors_prod` has 52 leftover `.lock` files from its build.
  Inert; the builder should remove them.
- `rerun_paper.sh` prints one line per run **on completion**, so a long
  job looks like a hang for hours. It should heartbeat.
- The resume directory keeps every checkpoint's pickle — 27 files,
  5.86 GB — instead of pruning. Correct but wasteful.
- `VERIFY.md`'s expectations section still describes the v1->v2
  correction and quotes a 1e-3..1e-2 bpc tolerance. Too loose for this
  round; it wants the 1e-4 rule.
- `PMM_GC_TRACE=1` and the `[phase]` lines in `pooled_lags.py` are
  instrumentation added to diagnose the quadratic memo. Keep or remove
  deliberately.
- The old cache `tables/universal_v2` (86 GB) is still on disk and still
  carries the right-series error. Nothing reads it under the new
  configuration. It is the only comparison point for the v2/v3 table
  above, so do not delete it until that comparison is finished.

## 2026-08-02 (evening) --- Ladder exonerated; the real defects: evaluator Laplace curvature (fixed) and universal_v2 large-r contamination (open)

Supersedes section 0 of the previous entry. Everything here is
marked **MEASURED** or **INFERRED**, same discipline as the morning file.

### 0. The stop-work reasoning was wrong — the ladder is fine

The morning file said the ladder's interpolation was too coarse (1.5e-3
nats at L=33, 0.52 bits at counts of 10,000) and that spacing and floor
had to be re-chosen. MEASURED today: after fixing the real defect (an
evaluator bug, §1) and building a trustworthy reference (§3), the ladder
in its production configuration —

    tables/anchors_prod, factor 8.3% (verified from anchors.json),
    dense floor 256, degree 11, EVERY=1, expansion at L>=54

— passes the codelength requirement on **all 22 instrument profiles,
worst |delta| = 5.6e-6 bits (c17M)**, 18x inside the 1e-4 requirement.
Domain covered: top counts 200..17,005,209 (the largest r any real run
ever requested, from the universal_v2 indices), d up to 300,000,
l_max to 53. `output/compare_fixed.jsonl` has the rows.

**Do not rebuild the store. Do not change the spacing.** MEASURED
margin: read-time decimation every=2 (16.6%) fails exactly one case,
c17M at +2.4e-4 bits. One octave of margin, no more. The graded c*
rows now scale cleanly with top count, as truncation should.

Stage B stays blocked, but for a different and smaller reason: Stage A
should be re-verified under the corrected evaluator first (§4).

### 1. The real defect: Laplace curvature was noise at large counts

The scan integrates each interior peak either by bracketed-solve +
Laplace or by grid trapezoid. The Laplace curvature was computed as
`sum(count * (rho + rho^2 - raw_second))` — a difference of near-equal
exponentials (~1e11 at c1M profiles) whose exponents carry the O(h^2)
error of linear interpolation between u-grid nodes.

MEASURED (profile `(1e6, 250k, 62.5k, ..., 1)`, d=1024, exact store):

- solver curvature at L=33: **-5.494e3 against a true -38.6**; at
  neighbouring points +210, +2074, -3572, +3878 — sign flips included.
- consequence: the Laplace contribution was 4.3 nats (~6 bits) low
  wherever that branch fired, and whether it fired flipped per level
  and per store on 1e-7-scale perturbations. Per-level deltas for c1M
  were bimodal: ~1e-7 bits where both evaluators took the same method,
  ±2..6 bits where they differed.
- adjudication: dense quadrature (40,001 points across the peak) agreed
  with the **trapezoid** to 0.02 nats and with the node-based Laplace
  to 0.15 nats; peaks there are wide (w = 0.16–0.70 nats vs grid step
  0.033), so the grid resolves them.

Fix (in `layered.py`, both `_scan_from_psi` and `_scan_sparse`, shared
helper `_integrate_grid_peaks`):

- method choice per peak from a **node-based curvature** (second
  difference of psi at grid nodes — cancellation-free);
- resolved peaks (width >= 2 grid steps) integrate by **trapezoid over
  a window extended to a 45-nat drop**, windows merged so mass is never
  double-counted;
- unresolved peaks fall back to solve + node curvature and the note
  says **NARROW** — none observed on any instrument profile so far;
  if NARROW appears in production, that peak needs master-grid
  refinement before its value is trusted;
- `PMM_SCAN_LEGACY=1` restores the old behaviour for A/B.

The compiled kernel still contains the old curvature arithmetic; it is
bypassed for method choice but its root-finding is still used. Cleaning
that up is open.

MEASURED adjudication of the fix: per-level dense integrals for
profile (1,1,1) d=4 match the new evaluator to <=1e-8 bits at every
level; c1M per-level deltas collapse from ±bits to ~1e-7 bits with
identical methods on both stores.

**Absolute codelengths moved with the fix** (the old Laplace was wrong
everywhere, mildly): tiny/shallow by 0.015 bits, medium/deep by 2.6e-3.
The published pipeline used the old evaluator, so published numbers
inherit these per-profile errors; §4 measures what survives to corpus
scale.

### 2. universal_v2 is NOT clean at large r

The morning file's premise "the old cache is clean" is false above
r ~ 1e4. MEASURED (5,431 common columns, probe_exact vs universal_v2):
smooth error growth ~1e-8 nats below r=100, ~1e-5 at r=1e4, 5e-4 at
250k, ~2e-3 at 1e6, **1.8e-2 nats at r=17,005,209**, across all levels.
One large column (r=16,606) is nearly clean (4e-10) — the cache mixes
build eras. INFERRED (mechanism, unverified): series-era builds carry
the right-pole certificate bug; the morning "200 samples, zero bad"
verification ran check_store's contour confirm **without**
PMM_BUILD_EXACT, so at large r the reference took the same series
branch and confirmed the bug against itself.

Consequences: the morning v2-vs-v3 attribution table compared against a
contaminated reference at large counts; `ctree_fullvocab`'s published
number is doubly suspect (cache error + evaluator error). Nothing was
written to universal_v2 today and nothing should be, ever.

### 3. New store: tables/probe_exact (the reference instrument)

Built today, on demand, by `compare_evaluators.py` pointed at an empty
dir with PMM_BUILD_EXACT=1: every column the 22 instrument profiles
need, computed by contour with the suspect series branch disabled.
~21 MB + the large-r columns. MEASURED verification: check_store --all
— every column bit-identical to a fresh contour evaluation; saddle
screen (independent method) agrees to <1e-2 nats on 209/210 columns per
level at L>=20.

**TODO (one command, not yet run):** freeze it —
`chmod -R a-w tables/probe_exact`. Treat it exactly like universal_v2:
never modified once filled and verified. If a future profile needs
columns it lacks, unfreeze, extend with PMM_BUILD_EXACT=1, re-run
check_store --all, re-freeze.

### 4. Published numbers: state of verification under the fixed evaluator

- MEASURED: `state256` re-run end to end (v4, fixed evaluator, ladder
  store): family mixture **3.7198465** vs published 3.719846. The
  published digits hold; the old evaluator's v3 value (3.719848) was
  the one slightly off. Per-profile absolute corrections (~1e-2 bits)
  wash out at corpus scale here.
- OPEN: the other Stage A runs, cheapest first, same env as
  `rerun_paper.sh` exports, out-dirs `output/v4_*`. `ctree_fullvocab`
  and `pooled_v1024` are the two that moved in the morning sweep and
  the two that matter; the morning shifts are now expected to be
  explained by evaluator method flips (INFERRED until the v4 runs land).
- Attribution rule from the morning file still applies, but compare
  v4 against **published**, not against v3: v3 carries the old
  evaluator's errors.

### 5. Instrument changes today

- `scripts/compare_evaluators.py`: per-profile `(profile, l_max, d)` —
  **d is load-bearing**: at the old d=max(len+1,4) the scan refuses
  large-N profiles (integrand right edge within 10 nats of peak), which
  is why the morning version had never actually compared anything above
  counts of 1e4. 15 realistic rows added (c200..c17M at d=1024,
  fv/1M at d=300k, l53, d256 control). PMM_BUILD_EXACT forced in both
  children. `--every/--only/--out` flags. PASS/FAIL vs 1e-4 bits.
- `scripts/smoke_ladder.py`: new check — large-count profile
  (3500, 800, 90, 7, 1) at d=1024 through every=1 vs decimated ladder
  must agree to <1e-3 bits. This is the class every previous smoke
  missed. Passes in ~5 s.
- `src/.../layered.py`: §1.

### 6. Open questions, updated

- **Right-series certificate**: RESOLVED late 2 Aug --- see the next
  log entry.
- **Kernel curvature arithmetic**: superseded in Python, still present
  in `_kernel.c`. Remove or fix; until then PMM_SCAN_LEGACY must stay
  off in production.
- **NARROW peaks**: none observed yet; if one appears, master-grid
  refinement is unimplemented.
- **Dense floor 256 / pad 8**: both now MEASURED-adequate end to end
  (c200/c256 rows exact; c17M passes at every=1 with r_max 2e8 slack).
  Floor dominates store size; shrinking it was not tested and is not
  worth the risk at 2.8 GB.
- **check_store thresholds**: CONFIRM at 1e-4 nats and screen at 1e-2
  are too coarse to catch §2-scale contamination at mid r. Run it with
  PMM_BUILD_EXACT=1 always (otherwise the reference can reproduce the
  series bug). A contamination map of universal_v2 (which levels/r are
  series-era) is open — matters only for archaeology of published v2
  numbers.
- Repo root has junk zero-byte files (`--degrees`, `--levels`, `--out`,
  `0.50%,`, `8.31%` etc.) from a mangled shell command. Delete freely.

## 2026-08-02 (late) --- right-series certificate explained, fixed, verified

The morning file's most important open item (§5: certificate reports
4.5e-11 where the truth is 3.7e-4) is closed.  The full chain, each
step MEASURED at (r=1045889, L=10, u=14.52):

1. The residue series FORMULA is correct: summing the first five
   residues in 60-digit arithmetic (mpmath) reproduces contour to
   9e-10 nats.
2. The regime is cancelling, not convergent-fast: terms j=0 and j=1
   have opposite signs and magnitudes within 0.6% (the docstring's
   (r+1+j)/(t (j+1)^L) ratio estimate does not hold near t ~ r+1),
   so the sum is ~1e-5 of the terms --- 5 digits lost.
3. The float64 coefficient recursion is CLEAN: p[n] matches the
   60-digit Taylor coefficients to ~1e-16, majorant p_abs/|p| ~ 1.
4. The error entered in the EXPONENT ASSEMBLY: lam_j = -a u +
   lgamma(a) + ln|coeff| differences ~1.5e7-magnitude floats, giving
   ~1e-9 nats of INDEPENDENT per-term rounding; amplified by
   total_abs/total ~ 2e5 -> the observed 3.7e-4.  The certificate
   charged that amplification at 2.2e-16 (exact-exponent assumption):
   understatement factor ~5e6 --- the mystery number.

Fix (mellin.py, both log_phi_right_series and right_series_column):
differential exponent assembly --- lam_j - lam_0 computed from small
quantities (-j u + sum ln(r+1+i) - L lgamma(j+1) + coefficient ratio),
so per-term differential rounding is ~1e-14; term_0's absolute
rounding shifts all terms identically and cancels in the final log.
The certificate now also charges the differential-exponent rounding
under cancellation.

Verified against independent contour (PMM_BUILD_EXACT=1) over
r in {1e3, 1e6, 1e8} x L in {2,5,10,33} x t/(r+1) in {1.5,2,5,50}:
point of record improves 3.7e-4 -> 2.1e-9 nats (cert 2.1e-8, honest);
36 accepted points within 3x cert plus representation floor; the
boundary/L=33 regime now carries loud certificates or refuses, which
the 1e-10 acceptance threshold correctly rejects (contour serves it).

KNOWN FLOOR, not a defect: the returned ln phi is assembled from
intermediates of magnitude ~r u, so its absolute error floor is
~eps * r * u (~1e-6 nats at r=1e8).  Contour carries the same floor.
Three orders below anything the 1e-4-bits requirement can feel.

Also this session (cleanup): dead evaluators removed
(log_q_lambda_laplace / _grid / compute_log_q_by_partition, zero
callers); [phase] checkpoint lines gated behind PMM_TIMING=1;
build_anchor_store removes its level locks; VERIFY.md expectations
rewritten to the 1e-4 rule; v3 pooled resume pickles (5.9 GB), stale
locks and universal_tables.py.bak deleted; compiled kernel .so
binaries untracked and gitignored; probe_exact frozen read-only.

## 2026-08-03 --- Pairwise order-two: first run of the new program

New experiment (scripts/pairwise_experiment.py, paper section "Memory
of order two from pairwise statistics" + appendix "The calibrated
pairwise predictor"): predict x_t from (x_{t-1}, x_{t-2}) using only
the three pairwise tables, combiners compared inside one checkpointed
sequential code (C=32, count-smoothed tables, no moment store).

MEASURED (text8, V=1024, 19.4M tokens, bits/token on the reduced
stream): star product beta=1 5.3917 (LOSES to lag1 5.3393 --- double
counting), best mixture 5.2976, calibrated product 5.2827 (no free
parameters), best tempered product 5.2746, markov2-with-backoff
5.2446.  Pairwise recovers ~2/3 of the order-two gain at ~3V^2/V^3 of
the parameter scale.  KEY LEAD: markov2 wins here (share-nothing
order-2 loses in Table order-two) because of its backoff to the
lag-1 row --- graded borrowing, the mechanism the excess analysis
said was missing.  Hierarchy is the next thread.

Debug history (smoke, V=64): IPF cycled on inconsistent estimated
margins (fixed: Sinkhorn projection of each pair table to common
unigram margins); calibrated then overfit because its joints were
effectively unsmoothed (fixed: joints built from the SAME smoothed
conditionals all schemes use).  After both fixes calibrated tied
markov2 on the smoke stream.  IPF core verified against dense
enumeration at small V (margins 1e-14; copy-lag extreme reproduces
p(x0|x1) to 1e-11 where the star errs by 0.36).

Also this session: paper foundations verified under the corrected
evaluator (4 fourth-decimal cells), order-two got its own section,
Pairs demoted to an appendix, model section rewritten (assignment
view, consistency identity, posterior-weighted conditional), forecast
paragraph removed.

## 2026-08-03 (later) --- Layered pairwise fleet in flight; exact order-2 reference added

STATE OF THE RUNS (three machines; results flow via git --- the
.gitignore now tracks output/*/results.json and nothing else under
output/):

- Laptop: pairwise_experiment.py --tables layered --order2 both on
  text8 V=1024, C=32, ~500 s/checkpoint, IN PROGRESS (hours).
  Output: output/pairwise_layered_v1024.
- Second Mac (rudiger@iscpc71): same but enwik8, running overnight.
  Store tables/anchors_prod transferred by tar+http; Stage B
  verification all green there.  Output:
  output/pairwise_layered_enwik8.
- Server (urbanke@lth.epfl.ch): three COUNTS-mode runs
  (pairwise_v4096, pairwise_enwik8, pairwise_enwik9).  No moment
  store on the server; counts mode does not need it.
- Collection, on each machine AFTER its runs finish:
  git pull --rebase && git add output && git commit -m results &&
  git push

WHY CHECKPOINTED LAYERED RUNS TAKE HOURS (question settled): the
markov2-layered member dominates.  A single full pass over the pair
contexts is itself on the order of ten minutes (hundreds of
thousands of distinct context-pair profiles, most shared with no
other context), and the 32 refreshes repeat it at 32 count
snapshots; frequent contexts change profile every block, so the
(profile, count) memo does not save them.  Checkable: one checkpoint
with --order2 backoff should take well under a minute.

NEW MEMBER, markov2-layered-exact (decided today): the layered
order-2 reference is a mixture, so its EXACT sequential code (tables
updated after every token) telescopes to one evaluation per context
pair at the final counts --- about the cost of one checkpoint, no
staleness.  The gap exact vs checkpointed measures what the C=32
schedule costs; if it is large, 32 checkpoints are not enough.
Implemented in scripts/pairwise_experiment.py as --exact
{off,add,only}: 'add' computes it alongside a full run; 'only'
computes just this number and MERGES it into an existing
results.json in --out (recomputes family code and best member;
asserts coded_positions match).  Paper section "Memory of order two
from pairwise statistics" updated accordingly (references bullet:
the order-2 reference comes in two forms; new paragraph on the
exception and what the gap means).

DEPLOYMENT CAUTION: the edited script sits on the laptop as
scripts/pairwise_experiment.py.new.  Do NOT replace
pairwise_experiment.py while any run using it is live --- spawned
workers re-import the file from disk (this exact failure mode
crashed the pooled run on 2 Aug).  When the laptop run finishes:
  mv scripts/pairwise_experiment.py.new scripts/pairwise_experiment.py
then commit and push; other machines pull only after their own runs
finish.  Then, per layered run, add the exact number with the SAME
--ids/--top-k/--n/--cap flags as the original run plus
  --tables layered --exact only --jobs <cores> --out <same out dir>

NEXT STEPS QUEUE:
1. Collect the fleet's results.json files via git; assemble the grid.
2. Run --exact only for pairwise_layered_v1024 (laptop) and
   pairwise_layered_enwik8 (second Mac).
3. Write the layered comparison into the paper section (placeholder
   line "The measured comparison will be reported here...");
   sanity-check against the count-based numbers recorded in the
   previous entry.
4. THEN the step-3 discussion (hierarchy / graded borrowing built
   into the estimator) BEFORE any implementation.  Constraint from
   Ruediger: keep the philosophy --- mixtures and algorithms, not
   learned weights.

Minor open items: the "2.2 bits/char at twenty observations"
sentence in the order-two section still awaits his choice
(per-bucket column vs explanatory sentence); the "MAP split contexts
49" cell in legacy.tex could not be re-derived (likely moot).


## 2026-08-03 (later still) --- Scope note: why the pairwise round runs at V=1024 (supersedes anything suggesting otherwise)

Agreed with Ruediger; this supersedes anything above that suggests
otherwise.  Why the pairwise experiment runs at V=1024, stated
correctly: it is a controlled first round for RANKING the combining
rules, nothing more.  At small V we can afford every member,
including the calibrated one (dense V x V IPF) and both memory-2
references, so the rules can be ranked against a visible ceiling.
The small alphabet is NOT scientifically necessary, and cost is NOT
a fundamental barrier to the full vocabulary:

1. The mixture and product members run at full vocabulary in sparse
   form: predictive rows are "shared background + corrections at
   observed symbols" (as in the memory-1 experiments, T7); the
   product of two such rows lives on the union of the supports, so
   normalization costs the observed entries, not V.  The dense
   tables in scripts/pairwise_experiment.py are a small-V
   convenience only.
2. The scientific payoff is AT the full vocabulary, not away from
   it: full memory 2 loses there because of the V^3 learning cost,
   and the pairwise construction exists precisely to avoid that cost
   while keeping most of the benefit.  So the follow-up that
   matters, after the rules are ranked at V=1024, is the winning
   rules at full vocabulary in sparse form, where the question is
   simply whether pairwise beats lag 1.
3. Open question for that follow-up: whether the calibrated member
   can be made sparse (the IPF potentials go dense under the
   multiplicative updates as implemented).  If not, it may only be
   testable at moderate alphabet sizes.

Do NOT start the full-vocabulary follow-up yet; finish the queue in
the previous entry first (collect the fleet's results, add the exact
memory-2 numbers, write the layered comparison into the paper).

Also this session: scripts/assemble_pairwise_grid.py added
(read-only grid assembler over output/pairwise_*/results.json;
imports nothing a running job uses; verified against
pairwise_v1024, which reproduces the entry's numbers exactly).
Measured laptop state at the time: no output/pairwise_layered_v1024
directory yet, so the live layered run had written nothing to disk.
Noted for later: in pairwise_smoke_layered, markov2-layered (3.0744)
is WORSE than plain markov2 (3.0238) --- C=8 staleness at V=64 or
something real; the exact member will say.

## 2026-08-03 (evening) --- Appendix E added: three Bayesian routes to long memory

New appendix "Three routes to long memory, in the same calculus"
(app:routes) written into paper/main.tex, per Ruediger's request,
distilling the design discussion: Route 1 hierarchy (CTW with our
layered estimator per node; collapses to one exchangeable evaluation
per node, no sequential walk), Route 2 retrieval (Bayesian copy
pointer, forward recursion, needs the checkpointed walk; only route
cheap at full vocabulary), Route 3 switching (mixture over sequences
of family members; drop-in for the uniform family average).  Each
with mechanism, fit, cost growth, evidence.  Bibliography extended
(ctw95, memoizer, ppmstar, steinruecken, paq, switching, volf).
Compiles clean (pdflatex, 0 errors, refs resolve; now 18 pages).
NOTE: the 0.044 bits/token lag-tail figure cited under Route 2 is
quoted from the pooled-lag run --- verify against
output/pooled_lags_v4096 before submission.  Discussion of these
routes with Ruediger is pending (he will read and return with
questions); no implementation started, per the step-4 constraint.

Appendix E rewritten sequence-first after discussion with Ruediger:
each route now opens with "Prediction at position t" (what is
predicted, from which past counts), then "The codelength" (the
closed-form trick where one exists --- CTW's two telescopes --- or
the sequential walk where none does), then fit/cost/evidence.  CTW
prediction now presented as graded backoff with posterior blending
weights along the D+1 path nodes.  Compiles clean; main.tex and
main.pdf committed to the laptop.

## 2026-08-03 (late) --- CTW next step agreed: ctree D=3 at V=1024; context counts measured

Appendix E discussion led to reviving the context-tree line (the
implementation in src/product_model_with_memory/context_tree.py IS
Route 1 with layered leaves; tests incl. brute-force enumeration).
MEASURED on the reduced text8 stream (V=1024, n=17,005,207):
depth 1: 1,024 contexts; depth 2: 307,076 (124,287 singletons,
35,775 with >=20 obs); depth 3: 1,966,048 (1,258,381 singletons,
72,512 with >=20 obs).  Consistency: 1+1024+307,076 = 308,101 =
the ctree_v1024_d2 run's reported context count.  Cost estimate for
D=3 by context-count scaling: ~2-4 h on 20 laptop cores.
QUEUED (after the layered pairwise run frees the laptop and the
pairwise_experiment.py.new swap is done):
  caffeinate -i .venv/bin/python scripts/context_tree_experiment.py \
    --corpus data/text8 --top-k 1023 --depth 3 --jobs 20 \
    --out output/ctree_v1024_d3
Sanity on arrival: depth-0/1/2 baselines must reproduce
6.0636/5.0564/5.1309; then compare family vs 4.9507 and the MAP
leaf histogram vs 57 split contexts (does memory saturate at two?).
Ruediger's framing: V=1024 runs are behavioral (ranking), not a
compression scheme for the file; real-scheme claims live at full
vocabulary.

Appendix D (app:store) rewritten at Ruediger's request: it now
describes the system as built and verified --- the anchor store
(dense floor to 255, anchors at 8.3% spacing to 2e8, degree-11
interpolation in ln(r+1), expansion at L>=54), the measured trust
chain (probe_exact contour reference, 1e-4-bits requirement, worst
5.6e-6 over the 22 instrument profiles, one-octave margin), and the
saddle-rejection measurement kept in condensed form.  The old text
still called the storage problem open; numbers taken from the
2026-08-02 entries.

EFFICIENCY NOTE (Ruediger, from Activity Monitor during the layered
v1024 run): the run is mostly ONE Python process at ~14% of a core,
with occasional short bursts of ~5 workers at ~40% each; the machine
sits ~88% idle.  Measured from screenshots 23:18/23:19, checkpoint
~28.  So the checkpointed sequential evaluator uses roughly a tenth
of the laptop.  Fine for this run; NOT fine for the constraint-model
program (Appendix E Route 2), which uses the same machinery.  Before
those runs: batch per-context evaluations into array ops, keep a
persistent worker pool busy across the whole checkpoint, transfer
tables to workers once.  Correctness first, then this.

## 2026-08-04 --- LAYERED PAIRWISE v1024 DONE: calibrated WINS; markov2-layered anomalous

MEASURED (output/pairwise_layered_v1024/results.json, 29,622 s,
8.2 h wall; laptop, --jobs 12; text8 V=1024, C=32, cap=freq,
capped=6,658,395):
  calibrated 5.1983 <- BEST (posterior 1.0)
  markov2 5.2152, prod:1,0.5 5.2338, lag1 5.3146,
  markov2-layered 5.4031 <- WORST, below lag1.
THE RANKING FLIPPED vs counts mode (there: markov2 5.2446 best,
calibrated 5.2827).  With layered tables the maxent/calibrated
construction beats the order-2 backoff reference --- direct support
for the constraint-model route (Appendix E, Route 2).  All members
improved under layered tables.
ANOMALY: markov2-layered worse than lag1, same pattern as the
V=64 smoke (3.0744 vs markov2 3.0238).  Candidate explanation:
C=32 refreshes too few for the per-pair layered code.  The
--exact only run (launched next, same flags) computes the exact
sequential version; a large gap exact-vs-5.4031 convicts the
checkpoint schedule, a small one convicts the member.
Next on laptop after exact-only: ctree D=3 V=1024 (queued command
in the 2026-08-03 (late) entry).

EXACT NUMBER IN (--exact only, ~25 min): markov2-layered-exact
5.2053.  MEASURED verdicts: (1) checkpointing convicted --- the
0.198 gap to the checkpointed 5.4031 is the price of C=32 for the
per-pair layered member; conclusions about such members need the
exact form or a much finer schedule.  (2) calibrated 5.1983 still
beats even the exact order-2 layered reference, while itself paying
staleness.  Results committed and pushed (commit 3256b60).
PAPER: new section "The pairwise rules with layered tables"
(sec:pairwise-layered) added after sec:pairwise, with the two-round
table (counts vs layered), the three findings, and the forthcoming
runs marked; the placeholder in sec:pairwise now points there.
A future CTW section is intended to go BEFORE it (Ruediger).
NOTE: the ctree D=3 launch FAILED on first attempt (truncated
--out argument in the pasted command); relaunched separately ---
check for output/ctree_v1024_d3/results.json.

OVERNIGHT LAPTOP QUEUE (revised; single-line chain, ';' separated):
ctree D=3 layered -> ctree D=3 KT ablation -> pairwise counts
enwik8 (V=1024) -> pairwise counts enwik9 (V=1024).  The two enwik
counts runs were the server's unstarted jobs; the SERVER SHOULD NOT
run enwik8/enwik9 counts anymore (same out dirs, git conflict) ---
its remaining job is pairwise_v4096 only.

## 2026-08-04 (overnight) --- ctree D=3 in (80 s!); enwik pairwise runs done but CALIBRATED IS NAN

MEASURED, ctree D=3 V=1024 with the anchor ladder (80 s wall ---
the ladder makes these runs near-free; the KT ablation took 23 s):
  fixed depths: d0 6.0636, d1 5.0564, d2 5.1306, d3 5.9071
  family 4.9390 (D=2 family was 4.9507: depth 3 adds 0.0117)
  MAP 4.9396, leaves {1:969, 2:21817, 3:17698} -> depth 3 heavily
  used; gains per depth collapsing (0.106 then 0.0117).
  Sanity: d0/d1 exactly reproduce the old run; d2 5.1306 vs old
  5.1309 (corrected evaluator).
  KT ablation: family 5.0825, leaves {1:1020, 2:1827, 3:810}.
  D=4/D=5 now cheap to try (context-count check first).

MEASURED, pairwise counts enwik8 (999 s) and enwik9 (8,084 s),
V=1024 C=32 cap=freq, laptop:
  enwik8: markov2 4.3096 best, prod:0.75,0.5 4.4222, lag1 4.6454;
  enwik9: markov2 3.7360 best, prod:1,0.5 3.9879, lag1 4.3777;
  *** calibrated = NAN on BOTH, family/posterior NAN. ***
BUG: RuntimeWarning invalid/overflow at pairwise_experiment.py:119
(psi12 *= P12 / np.maximum(M12/M12.sum(), 1e-300)) at checkpoint 6
on both files; ipf_residual_l1 = nan from then on.  IPF was already
at the 300-sweep cap with resid ~5e-4 on enwik8 BEFORE the nan
(text8 converges to 1e-9).  Non-calibrated members come straight
from tables and look sane.  WARNING: the server pairwise_v4096 run
uses the same code path --- check its output for the same failure.
TODO next session (in order): fix the overflow (log-domain or
renormalize psi12 per sweep; investigate the poor convergence at
the same time); verify the fix reproduces text8 counts calibrated
5.2827 and layered 5.1983; rerun enwik8/enwik9; then collect
second-Mac + server results and assemble the full grid
(scripts/assemble_pairwise_grid.py).

PAPER: preliminary section "Adaptive depth: the context-tree family"
(sec:ctree, tab:ctree-v1024) added between sec:ordertwo and
sec:pairwise, with the D=2/D=3/KT table; the layered-round section
now reports the enwik counts numbers and the calibrated-nan status
in prose.  Both marked as preliminary/forthcoming where applicable.

## 2026-08-04 (morning) --- Layered enwik8 in (calibrated nan AGAIN); server v4096 diagnosis; duplicate-run trap

MEASURED, pairwise_layered_enwik8 (second Mac, 32,651 s, results
pulled to laptop and pushed): markov2 4.3017 best; prod:0.75,0.75
4.3963; markov2-layered 4.4194 (NOT the text8 disaster of 5.4031;
exact number pending on that Mac); lag1 4.6378; calibrated NAN ---
third run lost to the pairwise_experiment.py:119 overflow.  Layered
minus counts gains on enwik8 are small (markov2 -0.008): more data
per table, less estimation benefit.  Paper's layered section
updated accordingly.

SERVER v4096 (log seen at checkpoint 26/32, ~2,200 s/checkpoint,
~16 h elapsed, ~4 h to go): single core because the command had no
--jobs (default 1).  BUT the dominant phases (IPF sweeps, coding
loop) are single-core regardless of --jobs --- confirms the
efficiency note as the gate to the constraint-model program.
No nan so far on text8 V=4096 (resid ~6e-6 at the 300-sweep cap;
compare 1e-9 at V=1024 --- convergence degrades with V).

TRAP: the server shell has enwik8 queued as typed-ahead input, then
a nohup'd enwik9 --- both would duplicate laptop runs already
pushed.  When v4096 prints its member table: Ctrl+C the starting
enwik8, then pgrep -af pairwise and kill survivors; git add ONLY
output/pairwise_v4096 on the server.

## 2026-08-04 (day) --- IPF overflow FIXED; scoring PARALLELIZED (threads); server run killed

The entry-node v4096 run was killed (wrong place to compute; also
counts mode ignores --jobs entirely --- it is consumed only by the
layered builder, MEASURED from the code).  Two changes to
scripts/pairwise_experiment.py, installed on the laptop:

1. ipf_triangle: each psi factor rescaled to max 1 after its update
   (model provably invariant --- every use normalizes over the
   target); floor 1e-300 -> 1e-150; degenerate margins reset the
   factors and report resid=inf, never nan.  MEASURED: smoke
   reproduces published numbers EXACTLY (calibrated 3.0307,
   posterior 0.5296); v1024 fixcheck checkpoints 1-4 reproduced the
   original run's IPF diagnostics digit for digit before Ruediger
   killed it in favour of the parallel version.

2. Scoring loop threaded (ThreadPoolExecutor over position chunks;
   numpy releases the GIL in the array arithmetic).  Partials merged
   in chunk order -> BIT-IDENTICAL to serial (verified in the cloud
   sandbox: jobs 1 vs jobs 2 results.json equal bit for bit).
   markov2-layered excluded from the pool (shared memoized builder);
   it runs in a separate serial pass --- parallelizing the layered
   member is still open.  IPF matmuls also still serial --- whether
   the numerical library threads them is machine-dependent (probe
   outstanding).

VALIDATION RUN (in flight or next): v1024 counts fixcheck with
--jobs 12; must reproduce calibrated 5.2827 and the full published
table.  THEN rerun enwik8 (~17 min serial, less now) and enwik9,
then v4096 counts on the laptop.  Server: nothing running; cluster
use deferred until the program parallelizes properly.

## 2026-08-04 (midday) --- Fix VALIDATED (5.2827 exact); 2.3x measured; phase economics understood

MEASURED: v1024 counts fixcheck (--jobs 12, threaded-scoring version)
reproduces the published table digit for digit --- calibrated 5.2827,
markov2 5.2446, lag1 5.3393, family 5.2446, posterior 1.0 --- in
354 s vs 799 s serial.  MEASURED phase breakdown (4-checkpoint slice,
output/pairwise_timing_probe): ipf 23.7 s, score 16.1 s, everything
else 0.5 s.  BLAS probe: numpy on Accelerate; 10 matmuls 2048^2 in
0.48 s (~360 GFLOPS via the AMX coprocessor) --- matmuls are already
on the fastest unit and show as ONE busy core in Activity Monitor;
visual utilization is the wrong instrument, wall time is the right
one.  Further script versions installed since the fixcheck ran:
bincount counting, sparse gather hoisted out of threads, IPF
elementwise threaded via one shared pool (all verified bit-identical
on smoke in the cloud sandbox).  Remaining serial floor: AMX matmuls
+ IPF sweep dependency; next lever would be ALGORITHMIC (sweep cap),
which changes numbers --- Ruediger's call only.
QUEUED on laptop (single line, sequential): enwik8 counts rerun,
enwik9 counts rerun (both replace nan-calibrated results), then
pairwise_v4096 (never completed anywhere).  Read results.json as
each lands; then grid, paper, handover.

FIXCHECK COMPLETE (instrumented version): full published v1024
counts table reproduced digit for digit, incl. per-checkpoint IPF
residuals.  MEASURED full-run phases: wall 369 s; ipf 255 s (69%),
score 109 s (threaded), tables 3 s, reveal 1 s.  PIPELINE VERSION
INSTALLED after this run: fitting of checkpoint c+1 overlaps scoring
of checkpoint c (state snapshotted; merges in checkpoint order;
verified bit-identical on smoke, jobs 1 and jobs > 1).  Expected
v1024-class wall ~270 s (3x vs 799 serial).  NOTE: with overlap on,
phase_seconds may sum to MORE than wall (concurrency) --- wall is
the metric.  Remaining floor: the IPF sweep chain (sequential, AMX).
Algorithmic lever documented and DECLINED by default: lowering the
300-sweep cap changes calibrated digits (Ruediger's call only).
CAUTION for the next --tables layered run: the layered member's
serial pass was carried through the pipeline refactor unchanged and
overlap is disabled when it is present, but run the layered smoke
first (--n 300000 --top-k 63 --tables layered --order2 both) as a
cheap check before any long layered run.
Chain queued on laptop (pipelined script): enwik8, enwik9, v4096.

## 2026-08-04 (afternoon) --- Pipeline post-mortem and the lag-k fit chains

MEASURED, pipecheck: the first overlap attempt delivered NOTHING
(373 s vs 369 s) --- the fitter's threaded elementwise helpers
shared the scoring pool and queued behind scoring chunks, slowing
ipf 255 s -> 360 s.  Fix: fitter off the shared pool (serial
elementwise on its own thread).

NEW: --ipf-lag k (Ruediger's idea): warm-start each calibration fit
from the fit k checkpoints back -> k independent fit chains running
concurrently on a k-worker fitter pool; scoring merges strictly in
checkpoint order behind them; bounded window (lag+1) keeps memory
flat.  Validity unchanged (fits use only prefix data).  MEASURED on
smoke: lag 1 BIT-IDENTICAL to all previous versions (default;
comparability preserved); lag 2 changes ONLY calibrated, by 3e-11
bits/token (start-independence where fits converge).  Digits move
materially only where the 300-sweep cap binds --- and the runs where
it binds (enwik8/9, v4096) have no surviving calibrated number, so
nothing comparable breaks.  Layered member forces the strict serial
path (sync_mode).
--ipf-iters remains the other knob: raising it is free of
comparability cost on exactly those same runs.
Verification queued: v1024 lagcheck at --ipf-lag 3 --jobs 12;
expect all members identical, calibrated ~5.2827 to 4 decimals,
wall 130-160 s vs 369.  THEN the enwik8/enwik9/v4096 chain (use
--ipf-lag 3 there too; for v4096 consider --ipf-iters 1000 if the
lagcheck diagnostics look clean, since fitting is its dominant
phase and nothing published constrains it).

## 2026-08-04 (late afternoon) --- Anderson-accelerated IPF; concurrent fleet launch

MEASURED, lagcheck (--ipf-lag 3): 244 s vs 369 s (34%); full table
incl. calibrated 5.2827 reproduced at 4 decimals; fit thread-time
tripled (698 s over 3 chains) --- AMX contention confirmed; lag-k is
capped well short of k-fold.
NEW: --ipf-solver anderson (default remains "ipf", bit-identical).
Same fixed-point map, same residual test, same unique optimum;
Anderson mixing over a small history.  Three implementation lessons,
all MEASURED in the sandbox: (1) naive version lost the win to
per-sweep log/exp and history re-stacking (235 s ipf despite 5x
fewer sweeps); (2) float32 history fails in BOTH domains (linear:
underflow across ~300 orders of magnitude; log: floored entries at
magnitude ~700 x float32 eps = 4e-5 noise, killing convergence ---
sweeps back to cap); (3) gauge consistency: never renormalize the
state without updating its logged copy (the map is scale-invariant,
so skip renormalization entirely).  Final form: linear-domain map,
float64 log-domain history, incremental ring updates, no m x n
temporaries.  Smoke: sweeps 300 -> 20-55, optimum agreement 2.6e-10,
default path bit-identical.
LAUNCHED concurrently on the laptop (Ruediger's suggestion ---
complementary bottlenecks: enwik9 scoring-bound on cores, v4096
fit-bound on AMX): enwik8 (jobs 4), enwik9 (jobs 8), v4096 (jobs 4),
all --ipf-solver anderson, logs at output/log_*.txt.  All three had
no surviving calibrated number, so the anderson digits break no
comparability.  v1024-class published rounds keep solver=ipf.

FLEET MID-RUN FINDING (from logs, runs untouched): enwik8/enwik9
fits hit the 300-sweep cap at a residual FLOOR ~8e-4/1.2e-3 under
BOTH solvers (plain IPF stalled at 5e-4 pre-fix; anderson stalls
similarly).  INFERRED (strong): the three pair tables are not
mutually consistent on enwik --- project_margins' fixed 500 Sinkhorn
rounds (tol 1e-13) suffice for text8 but not for enwik's heavier
tail; an inconsistent triangle has NO exact fixed point, so no
solver can pass the floor.  Check after the runs: resid constant
across checkpoints = floor, falling = convergence.  Fix candidate
for the next enwik round: raise project_margins iters (or make it a
flag) and re-measure the floor.  Today's calibrated numbers remain
valid sequential codes, conservatively fitted.

## 2026-08-04 (evening) --- CTW grid COMPLETE (M4 Max, ~2 min/run); full-vocab integrity settled

MEASURED (second Mac, anchor ladder, corrected evaluator; results
pulled to laptop and in git):
  depth ladder V=1024: family 4.9507 (D2) -> 4.9390 (D3) -> 4.9384
  (D4); added value 0.1057 / 0.0117 / 0.0006 --- geometric collapse,
  saturation at depth 2-3.
  alphabet trend (best D): V=1024 gain +0.1180; V=4096 +0.0366
  (family 6.6766, D3 adds ~0.004); V=16384 +0.0136 (family 8.2382,
  reproduces July to 4th decimal); FULLVOCAB -0.0011 (family
  10.0926 LOSES to fixed d1 10.0915) --- depth wrong axis at full V,
  now confirmed under the corrected system.
  INTEGRITY: ctree_fullvocab question CLOSED --- correct value
  10.0926; the published 10.0919 was low by ~7e-4 (old evaluator +
  contaminated cache).  Update legacy references accordingly.
  Cross-check: d1@4096 = 6.7132 = verified state4096 number.
PAPER: sec:ctree rewritten --- two tables (tab:ctree-depth,
tab:ctree-vocab), integrity notes, closing verdict: the tree is how
to SPEND memory of a few symbols, not how to acquire more; reach
must come from constraint models (V^2) and retrieval (V-free).

EDITORIAL RULE (Ruediger, adopt paper-wide): every section must open
with the scoreboard question --- does this construction move the
best honest bpc number toward Table 1's published targets --- and
answer it in comparable units; reduced-stream tables must be labeled
as the mechanism arena, explicitly not convertible to bpc.
sec:ctree rewritten to this form (full-fidelity verdict: adaptive
depth loses 0.0002 bpc at full vocabulary; arena tables demoted to
instrument status; the ~30k paying contexts flagged as the synergy
map for the constraint models).  STILL TO CONVERT to this form:
sec:pairwise and sec:pairwise-layered (their scoreboard line: no
full-fidelity claim yet by design --- the scope note --- with the
sparse full-vocab follow-up as the payoff), and a check that every
other section leads with its bpc consequence.  enwik8/enwik9 ctree
grids running on the second Mac will extend tab:ctree-vocab to
three corpora.

## 2026-08-04 (evening, cont.) --- enwik8 counts rerun DONE: calibrated finite at 4.4836

MEASURED (2,464 s under 3-way contention): markov2 4.3096 best
(reproduces the damaged run's non-calibrated members), calibrated
4.4836 --- FINITE, mid-pack: behind tempered products (best 4.4222),
ahead of star/mixes/lag1.  Residual trace: erratic, 8e-4..7e-1
across checkpoints --- consistent with the margin-inconsistency
floor, NOT with clean convergence; the calibrated number is a
conservatively-fitted lower bound.  Paper's layered-round paragraph
updated accordingly (caveat included).
NEXT-ROUND FIX (before drawing conclusions about calibrated at
scale): raise project_margins rounds (make it a flag), re-measure
the floor on enwik8, rerun.
IN FLIGHT: enwik9 at checkpoint 29/32; v4096 SLOW --- anderson at
cap 300 every checkpoint with resid floor ~3e-5, ~2,200 s/checkpoint
under contention -> overnight job; leave it.  ctree enwik8 grid
pushed from second Mac (git divergence resolved via pull --rebase);
enwik9 ctree grid queued there.

## 2026-08-04 (night) --- enwik9 pairwise done; enwik8 CTW grid in; paper two-corpus

MEASURED, pairwise enwik9 counts rerun (2,874 s): markov2 3.7360
best; calibrated FINITE 4.0920, behind all tempered products (best
3.9611) and the star; residuals ~1e-3..1e-4 (same inconsistency
floor as enwik8).  Both enwik calibrated numbers are provisional
lower bounds until project_margins is strengthened.

MEASURED, ctree enwik8 grid (second Mac, in git):
  V=1024 depth ladder: family 3.9844/3.9385/3.9265 (D2/D3/D4);
  added-by-depth 0.1194/0.0459/0.0121 --- collapse ~1/3 per depth,
  much slower than text8 (~1/10): markup structure runs deeper.
  D=5 at V=1024 on enwik8 likely still pays ~0.004 --- cheap, worth
  running.  V=4096 D3 gain +0.0844 (double text8's).  V=16384 gain
  +0.0020 (nearly dead --- earlier than text8's +0.0136).
  FULLVOCAB (V=300,001): family 10.8075 vs fixed d1 10.8050: loses
  0.0025 b/t --- the full-fidelity verdict holds on BOTH corpora.
PAPER: sec:ctree tables now two-corpus (tab:ctree-depth,
tab:ctree-vocab); conclusions note the file-dependence of the depth
collapse; layered-round paragraph carries enwik9 calibrated.
IN FLIGHT: v4096 pairwise --- cp3 converged at 98 sweeps/1e-9 (warm
start engaged; cp2's 3e-5 cap-hit was the cold chain) -> ~700-800 s
per checkpoint, ETA ~6 h, overnight.  enwik9 CTW grid queued on the
second Mac.

## 2026-08-04 (late) --- TOKENIZER DECREE + measurement harmonization

RUEDIGER'S STANDING DECISIONS (supersede anything contrary):
1. ONE measure: bits per character of the original file, honest
   complete codes.  Reduced-alphabet runs must be COMPLETED (price
   the catch-all contents + vocabulary charge + boundary) so every
   table reports bpc; rankings survive (completion is
   member-independent per run).
2. ONE tokenization: the LLM tokenizer (the bpe_* streams).  All
   word-tokenizer results are SCRATCHED for the paper: the entire
   ctree grid (July + today, both corpora), state_family_fullvocab,
   sf-mid (1.7503/1.6979 chain).  Pairwise experiments already live
   on bpe streams and survive untouched.
STATUS of the redo machinery: scripts/context_tree_experiment.py now
takes --ids <stream dir> (LLM streams); smoke-verified in sandbox.
Ctree grids to REDO on bpe_text8/bpe_enwik8 (laptop, alongside the
v4096 run) and bpe_enwik9 (heavier).  scripts/complete_bpc.py (the
bpc completion tool) still TO BE BUILT --- next session's first
task, then the paper-wide table rewrite in bpc.
Paper's sec:ctree numbers are word-tokenizer: to be REPLACED by the
bpe redo (skeleton/prose stays).  Mac's enwik9 word-level ladder:
STOP if still running.

## 2026-08-04 (midnight) --- CTW ON THE LLM CHAIN: THE VERDICT FLIPS; one honest table in the paper

MEASURED (ctw_* runs, bpe streams, top-k 100276 = no catch-all;
bpc = (family_bits + boundary at d0 rate)/chars + vocab charge
0.0621):
  VALIDATION: depth-1 rows reproduce tab:firstorder EXACTLY
  (text8 1.8298, enwik8 2.1135) --- accounting closed.
  text8:  D2 1.8185, D3 1.8182 bpc (-0.0113/-0.0116 vs first order)
  enwik8: D2 2.0676, D3 2.0606 bpc (-0.0459/-0.0529)
  Complete fixed depths >1 lose badly (d2 alone 2.31/2.53) --- all
  gain is per-context depth choice.  BEST HONEST NUMBERS IN THE
  PAPER so far on both files.
THE FLIP: on word tokens adaptive depth showed ~no gain at full
fidelity; on subword tokens it pays clearly.  Ruediger's tokenizer
decree changed the scientific conclusion, not just the bookkeeping.
PAPER: sec:ctree = ONE table (tab:ctw), honest bpc, three corpora,
word-chain results explicitly superseded in the text.
RUNNING: ctw_enwik9_d2/d3 (laptop, --jobs 12) -> fills the enwik9
rows (first order there 1.8371 bpc); v4096 pairwise still grinding
(cp7, resid stalls ~2e-4 at later checkpoints --- margin
inconsistency at V=4096 too; parked until pairwise phase resumes).
Next for CTW per Ruediger: nothing else --- this table IS the CTW
deliverable; D4 rows optional if he asks.

## 2026-08-05 --- Pairwise sections merged; program redefined to the one standard

PAPER: old sections 7+8 (pairwise counts round + layered round)
replaced by ONE section (sec:pairwise): design, protocol,
accounting; measured table forthcoming.  Standard: layered tables
ONLY (no count-smoothed variants), full LLM vocabulary, complete
codes, bpc.  The reduced-V rounds are cited as the exploratory round
that fixed three design decisions (layered tables mandatory; exact
form for the share-nothing layered reference --- the 0.198 b/t
staleness measurement; margin consistency before fitting) with NO
raw numbers in the paper.
IMPLEMENTATION TODO before the pairwise campaign can run:
1. Sparse pairwise machinery at full vocab (rows = background +
   corrections; products on support unions) --- currently the script
   is dense (V=1024/4096 only).
2. project_margins strengthening (iters flag + measure the floor).
3. Exact layered order-2 at full vocab: telescopes, needs profile
   dedup over ~millions of context pairs --- feasibility check.
4. Calibrated at full vocab: sparse IPF open; else moderate-V
   COMPLETE code (side-channel priced) per the section text.
V4096 COUNTS RUN: cannot produce a paper number under the standard
(counts tables, reduced V) --- recommended KILL; Ruediger to
confirm.  CTW: awaiting ctw_enwik9_d2/d3 to finish tab:ctw; then
CTW is done per Ruediger (D4+ on enwik9 only if D3's increment
warrants).

CTW enwik9 D2 IN (1,314 s): family 6.1310 b/t = 1.6840 bpc,
-0.1531 vs first order 1.8371 --- LARGEST GAIN of the campaign,
2.75M depth-2 leaves; d0/d1 rows validate exactly (2.9775/1.8371).
Best honest number in the project.  tab:ctw updated.  D3 running;
D4 on enwik9 clearly warranted after D3 (watch counting-phase
memory).  v4096 counts run KILLED by Ruediger (cannot enter the
paper under the standard).

CTW enwik9 D3 IN (2,282 s; 76.8M contexts, 933k profiles): family
6.0424 b/t = 1.6598 bpc, -0.1773 vs first order; D3 adds 0.0242
over D2 (enwik8's D3 added 0.007); 2.81M depth-3 leaves.  CURVE NOT
SATURATED on enwik9 -> D4 is the next measurement there.
IMPLEMENTATION PREREQUISITE for D4 at full vocab: the T4 packed
context keys overflow int64 at V^4 (100277^4 ~ 1e20 > 2^63) ---
counting needs per-level key remapping (rank contexts against the
observed depth-3 set) or 128-bit keys before a D4 run can exist.
Bounded work; do BEFORE promising the run.  tab:ctw is otherwise
COMPLETE (3 corpora x D2/D3, all validated against tab:firstorder
/ tab:memoryless anchors).  main.tex committed; PDF recompile
pending (sandbox Bash briefly unavailable).

SCOREBOARD added at the top of main.tex (unnumbered section before
Sec. 1): bytes / LLM memoryless / first order / context tree (bold
current bests 1.8182, 2.0606, 1.6598) / pairwise (in progress) vs
published records, all honest bpc.  RULE: every section that
produces a number updates this table.  D4 skipped per Ruediger
(after tiny-margin assessment).  Published-target verification:
nncp v3.2 enwik9 = 107,261,318 bytes WITH decompressor = 0.858 bpc,
same rules (lossless whole file, decompressor counted, no external
data; model trains online on the file) --- confirmed from LTCB
(mattmahoney.net/dc/text.html) on 2026-08-05.  Bash sandbox
recovered; PDF current again.

## 2026-08-04 (afternoon 2) --- CHUNKING-COST STUDY: first measurements are decisive

New instrument scripts/chunking_cost.py (exact telescoped reference
vs chunked evaluation; C grid; equal vs geometric spacing; per-block
batched evaluations; incremental saving added mid-study).  MEASURED
so far (bpe_text8, order 1, V=1024; exact reference 5.1443 b/t ---
first exact number on this reduced stream, no prior anchor):
  C=8 equal: +0.6208 b/t.  C=8 geometric: +0.0682.  NINE times
  less staleness from checkpoint PLACEMENT alone (small blocks
  early where tables learn fastest).  Smoke (V=64 order 0, C=4):
  equal +0.810 vs geo +0.109.
Runtime note: each configuration costs millions of fresh ladder
evaluations (every checkpoint re-freezes every profile) -> ~10-20
min per config at this size, NOT minutes total; this is the physics
of the measurement.  Remaining: C=32, C=128 both spacings, then
full-vocab orders 0/1 on text8+enwik8.
IMPLICATION already visible: the exploratory pairwise rounds ran
C=32 EQUAL --- much of their staleness was avoidable; geometric
checkpoints become the default for every future chunked experiment
(pairwise campaign included), pending the full grid.
Paper: bits-per-character = bits-per-byte convention paragraph
added to Sources (Ruediger).

CHUNKING STUDY CLOSED EARLY (Ruediger; grid truncated on evidence).
MEASURED, bpe_text8 order 1 V=1024 (exact 5.1443):
  C=8  equal +0.6208 (596s)  | C=8  geo +0.0682 (150s)
  C=32 equal +0.1703 (4393s) | C=32 geo +0.0319 (630s)
CROSS-VALIDATION: C=32 equal = 5.3146 = the pairwise layered
round's lag1 member EXACTLY (independent program, same protocol).
CONCLUSIONS (standing):
1. GEOMETRIC spacing is the default for all future chunked runs
   (9x less staleness at C=8, 5x at C=32; also CHEAPER to evaluate
   --- fewer distinct profile snapshots).
2. The exploratory pairwise rounds (C=32 equal) carried ~0.17 b/t
   avoidable staleness on lag1-class members at this scale; treat
   their absolute numbers accordingly.
3. C=128 rows and the full-vocab grid SKIPPED --- the geo curve
   (0.068 -> 0.032) is already in diminishing returns; each future
   chunked campaign can measure its own gap cheaply at small scale.
TODO before the pairwise campaign: add --spacing {equal,geo} to
scripts/pairwise_experiment.py (edges as in chunking_cost.py);
default geo.  The killed run wrote no results.json (predates the
incremental-save fix); the numbers above, from the terminal, are
the record.
CORRECTION (Ruediger): the 4393 s for C=32 equal includes a period
with the laptop lid closed --- wall time inflated, not
representative.  The gap numbers are unaffected (codelengths are
clock-independent).  Honest timing comparison: C=8, equal 596 s vs
geo 150 s (~4x).

RENAME (5 Aug): the dense reduced-alphabet script pairwise_experiment.py
-> pairwise_arena.py (mechanism arena; keeps the --exact merge mode).
The tables-free campaign evaluator pairwise_sparse.py ->
pairwise_experiment.py.  All handover mentions of
pairwise_experiment.py BEFORE this line mean the arena script.

## 4 Aug 2026, evening: tables-free evaluator rewritten, validated, fast
- scripts/pairwise_experiment.py is now the vectorized + threaded
  tables-free evaluator (no Python loop walks tokens, pairs, or
  triples; numpy on whole blocks, chunked over a thread pool; one
  batched builder request per block).  The old slow tables-free
  version is superseded; scripts/pairwise_fast.py is an identical
  copy left from testing and can be deleted.
- Validated on the laptop, bpe_text8 full vocabulary (V=100,253),
  4M-token slice, C=16 geo: every member agrees with the slow
  reference to 8.6e-12; wall time 1557 s -> 802 s at --jobs 8.
  Earlier check at V=64: agreement to 2.4e-14.
- Measured phase split of the 802 s: layered-table work 792 s,
  everything else 10 s.  The table phase is the parallel part; use
  --jobs 12 on the 12-core laptop.  New sub-timers prep_py/prep_tab
  report the serial-vs-parallel split inside the table phase on
  every future run.
- Science note (slice, to be confirmed by the campaign): at full
  vocabulary the mixes win (best 9.75 bits/token), products collapse
  (up to 15.4), lag1 10.65 --- the reverse of the V=1024 arena
  ranking.

## 4-5 Aug 2026, night: performance ledger of the pairwise evaluator
Benchmark: bpe_text8 full vocabulary, 1M tokens, C=8, --jobs 12.
Every step's members agreed to at least 1e-11; only time moved.
- Baseline barrier scheduling + per-checkpoint worker start: 123 s.
- KEPT: persistent worker pool, definitions once per call via shared
  memory, 4 work pieces per worker per level, task-time accounting
  (eval vs eval_cpu columns) -> 99 s.  This is the standing version.
- TRIED AND REVERTED (all correct, all slower): no-barrier task
  scheduling (269 s: level early-stop lost, giant families computed
  all 54 levels); + worker-side early stop and row cache (295 s:
  stop rule almost never triggers at these profile sizes); giant
  families as concurrent level-waves (100 s: freed eval time went to
  queue contention); + fill prefetch and longer waves (112 s: wave
  overshoot).  Lesson written in blood: scheduling changes traded
  seconds around a ~100 s plateau; do not retry these shapes.
- Measured structure of the 99 s: eval wall 60, of which true worker
  work 427 cpu-s (7.1 of 12 busy; the level barrier and indivisible
  giant families set the pace); fill 17; ~22 s parent-side Python
  BEFORE workers start (collecting/validating families, needed_r
  unions, window/setup) --- the largest single untouched item and
  the right next target, needs restructuring not knobs.
- py-spy cannot attach on this macOS (timeout even with sudo); the
  in-code eval/eval_cpu accounting is the working substitute.
- 4M slice, C=16, jobs 8 standing: 802 s (was 1557 before the
  vectorized script).  Campaign not yet launched.

## 5 Aug 2026: state of play for a fresh start on evaluator performance

For whoever works on this next. Read the ledger entry above first;
it lists four scheduling designs that were built, measured, and
reverted. Do not rebuild them from theory.

WHAT STANDS (all validated, member values stable to 1e-11 across
every variant tried):
- scripts/pairwise_experiment.py: tables-free pairwise evaluator,
  numpy on whole blocks, threaded. Its own cost is ~10 s of the
  benchmark; it is NOT the problem.
- src/product_model_with_memory/codelength.py: families evaluation
  with a persistent worker pool, per-level scheduling, self-reported
  timing columns per checkpoint line (tab[...]).
- Benchmark command (the ONLY agreed yardstick, ~99 s standing):
  export PMM_UNIVERSAL_TABLES=tables/anchors_prod \
    PMM_PHI_LADDER_EVERY=1 PMM_PHI_LADDER_DEGREE=11 \
    PMM_PHI_SADDLE_MIN_L=54 ; \
  .venv/bin/python scripts/pairwise_experiment.py \
    --ids output/streams/bpe_text8 --n 1000000 --checkpoints 8 \
    --jobs 12 --out output/pws_bench_1M

THE OPEN PROBLEM, measured (from the 99 s run and its columns):
- eval wall 60 s vs true worker work 427 cpu-s: 7.1 of 12 cores.
  Pace is set by the per-level barrier (54 levels, two pool
  round-trips each) and by giant families (unigram row, ~35k-entry
  profile; common states with >512 distinct successors) that are
  indivisible within a level. Splitting a giant family's SCAN across
  the grid (inside layered.py: log_q_lambda_scan_family) is the
  untried move that would actually break the straggler; every
  scheduling-level workaround was tried and failed.
- ~22 s parent-side Python inside each _ensure_families call before
  any worker starts: needed_r unions, validation, window and chunk
  setup over tens of thousands of families. Untouched; second target.
- fill (store row reads) 17 s: prefetching it into the eval tail was
  NET NEGATIVE at the benchmark (queue contention) - see ledger.
- The level early-stop (_LevelWindow, 80-bit drop rule) is essential:
  removing it multiplied worker cpu ~6x. Any redesign must keep
  sequential level order per family, or replicate the rule exactly.
CONSTRAINTS that cost a night to learn:
- macOS: worker process start is expensive (spawn); py-spy cannot
  attach (timeout, even sudo). The in-code timing columns are the
  measurement instrument. The cloud sandbox has 2 Linux cores:
  correctness transfers, parallel behavior does not - benchmark only
  on the laptop.
- RUSAGE_CHILDREN only counts exited workers: useless with a
  persistent pool; that is why eval_cpu comes from the tasks.
- Stale git locks may need: rm -f .git/HEAD.lock \
  .git/objects/maintenance.lock

SCIENCE QUEUE (untouched by all of this, ready to run):
- text8 campaign: full vocabulary, C=32 geometric, jobs 12,
  all members; slice results (4M: mixes ~9.75 best, lag1 10.65,
  products collapse) await confirmation at full length; then the
  paper's pairwise section gets its measured table.

Correction to the ledger, for honesty of the record (Ruediger's
point, 5 Aug): the 1557 s "baseline" was itself degraded by a
repeated known mistake (per-item builder requests - the same bug
introduced four separate times this week). The week's hand-tuned
programs ran at ~80% utilization; no version of the new evaluator
has reached that. The night's "factor 2" recovered self-inflicted
loss; the actual standard to meet is the 80% of the existing
programs, and it has not been met.

## 5 Aug 2026: three-pair calibration reformulated conditionally

Ruediger's decision: the calibrated model using ALL THREE pair
marginals is the point of Section 7. Mixtures/products that omit the
past--past pair are controls, not the scientific deliverable. Do not
launch the full campaign without a scalable calibrated member.

Added paper/graphical_model_prediction_note.tex unchanged from
Ruediger's supplied note. Added
src/product_model_with_memory/graphical_calibration.py: a dense
small-alphabet reference for conditional IPF. It eliminates the
past--past potential analytically by writing

  q(y,a,b) = P_ab(a,b) r(y|a,b),
  r(y|a,b) proportional to f_ya(y,a) f_yb(y,b),

then alternately corrects the produced (Y,A) and (Y,B) margins. P_ab
is exact after every update by construction. This is algebraically
the same three-pair maximum-entropy triangle, not a new estimator.

VALIDATED in tests/test_graphical_calibration.py:
- agrees with literal V^3 joint IPF to <2e-11 per joint cell;
- agrees with scripts/pairwise_arena.py's existing three-factor IPF
  to <2e-11;
- duplicate-lag case returns the one-lag conditional rather than
  squaring the evidence;
- rejects inconsistent pair margins.
Four tests pass; ruff passes.

NEXT: implement the same conditional-IPF operator on the layered
row representation (implicit background + observed corrections).
The remaining question is closure: after a calibration update, do
inactive target cells share an aggregable correction? If not, use an
explicit active-set/coarsened-feature definition and measure it
against the dense reference at small V. The current fast
pairwise_experiment.py still has NO calibrated member.

STRUCTURE PROBE COMPLETED (real bpe_text8 prefix, layered margins):
- New scripts/calibration_structure_probe.py.
- V=32, n=100k: full conditional IPF converged in 60 sweeps with all
  three L1 margin residuals below 8e-12.
- V=64, n=300k: converged in 350 sweeps, residuals below 1e-11.
- VERDICT: the uncoarsened exact triangle is NOT closed under one
  implicit background per factor row. After the best possible shared
  target gauge, inactive-cell log-factor RMS/max errors were
  0.331/1.726 at V32 and 0.485/3.413 at V64. Do not try to force the
  full solution into the current sparse-row shape.

Implemented the principled alternative in graphical_calibration.py:
grouped_conditional_ipf. It defines an exact coarsened exponential
family: every observed target--lag cell is constrained separately;
all unobserved target cells in one context row form one aggregate
constraint; P_ab is still enforced exactly. This representation is
closed under background + observed corrections by construction.
Validation test confirms every active cell, every inactive aggregate,
and P_ab are matched.

FIRST COST MEASUREMENT (V32, n=100k): after 20k plain sweeps the
grouped residual was 5.3e-8 (strict 1e-11 stop not yet reached), while
KL(grouped || full triangle) was only 0.0001211 bits/record. Very
promising, but the slow convergence means the next implementation
step is an accelerated/matrix-free grouped solver, then V64 and
larger comparisons. Five calibration tests pass; ruff passes.

WARM START + INTERLEAVING (same day): grouped_conditional_ipf now
accepts prior log factors. A checkpoint-like regression confirms a
warm fit reaches the same joint as a cold fit and uses fewer sweeps.
Added GroupedCheckpoint + fit_grouped_checkpoints: with interleave=k,
chain j fits j,j+k,j+2k,... sequentially, warm-starting within the
chain, while k chains run concurrently; results return in checkpoint
order. Regression: k=3 and k=1 produce the same six checkpoint joints
to <2e-10. Seven calibration tests pass; ruff passes. This preserves
the earlier pairwise evaluator lesson: a slightly older warm start is
worth using when multiple serial calibration chains keep the CPU busy.

TARGET MAIN EFFECT CORRECTION: the grouped model must separately
preserve the full Y marginal; active cells + one inactive aggregate
per row do not imply it. Added an explicit global log_base_y factor
and an exact Y-margin IPF update. The scalable parameterization is now
layered unigram q0(y) times a fitted Y correction times sparse lag
corrections. Since P_Y is constrained, using q0 rather than uniform
changes the parameterization/initialization, not the optimum. All
seven tests still pass. On the V32 real-prefix probe the corrected
grouped model's KL from the full triangle is 4.95e-5 bits/record
(better than the earlier incomplete model's 1.21e-4); plain IPF is
still slow (grouped residual 1.34e-6 after 20k sweeps).

SCALING BOUNDARY IDENTIFIED: sparse evaluation of one q(y|a,b) is
not enough. Exact enforcement of the layered, full-support P_ab asks
for averages over every a,b, including unobserved context pairs: V^2
context combinations. No table need be stored, but the computation is
still prohibitive. A scientific choice is required before the
matrix-free implementation: either (a) calibrate on the observed
context-pair support and use an explicit fallback on new/unseen pairs,
or (b) introduce a structured/coarsened model for P_ab's unobserved
mass. Do not silently present either as the uncoarsened full-support
triangle.

DECISION (Ruediger, 5 Aug): take option (a) now. Calibrate only on
context pairs observed in the decoded prefix; for an unseen context
pair use the two-pair maximum-entropy product as an explicit fallback.
This is a deliberate, revisitable model choice, not claimed to equal
the full-support triangle. Added restrict_margins_to_observed_contexts:
restrict/renormalize layered P_ab, then minimum-KL project P_ya and
P_yb to its new A/B margins while retaining their common Y marginal.
Regression verifies consistency. V32 real-prefix result: 635 observed
context pairs retain 0.999109 of layered P_ab mass; the complete gated
predictor is 0.0001485 bits/record KL from the full triangle.

MATRIX-FREE REFERENCE: added SparseGroupedProblem +
sparse_grouped_ipf. It stores only target baseline, observed context
edges, and active YA/YB corrections; no YA, YB or YAB probability
array. Each edge normalizer uses the union of its active corrections
plus an analytic background mass; a stable dense-softmax fallback is
used only when that union covers >99% of Y (tiny-V diagnostic regime).
The sparse solver matches the dense grouped predictor on every tested
supported context to <2e-9. Nine tests pass; ruff passes.

PERFORMANCE STATUS: the transparent Python-loop sparse reference is
far too slow (V32 real-prefix timing exceeded two minutes and was
stopped). Correctness is established; next work is a vectorized
sorted-array/join implementation plus acceleration of the 20k-sweep
plain IPF convergence. Do not benchmark this reference on the laptop.

VECTORIZED SPARSE SWEEPS (later 5 Aug): implemented a one-time
edge--correction incidence plan; every repeated sweep now uses
segmented maximum/reduction, bincount margin accumulation, and the
analytic baseline normalizer. Near-full-union edges use one bounded
batched dense softmax (the earlier per-edge fallback defeated
vectorization at tiny V). All dense/sparse prediction equivalence
tests still pass to <2e-9. Real V32, 635 context edges: 1000 sweeps
now 4.2--4.4 s instead of >2 minutes. Ten tests passed at this point.

SOLVER TRIALS, measured on the same V32/100k prefix:
- plain IPF, 1000 sweeps: YA/YB/Y residuals .0354/.0315/.0280.
- old-style Anderson (log history, regularized small system), 1000:
  .0226/.0229/.0185. Helpful but not the old full-triangle speedup.
- L-BFGS convex-dual attempt + IPF polish: 1000+1000 gave
  .000254/2.19e-5/8.93e-5 in 5.5 s, a large improvement; letting
  L-BFGS run longer was NON-MONOTONE in the certified residual and
  worse (5000+5000: .00178/.000181/.000702). Do not adopt unchecked.
  A short 200-step L-BFGS initializer was also worse after polishing.
Eleven tests pass; ruff passes. Standing conclusion: vectorization is
solved for the small reference, convergence is not. Keep the true L1
margin certificate authoritative. Next: exploit causal checkpoint
warm starts/interleaved chains on an actual geometric checkpoint
sequence and measure whether only the cold first fit is difficult;
do not spend more time tuning isolated-snapshot L-BFGS first.

HYBRID SOLVER DECISION (later 5 Aug): use dual L-BFGS to approach the
solution, then switch to ordinary marginal-scaling/IPF sweeps for the
final certificate. Both phases now retain the iterate with the best
true certificate max(L1 Y, L1 YA, L1 YB), rather than trusting the
last iterate: neither scipy's dual stopping test nor the combined L1
certificate is monotone along every numerical step. On the same
V32/100k prefix, 1000 L-BFGS iterations followed by at most 5000 IPF
sweeps returned YA/YB/Y residuals 6.99e-5/5.35e-6/2.36e-5 in 22.46 s,
instead of the 5e-3--8e-3 late-iterate regression. It did not meet the
deliberately severe 1e-8 test tolerance. Twelve tests pass. Next use
this hybrid with transferred sparse warm starts on the geometric
checkpoint sequence, and measure tolerances against predictive loss
before choosing the production stopping threshold.

GEOMETRIC SPARSE CHECKPOINT PROBE (later 5 Aug): added
scripts/calibration_checkpoint_probe.py and
fit_sparse_grouped_checkpoints. Each interleaved chain transfers the
global target factor and corrections for still-active YA/YB cells;
new cells start at zero. An early causal checkpoint exposed exact-zero
A/B margins for context states not yet observed. The minimum-KL margin
projection is now zero-margin safe, with a regression test (13 tests
pass; ruff passes).

Measured bpe_text8 V32, n=100k, eight geometric checkpoints, hybrid
solver, at most 2000 iterations per phase, provisional tolerance 1e-4:
- interleave=2: fitting 10.57 s; 7/8 certificates passed. Prefix 65,410
  missed at max residual 3.53e-4; final prefix passed at 6.75e-5.
- interleave=1: fitting 12.33 s; 7/8 passed. Prefix 65,410 passed at
  4.28e-5; final narrowly missed at 1.11e-4.
Interleaving therefore gave a modest 14% fit-time reduction here and
no systematic accuracy loss, though each warm-start path can have a
different difficult checkpoint. All results from either run are below
5e-4, but do NOT yet adopt 5e-4: choose the production certificate by
measuring predictive-loss sensitivity. Margin construction was more
expensive than fitting because this diagnostic rebuilds every prefix;
the 32-checkpoint experiment should update counts causally and avoid
unnecessary repeated construction before scaling V.

CAUSAL SCORING + THIRD-PAIR RESULT (later 5 Aug): added
sparse_gated_log_probabilities and star_log_probabilities. Supported
contexts are scored from baseline mass plus sparse correction unions;
no context-by-target or triple table is built. Unseen contexts use the
declared two-pair maxent/star fallback. A direct dense-conditionals
regression passes (14 tests total; ruff passes). On bpe_text8 V32,
n=100k, eight geometric checkpoints, interleave=2, tolerance 5e-4,
the seven following blocks (97,950 records) measured:
- calibrated gated three-pair: 2.09403068 bpc
- two-pair star used everywhere: 2.12954671 bpc
- third-pair gain: 0.03551603 bpc
Every block improved (0.01757--0.05754 bpc). Observed-pair coverage
rose from 95.92% after the first fit to 99.80% after the last scored
fit. This is the first direct evidence that the third pair adds useful
held-out information in the implemented model.

TOLERANCE CAVEAT: independently rerunning L-BFGS or over-polishing
failed checkpoints can produce extreme held-out probabilities even
with a superficially small aggregate margin residual. Stable scoring
fixed cancellation in the normalizer, but confirmed this is a factor
path/fit issue, not merely scoring arithmetic. For the five checkpoints
where independent IPF polishing from each 5e-4 candidate genuinely
reached 1e-5, the loose-vs-tight difference was only -7.29e-5 bpc
(the loose fit was slightly better on the next blocks); maximum absolute
block difference was 1.35e-4 bpc. Two polishes failed and must not be
treated as references. Provisional 5e-4 is therefore adequate for the
small predictive experiment, but retain both the convergence flag and
held-out sanity checks; residual alone is not a sufficient numerical
health certificate.

CHECKPOINT CONSTRUCTION: the probe now updates unigram, YA, YB and AB
counts only from the newly revealed slice instead of recounting every
prefix. A V16/n10k/three-checkpoint causal smoke run passes. Layered
probability construction remains the dominant setup cost and is the
next scaling target; incremental counts remove only the avoidable raw
statistics work.

32-CHECKPOINT RESULT + PERSISTENCE (later 5 Aug): four-worker layered
family construction uses the existing cross-checkpoint builder memo and
persistent worker pool. At V32/n100k, C=8 it cut construction to 22.24 s
(about half the one-worker time); calibration was 4.63 s. The full
C=32 geometric run with four interleaved fit chains took 104.50 s for
layered construction and 23.50 s for calibration. All 32 checkpoints
passed tolerance 5e-4; worst residual 1.17e-4, median 7.88e-5.

The C=32 causal held-out comparison over 97,950 records is:
- calibrated gated three-pair: 2.08177023 bpc
- two-pair star everywhere: 2.11831991 bpc
- third-pair gain: 0.03654968 bpc
All 31 scored blocks improve (range 0.000376--0.117523 bpc; median
0.030891 bpc). This confirms the C=8 result under the intended paper
checkpoint schedule. The harness now persists every sparse problem,
fitted factors, and fallback pair margins as compressed per-checkpoint
NPZ state. All 32 states total 1.35 MB, allowing later scoring work
without rebuilding the layered estimator. Result directory:
output/calibration_prediction_v32_c32_i4_jobs4/.

FIRST VOCABULARY SCALE STEP (later 5 Aug): V64/n100k/C=8,
interleave=2, jobs=4, tolerance 5e-4. Layered construction took 33.51 s
and calibration 50.45 s, so calibration becomes the bottleneck at V64.
Six of eight checkpoints passed. Prefix 2,050 narrowly missed at
5.23e-4; the final (unscored) prefix missed at 1.83e-3. Nevertheless
the causal prediction result over 97,950 records is strong and sane:
- calibrated gated three-pair: 2.75360432 bpc
- two-pair star everywhere: 2.80362067 bpc
- third-pair gain: 0.05001635 bpc
All seven scored blocks improve (0.01949--0.07075 bpc, median 0.03936).
Thus the third-pair benefit strengthens from V32 to V64. Do not launch
V64/C32 yet: first add reload/refit support for persisted checkpoint
states and diagnose continuation of the two failed V64 fits without
rebuilding layered margins. Result directory:
output/calibration_prediction_v64_c8_i2_jobs4/.

PERSISTED V64 REFIT DIAGNOSTIC (later 5 Aug): added
scripts/calibration_refit_states.py, which reloads selected sparse
checkpoint problems/factors and continues them without layered
reconstruction; it also rescored the following block. Plain-IPF results:
- prefix 2,050: 5,030 sweeps / 5.02 s moved max residual from 5.23e-4
  to 4.9999e-4, but worsened the next-block loss by 0.015307 bpc.
- final prefix 100,000: 10,000 sweeps / 192.32 s still failed; best
  max residual remained 1.825e-3 (no following block to score).
Therefore do not enforce 5e-4 mechanically at V64. The narrowly failed
early fit was predictively better, showing useful early-stopping
regularization; the expensive final fit does not enter causal loss.
Use held-out predictive sanity plus a broad residual guard, and treat
the residual as a diagnostic rather than the sole acceptance rule.
Refit output: output/calibration_prediction_v64_c8_refit_ipf/.

V64/C32 PRODUCTION-STYLE RUN (later 5 Aug): applied the solver lesson
by using 1000 L-BFGS iterations and a broad, explicit 2e-3 residual
guard, with no forced long polish. Four interleaved chains accepted
all 32 fits (worst residual 1.958e-3, median 6.87e-4). Construction
took 151.78 s and calibration 41.84 s, less calibration time than the
C=8 run that forced 5e-4. Causal result over 97,950 records:
- calibrated gated three-pair: 2.69952421 bpc
- two-pair star everywhere: 2.75137807 bpc
- third-pair gain: 0.05185386 bpc
All 31 blocks improve (0.001257--0.194952 bpc; median 0.049814).
This is the current production stopping policy at V64, explicitly
provisional and to be rechecked when V changes. Result directory:
output/calibration_prediction_v64_c32_i4_guard2e3/.

PAPER APPENDIX UPDATED: paper/main.tex no longer describes the obsolete
dense full-support matrix-product implementation as the measured code.
Appendix ``The calibrated pairwise predictor'' now derives the
conditional elimination of the context potential, states the observed-
support and grouped-inactive approximation explicitly, documents the
two-pair fallback, sparse normalizers, hybrid solver, best-certificate
retention, support-aware warm starts and interleaved chains, and warns
that residual is not a held-out-loss oracle. It distinguishes checks of
the implemented sparse model from the approximation choice. main.tex
compiles successfully in two pdflatex passes (24 pages).

SAME-BLOCK ARENA COMPARISON (later 5 Aug): ran the established
pairwise_arena at V64/n100k/C32, then added
scripts/calibration_compare_states.py to remove the arena's cold first
block and score baselines on exactly the new model's 97,950 records.
The scorer reproduces the stored star result at beta=(1,1) to 6.2e-15
bpc. Apples-to-apples numbers:
- tuned product beta=(.75,.5): 2.68251647 bpc
- count-backoff Markov-2: 2.69003788 bpc
- pure three-pair calibration: 2.69952421 bpc
- star beta=(1,1): 2.75137807 bpc
Thus the third pair recovers 0.05185 bpc from star but pure maximum-
entropy calibration is over-strong and does not yet beat the tuned
product or Markov baseline. The raw arena totals are not directly
comparable because they include the cold first 2,048-token block.

EXPLORATORY REGULARIZED THIRD-PAIR DIAGNOSTIC (DO NOT PROMOTE YET):
tested a fixed arithmetic mixture
q_w=(1-w) q_product + w q_calibrated, which remains causal, normalized,
and uses the third pair for every w>0. At V64 the broad optimum is
w=.35 (2.67491066 bpc), beating pure product by .007606 and Markov-2
by .015127. Treating weights {0,.1,.15,.2,.25,.3,.35,.4,.45,.5,1}
as an equal-prior Bayesian family costs 2.67494574 bpc, only 3.51e-5
above the selected member. The identical grid and exponents replicate
at V32/C32: product 2.08051827, Markov-2 2.07621293, pure calibration
2.08177023, best mixture w=.5 2.07271899, family 2.07275416 bpc.
On these particular strongly quantized 100k-token streams, regularized
use of the third pair beats both comparison baselines. This is only a
mechanism diagnostic. The product exponents, interpolation grid, and
mixture weights were inspected on the same data, while V32/V64
quantization changes support and redundancy drastically. Even though a
fixed equal-prior mixture has a legitimate finite-family coding cost,
that fact does not turn this hand-designed family into the universal,
parameter-free construction sought by the project. Do not put the
selected weights or the claimed win into the paper as a general result.
Use this observation only to diagnose that the third-pair correction
contains signal but is too strong in this small setting. Next seek a
predeclared/Bayesian shrinkage rule derived from the model, and validate
without retuning across substantially larger vocabularies and streams.
Results:
output/calibration_mixture_v64_c32_refined/ and
output/calibration_mixture_v32_c32_refined/.

DATA-SCALING CORRECTION + V128 (later 5 Aug): Ruediger correctly
stopped further interpretation of tiny strongly quantized runs: raising
V without raising n mainly creates rare cells and fallback dependence,
and cannot establish a need for shrinkage. Pure calibration remains the
central object. V128/n100k/C8 was retained only as a technical smoke:
coverage 89.4%--96.0%, one of seven blocks regressed, although aggregate
calibration beat star by .06539 bpc. V128/n1m/C8 improved coverage to
88.8%--99.34%; only the first 4,555-record block regressed, and aggregate
gain was .06741 bpc.

The genuine full bpe_text8 V128 run used n=19,429,294, C=8,
interleave=2, jobs=4, 1000 L-BFGS iterations and broad guard 5e-3.
Construction took 159.74 s, fitting 190.06 s; all 8 fits passed (worst
residual 3.44e-3). Over 19,427,244 causal records:
- pure calibrated three-pair: 3.02584592 bpc
- star: 3.09337028 bpc
- gain: 0.06752437 bpc
All seven blocks improve (0.00430--0.07707 bpc). Coverage rises from
88.12% in the first scored block to 99.959%; final observed AB support
has 13,811 edges and retains 0.9999468 of layered AB mass. This is the
first substantial-data result and shows the small-n early regression
disappearing. Result:
output/calibration_prediction_v128_n19m_c8_i2_guard5e3/.

SCORING SCALE FIX: star_log_probabilities and
sparse_gated_log_probabilities now stable-sort records once by context
and process contiguous groups, replacing one full-block boolean scan per
distinct context. Tests remain 14 passed. Note: directory
output/calibration_prediction_v128_full_c8_i2_guard5e3/ is an accidental
100k repeat caused by the probe's default --n; do not use it as a full
run.

V256 PROGRESSION (later 5 Aug): kept pure calibration central and
increased data rather than tuning shrinkage. V256/n1m/C8 used guard
1e-2; construction 73.12 s, fit 267.30 s, all fits passed. Pure
calibration 3.82625404 versus star 3.90685634: gain .08060230 bpc;
all seven blocks improved despite coverage beginning at 81.82% and
ending at 98.21%. This was an intermediate scaling check only.

Full bpe_text8 V256 used n=19,429,294, C=8, interleave=2, jobs=4,
1000 L-BFGS iterations and broad guard 1e-2. Construction took
171.48 s; fitting took 1070.79 s (17.85 min), now the decisive
bottleneck. All 8 fits passed (worst residual 9.08e-3, median 4.68e-3).
Over 19,427,244 causal records:
- pure calibrated three-pair: 3.61843123 bpc
- star: 3.69715445 bpc
- gain: 0.07872321 bpc
All seven blocks improve (0.02121--0.09001 bpc). Coverage rises from
81.13% to 99.793%; final support has 46,052 AB edges, retains
0.9996834 of layered AB mass, and stores 46,052/60,994 YA/YB
corrections. The full-data pure-calibration gain therefore increases
from .06752 bpc at V128 to .07872 at V256; these results do not support
asserting that shrinkage is intrinsically needed. Result:
output/calibration_prediction_v256_n19m_c8_i2_guard1e2/.

DO NOT launch V512 with the current harness. Before the next vocabulary
step, optimize the calibration stage (V256 fitting is 6.2x construction)
and remove the remaining dense V-by-V upstream count/probability arrays
used by the diagnostic margin builder and fallback. Calibration sweeps
are sparse, but real tokenizer V~100k remains impossible until the
inputs are sparse too.

LOCAL HARDWARE NOTE (Ruediger): the development machine has 12 CPU
cores. Recent scaling runs deliberately/accidentally underused it
(layered jobs=4, calibration interleave=2). Construction and fitting
are sequential phases, so future benchmarks should test up to 12
layered workers and up to min(checkpoints,12) interleaved calibration
chains. Pin BLAS/OpenMP numerical-library threads to one per fit chain
to avoid 8--12 outer chains each spawning an inner worker team. Measure
jobs/interleave scaling rather than assuming 12 is automatically best;
memory bandwidth and warm-start distance may make a smaller chain count
faster or statistically/numerically preferable.

12-CORE SCALING BENCHMARK (later 5 Aug): V256/n1m/C8, fixed 1000
iteration/guard setup. Original jobs=4/interleave=2: construction
73.12 s, fit 267.30 s, total 340.42 s, median residual 3.58e-3,
3.82625404 bpc. With BLAS/OpenMP pinned to one inner thread:
- jobs=12/interleave=8 (all cold): 51.27 + 218.06 = 269.33 s,
  median residual 4.23e-3, 3.82655886 bpc.
- jobs=12/interleave=4: 53.21 + 218.69 = 271.89 s,
  median residual 4.20e-3, 3.82699675 bpc.
- jobs=12/interleave=2: 56.09 + 277.98 = 334.07 s,
  identical residuals/predictions to the original warm path.
Ruediger observed the CPU trace: 4/8-chain runs had high peaky usage
(about 60% peaks); the 2-chain fit was flat around 10%, as expected
with two single-threaded outer chains on 12 cores. Conclusion: 12
workers are useful for layered construction, but cold checkpoint
parallelism trades away warm-start quality and saturates by four
chains. The next optimization must parallelize margin/gradient work
inside each warm-started fit (roughly 6 inner workers per each of two
chains), rather than add more outer chains. Benchmark outputs are the
three output/calibration_prediction_v256_n1m_c8_i{2,4,8}_jobs12_*
directories.

INNER-FIT PROFILING/EXPERIMENT (later 5 Aug): cProfile on the largest
persisted full-V256 checkpoint, 20 IPF sweeps, found 54.1/61.0 s in 81
margin evaluations; segmented np.at reductions and repeated dense-
fallback indexing were visible costs. Raising the dense fallback cutoff
from .99 to 1-1e-12 made 20 sweeps faster (41.5 s) but convergence much
worse: at tolerance 5e-4 the .99 path converged in 2 sweeps/7.45 s,
whereas the aggressive sparse path failed after 100 sweeps/205.0 s.
The cutoff is now an explicit dense_fallback_mass parameter but remains
.99 by default. Do not optimize per-sweep time at the expense of the
certificate trajectory.

Added optional margin_workers that parallelizes mathematically
independent dense-fallback edge blocks. On the isolated largest
checkpoint, 6 workers reduced the exact two-sweep continuation from
7.45 to 5.43 s with identical residuals. End-to-end V256/n1m with
jobs=12, interleave=2, margin_workers=6 preserved predictions/residuals
exactly but fit time was 270.85 s: only slightly below the pinned
single-inner 277.98 s and above the original unpinned 267.30 s. Thread
pool overhead on small checkpoints plus memory-bandwidth saturation
erase the isolated gain. Keep margin_workers optional/default 1; this
is not the production scaling solution. Next reduce work per objective
evaluation and build sparse upstream pair margins rather than relying
on more threads.

EXPERIMENT-LEVEL PARALLELISM (Ruediger): do not obsess over saturating
CPU inside one fit. Independent experiments can run concurrently while
each preserves its two warm-start chains. Added peak_resident_bytes to
checkpoint and persisted-refit JSON. Loading/evaluating the largest
full-V256 checkpoint measured 2,347,630,592 bytes peak RSS (~2.35 GB)
for one process, before allowing extra margin for Python, layered worker
processes and the OS. The sandbox could not read installed RAM. Until
memory headroom is confirmed, run at most two V256 experiments at once
(roughly 5 GB plus overhead), each with 2 fit chains and 4--6 layered
workers; avoid two simultaneous jobs=12 construction bursts. V128 jobs
are cheaper and can be paired more freely. Monitor memory pressure/swap,
because swapping would erase any CPU-throughput gain.

SPARSE-UPSTREAM IMPLEMENTATION (later 5 Aug): the development machine
has 24 GB RAM; Ruediger also has access to a 240 GB, 64-core server whose
individual cores may be slower. The production target is explicitly the
full tokenizer alphabet on the whole sequence. Neither machine should
use dense V-by-V pair tables: at V around 100k even one float64 table is
about 80 GB, and the pipeline needs several such objects and temporaries.
The server's memory is useful for large observed-support arrays and
concurrent/sharded work, not for reverting to dense pair matrices.

Added SparseCountRows and sparse layered-row construction, an implicit
SparseProjectedPair representation and Sinkhorn projection, sparse
restriction to observed AB contexts, analytic sparse star/gated
fallback scoring, and a --sparse-upstream checkpoint-probe path. Thus
the upstream counts and projected YA/YB/AB margins no longer require
V-by-V arrays. The dense path remains only as a small-case reference.
The combined graphical/layered suite passes 22 tests.

An end-to-end V16/n10k/C3 equivalence run gave exactly the same retained
AB masses and support/correction counts at every checkpoint. Sparse and
dense star scores agree to numerical precision. With a deliberately
loose 1e-3 solver tolerance, calibrated aggregate scores differed by
2.20e-5 bpc because tiny floating-point differences changed the IPF
stopping trajectory; this is below the requested tolerance and is not a
different statistical construction. Do not tune against this tiny run.

NEXT SCALING PLAN: first harden/profile the sparse path at progressively
larger vocabulary and data sizes on the 24 GB machine, recording peak
RSS, construction time, fit time, support sizes and convergence. Keep
warm starts along each checkpoint chain; do not parallelize all
checkpoints cold. Parallelism for the 64-core server should shard
observed edges and sparse row/profile construction inside a small
number of warm chains, with reusable worker pools and deterministic
reductions. Independent full experiments can then occupy otherwise idle
cores subject to measured memory. The acceptance test remains a pure
three-pair full-alphabet, whole-sequence experiment—not a reduced-V
benchmark.

MODEL-COMPLEXITY CONTINUATION IDEA (Ruediger, later 5 Aug): in addition
to warm-starting along data checkpoints, initialize the full model from
a simpler fitted model and add factors/lags progressively. Implemented
an optional --initialization first_pair mode. It exactly embeds P(y|x1):
the YA margin and unigram target margin agree to machine precision before
the first full-model iteration, while the YB correction starts neutral.
This is a universal continuation device and may extend naturally to
larger memories by adding one lag/factor at a time.

A first V64/n100k controlled probe is mixed, so first_pair is NOT yet the
default. It reduced the first checkpoint's reported iterations from
1004 to 606, but total fit time was 27.36 versus 26.68 seconds because
later warm-started checkpoints still hit their iteration caps. The last
unscored checkpoint also had a worse certificate on this run. Re-test at
a meaningful scale and inspect convergence trajectories; consider a
two-pair intermediate rung rather than assuming the one-pair seed alone
solves the production bottleneck.

INITIALIZER FAMILY / PARALLEL PORTFOLIO (Ruediger, later 5 Aug): do not
choose only one side. Added second_pair, pair_midpoint and pair_product
starts alongside unigram and first_pair. pair_midpoint averages the two
one-sided natural-parameter vectors, hence starts from
q(y|a,b) proportional to sqrt(P(y|a) P(y|b)); pair_product uses both
pair exponents at full strength. All are table-free and exactly embedded
in the sparse factor representation.

On the small V64/n100k/C8 screen, pair_midpoint was the clear candidate:
all checkpoint certificates were below 1e-3, the first checkpoint used
573 iterations versus 1004 from unigram, and a clean standalone run cut
fit time from 26.68 to 10.88 seconds. A concurrent run reproduced the
same parameters/results and took 11.42 seconds. This is promising but is
not grounds to tune production to the strongly quantized diagnostic.
Carry the initializer family to a meaningful-scale comparison.

Potential server strategy: at each warm chain's first checkpoint, run a
short fixed-budget portfolio of unigram/x1/x2/midpoint/product states on
separate cores. Select by the actual maximum margin certificate (and,
where available, dual objective), not intuition. Also evaluate consensus
averages in natural-parameter space; retain an average only if its
measured objective/certificate improves. Then continue only the selected
state serially along that warm checkpoint chain. This can use the
64-core server without reverting to cold independent checkpoint fits.

SOLVER DIAGNOSTICS / EXACT GROUPED STARTS (later 5 Aug): added optional
dual/certificate traces with gradient norms, limiting margin and factor
quantiles, plus exact margin-evaluation counts. A 20-sweep audit exposed
a real saturated-row bug at V256 checkpoint 6: a structurally full YA
row had a fictitious 2.0e-13 inactive mass from floating-point
cancellation, producing factors of 700--1300 and certificate explosions.
Saturation is now detected exactly from active support count. On the same
persisted unigram/midpoint states, 20 sweeps now reduce certificates from
.00461/.00424 to .000952/.000913 while maximum factors remain near 11.

Added a small-reference grouped feasibility LP over retained AB edges.
It accepts a compatible binary example and rejects the pairwise-consistent
but jointly impossible requirements Y=A, Y=B, A!=B. All three checkpoints
of the actual V16 sparse pipeline are feasible, with LP equality residuals
around 2.5e-13. This is diagnostic only and must not be used at full V.

The earlier first/second/product/midpoint starts were exact only with all
cells active. SparseGroupedCheckpoint now retains the O(V+active) projected
pair decompositions, and exact grouped tree factors are constructed from
their right/background/delta components. A partial-active unit test gives
YA and Y residuals below 2e-14 before optimization. Tree parameters are
put in a canonical global and saturated-row gauge.

L-BFGS now removes the obvious exact gauges (one global baseline constant
and one correction per structurally saturated A/B row). Sparse results
report margin_evaluations, the comparable work unit across L-BFGS and IPF.
The focused suite currently passes 26 tests.

EXACT-START BENCHMARKS: on a five-way concurrent V64/n100k screen, total
margin evaluations were unigram 8,821; first 8,818; second 8,411;
midpoint 8,506; exact product 8,000. Exact product therefore saved 9.3%
versus unigram and cut first-checkpoint evaluations from 1,144 to 623.
On a matched concurrent V256/n1m comparison, gauge-reduced unigram used
8,886 evaluations/283.25 s/worst certificate .00940; exact product used
8,017/278.40 s/.00651. Product improves work by 9.8%, although large later
checkpoints erase most wall-time savings.

Added optional tree_delta checkpoint transfer:
transfer(fitted previous)+current exact product tree-transfer(previous
tree). It gives newly active cells data-based factors in a consistent
gauge. At V64 it used 7,839 evaluations versus 8,000 for copy transfer and
improved the worst certificate (.000768 versus .000844). A standalone
V256 tree_delta test was launched next; do not infer its result until its
results.json exists.

TREE-DELTA FOLLOW-UP: the first V256 delta run canonicalized the initial
tree as well as the tree difference, confounding the first-checkpoint
L-BFGS trajectory. Canonicalization is now restricted to difference terms;
ordinary exact initialization keeps its original coordinates. A corrected
V64 comparison has identical first-chain starts and worst certificate:
copy uses 8,000 margin evaluations, tree_delta 7,833 (2.1% saving), mainly
at intermediate checkpoints. Accepted bpc differs by .00114 at the loose
1e-3 tolerance, so do not interpret prediction ordering. The confounded
V256 delta run used 8,409 evaluations/278.99 s/worst .00432; based on its
extra 536 first-checkpoint evaluations the corrected form may be near
7,873, but this is an inference and must not be reported as a measured
V256 result. Tree delta remains optional pending a controlled V256 run or
a tighter-tolerance comparison.

MAIN SPARSE-MARGIN SHARDING (later 5 Aug): Activity Monitor confirmed the
calibration phase was essentially one core (one Python process around
100%, 86% machine idle). The old margin_workers only covered rare dense
fallbacks. It now creates a persistent pool for each fit, partitions AB
edges by sparse-union workload, parallelizes conditional normalization,
and shards YA/YB/Y reductions into disjoint output ranges. Workers return
only their slices, not full margin vectors, so memory does not multiply by
the complete active-feature count. Serial versus 3-worker regression
agrees in factors/residuals to 1e-12.

On the largest persisted V256/n1m checkpoint, fixed 20-sweep timings were:
1 worker 13.91 s, 2 workers 11.69 s, 4 workers 9.52 s, 6 workers 9.09 s;
all used 81 margin evaluations and produced the identical .000835443
certificate. Scaling flattens after four workers, likely from memory
bandwidth plus remaining serial assembly.

End-to-end V256/n1m exact-product with two warm chains and five persistent
margin workers per chain: construction 57.08 s, fit 178.70 s, peak RSS
1.602 GB. The matched serial exact-product fit was 278.40 s, so calibration
falls 35.8% while predictions/certificates are identical. Total runtime
falls from about 333 to 236 seconds. This is the first useful inner-fit
parallelization. Next remove nested idle pools at L-BFGS/IPF handoff,
profile remaining serial assembly, and benchmark 3--5 workers per chain;
do not simply raise worker count.

V512/N4M SCALING + SCORING BUG (later 5 Aug): launched exact-product,
two warm chains, four margin workers at V512/n4m. The eight fitted states
completed and were persisted, but final scoring spent tens of minutes at
one core. Stack sampling showed sparse_gated_log_probabilities repeatedly
called sparse_star_log_probabilities once per unsupported context, and
the latter rebuilt complete Python pair-row dictionaries every time.
Stopped the parent after about 40 minutes; calibration states survived.
Scoring now collects all unsupported records in a block and constructs
the fallback maps once. A recovery scorer loaded the states and finished
all 3,997,950 records in 12.58 s:
- calibrated 4.44282442 bpc
- star 4.54717605 bpc
- pure three-pair gain .10435163 bpc
All seven blocks improve (.01323--.11496); coverage rises 72.09--98.09%.
The interrupted process's exact construction/fit timing was not persisted
and cannot be reported. Add checkpoint progress/intermediate results
before future long server runs.

FULL-TOKEN INCIDENCE AUDIT: bpe_text8 has 19,429,294 tokens, cl100k_base
alphabet 100,277, 35,767 distinct observed symbols. The definitive run
uses top_k=100276 so V=100277 while preserving unused vocabulary entries.
On the full uncollapsed sequence there are 3,663,366 YA cells, 5,732,416
YB cells and 3,663,366 AB edges. The present per-edge union expansion has
an upper bound 12,785,967,483 incidences (3,490 per AB edge), impossible
even on the 240 GB server. A 100k-edge sample estimates actual YA/YB
correction intersections at mean 116, median 59, p99 879, total about
426 million. Intersection-only algebra is roughly 30x smaller but still
too costly for thousands of complete evaluations.

Added and dense-validated an independent intersection-factorized margin
reference. With exp(c1)=1+r1 and exp(c2)=1+r2,
Z_ab=1+S1_a+S2_b+sum_y p_y r1_ya r2_yb; margins follow from AB-weighted
row/column sums plus correction intersections. Random dense tests match Y,
active YA/YB and every log normalizer below 2e-14 without materializing
per-edge active unions. Next production architecture must stream or use a
heavy/light scheme for intersections and drastically reduce full passes
via continuation, block/incremental updates, or a stronger optimizer.

COMPILED INTERSECTION EVALUATOR (later 5 Aug): the factorized algebra now
has a production array plan built by chunked SciPy CSR multiplication, not
Python dictionary intersections. The plan stores edge, target and the two
active-correction indices for each nonempty intersection. Largest saved
checkpoints: V256 has 1,068,000 intersections (17.1 MB, .032 s build) and
V512 has 8,356,483 (133.7 MB, .208 s build). Direct margin evaluations are
.0228 s and .1780 s respectively. The factorized evaluator is selectable
in sparse_grouped_ipf, checkpoint fitting, the checkpoint probe and the
state diagnostic; union remains the default until broader validation.
The same plan is reused if L-BFGS hands off to IPF polishing.

Controlled V256 continuation from identical saved factors, 50 IPF sweeps
and 201 margin evaluations: union took 33.19 s inside margins, factorized
4.54 s (7.3x). Objectives and all final residuals agree at roughly 1e-14;
both end at certificate .000625539557. At V512, 20 sweeps/81 factorized
evaluations took 14.26 s inside margins and reached certificate
.001196591654. This is still single-core, explaining low whole-machine CPU
use, but it removes much more work than union sharding. Next benchmark a
complete checkpoint ladder with evaluator=factorized, then parallelize or
stream its intersection plan and reduce the number of global passes before
attempting the full-token run.

NUMERICAL SAFEGUARD + MATCHED LADDER (later 5 Aug): a first complete
factorized ladder exposed overflow only in extreme exploratory L-BFGS
line-search proposals; the optimizer recovered, but those evaluations are
not acceptable at full scale. L-BFGS now has a per-phase displacement
trust region of 16 natural-log units around its warm start. Its ordinary
line search then adaptively mixes the old and proposed natural parameters;
IPF polishing remains unrestricted, so this does not cap the estimator.
The radius is a solver option, not a data-dependent fitted choice.

A clean matched V256/n1m run (exact pair-product start, two interleaved
chains, tolerance .01) produced no numerical warnings: construction
56.60 s, fit 41.92 s, peak RSS .599 GB. The prior union/sharded-5 run was
178.70 s fit and 1.602 GB, so factorization gives a 4.26x complete-fit
speedup and 63% lower peak memory. All checkpoints meet the requested
certificate. Most still use about 1,100 margin evaluations because SciPy's
componentwise stopping rule is much stricter than the grouped L1 margin
certificate. Next stop L-BFGS at an accepted iterate once that actual
certificate is met, then measure pass-count reduction before adding inner
parallelism.

CERTIFICATE-AWARE MIXING (later 5 Aug): L-BFGS now checks the actual grouped
margin certificate after each accepted line-search iterate and stops as soon
as it meets the requested tolerance. This is the adaptive old/new natural-
parameter mixture discussed with the user, inside the numerical trust
region; it avoids a fixed damping coefficient. The matched V256/n1m ladder
now fits in 31.00 s (construction 57.80 s, peak .604 GB), versus 41.92 s
without certificate stopping and 178.70 s for union/sharded-5. Thus current
fit is 5.76x faster than union. Evaluations across the eight checkpoints are
109, 155, 318, 400, 517, 609, 773 and 912 rather than about 1,100 each; all
certificates are below .01. Activity Monitor showed Python around 171%, as
expected for two single-threaded interleaved warm chains. Do not optimize
CPU percentage by adding cold chains without timing: earlier work showed
that extra cold-start iterations can erase nominal parallelism.

V512/N4M EARLY-STOP SCALING (later 5 Aug): the clean factorized run used
105.30 s construction, 251.97 s fit and 2.303 GB peak RSS. The eight margin-
evaluation counts were 152, 293, 491, 623, 786, 1102, 1106 and 1099; all
certificates meet .01. Activity Monitor clearly separates repeated high CPU
construction peaks (12 layered-estimator workers) from a long flat ~15%
calibration phase (two single-threaded warm chains). The last three large
checkpoints genuinely first reach the certificate late in L-BFGS; this is
not a missed IPF early-stop condition. The next change should parallelize
intersection reductions inside each factorized evaluation, retaining two
warm chains rather than adding cold chains.

FACTORIZED INNER PARALLELISM (later 5 Aug): intersection cross-normalizer
and YA/YB/Y correction reductions are now sharded across a persistent thread
pool. Partial arrays are reduced by the parent; this adds bounded memory per
worker and preserves two warm chains. Serial/parallel/union trajectories
agree within 1e-12 in tests. Largest V512 checkpoint, 20 IPF sweeps/81
evaluations: factorized time is 11.996, 7.332, 5.749, 4.909 and 4.436 s for
1--5 workers, respectively (2.70x at five); certificates agree around 3e-14.

The complete matched V512/n4m run with two chains x five inner workers used
94.53 s construction, 110.74 s fit and 2.814 GB peak RSS. The one-worker
factorized run used 105.30 s construction, 251.97 s fit and 2.303 GB, so fit
improves 2.28x and complete construction+fit falls 357.27 -> 205.26 s (42.5%)
for 22% more peak memory. All checkpoints meet tolerance .01. Recovered
scoring took 13.62 s: calibrated 4.44009273 bpc, star 4.54717605, pure
three-pair gain .10708332 bpc; all seven blocks improve. The earlier serial
fit scored 4.44282442/.10435163, so loose-tolerance solver paths differ by
.00273 bpc but support the same conclusion. Tighter tolerance must be used
for final scientific comparisons.

V1024/N8M LAPTOP SCALING (later 5 Aug): doubled vocabulary and sequence
together, retaining two chains x five inner workers. Construction took
175.84 s, fit 405.02 s, total 580.86 s (9.68 min), process peak RSS 5.517
GB; all eight checkpoints meet .01. Largest checkpoint has 232,245 AB/YA
cells and 366,390 YB corrections. Its exact intersection plan has
35,508,632 entries, 568.1 MB in four int32 arrays, and builds in 1.16 s.
System-wide Activity Monitor nevertheless reached 23/24 GB used with 8.4
GB compressed during the run, so laptop headroom is limited even though
the parent RSS is moderate. Construction used about 10.7 effective cores;
later calibration became memory/bandwidth constrained.

Recovered V1024 scoring: 53.64 s, calibrated 5.23486228 bpc, star
5.36242730, overall pure-three-pair gain .12756502. Six of seven blocks
improve. The first prefix (only 2,050 tokens, 57.9% context support) loses
.01380395 bpc; do not tune this away, since it demonstrates the expected
data scarcity when V grows. Before V2048, add incremental phase/checkpoint
progress and consider releasing construction intermediates or reducing
simultaneous worker/chain memory; the factorized plan itself is not yet the
sole barrier.

TWO-CHECKPOINT STREAMING + FULL V1024 (later 5 Aug): the checkpoint probe
can now construct, fit, persist and release batches of two checkpoints.
Only compact encoded YA/YB feature keys and fitted corrections survive for
the two warm-start chains. A matched V32 smoke test reproduces the former
all-in-memory iterations/evaluations/residuals exactly. Each completed
checkpoint is announced and persisted, so long runs are observable and
recoverable.

V1024 on all 19,429,294 bpe_text8 tokens completed: construction 239.24 s,
fit 813.33 s, elapsed 1053.85 s (17.56 min), peak parent RSS 5.826 GB. All
eight certificates are below .01. Final checkpoint has 338,237 AB/YA cells
and 537,579 YB corrections. Streaming therefore removes checkpoint-count
memory accumulation; the largest pair is now limited mainly by intersection
reduction bandwidth (observed about 3.5 effective cores late in the run).

HONEST BPE ACCOUNTING: calibration_score_states now adds the initial prefix
under a fixed-alphabet Jeffreys/KT code, the external tokenizer vocabulary
charge from stream metadata, an enumerative selected-subset description,
and a causal KT payload for original IDs behind escape. Full V1024 results:
reduced predictive 5.11368479 bits/BPE token; vocabulary description
.31994890 (6,208,152 tokenizer bits + 8,229 subset bits); escape payload
4.71630699; honest total 10.14994068 bits/BPE token = 1.97206182 bits per
original text8 byte/character. Empirical escape oracle is 4.69543894, so
the causal escape penalty is .02086806 bpt. Calibrated suffix 5.11354447,
star 5.23696052, pure-three-pair gain .12341605. First tiny block loses
.01636; all later blocks improve. Current reduce_ids is whole-corpus
frequency selection; it is made into a valid two-part code by transmitting
the subset, but also report fixed tokenizer-ID selection as a comparison.

V4096 FULL FEASIBILITY RUN (5 Aug evening): the original 2,050-token start
was numerically/data sparse (checkpoint certificate .03 and factorized
cancellation). The accepted test uses first prefix 16V=65,536, eight
checkpoints, auto union for <=5m incidence upper bound, otherwise parallel
factorized evaluation, two checkpoint batches x five workers. All eight
certificates are below .01. Construction 810.92 s, fitting 4801.49 s,
elapsed 5621.06 s (93.68 min), peak parent RSS 17.080 GB. Final problem has
1,399,764 AB/YA cells and 2,288,630 YB corrections. Late Activity Monitor
showed only ~3--4 effective Python cores plus high kernel CPU: the large
factorized reductions are bandwidth/allocation limited. Several exploratory
factorized trials still emitted cancellation warnings, although retained
final states are finite and certified; solve this before production.
Scoring with the present Python lookup implementation became a separate
single-core multi-minute bottleneck and was still pending when recorded.
The run revises laptop expectations: V4096 is feasible but already near a
safe 24GB ceiling; do not jump to V8192 with two simultaneous full problems.

PRODUCTION SCALING CHECKLIST (agreed with user, 5 Aug): target is full
tokenizer vocabulary, 32 checkpoints, full text8 and eventually enwik9.
Priority is computation, especially calibration; 240GB server memory may
already suffice. Implement/measure in this order: (1) fixed-manifest cache
for corpus count increments, layered row profiles/evaluations and prepared
checkpoint problems; strict keys include corpus/tokenizer hashes, V,
boundaries, l_max, table/code versions; (2) vectorized sorted-array scorer,
removing Python dictionaries/membership loops; (3) certificate/pass profile
per optimizer phase and checkpoint; (4) stronger continuation across 32
nearby checkpoints and progressive V=256->512->1024->4096->full, with all
vocabulary messages charged; (5) numerically safe factorized dual trial
steps; (6) output-sharded worker reductions rather than full YA/YB arrays
per worker; (7) fit one full checkpoint at a time if memory requires it,
retaining compact starts for both chains; (8) block-stream or memory-map the
intersection plan for full vocabulary; (9) benchmark thread counts on the
64-core server, expecting bandwidth rather than core count to limit inner
scaling. Do not mistake the V4096 eight-checkpoint/16V-start result for the
final codelength.

## 2026-08-06 --- V1024/C32 stochastic chain succeeds; memory is now the target

The workflow is now split cleanly into reusable checkpoint construction and
calibration. `scripts/calibration_checkpoint_probe.py --construct-only`
builds/persists problems without fitting. The full bpe_text8 V1024/C32 build
is in `output/calibration_problems_v1024_full_c32_jobs12`: 19,429,294 tokens,
32 geometric checkpoints, 12 layered-estimator processes, 871.05 s total,
5.44 GB peak RSS, and 449 MB of uncompressed reusable states.

`scripts/calibration_fit_precomputed.py` loads those problems one at a time,
uses pair-product initialization at checkpoint 0, transfers the preceding
certified solution thereafter, tries 12-worker block-SVRG/Adam first at every
checkpoint, reduces the learning rate on an exact-certificate plateau, and
falls back to exact factorized L-BFGS from the stochastic candidate only if
needed. Output is
`output/calibration_v1024_full_c32_stochastic_plateau_w12_fitted`.

MEASURED: all 32 checkpoints certified below .01; zero exact fallbacks, zero
nonfinite rejections, 20,250 stochastic updates. Calibration-only elapsed
142.51 s. Breakdown: sampled gradients 96.36 s, reference cache 15.72 s,
optimizer 9.48 s, exact certificates 5.99 s. CPU utilization ramps upward
with checkpoint size and reaches respectable ~50% whole-machine peaks. Peak
RSS was 10.25 GB.

MEMORY DIAGNOSIS: checkpoint accumulation is not the cause. The driver keeps
only the current and immediately previous problems for transfer. At the final
checkpoint the stored problem arrays are 29.15 MB. The exact intersection
plan has 83,702,774 entries and four int32 arrays totaling 1.339 GB, but its
current SciPy sparse construction peaks at 7.778 GB. Stochastic reference
caches and concurrent worker outputs add further bounded memory. NEXT TASK:
construct/stream the compact plan directly, without the large SciPy
temporaries; then reconsider V2048/V4096 on the 24 GB laptop. Also vectorize
the serial scorer, which took 256.82 s for 31 blocks.

HONEST C32 SCORE (`.../scoring.json`): calibrated suffix 5.06188255 versus
star 5.17328182 bits/BPE token, gain .11139928. Complete reduced predictive
charge 5.06202832 bpt; vocabulary description .31994890; escape payload
4.71630699; honest total 10.09828421 bpt = 1.96202533 bits per original
text8 byte/character. The corresponding honest star number is 1.98366714
bpc. Every one of the 31 predicted blocks improves. Compared with the prior
C8 honest result 1.97206182, C32 improves by .01003649 bpc. Treat these as
provisional tolerance-.01 numbers until the final experimental protocol is
fixed, but the computational and predictive result is unambiguously positive.

### 2026-08-06 memory/scoring implementation pass (while V4096/C32 constructs)

The SciPy sparse-product intersection builder has been replaced, when the C
extension is available, by a direct two-pass compact builder. It indexes the
two factor supports by context, merge-intersects their sorted target lists for
each observed AB edge, counts first, then allocates/fills exactly the four
final int32 arrays. The SciPy implementation remains the fallback/reference.
Randomized tests require bit-for-bit equality of all four arrays, and the
memory-limit guard is tested.

MEASURED on saved V1024 checkpoint 20 (4,332,210 intersections; 69.3 MB final
plan): SciPy 0.462 s and 1.134 GB peak RSS; direct C 0.185 s and 160.3 MB peak
RSS. Thus this checkpoint is 2.5x faster and 7.1x lower peak memory. Do the
final checkpoint benchmark after the concurrent V4096 construction exits;
do not create its additional 1.34 GB plan while the laptop is pressured.

The block-SVRG reference cache no longer retains full YA/YB vectors for all
128 blocks. Each cached block stores only corrections whose contexts occur in
that block, with int32 positions. On checkpoint 20 this is about a 2x cache
reduction; expect roughly 0.4--0.5 GB saved at final V1024. Sampled worker
outputs remain full-sized but exist only for the active concurrent calls.

The plateau scheduler now terminates stochastic fitting when it plateaus at
the minimum learning rate; the calibration-only driver then invokes exact
fallback. It no longer burns the remaining maximum-step budget. The driver
also reports current macOS RSS before load/fit, after stochastic fitting, and
after release, separately from lifetime peak RSS; it records exact plan and
reference-cache byte counts.

The experimental factorized-normalizer scorer was rejected after the full
run: although faster, cancellation changed the aggregate codelength slightly.
The production scorer retains the exact positive sparse-union log-sum-exp
calculation and parallelizes the 31 independent checkpoint intervals across
processes reading one memory-mapped reduced stream. Four workers reproduce
every serial row and the complete honest accounting object exactly in 87.65 s
versus 256.82 s, a 2.93x elapsed-time speedup.

FULL V1024/C32 MEMORY VALIDATION: the direct plan builder plus compact
reference cache reproduced all 32 fitted states, update counts and
certificates exactly; no exact fallback or nonfinite rejection occurred.
Calibration took 151.27 s versus 142.51 s before (6.2% slower), while measured
peak RSS fell from 10.25 GB to 5.12 GB. At the final checkpoint the plan is
1.339 GB, the cache .776 GB, and parent RSS after release 2.72 GB. Thus the
main memory change is validated and roughly halves peak memory. Python's
allocator still retains arenas between checkpoints, so RSS after release is
larger than the live saved problem; checkpoint subprocess isolation remains
an optional future measure, not a correctness requirement.

Validation state: 37 calibration tests pass; selected source/scripts/tests
pass Ruff and `git diff --check`. Repository-wide pytest still has an
unrelated pre-existing collection failure because `tests/test_layered.py`
imports absent `log_q_lambda_grid`; do not conflate that with this work.

### 2026-08-06 --- Full V4096/C32 calibration and honest score

The reusable problems in `output/calibration_problems_v4096_full_c32_jobs12`
were fitted sequentially with 12 stochastic workers and one warm-started
chain into `output/calibration_v4096_full_c32_compact_w12`. All 32
checkpoints certified below .01; zero exact fallbacks and zero nonfinite
rejections. Calibration took 1330.63 s (22.18 min), 20,600 updates, and
13.887 GB measured peak RSS including workers. Final direct plan: 4.477 GB;
compact reference cache: 3.187 GB. Breakdown: sampled gradients 786.64 s,
exact certificates 253.73 s, reference caches 114.60 s, optimizer 61.70 s.

HONEST RESULT: calibrated suffix 6.92395455 versus star 7.09408171 bpt,
pure-three-pair gain .17012716. Reduced predictive 6.92895566 bpt;
vocabulary description .32079521; escaped payload 2.33023552; honest total
9.57998639 bpt = 1.86132372 bits/original text8 byte. Treat as provisional
tolerance-.01, but it improves the analogous V1024/C32 honest result
1.96202533 by .10070161 bpc.

Exact scoring keeps the positive/log-sum-exp reference path. Intervals are
now scheduled longest-first across processes reading a shared memory-mapped
reduced stream, then restored to causal order before accumulation. V4096
times: old ascending four-worker 647.69 s; LPT four-worker 553.57 s; LPT
eight-worker 321.63 s; LPT twelve-worker 270.81 s. Every row and accounting
total is exactly identical. Twelve workers are the current laptop default
for extraction. Remaining scoring speedup requires splitting large
intervals into record ranges, since a single large interval is still handled
by one worker.

### 2026-08-06 --- Next server experiment prepared

`cluster/job_graphical_enwik8_v4096.sbatch` is the staged first test of the
new three-pair pipeline on the EPFL server. Submit only through
`urbanke@lth.epfl.ch`; it requests node14, 64 CPUs and 220 GB on
`slurm-cluster`. It reuses the existing Python >=3.11 on the submitted
environment's `PATH` (server reports 3.13.9, numpy 2.3.5, scipy 1.16.3) and
compiles the native extension in place without installing packages,
prepares/reuses the 25,793,085-token `bpe_enwik8` stream, constructs V4096/C32
problems with 64 workers, calibrates with the validated fixed batch of 12
replicas/workers, and exact-scores with 31 workers. Increasing replicas merely
to occupy cores would change the stochastic algorithm and repeat the batch-size
mistake diagnosed on the laptop. BLAS inner threading is forced to one. Stages are restartable
through separate `results.json` files under
`output/calibration_enwik8_v4096_server`. The prepared `bpe_enwik8` stream
must already exist; the job stops instead of installing `tiktoken`. The laptop concurrently builds
V8192/text8 problems with four workers; do not confuse the two runs.

Server job 2186 constructed checkpoints 0--16, then the sparse Y--lag-2
Sinkhorn projection at checkpoint 17 exhausted its old 10,000-iteration,
1e-12 tolerance. This was not memory failure (MaxRSS 24.8 GB). The projection
controls are now explicit; the enwik8 job uses tolerance 1e-10 and at most
100,000 iterations. The permitted final pair-margin discrepancy is still at
most about 1e-9 L1, eight orders below the .01 calibration certificate.
Failures now report achieved residual/tolerance/iterations.

Construction now has checkpoint-level restart support.  The server job passes
`--resume-streamed`: after an interrupted construction it replays the cheap
cumulative token counts, validates each existing state against the expected
prefix and vocabulary, and skips layered construction and pair projection for
every completed checkpoint.  A `construction_fingerprint.json` binds reuse to
the stream path, length, vocabulary, checkpoint edges, projection controls and
layered-table settings; incompatible partial output is rejected.  Truncated
checkpoint files are discarded and recomputed.  A two-checkpoint local test
reused both states and a deliberately changed run configuration was rejected.

### 2026-08-06 --- Full V8192/C32 text8 result and paper placement

The laptop V8192 construction completed in 9361.57 s with four workers.
Calibration into `output/calibration_v8192_full_c32_compact_w4_r12` completed
all 32 checkpoints in 4590.82 s with the fixed 12-replica stochastic batch,
four worker processes and 21,500 updates.  Every certificate is below .01
(maximum .0098082), with zero exact fallbacks and zero nonfinite rejections;
peak RSS was 17.915 GB.  Exact positive-log-sum-exp scoring with 12 workers
took 544.52 s.  The calibrated suffix is 7.87993175 versus 8.07782087 bpt for
the star rule, a .19788911-bpt gain.  Honest accounting: reduced prediction
7.89141144, vocabulary .32163113, escape payload 1.20515793, total 9.41820049
bits/BPE token = 1.82988986 bits/original text8 character.

`paper/main.tex` now places the complete calibrated-pairwise section before
the `\\appendix` boundary.  The obsolete V4096/C8 discussion and numbers were
removed.  One expandable two-panel C32 table reports V1024, V4096 and V8192.
Both panels use bits per original character: the upper gives the fitted-suffix
contributions of the star rule, calibrated rule and third-pair gain; the lower
gives reduced prediction, vocabulary description, escape payload and their
honest total.  The front scoreboard now uses
the V8192 result.  The PDF was rebuilt and pages 14--15 were visually checked:
the table is legible and Appendix A starts after the Section 7 discussion.

### 2026-08-06 --- Active scaling runs and next solver branch

Two scaling runs are active.  EPFL server job 2187 is constructing
enwik8/V4096/C32 with 64 workers and has passed the projection failure point
from job 2186; at checkpoint 29 it had accumulated 2.67 hours of construction.
Its printed `peak_resident_bytes` is actually Linux `ru_maxrss` in KiB, so
16,737,532 means about 16.0 GiB.  It will automatically continue through the
fixed-12-replica/12-worker calibration and 31-worker scoring stages.  The M4
Max (16 cores, 36 GB) is running text8/V16384/C32: eight construction workers,
then fixed 12 replicas/workers for calibration and 15 scoring workers.  At
checkpoint 18 it had accumulated .80 hours and peaked at 13.39 GB.  Its runner
is `scripts/run_graphical_text8_v16384_m4.sh`; it deterministically builds the
cl100k stream and exact production anchor store when absent.

The user now also has SCITAS access to 72-core Xeon nodes with 512 GB--2 TB
RAM.  These make a first full-token enwik9 structural/baseline run feasible,
but a week per experiment is unacceptable because longer-memory models are
the real target.  Optimization priority is therefore explicit: (1) remove
computations and data first; (2) only then execute the remaining work faster.
The two leading exact-model ideas are matrix-free Newton--CG to remove the
long stochastic tail and lazy sampled-block intersections to avoid building
unused plan records.  Newton--CG comes first because it attacks the dominant
number of margin passes, can be tested against existing saved checkpoints,
and may change the access pattern that the block store should optimize.

NEXT IMPLEMENTATION: derive an analytic sparse-dual Hessian-vector product;
validate it against finite differences and literal dense small problems;
then use a standard trust-region/Newton--CG method from an existing stochastic
warm iterate.  Test transitions from certificates roughly .05/.02 to .01 and
count full-plan-equivalent passes as well as elapsed time.  Continue only if
about 5--20 Hessian-vector passes replace a material part of the present
hundreds of stochastic updates; stop early if inner CG needs many full passes
or conditioning is poor.  Preserve gauge removal, damping/globalization and
the exact marginal certificate.  Sampled-block construction remains the next
memory branch regardless of the Newton outcome.

The Newton experiment is now complete enough to decide.  The analytic exact
Hessian-vector product agrees both with centered finite differences and with
an independently constructed literal covariance Hessian on small problems.
The first unpreconditioned real checkpoint-23 trial spent more than 150 s in
the inner CG loop without completing a useful outer step.  Curvature probes
showed severe scaling: normalized-gradient curvature was about 0.0742 whereas
a random-direction curvature was about 6.45e-6.  A standard target-Fisher
Jacobi coordinate scaling plus a hard Hessian-product budget fixed this
pathology.  On the genuine checkpoint-22 to checkpoint-23 transfer it reduced
the certificate .06035 -> .02871 with 40 products/20.69 s, -> .01570 with 80
products/42.61 s, and -> .01117 with 140 products/69.42 s.  The existing
fixed-batch stochastic fit reached .00772 in about 5 s.  On the much harder
checkpoint-0 to checkpoint-1 transfer, a conservatively capped preconditioner
used 100 products but only reduced .46384 -> .14974, while stochastic reached
.00929 in 2.83 s.  An aggressive scale cap also exposed nonfinite exploratory
products at that early checkpoint.

DECISION: retain the validated, budgeted Newton-CG implementation and probe
script as a documented experimental/fallback branch, but do not put it in the
production checkpoint solver.  Exact curvature passes do not remove enough
work to beat sampled gradients.  The next implementation is lazy sampled-block
intersection construction: stochastic fitting must not first materialize the
full intersection plan/reference cache when it will touch only sampled edge
blocks.  This directly targets both computation and peak memory and therefore
matches the agreed remove-work-first priority.

Potential future fallback hierarchy (record this exactly, but do not implement
it yet): run the normal stochastic fit from the best certified warm start; if
its scheduler stalls above tolerance, try preconditioned Newton--CG under a
small hard Hessian-product/time budget; accept the Newton candidate only when
an exact certificate improves sufficiently and is finite; otherwise pass the
best valid candidate to the existing reliable L-BFGS fallback.  Newton must
never replace that final safety net.  Controlled checkpoint-23 finishing tests
show why it can be a useful rescue but not the default: from certificate
.02141 it crossed .01 in one accepted step (.00675), but required 37 serial
Hessian products/18.3 s versus 3.5 s for stochastic continuation.  Starts at
.02742 and .03379 exhausted 100/120-product budgets without crossing .01, and
some closer states were also poorly conditioned.  Thus certificate alone does
not identify the one-step basin; a bounded attempt and exact acceptance test
are essential.  Parallel Hessian products or a future tighter required
tolerance could change this assessment.

### 2026-08-06 --- Single depth-layered intersection graph audit

The proposed topology is a bipartite graph whose left nodes are active
`(y,a)` corrections and right nodes are active `(y,b)` corrections.  A graph
edge exists when the corresponding `(a,b)` context edge is retained; it does
not require the empirical triple `(y,a,b)` itself to have occurred.  Such an
edge is exactly one present intersection/colored support triangle.  Grouping
edges by their left correction node makes `y` and the left correction index
implicit: a CSR edge need only store the right-correction index and AB-edge
index.

`scripts/intersection_topology_audit.py` tested the idea on the real
V1024/C32 checkpoint chain.  YA, YB and AB supports are exactly monotone over
all 32 checkpoints.  Giving every pair edge its first active checkpoint and
every triangle birth depth equal to the maximum of its three pair-edge births
reconstructs every checkpoint exactly: all 32 cumulative birth counts equal
the independently recorded intersection-plan lengths.  The final supports
have 338,237 YA, 537,579 YB and 338,237 AB edges.  There are 213,958,699
candidate blue-red paths, of which the AB mask retains 83,702,774 triangles.

The present solver constructs 372,522,181 triangle records cumulatively over
the 32 checkpoints, 4.45 times the final topology.  Its final four-int32 plan
is 1,339,244,384 bytes (1.247 GiB).  One 32-layer CSR topology with two int32
values per graph edge and one int64 row-pointer array per layer is estimated
at 756,211,120 bytes (0.704 GiB), a 43.5% reduction even at the final
checkpoint, while constructing topology only once.  A node-major two-int32
plus uint8-depth layout is essentially the same size.  The audit output is
`output/calibration_v1024_full_c32_compact_w12_validation/intersection_topology_audit.json`.

NEXT: implement a small correctness reference for the layered CSR store and
show that summing layers 0..k gives bit-identical margins/gradients to the
current checkpoint-specific plan.  Then add a native sequential traversal and
benchmark it before changing production construction or saved-state formats.

The correctness reference and native sequential traversal are now implemented
on branch `codex/layered-intersection-graph`.  `LayeredIntersectionGraph`
stores one CSR layer per birth depth; its row is the global YA-correction
index, while each edge stores only the YB-correction and AB-edge indices.
Expansion reconstructs all explicit triangle fields exactly.  A direct NumPy
reference and the native C traversal reproduce all model-margin families and
normalizers on independent small problems; the complete calibration test file
has 40 passing tests.

On the real V1024 final checkpoint (83,702,774 triangles), explicit topology
uses 1,339,244,384 bytes and the layered graph 756,211,120 bytes.  One native
sequential evaluation took .610 s for the existing plan and .510 s for the
stabilized layered graph, a 16% speedup before parallelism.  Construction of
the old final plan took 1.87 s and conversion to the Python-built layered
prototype 5.75 s; production must build layers directly rather than convert.

Changing summation order exposed cancellation in the old expanded evaluator.
The layered kernel now uses compensated signed accumulation, omits edges whose
cancellation ratio exceeds 1e10, and adds those few edges back with positive
log-sum-exp.  At the worst real edge, old explicit log Z was 6.725633702,
initial layered 6.724733477, and direct truth 6.725007924.  Stabilized layered
equals 6.725007924.  Across the 100 edges with largest old/new differences,
old maximum/total direct error were 6.26e-4/6.59e-3; stabilized layered error
was zero at displayed double precision.  NEXT: checkpoint this sequential
kernel, then add parallel traversal with controlled thread-local reductions;
do not trade the topology memory saving for unbounded workers-times-margin
scratch arrays.

The parallel layered kernel is now implemented with a hard scratch-memory
budget.  Forward workers keep private AB cross accumulators; reverse workers
own disjoint YA rows and keep private YB/target accumulators.  Requested
workers are capped so scratch is at most the configured budget, with exact
requirement `8 W (|E_AB|+|E_YB|+V)` bytes.  At V1024/C32 this is only about
84 MB for 12 workers.  Stabilized final-checkpoint timings (three repeats):
old explicit sequential .609--.611 s; layered W=4 .245--.272 s; W=8
.172--.174 s; layered W=12 .141--.143 s.  Twelve workers therefore give a
4.3x wall-clock speedup over the old sequential evaluation, alongside the
43.5% topology reduction.  Direct-log-sum-exp validation remains exact at
displayed double precision for the 100 most cancellation-sensitive edges.

NEXT: construct the layered CSR topology directly from pair-edge birth depths
rather than first building the final four-index plan and converting it.  Then
persist/memory-map the one shared graph and adapt checkpoint problems to stable
global YA/YB/AB identifiers before attempting a complete fitting trajectory.

Direct native construction is now implemented and validated against the
converted graph layer-by-layer on independent problems.  On the real V1024
final topology it takes 1.95 s.  The prototype conversion path took 1.85 s to
allocate/build the old 1.339-GB plan plus 5.70 s to convert it; the direct
builder never allocates that plan and emits the 0.756-GB graph immediately.
The next format change is stable birth-major global IDs for YA, YB and AB
edges.  With every checkpoint support represented as a prefix of those global
orders, layers 0..k can use the same stored CSR indices directly.  Checkpoint
numerical margins/factors remain checkpoint-specific prefix arrays; future
coordinates must not enter the optimizer.  After this invariant is tested,
persist the graph as uncompressed/memory-mappable arrays and run one complete
V1024/C32 fitting trajectory against the existing result.

Birth-major IDs and persistence are now implemented.  The real V1024/C32
store in `output/layered_intersection_v1024_c32` contains the complete shared
topology (83,702,774 triangles) in 756,211,120 bytes; direct construction took
2.50 s and uncompressed persistence .15 s.  Each layer is a separate `.npy`
array and loading uses memory mapping, so the operating system can page layers
without copying the whole graph into each process.

Real checkpoints 0, 7, 15, and 23 were independently rebuilt in their old
explicit representation and compared with layers 0..k of the stored graph.
Triangle counts and all four indices `(AB,y,YA,YB)` agree exactly at every
checkpoint.  Apparent numerical disagreements with the old evaluator were
not topology errors: at the AB edge of largest disagreement, direct positive
log-sum-exp agrees with the stabilized layered evaluator to displayed double
precision, while the old expanded evaluator errs by .02795, .45249, and
.30744 in log Z at checkpoints 7, 15, and 23 respectively.  The stricter
cancellation test is therefore detecting real old-path error.

`sparse_grouped_ipf` now accepts `evaluator="layered"` with a shared graph and
checkpoint depth, including its L-BFGS-to-IPF polishing recursion.  All 41
calibration tests pass.  NEXT: route the stochastic solver's periodic exact
certificate through this evaluator and provide an offline fitting driver that
loads checkpoint problems, aligns them to the persisted birth-major support,
and warm-starts the complete C32 trajectory.  Sampled block gradients still
need their own lazy intersections; they must not force construction of the old
full explicit plan merely to prepare stochastic blocks.

The stochastic exact records now use the shared graph, and a first bounded
lazy sampled-block implementation removes the old eager setup allocation.
AB-block boundaries are balanced from per-edge triangle counts obtained by
scanning the active graph layers.  Local four-index plans and their SVRG
reference margins are constructed on demand and retained in independent LRU
caches (default 16 blocks).  All 41 calibration tests pass, including an
actual parallel stochastic update through the cache-size-one path.

Real checkpoint-23 measurements with 128 blocks and 12 workers quantify the
tradeoff.  Eager setup retains 165.6 MB of topology plus 212.6 MB of reference
margins.  A 16-block lazy cache peaked at 20.8 MB plus 24.5 MB in a 10-step
run: 8.3x less cache memory, with elapsed time 1.15 s versus .93 s.  Over 50
steps the cache remained 7.9x smaller (47.6 MB versus 378.1 MB) but random
access caused repeated reconstruction and elapsed time was 3.89 s versus
1.57 s.  An 8-block cache reached a 16.7x memory reduction but was slower.

NEXT: do not tune this LRU as the final answer.  Add a sampled AB-edge-range
view to the native layered evaluator so sampled gradients traverse the shared
graph directly and never reconstruct local four-index plans.  The bounded
cache remains a correct low-memory fallback and a measured baseline.  After
that, run the complete V1024/C32 warm-started trajectory from the persisted
store and compare certificates, elapsed time, and peak memory.

Cache misses for topology and reference margins were initially serialized
under Python locks.  They now construct concurrently and lock only for the
short insertion/eviction operation.  On the checkpoint-23 50-step probe this
reduced the 16-block lazy time from 3.89 s to 3.17 s (eager 1.51 s), with
about 47.0 MB versus 378.1 MB of cached arrays.  A 32-block cache took 2.23 s
and 92.8 MB, versus eager 1.50 s and 378.1 MB: about 1.49x elapsed time for
4.1x less cache memory.  Thus cache size is a useful explicit operating knob,
but direct sampled traversal remains the route to removing reconstruction
work rather than merely trading it against memory.

That direct route is now implemented as `ABMajorIntersectionGraph`.  It stores
an AB-edge pointer plus `(YA index, YB index, birth depth)` per triangle; AB
and target `y` are implicit.  Native construction matches the old explicit
plan exactly and builds the real V1024/C32 graph in 1.22 s.  Its size is
756,030,870 bytes (0.704 GiB), essentially identical to the YA-major layered
graph.  It has uncompressed save/load support and is memory-mapped by default.

The native AB-major evaluator accepts a contiguous sampled AB range and
checkpoint depth.  It matches the explicit evaluator for full and interior
block problems in all margin families and normalizers.  In the real
checkpoint-23 50-step/128-block/12-worker probe, eager sampling took .946 s
and retained 165.6 MB topology plus 212.6 MB references.  Memory-mapped
AB-major sampling took 1.020 s, retained zero topology cache and 26.2 MB of
bounded reference cache, and used the corrected stabilized exact certificate.
Thus direct sampling removes about 352 MB of checkpoint-local cached arrays
at essentially unchanged elapsed time.

NEXT: benchmark and parallelize full exact evaluation in AB-major order.  If
it matches the YA-major layered evaluator's speed under a bounded scratch
budget, keep only the AB-major shared graph rather than persisting two 0.704
GiB layouts.  Then integrate AB-major persistence into the offline C32 fitting
driver and run the complete V1024 trajectory.

Full AB-major exact evaluation is now implemented and parallelized with
bounded worker-private scratch.  It is numerically consistent with YA-major
evaluation (margin differences around 1e-10 and worst observed log-normalizer
difference 2.6e-7), but remains slower.  At checkpoint 31 with 12 workers,
YA-major took about .103 s and AB-major .192 s.  We therefore retain both
memory-mapped 0.704-GiB views: YA-major for exact certificates/fallback and
AB-major for sampled blocks.  This deliberate duplication buys fast exact
checks while avoiding all checkpoint-local sampled topology.

`scripts/fit_shared_graph_checkpoints.py` now provides the complete offline
warm-started fitting workflow.  It walks checkpoints sequentially, uses the
AB-major graph for stochastic sampled blocks, certifies with the YA-major
graph, invokes the exact layered L-BFGS fallback at any checkpoint that does
not certify, and writes scoring-compatible states in the original coordinate
order.  `scripts/ab_major_exact_probe.py` benchmarks the two exact layouts,
and `scripts/validate_layered_checkpoint_store.py --ab-major` can persist the
AB-major graph alongside a checkpoint store.

The complete text8 V1024/C32 run in
`output/shared_graph_fit_v1024_c32_scorable` finished all 32 checkpoints in
108.739 s with 12 workers and peak RSS 3,981,393,920 bytes (3.71 GiB).  It
used 9,650 stochastic updates, invoked exact fallback at 10 early/intermediate
checkpoints (2.28 s total fallback time), retained zero sampled-topology cache,
and used at most 103,895,216 bytes of sampled reference cache.  Every final
certificate is below .01; the maximum is .009988078 and the last is
.008877395.  The preceding compact-plan run took 142.507 s and peaked at
10,254,237,696 bytes, so the shared-graph route is about 24% faster and uses
about 2.6 times less resident memory.

Twelve-worker scoring took 52.291 s.  In bits per original text8 character,
the updated V1024 row is: star 1.005026, three-pair prediction 0.982837, gain
0.022189; honest accounting is reduced prediction 0.982970, vocabulary
0.062164, escape payload 0.916345, total 1.961478.  These replace the older
V1024 values in `paper/main.tex`.

NEXT: commit and push this production path.  Then port it to the V4096/server
workflow.  The main remaining fitting opportunity is the early trajectory:
several early checkpoints exhaust 500 stochastic updates and then finish
quickly under fallback, whereas later warm starts need only 100--300 updates.
Improve that scheduler only through generic, certificate-driven rules; do not
tune it to text8.  Retain the exact fallback at every checkpoint.  If much
tighter tolerances are later required, revisit the compensated-sum threshold
and the recorded Newton fallback proposal.
