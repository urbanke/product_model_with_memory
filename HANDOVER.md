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

- Rüdiger to specify the concrete idea for the memory scheme (how to beat the
  data-starved joint and the share-nothing context partition).

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
