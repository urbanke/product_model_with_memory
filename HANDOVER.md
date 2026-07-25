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
