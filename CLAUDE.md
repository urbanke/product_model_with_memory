# Standing instructions for every session in this repository

Ruediger's rules. Follow them without being reminded.

## Language

- Write in plain English. Short sentences. One idea per sentence.
- No jargon. If a technical term is needed, define it the first time.
- Before giving a command to run, say in plain words what it does and
  why. Do the same for every step of a multi-step command.
- Keep answers short.

## Working style

- Say plainly what is measured versus what is inferred.
- One command at a time. Ruediger runs it and pastes the output back.
- Fast, cheap checks before long or expensive runs.
- Never edit a file that a running job has imported. Spawned workers
  re-import files from disk. This has crashed runs before (2 Aug 2026).
- Correctness first, then efficiency.
- Read HANDOVER.md at the start of a session and append an entry when
  something notable is done, decided, or measured.

## The science

- Everything Bayesian: mixtures with priors, integrated exactly by
  algorithms. No learned or tuned weights.
- Production invariant: every data-bearing symbol sequence uses the
  depth-averaged layered product-simplex predictor. This includes initial
  prefixes and token identities behind escape. Never substitute KT/Jeffreys,
  Laplace add-one, plug-in probabilities, or another estimator as a coding
  convenience. A different estimator is allowed only as an explicitly named
  scientific comparison with separate provenance and accounting. Metadata
  descriptions such as an enumerative vocabulary-subset code are not symbol
  predictors and are the only exception.
- Small-vocabulary runs (for example V=1024) are behavioral: they rank
  methods in a controlled arena. They are not compression schemes for
  the original file. Real-scheme claims live at the full vocabulary.

## Measurement standards (decided 4 Aug 2026)

- One measure: honest bits per character (= bits per byte) of the
  original file, complete codes, vocabulary charge included.
- One tokenization: the LLM tokenizer (the bpe_* streams). Word-level
  results do not enter the paper.
- Exact telescoped evaluation whenever the scheme allows it (all
  mixtures per state). When chunking is unavoidable: GEOMETRIC
  checkpoint spacing, C = 32 (measured: ~0.03 bits/token residual vs
  0.17 for equal spacing at V=1024 first order). Always report the
  schedule next to the number.

## Performance rules (written after 4-5 Aug 2026)

- No plain-Python loop over tokens, pairs, or triples in any
  evaluator. Inner loops are numpy over whole blocks. Requests to the
  layered builder are batched: one call per block.
- Every performance claim is measured on Ruediger's machine, never
  assumed from the cloud box (2 cores, Linux: parallel behavior does
  not transfer; correctness does).
- One change at a time, against a fixed benchmark command, with the
  decision rule agreed BEFORE the run: beats the standing time or is
  reverted immediately.
- Programs report their own utilization (eval vs eval_cpu per
  checkpoint); Activity Monitor screenshots are corroboration, not
  the measurement.
