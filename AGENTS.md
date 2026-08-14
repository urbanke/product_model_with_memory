# Non-negotiable production estimator

Every data-bearing symbol sequence in production code must use the
depth-averaged layered product-simplex predictor identified by
`layered_depth_averaged_product_simplex_v1`. This includes the primary
predictive stream, checkpoint prefix, fallback pair laws, and token identities
transmitted behind an escape symbol.

Do not replace it with KT/Jeffreys, add-one/Laplace-rule, plug-in/count-table,
or any other estimator for convenience or speed. An alternative may be added
only as an explicitly named scientific comparison, with separate provenance
and accounting, after it has been discussed with the user. It must never be a
silent fallback or the default of a production entry point.

Metadata descriptions, such as an enumerative code for the selected
vocabulary subset, are not symbol predictors and are outside this rule.

# Non-negotiable production tokenizer

Every production corpus stream must be the complete ChatGPT
`cl100k_base` BPE tokenization of the named source file. The repository's
custom `ours` tokenizer, word tokenization, raw bytes, truncated prefixes, and
other representations are diagnostic scientific comparisons only. They must
never feed a production planner, schedule, result, or paper table.

Production stream preparation must validate `stream.json` and `ids.npy`,
require `representation=bpe` and `encoding=cl100k_base`, require the complete
token sequence rather than a prefix, and propagate immutable source-manifest
and source-ID hashes through the reduced-stream manifest, anchor plan, and
schedule. Production consumers must fail closed when this provenance is
missing or different. Shared enforcement belongs in
`src/product_model_with_memory/production_coding.py`.

Programs that intentionally compare tokenizers must say `DIAGNOSTIC ONLY` in
their description and write to clearly diagnostic output paths. Do not add an
override that makes a diagnostic stream production eligible.

Production artifacts must record the estimator identifier and consumers must
reject an explicitly different identifier. Put shared enforcement in
`src/product_model_with_memory/production_coding.py`; do not add independent
estimator-selection switches to production scripts.

# Non-negotiable production scheduling policy

Production checkpoint campaigns must use the dependency-driven scheduler. Its
schedule waves are stable priority hints, not phase barriers: launch every
dependency-ready job that fits the live CPU and private-memory budgets. Do not
restore rigid whole-phase synchronization.

Preserve fine-grained jobs and anchor-causal priority. In particular, ready
multi-core pair jobs from an earlier anchor must outrank later serial unigram
jobs; otherwise serial jobs can consume the memory budget while most CPU cores
remain idle. Use spare capacity to advance later independent work.

Resource contracts must reflect the job's actual scale. Prefix-dependent jobs
must not all be charged at their full-prefix peak. Keep separate CPU and memory
limits: the local production default is 13 workers with a 12 GiB private-memory
budget, multi-core pair jobs use four workers, and intrinsically serial
unigram jobs use one. These numbers may be retuned for another machine from
measurements, but the policy must not depend on vocabulary size.

The anchored schedule generators expose named resource profiles.  `laptop`
uses 13 workers and 12 GiB on the current 24-GiB Mac; `m4pro` uses 14 workers
and 24 GiB on the 36-GiB M4 Pro; `cpu64` initially uses 64 workers and 192 GiB
on node 14's 256-GiB Slurm allocation; `scitas` uses 72 workers and a 384-GiB
private-memory scheduling budget inside the current 440-GiB Jed Slurm
allocation.  On SCITAS, prefix-scaled marginal jobs reserve between 1 and
16 GiB each.  The former 3-GiB ceiling admitted too many late enwik9 marginals
and caused measured OOM failures at roughly 440 GiB RSS; do not reduce the
corrected reservations without a utilization and peak-memory replay.
Keep the mathematical DAG identical across profiles.  Distinct anchors are
independent after their prefix-count snapshots exist, so topology, fitting,
and scoring must run across anchors concurrently.  Tail work (score, fit,
topology, assembly) outranks opening further construction when both are ready,
while spare resources continue advancing later anchors.

This policy restored sustained use of roughly 12 cores in the text8 anchored
campaign after phase-wide unigram priority had reduced execution to four or
five serial cores. Do not redesign or remove these scheduling invariants
without explicit agreement with the user. Any change must include a small
utilization replay and scheduler regression tests before a production restart.
