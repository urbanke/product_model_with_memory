# Checkpoint scheduler refactor

The target workflow treats construction, fitting, and evaluation at each of
32 checkpoints as coarse, restartable jobs.  The scheduler minimizes predicted
experiment makespan subject to dependency, core, and memory constraints.  It
does not try merely to maximize instantaneous CPU utilization.

## Initial dependency policy

Construction and fitting each remain a single causal chain initially:

\[
C_{k-1}\longrightarrow C_k,
\qquad
F_{k-1}\longrightarrow F_k.
\]

The first edge reflects cumulative counting.  The second preserves the fitted
warm start.  A fit additionally requires its checkpoint problem and graph
layer.  Interval evaluation requires its own fitted state and the next
construction boundary:

\[
C_k,\,G_k,\,F_{k-1}\longrightarrow F_k,
\qquad
F_k,\,C_{k+1}\longrightarrow E_k.
\]

There are 32 construction jobs, 32 fits, and 31 scored intervals.  Initial
prefix coding and final honest-code aggregation are separate small jobs.

## Incremental graph contract

The present final-union graph delays every fit until all checkpoint problems
exist.  Its exact traversal is already persisted as 32 physical triangle
birth layers, but those layers are constructed together from the final support;
the AB-major stochastic graph remains monolithic with per-triangle birth
labels.  Refactor publication into an immutable initial chunk and 31 deltas:

\[
G_k=G_0\cup\Delta G_1\cup\cdots\cup\Delta G_k.
\]

An old edge is never moved.  A newly active YA, YB, or AB edge receives a
stable append-only identifier and is written to the current delta.  A triangle
is written at the checkpoint at which its last constituent edge becomes
active:

\[
b(y,a,b)=\max\{b(y,a),b(y,b),b(a,b)\}.
\]

All chunks are read-only after publication.  Exact and stochastic native code
must accept a list of chunk descriptors and traverse it in one native call;
Python must not invoke the numerical kernel once per layer per update.

This design requires monotone checkpoint supports.  The audit script
`scripts/audit_checkpoint_graph_deltas.py` checks this assumption before any
production representation is changed.

## Artifact publication

Every job writes to temporary paths, closes and flushes them, renames them
atomically, and publishes a completion manifest last.  Consumers depend on
the manifest, not the incidental existence of a data file.  Immutable graph
chunks and token streams are memory mapped so independent processes can share
physical pages through the operating system.

Each manifest records its job type and checkpoint, strict parent artifacts,
wall and CPU times, workers, private peak memory, mapped bytes, and observable
work sizes.  Relevant work sizes include new tokens and active supports for
construction, active edges/triangles and warm-start diagnostics for fitting,
and interval length for evaluation.

## Online scheduling

For each unfinished job and candidate worker allocation, the scheduler keeps
an estimated runtime and private-memory peak.  It also learns slowdown factors
for overlapping job classes such as C+F, C+E, F+E, and E+E.  Whenever a job
finishes, the scheduler updates these estimates, releases newly ready jobs,
simulates feasible continuations of the remaining dependency graph, and
chooses the immediate allocation with the smallest predicted makespan.

Only one fitting job runs at a time in the initial version.  Fitting need not
always run: construction or evaluation may lie on the predicted critical path.
Multiple fitting chains are a later option only if measurements show that the
single fitting chain dominates total completion time.

## Refactor order

1. Audit support monotonicity and graph births on existing checkpoint runs.
2. Implement append-only graph chunks with atomic manifests.
3. Add numerically identical multi-chunk exact and stochastic traversal.
4. Remove evaluation's artificial need for the next fitted state merely to
   discover the next checkpoint boundary.
5. Expose C, graph publication, F, and E as independently restartable jobs.
6. Measure isolated and overlapping job classes.
7. Implement the online resource-constrained scheduler.
8. Validate end-to-end code lengths against the current sequential workflow.

## Incremental executable path

`calibration_checkpoint_probe.py --stop-after-checkpoint k` publishes one
additional construction state.  Every state carries the cumulative unigram
and three sparse sufficient-statistic maps, so the next invocation begins at
the stored prefix and never replays earlier tokens.  A resumed
three-checkpoint construction is bitwise identical to uninterrupted
construction.

`publish_checkpoint_graph_delta.py` loads only the previous stable support
state and the new checkpoint problem, then publishes the new immutable graph
delta.  The fitter's `--delta-store` path memory-maps triangle payloads from
all deltas through k.  It expands only their small YA row directories for one
native exact call; it does not concatenate or copy triangle payloads.  Sampled
updates construct and cache intersections only for sampled AB blocks.  The
old cumulative materializer remains solely as a validation utility.

The fitter accepts `--start k --stop k+1`.  The interval scorer evaluates one
interval from `F_k` and an explicit next boundary, without waiting for
`F_{k+1}`.  `run_fixed_checkpoint_schedule.py` executes a JSON list of jobs in
explicit concurrent waves, checks dependencies and resource declarations,
verifies outputs, and publishes completion manifests.

`make_fixed_checkpoint_schedule.py` generates two deliberately simple fixed
orders.  Both enforce the serial chains

\[
C_0\to C_1\to\cdots,
\qquad
F_0\to F_1\to\cdots.
\]

The `phased` order completes all construction and graph jobs before fitting;
the `pipeline` order overlaps only different job types whose declared inputs
already exist.  A three-checkpoint text8 smoke test ran all 11 jobs in both
orders.  The three construction states and three fitted states were bitwise
identical, and both interval score records were identical after excluding
elapsed time.  This establishes scheduling-order invariance for the current
path on the test problem.  The comparison was repeated after removing count
replay and cumulative graph materialization: all C and F states remained
bitwise identical and both E records remained identical.  No cumulative
materialized graph directory was created.

## Portable analytic scheduler, phase one

The first automatic scheduler deliberately uses no measured wall-clock times.
It estimates work in transparent dimensionless primitive-visit units from the
known checkpoint prefixes, fitting parameters, and reduced unigram law.  A
small logarithmic histogram of unigram probabilities estimates expected
distinct pair supports and compatible triangles without constructing a
quadratic or cubic vocabulary table.  Its phase estimates are

\[
 W_C(k)=\Delta N_k+3\Delta D_2(k),\qquad
 W_G(k)=\Delta T(k),\qquad
 W_E(k)=N_{k+1}-N_k,
\]

and fitting is the active triangle count times the analytically known number
of sampled and exact passes.  These quantities establish priorities; they are
not reported as seconds.

The initial portable speedup prior exposes modes 1--4 with speedups
\(1,1.9,2.25,2.4\), saturating thereafter.  This encodes the agreed
conservative qualitative model rather than a benchmark of the current host.
G and E presently expose only their implemented one-worker modes.  An
event-driven critical-path list planner assigns a fixed worker count when each
job launches.  The executor tracks its own allocated workers, and whenever a
process finishes it immediately starts the highest-priority ready jobs that
fit the released capacity.  It does not wait for fixed waves and does not use
noisy instantaneous system CPU utilization.

`plan_analytic_checkpoint_schedule.py` writes the analytic profile, task DAG,
worker allocations, and predicted dimensionless event times.
`run_analytic_checkpoint_schedule.py` combines that plan with the existing
restartable job definitions.  On the local enwik8 V1024/C32 planning case the
127-job plan respects every dependency and serial C/G/F chain and peaks at
nine of the twelve available workers; the remaining capacity follows from the
current four-worker cap for C/F and one-worker G/E modes, not an executor
barrier.  Phase two will compare completion manifests with the analytic model
and may update phase costs and speed curves online.  That measurement-based
adaptation is intentionally not part of phase one.

The first automatic end-to-end smoke test used three text8 checkpoints and
eleven C/G/F/E jobs.  Against a matching fixed phased execution, every array
in all three construction states and all three fitted states was bitwise
identical.  Both interval score records were identical after removing elapsed
time.  Thus the event-driven executor passes the same scheduling-order
invariance criterion as the earlier hand-authored pipeline.
