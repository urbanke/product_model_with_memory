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
layer, and evaluation requires its own fitted state:

\[
C_k,\,G_k,\,F_{k-1}\longrightarrow F_k,
\qquad
F_k\longrightarrow E_k.
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

## Current correctness bridge

The first implementation intentionally favors independently testable jobs over
optimal execution.  `calibration_checkpoint_probe.py --stop-after-checkpoint k`
publishes one additional construction state; it currently replays cumulative
counts from the beginning when invoked again.  `publish_checkpoint_graph_delta.py`
publishes the append-only graph delta, and
`materialize_checkpoint_delta_store.py` temporarily assembles deltas through
`k` into the legacy fitter format.  The existing fitter already accepts
`--start k --stop k+1`.  `score_checkpoint_interval.py` evaluates one interval
from `F_k` and an explicit next boundary, without waiting for `F_{k+1}`.

`run_fixed_checkpoint_schedule.py` executes a JSON list of jobs in explicit
concurrent waves.  It checks strict dependencies, worker capacity, and private
memory declarations, verifies expected outputs, and publishes one completion
manifest per job.  This bridge permits correctness comparisons among fixed
orders before the compatibility materialization and count replay are replaced
by direct incremental traversal and persisted count states.
