# Calibration optimization registry

This file prevents repeated implementation of ideas already measured.  Read
it before changing the three-pair calibration evaluator or solver.  Detailed
numbers and chronology are in `HANDOVER.md`.

## Tried: do not repeat as a new idea

| Family | Implementations already measured | Verdict |
|---|---|---|
| Dense/union margins | Serial union expansion; AB-edge sharding; disjoint output shards | Correct, but substantially more work than intersection factorization. |
| Intersection layouts | Explicit four-index plan; depth-layered YA-major CSR; node/birth-major representation; AB-major representation | YA-major is fastest for full exact passes; AB-major is useful for sampled blocks but slower for full passes. |
| Lazy topology | LRU block intersections with 8/16/32-block caches; concurrent cache misses; direct sampled AB-range traversal | Direct AB-major sampling won.  LRU reconstruction trades memory for time. |
| Traversal fusion/indexing | Indexed AB edges; fused current/reference traversal; current-only cached-reference traversal; branch-free final-checkpoint traversal | All slower than the worker-local fixed-batch route. |
| Accumulation/reduction | Full worker accumulators; sparse NumPy accumulators; contiguous worker accumulators; output sharding; centralized context sums; selected-YA scans; serial and parallel reductions | Rearrangements did not remove the dominant random triangle traffic. |
| Work partitioning | Equal rows; triangle-balanced rows; 64/128/256 blocks; 1/4/8/12 workers | Triangle balance and parallel reductions help modestly, but full traversals still saturate memory bandwidth. |
| Exact Hessian layouts | Layered YA-major native Hessian; AB-major Hessian with complete contexts | AB-major Hessian was about 2.7x slower.  Layered Hessian reaches only 4.09x at 12 workers. |
| First-order solvers | IPF; Anderson IPF; L-BFGS; certificate-aware stopping; trust-region displacement; stochastic fixed batches; SVRG refresh intervals; adaptive schedules | Current stochastic/SVRG route is the production baseline.  Anderson and extensive hand-tuned scheduling did not solve the large-support tail robustly. |
| Exact Newton | Analytic Hessian products; finite-difference products; trust-region Newton-CG; Fisher/Jacobi scaling; multiple product budgets; relaxed objective | Exact Newton can rescue some near-solution states, but full Hessian passes are too expensive.  It is a bounded fallback, not the default. |
| Parallel execution | Multiple chains, interleaving, inner margin workers, worker-local gradient reduction, native C traversal | Useful but limited by extra cold iterations or bandwidth.  More cores alone are not the answer. |

## Proposed before but not sufficiently distinct

Two-dimensional tiling, A-context tiling, and another locality-preserving
ordering have been proposed repeatedly.  They are too close to the measured
YA-major/AB-major/indexed/fused variants to justify implementation without a
new mathematical argument showing that they eliminate triangle products or
asymptotically reduce bytes read.  A claim of better cache locality alone is
not enough.

## Genuinely untried candidate

### Subsampled Newton--CG

Use an exact gradient/certificate at the outer iterate, but form
Hessian--vector products from the existing sampled AB blocks.  Solve the
sampled Newton system only approximately under a hard product budget, then
accept a proposed step only after an independent full relaxed-objective
evaluation.  Increase the Hessian sample only when its measured model quality
requires it.

This differs fundamentally from prior work: it reduces the triangles touched
by each curvature product rather than changing the layout of a full product.
It is a standard finite-sum optimization method, not a project-specific
learning-rate rule.  Relevant primary references include Bollapragada, Byrd,
and Nocedal, *Exact and Inexact Subsampled Newton Methods for Optimization*
(IMA Journal of Numerical Analysis, 2019), and Berahas, Bollapragada, and
Nocedal, *An Investigation of Newton-Sketch and Subsampled Newton Methods*
(Optimization Methods and Software, 2020).

Before production integration, a bounded probe must answer:

1. Does a fixed sampled Hessian give a descent direction for the independently
   evaluated full relaxed objective?
2. How many sampled triangle products replace one full Hessian product?
3. Does increasing the sample improve the direction predictably?
4. Does the complete stochastic-plus-subsampled-Newton finish beat stochastic
   continuation from the identical saved state?

If these fail, stop.  Do not return to another full-traversal layout trial.
