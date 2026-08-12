# Scheduled pipeline audit (9 August 2026)

## Scope and preservation rule

This audit compares the scheduled pipeline with historical implementations.
All established production roots were treated as read-only.  Diagnostic runs
write only below `output/check_*` or `output/audit_*`.  No production state,
score, accounting file, job manifest, or scheduler event log was overwritten.

## Results

### Incremental counts and the one-lag reference are correct

The old Markov-1 evaluator was replayed on the exact current checkpoint
boundaries.  At V=1024, all 31 intervals and 19,363,758 scored tokens agree
with scheduled `pair1` to 0.00183 bit in total.  At V=4096, representative
early, middle, and final intervals agree to relative error below 1.3e-10.  At
V=16384 the corresponding errors are below 2.6e-12.  Direct stream recounts
also show that stored delta counts equal interval counts and their cumulative
sums equal full-prefix counts.  The suspected delta-versus-prefix bug is
therefore ruled out.

### The scheduler does not change an F result

V=4096 checkpoint 15 was rerun by invoking the production F command directly,
outside the scheduler, with the same two workers, 12 total replicas,
128/128/50 batch geometry, seed, and 1,000-step cap.  The saved `log_base_y`,
`correction_ya`, and `correction_yb` arrays are byte-for-byte identical to the
scheduler-produced state.

### A real scorer-normalization defect was found

For supported contexts, `sparse_gated_log_probabilities` reconstructed the
uncorrected baseline mass as `1 - corrected_mass`.  When corrected targets
carry nearly all baseline mass, cancellation can create a false background
larger than the true complement.  The resulting conditional need not sum to
one.  The future scorer now sums the complement explicitly in log space in
that regime, and a deterministic regression test covers the failure.

The defect is real but too small to explain the large-V result.  Re-evaluating
all V=1024 intervals changes the candidate code by -11,083.0 bits, or
-0.00011083 bits per original character.  Representative V=4096 intervals
change by only 0.000257 bit in total.  Existing production accounting remains
unchanged pending an explicit rerun decision.

### Larger-V fitting is cap-limited, but more fitting is not better prediction

At V=1024, 30 of 32 fits stop by the plateau rule and two hit the 1,000-step
cap.  At V=4096, all 32 hit the cap.  This does not imply that the cap should
be raised.  At V=4096 checkpoint 15:

| cold fit | stationarity | held-out bits/reduced token |
|---:|---:|---:|
| 500 steps | 0.00213319 | 7.172799109 |
| production 1,000 steps | 0.00213319 | 7.172799106 |
| 2,000 steps | 0.00166599 | 7.378468911 |

The 500-step and production states give the same held-out code to 3e-9
bits/token; the 2,000-step state is closer to the regularized training optimum
but predicts substantially worse.  Thus tighter calibration is overfitting
this finite prefix rather than recovering missing predictive gain.

At V=1024, the fitted suffix costs 5.10374 bits/reduced token under the
normalization fix, compared with 5.21028 for its unfitted pair-product
initializer and 5.17029 for the one-lag reference.  Fitting is beneficial
overall, but the one-lag model wins early and the fitted three-pair model wins
later.  At larger V the transition is delayed because many more pair cells
must be estimated from the same early prefixes.  This is consistent with the
V=16384 candidate losing to `pair1`; it is not evidence of scheduler drift.

### Posterior-weighted checkpoint staleness is small

The production one-lag table is not a uniform average of per-depth
predictives.  At every checkpoint it evaluates a ratio of depth-averaged
joint laws.  Equivalently, its depth-specific predictives are weighted by the
posterior probability of each depth given the complete prefix.  Production
then freezes that normalized, posterior-weighted table over the next
interval.  The exact sequential reference uses the same rule but updates the
posterior weights after every symbol; its interval cost is therefore the
difference between the layered prefix codelengths at the two boundaries.

`scripts/audit_markov1_interval_regret.py` computes those exact boundary
codelengths from the persisted cumulative YA counts and compares each
difference with the normalized `pair1_bits` already written by the production
scorer.  On cl100k/text8 V1024/C32 the scored suffix costs
100,116,224.245 bits under production and 99,577,630.338 bits under exact
sequential refreshing.  The complete checkpoint-staleness price is therefore
538,593.907 bits, or **0.00538594 bits per original character**.  Before the
first checkpoint production uses the layered memoryless code, which costs
394,444.298 bits, while the exact one-lag prefix costs 372,074.668 bits.  This
adds only 22,369.630 bits = 0.00022370 bpc.  The combined price of the initial
fallback and all frozen checkpoints is therefore 0.00560964 bpc.

The regret per reduced token is largest in the first intervals (0.315 in
interval 0 and about 0.19 in intervals 1--2), then declines to about 0.012 in
the final interval.  The late intervals nevertheless contribute substantial
absolute regret because the geometric blocks are much longer: intervals
24--30 contribute 31.2% of the total, versus 13.7% for intervals 0--3.
Thus staleness is distributed across the file in the expected way, rather
than arising from one bad boundary.  The measured 0.00539 bpc is far too small
to explain the roughly 0.15 bpc difference between reduced-V checkpointed
pair1 and the established full-vocabulary Markov-1 result.  That larger
difference must be sought in vocabulary reduction and escape-identity coding,
not in posterior weighting or checkpoint placement.

The identical audit at V4096 confirms that checkpoint loss itself grows with
the alphabet.  Its 31 scored intervals lose 0.01951635 bpc, and the initial
memoryless prefix adds 0.00006705 bpc, for 0.01958341 bpc total.  The four
successive groups of scored intervals 1--8, 9--16, 17--24, and 25--31
contribute 0.00376203, 0.00365099, 0.00560174, and 0.00650159 bpc.  The
corresponding V1024 values are 0.00126817, 0.00100352, 0.00143187, and
0.00168238 bpc.  Thus the larger-V loss is present throughout the file and is
about four times larger in the second half; it is not an anomalous initial
interval.  The geometric schedule still roughly balances loss across groups
containing equal numbers of checkpoints, but becomes somewhat tail-heavy at
V4096.  More context rows divide the same prefix among more evolving row
distributions, so freezing them for the same intervals is materially more
expensive.

## Current conclusion

The scheduled construction and F-task semantics reproduce the trusted
reference paths.  The only demonstrated correctness bug is the supported-
context normalization cancellation, now fixed in source but not retroactively
applied to stored accounting.  Exact prefix accounting also rules out
posterior weighting and checkpoint staleness as explanations of the much
larger full-vocabulary versus reduced-vocabulary Markov-1 gap.  The next audit
should isolate the cost of separating the reduced symbol stream from escaped
token identities.  Independently, a robust candidate-versus-pair1 coding rule
to discuss is a sequential Bayesian mixture: it is an honest distribution and
loses at most one bit in total to the better complete expert, without choosing
a corpus-specific threshold.
