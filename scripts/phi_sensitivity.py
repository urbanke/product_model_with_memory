#!/usr/bin/env python3
"""How many nats of error in ln phi can we afford?

Every accuracy decision in this project has been argued from errors in
ln phi --- the store is certified to 1.3e-5 nats, the saddle expansion
misses by 5e-3 at shallow depth, an r-ladder interpolates to below 1e-6
--- and none of those numbers has ever been connected to the quantity
actually reported, which is bits per character.  The thresholds were
therefore guessed.  This script measures the conversion factor instead.

It re-runs one real experiment several times, injecting a controlled
error field into every value the moment store serves, and reads off how
the codelength moves.  Two error models bracket what a real scheme does:

  WAVE  eps * cos(u / lambda + phase(L, r)): smooth in u, oscillating
        in sign, different in every column.  This is the shape a real
        interpolation or asymptotic error actually has.

  BIAS  exactly +eps everywhere.  Nothing cancels anywhere; the
        pessimistic end.

A real scheme lies between the two, so both are reported: a single
number here would be a guess wearing a measurement's clothes.

A FIRST VERSION OF THIS SCRIPT WAS WRONG in a way worth recording,
because it is the trap this whole exercise exists to avoid.  It used an
error field decorrelated between adjacent u grid points, on the
reasoning that independent signs are conservative.  They are not: the
evaluator DIFFERENTIATES the column to locate the Laplace peak, so a
field that jumps between neighbours is amplified by 1/H and 1/H^2 with
H = 0.02.  It reported that 1e-8 nats costs 7.5e-4 bits/token --- an
amplification of 1e5 --- and blew up the evaluator entirely at 1e-3.
That number is an artefact of the error model, not a property of the
estimator.  Errors of the shape our candidate schemes actually produce
are smooth, and the WAVE model is the corrected stand-in.

Because getting the model right is evidently not easy, the script also
runs the measurement that needs NO model:

  SADDLE  serve every level at or above a cutoff from the second-order
          saddlepoint expansion instead of the store, and sweep the
          cutoff.  This is a real candidate evaluator, measured end to
          end.  Where the perturbation study says what an error would
          cost, this says what a specific scheme does cost.

Trust the saddle sweep over the perturbation sweep where they disagree.

    python scripts/phi_sensitivity.py --ids output/streams/bpe_text8 \\
        --jobs 12 --out output/phi_sensitivity_text8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_EPS = "1e-6,1e-4,1e-3,1e-2"


def run_once(script: str, ids: str, jobs: int, workdir: Path,
             env_extra: dict[str, str], tag: str, top_k: int,
             m_grid: str, n: int | None) -> dict:
    """One full experiment with a perturbation applied, in a fresh
    process (the perturbation is read from the environment at import).

    The perturbation is applied when a value is READ, never when a
    column is built, so a perturbed run cannot contaminate the store.
    Run the baseline first anyway: it builds whatever is missing, and
    every later run then reads the same columns."""

    out = workdir / f"run_{tag}"
    env = dict(os.environ)
    env.update(env_extra)
    src = ["--corpus", ids[len("corpus:"):]] if ids.startswith("corpus:") \
        else ["--ids", ids]
    cmd = [sys.executable, script, *src, "--jobs", str(jobs),
           "--top-k", str(top_k), "--m-grid", m_grid, "--out", str(out)]
    if n:
        cmd += ["--n", str(n)]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        # One amplitude failing is a RESULT --- it means the evaluator
        # stops producing a probability at all, which is exactly what we
        # are trying to locate.  Record it and carry on; aborting the
        # sweep would throw away every larger and smaller point too.
        #
        # But WHICH failure it is matters, and the first version of this
        # took the last non-empty line, which on a run that emits a
        # numpy warning is "warnings.warn(" --- the continuation line of
        # the warning, carrying no information at all.  Take the real
        # exception: the last "Type: message" line of the traceback.
        text = proc.stdout + proc.stderr
        why = "(no output)"
        for line in reversed(text.splitlines()):
            s = line.strip()
            if s and not s.startswith(("File \"", "warnings.warn",
                                       "^", "~", "|")) \
                    and re.match(r"^[A-Za-z_.]+(Error|Exception|Warning):", s):
                why = s
                break
        else:
            tail = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if tail:
                why = tail[-1]
        full = out.parent / f"fail_{tag}.log"
        full.write_text(text)          # the whole thing, for when the
        return {"tag": tag, "seconds": time.time() - t0,   # summary is not enough
                "bits_per_token": None, "failed": why[:300],
                "log": str(full)}
    res = json.loads((out / "results.json").read_text())
    return {"tag": tag, "seconds": time.time() - t0,
            "bits_per_token": res["family_bits_per_token"],
            "chars_per_token": res.get("chars_per_token"),
            "results": str(out / "results.json")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", required=True)
    p.add_argument("--script", default="scripts/state_family_experiment.py")
    p.add_argument("--eps", default=DEFAULT_EPS)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--top-k", type=int, default=100276)
    p.add_argument("--m-grid", default="0,100277",
                   help="the member grid; the default is the two-member "
                        "grid of the first-order runs, which is what makes "
                        "this cheap")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--saddle-cutoffs", default="54,43,35",
                   help="serve levels >= cutoff without the store.  54 "
                        "(= l_max) is the deepest level only; 2 would mean "
                        "the store is not used at all.  Cost scales with "
                        "the NUMBER of levels served, so sweeping to 2 is "
                        "~50x the cost of the first row --- add lower "
                        "cutoffs deliberately, not by default")
    p.add_argument("--saddle-n", type=int, default=None,
                   help="run the substitution block on a prefix of this "
                        "many tokens.  The quantity of interest is a "
                        "DIFFERENCE from baseline, which converges far "
                        "sooner than the codelength itself; the baseline "
                        "for that block is recomputed on the same prefix "
                        "so the two are comparable")
    p.add_argument("--ladder-every", default=None,
                   help="comma-separated decimations m of the anchor "
                        "store's own grid, e.g. 1,4,8.  Needs --tables "
                        "pointing at a store from build_anchor_store.py.  "
                        "This is the correct form: a nominal FACTOR "
                        "recomputes round((1+f)^k), which does not "
                        "coincide with any decimation and would silently "
                        "serve a gappy ladder")
    p.add_argument("--ladder-cutoff", type=int, default=None,
                   help="levels at or above this are served by the "
                        "expansion while the ladder serves the rest, so "
                        "each row is the COMPLETE system.  Defaults to "
                        "the anchor store's top level + 1")
    p.add_argument("--ladder-degree", type=int, default=11,
                   help="degree 11 measured flat to 4%% spacing with no "
                        "outliers; degree 7 showed sporadic blowups two "
                        "to three orders above its own median")
    p.add_argument("--tables", default=None,
                   help="store for the BASELINE and perturbation rows "
                        "(sets PMM_UNIVERSAL_TABLES).  This must be a "
                        "store that holds every column the run asks for "
                        "--- i.e. the cache, not an anchor store")
    p.add_argument("--ladder-tables", default=None,
                   help="store for the LADDER rows only.  It must be an "
                        "anchor store, and the two cannot be the same "
                        "store: an anchor store holds only anchors, so an "
                        "unladdered run against it rebuilds every missing "
                        "column on demand --- which is the whole cache, "
                        "silently, as a hang")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # validate BEFORE the baseline runs.  These checks were originally
    # placed next to the ladder block, i.e. after a full baseline had
    # already been paid for --- and the failure they guard against is a
    # silent hour-long hang, so catching it late is barely catching it.
    if args.ladder_every:
        if not args.ladder_tables:
            raise SystemExit(
                "--ladder-every needs --ladder-tables pointing at an "
                "anchor store.\nDo NOT point --tables at one: the "
                "baseline row needs an exact column for\nevery r the run "
                "touches, an anchor store does not have them, and the "
                "store\nwould rebuild the entire cache on demand --- "
                "which looks exactly like a hang.")
        if args.ladder_tables == args.tables:
            raise SystemExit(
                "--tables and --ladder-tables must differ: the baseline "
                "needs the full\ncache, the ladder needs the anchor grid.")
        if not (Path(args.ladder_tables) / "anchors.json").exists():
            raise SystemExit(
                f"{args.ladder_tables}/anchors.json not found; that is "
                "not an anchor store.\nBuild one with "
                "scripts/build_anchor_store.py.")
        # An anchor store covers a level RANGE.  The run's l_max is
        # higher, so a ladder on its own would ask for levels the store
        # does not have and simply fail.  Pairing it with the
        # substitution cutoff at the store's top level + 1 is not a
        # convenience: it is the only configuration in which the ladder
        # is even runnable --- and it happens to be the complete system
        # we are trying to measure, ladder below, expansion above.
        _meta = json.loads(
            (Path(args.ladder_tables) / "anchors.json").read_text())
        _top = max(int(k) for k in _meta["levels"])
        if args.ladder_cutoff is None:
            args.ladder_cutoff = _top + 1

    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    eps_list = [float(x) for x in args.eps.split(",")]

    base_env = {"PMM_UNIVERSAL_TABLES": args.tables} if args.tables else {}

    def go(env_extra, tag, n=None):
        return run_once(args.script, args.ids, args.jobs, work,
                        {**base_env, **env_extra},
                        tag, args.top_k, args.m_grid,
                        args.n if n is None else n)

    def show(res):
        """bits/token, and bits/character beside it when the run
        reports one.  The paper quotes bits/CHARACTER; printing only
        bits/token under a line claiming it is 'the number the paper
        reports' invited exactly the confusion it caused."""

        bpc = (res.get("results") and
               json.loads(Path(res["results"]).read_text())
               .get("family_bits_per_character"))
        return (f"{res['bits_per_token']:.6f} bits/token"
                + (f"  ({bpc:.6f} bits/char)" if bpc else ""))

    print("Baseline (no perturbation).", flush=True)
    base = go({}, "base")
    b0 = base["bits_per_token"]
    print(f"  {show(base)}  ({base['seconds']:.0f}s)")
    print("  deltas below are in bits/token; the paper's headline figure "
          "is the\n  bits/character column.\n", flush=True)

    rows = [dict(base, eps=0.0, model="none", delta=0.0)]
    for model, var, what in (
            ("wave", "PMM_PHI_WAVE", "eps*cos(u/lambda + phase(L,r))"),
            ("bias", "PMM_PHI_BIAS", "constant +eps")):
        print(f"{model.upper()}: error field {what}")
        print(f"{'eps (nats)':>12} {'bits/token':>13} {'delta':>13} "
              f"{'delta/eps':>12}")
        for eps in eps_list:
            r = go({var: repr(eps)}, f"{model}_{eps:g}")
            if r["bits_per_token"] is None:
                rows.append(dict(r, eps=eps, model=model, delta=None))
                print(f"{eps:>12.1e} {'FAILED':>13}   {r['failed'][:60]}",
                      flush=True)
                continue
            d = r["bits_per_token"] - b0
            rows.append(dict(r, eps=eps, model=model, delta=d))
            print(f"{eps:>12.1e} {r['bits_per_token']:>13.6f} "
                  f"{d:>+13.3e} {d/eps:>12.3e}", flush=True)
        print()

    if args.saddle_cutoffs:
        print("SUBSTITUTION: levels >= cutoff served WITHOUT the store")
        print("        (certified series where it applies, order-2 saddle "
              "elsewhere --- a")
        print("         real evaluator measured end to end, with no error "
              "model in between)")
        sub_n = args.saddle_n or args.n
        sub_b0 = b0
        if args.saddle_n:
            # a delta against a full-stream baseline would be measuring
            # the prefix, not the substitution
            pref = go({}, "base_prefix", n=args.saddle_n)
            sub_b0 = pref["bits_per_token"]
            print(f"        on a {args.saddle_n:,}-token prefix; prefix "
                  f"baseline {sub_b0:.6f} bits/token")
        print(f"{'cutoff L':>12} {'bits/token':>13} {'delta':>13} "
              f"{'seconds':>9}")
        for cut in [int(x) for x in args.saddle_cutoffs.split(",")]:
            r = go({"PMM_PHI_SADDLE_MIN_L": str(cut)}, f"saddle_L{cut}",
                   n=sub_n)
            if r["bits_per_token"] is None:
                rows.append(dict(r, model="saddle", cutoff=cut, delta=None))
                print(f"{cut:>12} {'FAILED':>13}   {r['failed'][:60]}",
                      flush=True)
                continue
            d = r["bits_per_token"] - sub_b0
            rows.append(dict(r, model="saddle", cutoff=cut, delta=d,
                             n_tokens=sub_n))
            print(f"{cut:>12} {r['bits_per_token']:>13.6f} {d:>+13.3e} "
                  f"{r['seconds']:>9.0f}", flush=True)
        print()

    if args.ladder_every:
        print(f"COMPLETE SYSTEM: ladder below L={args.ladder_cutoff}, "
              f"expansion at or above it")
        print(f"        anchors from {args.ladder_tables}, degree "
              f"{args.ladder_degree}; NO exact column is read at any "
              f"level.")
        print("        This is the store we would ship, measured end to "
              "end on the real workload.")
        print("        (the other half of the small-store design, "
              "measured end to end --")
        print("         intrinsic error in nats does not convert to bits "
              "without a factor we\n         do not know to better than "
              "an order of magnitude)")
        if not args.tables:
            print("        WARNING: no --tables given, so this runs "
                  "against the default store.\n        Unless that store "
                  "has an anchors.json the run will fail; if it does\n"
                  "        not fail, check what it actually served.")
        print(f"{'every':>8} {'eff. spacing':>13} {'bits/token':>13} "
              f"{'delta':>13} {'seconds':>9}")
        for m in [int(x) for x in args.ladder_every.split(",")]:
            env = {"PMM_PHI_LADDER_EVERY": str(m),
                   "PMM_PHI_LADDER_DEGREE": str(args.ladder_degree),
                   "PMM_PHI_SADDLE_MIN_L": str(args.ladder_cutoff),
                   "PMM_UNIVERSAL_TABLES": args.ladder_tables}
            r = go(env, f"ladder_every{m}")
            if r["bits_per_token"] is None:
                rows.append(dict(r, model="ladder", every=m, delta=None))
                print(f"{m:>8} {'--':>13} {'FAILED':>13}   "
                      f"{r['failed'][:50]}", flush=True)
                continue
            d = r["bits_per_token"] - b0
            rows.append(dict(r, model="ladder", every=m, delta=d,
                             degree=args.ladder_degree))
            print(f"{m:>8} {'(see store)':>13} "
                  f"{r['bits_per_token']:>13.6f} {d:>+13.3e} "
                  f"{r['seconds']:>9.0f}", flush=True)
        print()

    payload = {"stream": args.ids, "script": args.script,
               "baseline_bits_per_token": b0, "rows": rows}
    (work / "results.json").write_text(json.dumps(payload, indent=2))

    print("How to read this.  In the perturbation blocks `delta/eps` is "
          "the conversion\nfactor: bits per token per nat of ln phi "
          "error.  Constant down a column means\nthe response is linear "
          "and the factor can be used to DERIVE an accuracy target;\nif "
          "it is not constant, something is amplifying or absorbing the "
          "error and that\nneeds explaining before any threshold is set "
          "from it.\n\nThe SADDLE block needs no such reasoning.  Each "
          "row is a real evaluator on the\nreal workload, and `delta` is "
          "simply how much that scheme costs.  A cutoff whose\ndelta is "
          "far below the last digit we report is a cutoff we can ship.")
    print(f"\nwritten: {work/'results.json'}")


if __name__ == "__main__":
    main()
