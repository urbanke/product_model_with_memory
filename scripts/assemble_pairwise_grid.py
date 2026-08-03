#!/usr/bin/env python3
"""Assemble the pairwise-experiment grid from output/pairwise_*/results.json.

Read-only: touches nothing a running job uses.  Usage:
    python scripts/assemble_pairwise_grid.py [--root output] [--tsv grid.tsv]
"""
import argparse, glob, json, os

KEY = ["lag1", "calibrated", "markov2", "markov2-layered", "markov2-layered-exact"]

def best(members, prefix):
    c = {k: v for k, v in members.items() if k.startswith(prefix)}
    if not c:
        return None, None
    k = min(c, key=c.get)
    return k, c[k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--tsv", default=None)
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, "pairwise_*", "results.json"))):
        run = os.path.basename(os.path.dirname(path))
        d = json.load(open(path))
        m = d["member_bits_per_token"]
        row = {"run": run, "V": d.get("V"), "n": d.get("coded_positions"),
               "C": d.get("checkpoints")}
        for k in KEY:
            row[k] = m.get(k)
        for prefix, name in [("mix:", "best_mix"), ("prod:", "best_prod")]:
            bk, bv = best(m, prefix)
            row[name] = bv
            row[name + "_id"] = bk
        row["star(prod:1,1)"] = m.get("prod:1,1")
        row["family"] = d.get("family_bits_per_token")
        row["best_member"] = d.get("best_member")
        row["seconds"] = round(d.get("seconds", 0))
        rows.append(row)

    cols = ["run", "V", "n", "C", "lag1", "star(prod:1,1)", "best_mix",
            "best_prod", "calibrated", "markov2", "markov2-layered",
            "markov2-layered-exact", "family", "best_member",
            "best_mix_id", "best_prod_id", "seconds"]

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    widths = [max(len(c), *(len(fmt(r.get(c))) for r in rows)) if rows else len(c)
              for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(fmt(r.get(c)).ljust(w) for c, w in zip(cols, widths)))

    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(fmt(r.get(c)) for c in cols) + "\n")
        print(f"\nwrote {args.tsv}")

if __name__ == "__main__":
    main()
