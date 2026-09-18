#!/usr/bin/env python3
"""E11b — is any E11 variant's gain over V0 real, or split noise?

Pools all 900 queries (3 splits), resamples queries within each class, and reports the
paired weighted-MRR difference vs V0 with a 95% interval. Weighted = 0.16/0.45/0.39
over per-class means, same as every other experiment.
"""
import json, os
import numpy as np
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
W = {1: 0.16, 2: 0.45, 3: 0.39}
rows = []
for s in ("seed0_n300", "seed1_n300", "seed2_n300"):
    rows += json.load(open(f"{SPLITS}/E11_perquery_{s}.json"))
# Empty-pool queries score identically under every variant: they count, with a zero diff.
# (First version dropped them, which inflated every per-class mean diff slightly.)
variants = [v for v in next(r for r in rows if "V0" in r) if v.startswith("V")]
d = lambda r, v: (r[v] - r["V0"]) if "V0" in r else 0.0
by = {c: [r for r in rows if r["cls"] == c] for c in (1, 2, 3)}
rng = np.random.default_rng(0)
def wdiff(v, idx):
    return sum(W[c] * np.mean([d(by[c][i], v) for i in idx[c]]) for c in (1, 2, 3))
full = {c: np.arange(len(by[c])) for c in (1, 2, 3)}
print(f"n per class: { {c: len(by[c]) for c in by} }  (all queries; empty pools = zero diff)")
print(f"{'variant':8s} {'Δweighted':>10} {'95% CI':>20} {'ΔC2':>8}")
for v in variants:
    if v == "V0": continue
    boots = [wdiff(v, {c: rng.integers(0, len(by[c]), len(by[c])) for c in by}) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    dc2 = np.mean([d(r, v) for r in by[2]])
    print(f"{v:8s} {wdiff(v, full):>+10.4f}   [{lo:+.4f}, {hi:+.4f}] {dc2:>+8.4f}")
