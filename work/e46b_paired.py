#!/usr/bin/env python3
"""E46b — the PAIRED half-width on the clean subset, which is what the threshold is about.

E46 measured the bootstrap interval of the ABSOLUTE weighted score and I then compared the
+0.020 continuation threshold against it. That is a category error: the threshold applies to a
DIFFERENCE between two arms scored on the same queries, and a paired difference cancels
per-query difficulty, which is what dominates the absolute interval. On the full set the same
distinction is a factor of five -- E37's paired interval was +/-0.0037 where the absolute
half-width is +/-0.0191.

This measures the paired half-width directly, using two arms that already exist and differ
per query (generation off vs on), so the number is observed rather than scaled from an estimate.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e22_ranker_v2 import rank_score, W
from e39_gate import CACHE, code_hash

WORK = os.path.dirname(os.path.abspath(__file__)); SPLITS = os.path.join(WORK, "splits")

got = pickle.load(open(CACHE, "rb")); assert got["hash"] == code_hash()
ev = got["ev"]
w = np.array(json.load(open(os.path.join(SPLITS, "ranker_w_fr.json")))["w"])
clean = json.load(open(os.path.join(SPLITS, "clean_eval_queries.json")))
ck = {q["query"] for lst in clean.values() for q in lst}

def paired(qs, seed=0, reps=4000):
    a, b = defaultdict(list), defaultdict(list)
    for q in qs:
        a[q["cls"]].append(rank_score(q, w, use_gen=False))
        b[q["cls"]].append(rank_score(q, w, use_gen=True))
    d = sum(W[c] * (np.mean(b[c]) - np.mean(a[c])) for c in (1, 2, 3) if a[c])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        boots.append(sum(W[c] * np.mean([b[c][i] - a[c][i] for i in rng.integers(0, len(a[c]), len(a[c]))])
                         for c in (1, 2, 3) if a[c]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d, (hi - lo) / 2, lo, hi

print(f"{'set':18s} {'n':>5} {'paired diff':>12} {'half-width':>11}  95% CI")
for label, qs in (("full set", ev), ("CLEAN", [q for q in ev if q["k"] in ck])):
    d, hw, lo, hi = paired(qs)
    print(f"{label:18s} {len(qs):>5} {d:>+12.4f} {hw:>11.4f}  [{lo:+.4f}, {hi:+.4f}]")
    if label == "CLEAN":
        print(f"\n+0.020 threshold is {0.020/hw:.1f}x this half-width "
              f"-> {'DETECTABLE' if 0.020/hw >= 2 else 'NOT RELIABLY DETECTABLE'}")
