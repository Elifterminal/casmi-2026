#!/usr/bin/env python3
"""E34 — score on a set that matches the test's INSTRUMENT, not just its chemistry.

Vanta's review flagged train/test instrument shift. Measured, it is stark:

    the real test set   100% Bruker timsTOF (1,213 of 1,213 spectra)
    our eval queries    0.2% timsTOF (12 of 5,291) -- overwhelmingly Orbitrap and QTOF

Only two libraries are timsTOF: enveda-180 (right instrument, synthetic chemistry) and
enveda-np-examples (right instrument AND natural products, 250 structures). Our NP-only filter
was right about chemistry and pushed us almost entirely onto the wrong instruments.

So this evaluates the shipped pipeline on the only set that matches the test on both axes:
queries drawn from enveda-np-examples, twin-safe, three seeds of 240. Everything else is
identical to E25 -- sqrt-p ground, generation off, ranker trained on the usual NP splits.

Those 250 structures are exactly the set the field validates on and gets burned by (E01: all 250
appear in other libraries, median 168 spectra elsewhere). Our twin-safe holdout removes every
spectrum of the whole scorer-identity group, so a Class 2 query here is genuinely spectrum-free.

PREDICTIONS (before the run, 2026-09-21):
  P1  The weighted score is LOWER than 0.3682. If instrument shift matters at all, a query
      recorded on the instrument our references mostly are not should be harder.
  P2  The drop is biggest in Class 1, which is the class that depends on matching a reference
      spectrum of the same molecule -- recorded, here, on a different machine.
  P3  It lands between 0.28 and 0.35, i.e. closer to our leaderboard 0.288 than our local 0.368.
      That would say instrument shift is a real part of the gap we have never accounted for.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import build, fit, rank_score, FEATS2, TRAIN, W

RESULTS = os.path.expanduser("~/casmi-2026/work/results")
TIMS = ("nptstimsseed30_n240", "nptstimsseed31_n240", "nptstimsseed32_n240")


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    st = load_structures()
    ev = build(TIMS, st, log=log, pools_prefix="fr_", generate=False)
    tr = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log, pools_prefix="fr_", generate=False)
    w, n = fit(tr)
    log(f"trained on {n:,} pairs")
    per = defaultdict(list); bysplit = defaultdict(lambda: defaultdict(list))
    for q in ev:
        r = rank_score(q, w, use_gen=False)
        per[q["cls"]].append(r); bysplit[q["split"] if "split" in q else "all"][q["cls"]].append(r)
    c = [float(np.mean(per[k])) for k in (1, 2, 3)]
    wt = sum(W[k] * c[k-1] for k in (1, 2, 3))
    L = ["E34 — timsTOF natural-product queries: the only set matching the test on both axes", "",
         f"queries: {sum(len(per[k]) for k in per)}   (C1 {len(per[1])}, C2 {len(per[2])}, C3 {len(per[3])})", "",
         f"{'set':34s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}",
         f"{'timsTOF NP (test-faithful)':34s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt:>9.4f}",
         f"{'mixed-instrument NP (our usual)':34s} {0.8245:>8.4f} {0.5250:>8.4f} {0.0000:>8.4f} {0.3682:>9.4f}",
         "", f"difference: {wt-0.3682:+.4f} weighted",
         f"leaderboard for this pipeline: 0.288", f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E34_timstof_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump(dict(C1=c[0], C2=c[1], C3=c[2], weighted=wt, n=len(ev),
                   weights=dict(zip(FEATS2, map(float, w)))),
              open(f"{RESULTS}/E34_timstof.json", "w"), indent=2)


if __name__ == "__main__":
    main()
