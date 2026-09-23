#!/usr/bin/env python3
"""E46 — the baseline on the 370 uncontaminated queries, established BEFORE the experiment.

E45 left 370 evaluation queries whose true structure is outside the open ICEBERG weights'
training fold. That is the only set on which an intensity-prediction gain can be read honestly.
But a gain is a difference, and a difference needs a reference: if I measure the ICEBERG arm on
the clean subset and compare it against 0.3718 -- which was measured on all 900 -- I would be
reading a change in the query set as a change in method. That is the same error as E36, where an
experiment measured its own construction, and I would rather not make it a third time.

So this fixes the reference first, with no new method in it at all.

WHAT IT ALSO SETTLES. I claimed the clean subset is adequate to detect the +0.020 continuation
threshold, from a square-root scaling argument: 900 -> 370 should widen intervals about 1.56x,
giving a half-width near 0.006 against a threshold of 0.020. That was arithmetic, not a
measurement. This bootstraps the clean subset directly and reports the actual half-width, so the
decision rule rests on a measured resolution rather than my estimate of one.

PREDICTIONS (before the run, 2026-09-23):
  P1  The clean subset scores LOWER than the full set. Its molecules are the ones nobody has
      published a spectrum for -- less-studied chemistry, sparser neighbourhoods, weaker
      neighbour votes. I would guess 0.32-0.36 against 0.3718.
  P2  The drop is concentrated in Class 1 and Class 2, where the neighbour vote does the work.
      Class 3 is near zero on both and has little room to fall.
  P3  The measured bootstrap half-width lands between 0.005 and 0.009, confirming that +0.020
      remains detectable on this subset.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e22_ranker_v2 import fit, rank_score, W
from e39_gate import CACHE, code_hash

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
WCACHE = os.path.join(SPLITS, "ranker_w_fr.json")


def ranker_weights(tr, log):
    want = code_hash()
    if os.path.exists(WCACHE):
        d = json.load(open(WCACHE))
        if d.get("hash") == want:
            log("ranker weights from cache")
            return np.array(d["w"])
    w, npairs = fit(tr)
    json.dump(dict(hash=want, w=list(map(float, w))), open(WCACHE, "w"))
    log(f"ranker refit on {npairs:,} pairs -> cached")
    return w


def score_set(queries, w, use_gen):
    per = defaultdict(list)
    for q in queries:
        per[q["cls"]].append(rank_score(q, w, use_gen=use_gen))
    wt = sum(W[c] * float(np.mean(per[c])) for c in (1, 2, 3) if per[c])
    return wt, per


def halfwidth(per, reps=4000, seed=0):
    """Bootstrap half-width of the weighted score, resampling queries within each class."""
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        boots.append(sum(W[c] * np.mean([per[c][i] for i in rng.integers(0, len(per[c]), len(per[c]))])
                         for c in (1, 2, 3) if per[c]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (hi - lo) / 2, lo, hi


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    got = pickle.load(open(CACHE, "rb"))
    assert got["hash"] == code_hash(), "cache stale -- rebuild it before trusting this"
    ev, tr = got["ev"], got["tr"]
    w = ranker_weights(tr, log)

    clean = json.load(open(os.path.join(SPLITS, "clean_eval_queries.json")))
    clean_keys = {q["query"] for lst in clean.values() for q in lst}
    log(f"{len(clean_keys)} clean query keys")
    ev_clean = [q for q in ev if q["k"] in clean_keys]
    ev_dirty = [q for q in ev if q["k"] not in clean_keys]
    log(f"{len(ev_clean)} clean / {len(ev_dirty)} contaminated of {len(ev)} eval queries")

    L = ["E46 — baseline on the uncontaminated subset, fixed before the intensity experiment", ""]
    rows = {}
    for label, qs in (("full set (900)", ev), ("CLEAN (370)", ev_clean),
                      ("contaminated (530)", ev_dirty)):
        wt, per = score_set(qs, w, use_gen=False)
        hw, lo, hi = halfwidth(per)
        rows[label] = dict(weighted=wt, halfwidth=hw,
                           **{f"C{c}": (float(np.mean(per[c])) if per[c] else float("nan"))
                              for c in (1, 2, 3)},
                           **{f"n{c}": len(per[c]) for c in (1, 2, 3)})
        L.append(f"{label:20s} n={len(qs):>4}  C1 {rows[label]['C1']:.4f}  C2 {rows[label]['C2']:.4f}  "
                 f"C3 {rows[label]['C3']:.4f}  weighted {wt:.4f}  +/- {hw:.4f}")
    L += ["", f"class counts on the clean subset: "
              f"C1 {rows['CLEAN (370)']['n1']}, C2 {rows['CLEAN (370)']['n2']}, "
              f"C3 {rows['CLEAN (370)']['n3']}",
          f"clean minus full: {rows['CLEAN (370)']['weighted'] - rows['full set (900)']['weighted']:+.4f}",
          "",
          f"measured bootstrap half-width on the clean subset: "
          f"{rows['CLEAN (370)']['halfwidth']:.4f}",
          f"  against the +0.020 continuation threshold that is "
          f"{0.020 / max(1e-9, rows['CLEAN (370)']['halfwidth']):.1f}x the half-width",
          f"  (full set for comparison: {rows['full set (900)']['halfwidth']:.4f})",
          "",
          "reading: THIS is the number any ICEBERG arm must be compared against -- not the 0.3718",
          "measured on all 900. Comparing a clean-subset arm to a full-set baseline would read a",
          "change of query set as a change of method, which is the E36 mistake in a new costume.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E46_clean_baseline_2026-09-23.txt", "w").write("\n".join(L) + "\n")
    json.dump(rows, open(f"{RESULTS}/E46_clean_baseline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
