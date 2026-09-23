#!/usr/bin/env python3
"""E40 — is the gate's failure my classifier, my label, or the axis itself?

E39's gate captured 5% of the prize and fired on 73.8% of the Class 2 queries it exists to
refuse. Its AUC was 0.7226, which sounds respectable and is not: Class 1 is 2.1% positive,
Class 2 is 6.2%, Class 3 is 100%, so an AUC computed over all three is dominated by the easy
Class 1 axis and says almost nothing about the axis that pays. Three explanations, and they call
for three different responses:

  the classifier   logistic regression on eight summary features is too blunt -> try better
  the label        "truth absent from the retrieval pool" is not what generation's value tracks
                   -> a perfect predictor of it would not be worth much either
  the axis         Class 2 and Class 3 are not separable by anything we can compute -> stop

This separates them, and it is cheap because E39 cached the build.

  1. AUC decomposed by axis: C1-vs-rest (easy, worthless) against C2-vs-C3 (hard, the whole game).
  2. Single-feature AUC on C2-vs-C3, so a real signal in one feature cannot hide inside a fitted
     combination -- and so that "these features carry nothing" is measured, not asserted.
  3. The LABEL-ORACLE arm, which E39 never ran. E39's oracle fires wherever generation happens to
     help, which includes queries where it helps by luck. A gate that fires exactly when the truth
     is absent is the real ceiling for the label I chose. If that ceiling is far below E39's
     +0.0190 outcome-oracle, my label was wrong. If it is close, the label was right.

PREDICTIONS (before the run, 2026-09-22):
  P1  C1-vs-rest AUC above 0.85; C2-vs-C3 AUC below 0.60. The headline 0.7226 is the easy axis.
  P2  No single feature exceeds 0.60 on C2-vs-C3. The information is not hiding in one place.
  P3  The label-oracle arm lands within 0.002 of the outcome-oracle's 0.3904, confirming the
      label was the right target and the classifier is what failed.
  P4  Taking P1-P3 together the honest conclusion will be "the axis", not "the classifier":
      this is the same discrimination failure E38 found, measured from the other side. Our
      scoring cannot separate the true structure from same-mass look-alikes, so it cannot tell
      when the true structure is missing either.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e22_ranker_v2 import fit, rank_score, W
from e39_gate import CACHE, GFEATS, code_hash, gate_features, truth_absent

RESULTS = os.path.expanduser("~/casmi-2026/work/results")
WCACHE = os.path.expanduser("~/casmi-2026/work/splits/ranker_w_fr.json")


def ranker_weights(tr, log):
    """Refit the ranker on the cached training build, or reuse the cached fit (990s saved)."""
    want = code_hash()
    if os.path.exists(WCACHE):
        d = json.load(open(WCACHE))
        if d.get("hash") == want:
            log("ranker weights from cache")
            return np.array(d["w"])
    w, npairs = fit(tr)
    json.dump(dict(hash=want, w=list(map(float, w))), open(WCACHE, "w"))
    log(f"ranker fit on {npairs:,} pairs -> cached")
    return w


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    got = pickle.load(open(CACHE, "rb"))
    assert got["hash"] == code_hash(), "cache is stale for the current code -- rerun E39"
    ev, tr = got["ev"], got["tr"]
    log(f"cache loaded: {len(ev)} eval, {len(tr)} train queries")
    w = ranker_weights(tr, log)

    X = np.array([gate_features(q, w) for q in ev])
    y = np.array([truth_absent(q) for q in ev])
    cls = np.array([q["cls"] for q in ev])
    log("gate features built")

    L = ["E40 — where the gate's failure actually lives", "",
         f"{'axis':22s} {'n':>6} {'positives':>10} {'AUC':>8}"]
    axes = {"all three classes": np.ones(len(y), bool),
            "C1 vs (C2,C3)": np.ones(len(y), bool),
            "C2 vs C3 (the axis that pays)": (cls == 2) | (cls == 3),
            "C1 vs C3": (cls == 1) | (cls == 3)}
    # the gate's own score, refit on train so this is the same object E39 evaluated
    from sklearn.linear_model import LogisticRegression
    Xtr = np.array([gate_features(q, w) for q in tr]); ytr = np.array([truth_absent(q) for q in tr])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    g = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    p = g.predict_proba((X - mu) / sd)[:, 1]

    out = {}
    for name, m in axes.items():
        yy = (cls[m] == 3).astype(int) if "vs C3" in name else y[m]
        if name == "C1 vs (C2,C3)": yy = (cls[m] != 1).astype(int)
        if len(set(yy)) < 2: continue
        a = roc_auc_score(yy, p[m])
        out[name] = float(a)
        L.append(f"{name:22s} {int(m.sum()):>6} {yy.mean():>9.1%} {a:>8.4f}")

    L += ["", "single features on C2 vs C3 -- can any one of them see the axis at all?",
          f"{'feature':16s} {'AUC':>8}  (0.500 is chance; <0.500 means it points the other way)"]
    m = (cls == 2) | (cls == 3); yy = (cls[m] == 3).astype(int)
    singles = {}
    for j, f in enumerate(GFEATS):
        a = roc_auc_score(yy, X[m, j])
        singles[f] = float(a)
        L.append(f"{f:16s} {a:>8.4f}")

    # the arm E39 never ran: fire exactly where the truth is absent
    per = defaultdict(lambda: defaultdict(list))
    for q in ev:
        lab = truth_absent(q)
        per["label oracle"][q["cls"]].append(rank_score(q, w, use_gen=bool(lab)))
        per["outcome oracle"][q["cls"]].append(max(rank_score(q, w, use_gen=False),
                                                   rank_score(q, w, use_gen=True)))
        per["retrieval only"][q["cls"]].append(rank_score(q, w, use_gen=False))
        per["always on"][q["cls"]].append(rank_score(q, w, use_gen=True))
    wt = {k: sum(W[c] * float(np.mean(v[c])) for c in (1, 2, 3)) for k, v in per.items()}
    L += ["", "what a PERFECT gate is worth, by what it is perfect at:",
          f"{'arm':18s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}"]
    for k in ("retrieval only", "always on", "label oracle", "outcome oracle"):
        v = per[k]
        L.append(f"{k:18s} {np.mean(v[1]):>8.4f} {np.mean(v[2]):>8.4f} {np.mean(v[3]):>8.4f} "
                 f"{wt[k]:>9.4f}")
    L += ["",
          f"label oracle - outcome oracle: {wt['label oracle']-wt['outcome oracle']:+.4f}",
          "If those two are close, 'truth absent from the pool' was the right thing to predict and",
          "the classifier is what failed. If the label oracle is far below, the label was wrong.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E40_gatediag_2026-09-22.txt", "w").write("\n".join(L) + "\n")
    json.dump(dict(axes=out, singles=singles, arms=wt), open(f"{RESULTS}/E40_gatediag.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
