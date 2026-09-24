#!/usr/bin/env python3
"""E48 — does ICEBERG's predicted spectrum earn a place in the ranker? Pilot, split 60 only.

The pilot cost 88 minutes for one split, so the remaining two have to be justified before they are
spent. This scores the arms on split 60 alone: small sample, wide intervals, but it gives the sign
and rough magnitude, which is what the go/no-go needs.

THREE ARMS, and the third is the one that matters most:

  baseline        our eight production features (FEATS2), no predictor
  + iceberg       one extra feature: sqrt-cosine between the candidate's PREDICTED spectrum and
                  the observed spectrum
  + iceberg_flat  identical predicted PEAKS, intensities flattened to a constant

The flattened arm is ChatGPT's causal control and it is the whole point of the exercise. If the
full predictions beat the flattened ones, the predicted relative INTENSITIES carry information --
which is exactly the quantity E43 showed our fragment features do not have. If both arms move
together, we have only learned that ICEBERG proposes a better PEAK LIST than our combinatorial
fragmenter, which is a smaller and different claim.

CALIBRATION. ChatGPT asked for weights fitted only on training folds. Predictions exist for one
eval split and generating them for the training splits is another 88 minutes each, so instead the
weights are CROSS-FITTED within split 60: fit on half the queries, score the other half, swap,
pool the out-of-fold reciprocal ranks. No query is ever scored by a model that saw it.

BOOTSTRAP. Paired, and clustered by MOLECULAR IDENTITY rather than by query -- the
pseudo-replication correction ChatGPT flagged and I promised in L-039 and never delivered.

PREDICTIONS (before the run, 2026-09-23):
  P1  +iceberg beats baseline, but the interval does NOT clear zero on one split (n is far too
      small; the paired half-width here should be near 0.03).
  P2  The point estimate is between +0.01 and +0.04 weighted. Below +0.01 I would not spend the
      remaining three hours.
  P3  +iceberg beats +iceberg_flat. If it does not, the gain is a peak-list effect rather than an
      intensity effect, and the headline claim would have to change.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import build, FEATS2, W
from e47c_validate import sparse_cos, TOP_OBS
from casmi_pipeline import key14, TOPN
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
SPLIT = "npxmtsseed60_n300"
PRED = "iceberg_pred_seed60.npz"


def load_pred():
    d = np.load(f"{EXPORTS}/{PRED}")
    byq = defaultdict(dict)
    for k in d.keys():
        q, c, kind = k.split("|")
        byq[q].setdefault(c, {})[kind] = d[k]
    return byq


def fit_w(qs, feats):
    X, y = [], []
    for q in qs:
        tk = [c for c, (_, s) in q["cand"].items() if key14(s) == q["truth"]]
        if not tk: continue
        t = np.array([q["fv"][tk[0]][f] for f in feats])
        for c in q["cand"]:
            if c in tk: continue
            d = t - np.array([q["fv"][c][f] for f in feats]); X += [d, -d]; y += [1, 0]
    if not X: return np.zeros(len(feats))
    X = np.array(X); sd = X.std(0) + 1e-9
    return LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X / sd, y).coef_[0] / sd


def rr(q, w, feats):
    sc = {c: float(np.dot(w, [q["fv"][c][f] for f in feats])) for c in q["cand"]}
    order = [q["cand"][c][1] for c in sorted(sc, key=lambda c: (-sc[c], c))] + q["tail"]
    seen, out = set(), []
    for smi in order:
        kk = key14(smi)
        if kk is not None and kk in seen: continue
        if kk is not None: seen.add(kk)
        out.append(kk)
        if len(out) == TOPN: break
    for i, kk in enumerate(out, 1):
        if q["truth"] is not None and kk == q["truth"]: return 1.0 / i
    return 0.0


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    st = load_structures()
    ev = build([SPLIT], st, log=log, pools_prefix="fr_", generate=False)
    log(f"{len(ev)} queries built")
    pred = load_pred()
    pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{SPLIT}.pkl", "rb"))}

    # attach the two ICEBERG features to every candidate
    have = 0
    for q in ev:
        p = pools[q["k"]]
        obs_mz, obs_it = np.asarray(p["allmz"]), np.asarray(p["allit"])
        if len(obs_mz):
            o = np.argsort(-obs_it)[:TOP_OBS]; obs_mz, obs_it = obs_mz[o], obs_it[o]
        pr = pred.get(q["k"], {})
        q["fv"] = {}
        for c, (vec, smi) in q["cand"].items():
            f = {name: float(v) for name, v in zip(FEATS2, vec)}
            v = pr.get(c)
            if v is not None and "mz" in v and len(v["mz"]) and len(obs_mz):
                f["iceberg"] = sparse_cos(v["mz"], v["it"], obs_mz, obs_it)
                f["iceberg_flat"] = sparse_cos(v["mz"], np.ones_like(v["it"]), obs_mz, obs_it)
                have += 1
            else:
                f["iceberg"] = 0.0; f["iceberg_flat"] = 0.0
            q["fv"][c] = f
    log(f"ICEBERG features attached to {have:,} candidates")

    ARMS = {"baseline": FEATS2,
            "+ iceberg": FEATS2 + ["iceberg"],
            "+ iceberg_flat": FEATS2 + ["iceberg_flat"]}

    # 2-fold cross-fit by MOLECULAR IDENTITY, so a molecule is never scored by a model that saw it
    ids = sorted({q["truth"] or q["k"] for q in ev})
    rng = np.random.default_rng(0); rng.shuffle(ids)
    fold = {i: (n % 2) for n, i in enumerate(ids)}
    per = {a: defaultdict(list) for a in ARMS}
    byid = defaultdict(lambda: {a: [] for a in ARMS})
    for name, feats in ARMS.items():
        for f in (0, 1):
            tr = [q for q in ev if fold[q["truth"] or q["k"]] != f]
            te = [q for q in ev if fold[q["truth"] or q["k"]] == f]
            w = fit_w(tr, feats)
            for q in te:
                s = rr(q, w, feats)
                per[name][q["cls"]].append(s)
                byid[q["truth"] or q["k"]][name].append((q["cls"], s))
        log(f"  {name:16s} weighted "
            f"{sum(W[c]*np.mean(per[name][c]) for c in (1,2,3) if per[name][c]):.4f}")

    def wt(name): return sum(W[c] * float(np.mean(per[name][c])) for c in (1, 2, 3) if per[name][c])

    L = ["E48 — ICEBERG predicted-spectrum agreement as a ranker feature (pilot: split 60 only)", "",
         f"{len(ev)} queries, ICEBERG features on {have:,} candidates", "",
         f"{'arm':18s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9} {'vs baseline':>12}"]
    for name in ARMS:
        c = [float(np.mean(per[name][k])) if per[name][k] else float("nan") for k in (1, 2, 3)]
        L.append(f"{name:18s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt(name):>9.4f} "
                 f"{wt(name)-wt('baseline'):>+12.4f}")

    # paired bootstrap CLUSTERED BY MOLECULAR IDENTITY
    idlist = [i for i in byid]
    rng = np.random.default_rng(1)
    L.append("")
    for name in ARMS:
        if name == "baseline": continue
        boots = []
        for _ in range(3000):
            pick = rng.integers(0, len(idlist), len(idlist))
            a, b = defaultdict(list), defaultdict(list)
            for j in pick:
                rec = byid[idlist[j]]
                for cls, s in rec["baseline"]: a[cls].append(s)
                for cls, s in rec[name]: b[cls].append(s)
            boots.append(sum(W[c] * (np.mean(b[c]) - np.mean(a[c]))
                             for c in (1, 2, 3) if a[c] and b[c]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        d = wt(name) - wt("baseline")
        L.append(f"{name:18s} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                 f"half-width {(hi-lo)/2:.4f}" + ("   <- clears zero" if lo > 0 else ""))
    L += ["", f"intensity effect: (+ iceberg) - (+ iceberg_flat) = "
              f"{wt('+ iceberg') - wt('+ iceberg_flat'):+.4f}",
          "  positive means the predicted relative INTENSITIES carry information beyond the",
          "  predicted peak list -- the quantity E43 showed our fragment features lack.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E48_arms_pilot_2026-09-23.txt", "w").write("\n".join(L) + "\n")
    json.dump({a: dict(weighted=wt(a), **{f"C{c}": float(np.mean(per[a][c])) if per[a][c] else None
                                          for c in (1, 2, 3)}) for a in ARMS},
              open(f"{RESULTS}/E48_arms_pilot.json", "w"), indent=2)


if __name__ == "__main__":
    main()
