#!/usr/bin/env python3
"""E50 — the corrected arms: rerank ONLY within the baseline top 25, with a coverage guard.

E48 reported +0.0866 and it was an artifact: predictions came from the pool file's candidate set
while scoring used the build's, so 94% of truths had a prediction against 5.8% of decoys and the
ranker learned the 16x indicator rather than any chemistry. This fixes the design and, more
importantly, makes that failure impossible to reproduce quietly:

  - candidates are the SCORER'S OWN top 25 by the baseline ranker, exported from the same objects
  - every candidate under comparison therefore has a prediction
  - a COVERAGE GUARD asserts the truth-vs-decoy prediction rate differs by under 5 points, and
    refuses to print a result otherwise

Reranking happens strictly inside those 25; candidates 26+ keep their baseline order, then the
spectral tail. That is also the only operation that can change MRR@25 without changing which
structures we submit.

Class 3 queries carry no predictions (generation is off, so none of their candidates can be the
truth). They are kept in the denominator with a uniform zero feature, so they contribute
identically to every arm and cancel in the paired difference -- rather than being dropped, which
would flatter the weighted score.

ARMS: baseline | + iceberg | + iceberg_flat (same peaks, intensities levelled -- the causal
control that exposed the artifact, and the only thing that distinguishes "ICEBERG predicts better
PEAKS than our fragmenter" from "predicted INTENSITIES carry information").

PREDICTIONS (before the run, 2026-09-23):
  P1  The corrected gain is far smaller than the artifact's +0.0866 -- I expect between 0.00 and
      +0.03, and I would not be surprised by a null.
  P2  The interval does not clear zero on one split (170 scored queries; the half-width should be
      near 0.03).
  P3  + iceberg and + iceberg_flat stay within 0.01 of each other, since the pilot already showed
      flattening does not hurt. If so, the honest claim concerns the peak list, not intensities.
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
WORKLIST = f"{EXPORTS}/iceberg_top25_d60_n300_c12.json"
PRED = f"{EXPORTS}/iceberg_pred25_seed60.npz"
GUARD_TOL = 0.05


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    d = np.load(PRED)
    pr = defaultdict(dict)
    for k in d.keys():
        q, c, kind = k.split("|")
        pr[q].setdefault(c, {})[kind] = d[k]
    wl = {q["key"]: q for q in json.load(open(WORKLIST))["queries"]}
    log(f"{len(pr)} queries with predictions, {len(wl)} in the work list")

    st = load_structures()
    ev = build([SPLIT], st, log=log, pools_prefix="fr_", generate=False)
    pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{SPLIT}.pkl", "rb"))}
    base_w = np.array(json.load(open(os.path.join(SPLITS, "ranker_w_fr.json")))["w"])

    # ---- attach features, restricted to the exported top 25 --------------------------------
    tr_hit = tr_tot = de_hit = de_tot = 0
    for q in ev:
        p = pools[q["k"]]
        obs_mz, obs_it = np.asarray(p["allmz"]), np.asarray(p["allit"])
        if len(obs_mz):
            o = np.argsort(-obs_it)[:TOP_OBS]; obs_mz, obs_it = obs_mz[o], obs_it[o]
        sc = {c: float(np.dot(base_w, v)) for c, (v, _) in q["cand"].items()}
        order = sorted(sc, key=lambda c: (-sc[c], str(c)))
        q["top25"] = order[:TOPN]
        q["rest"] = order[TOPN:]
        q["fv"] = {}
        got = pr.get(q["k"], {})
        in_worklist = q["k"] in wl
        for c in q["cand"]:
            f = {n: float(v) for n, v in zip(FEATS2, q["cand"][c][0])}
            v = got.get(str(c))
            # bool() is load-bearing: `a and b and len(x)` returns len(x), not True, so
            # `tr_hit += ok` was adding up to 40 per candidate and the guard reported 3162%.
            ok = bool(v is not None and "mz" in v and len(v["mz"]) > 0 and len(obs_mz) > 0)
            if ok:
                f["iceberg"] = sparse_cos(v["mz"], v["it"], obs_mz, obs_it)
                f["iceberg_flat"] = sparse_cos(v["mz"], np.ones_like(v["it"]), obs_mz, obs_it)
            else:
                f["iceberg"] = 0.0; f["iceberg_flat"] = 0.0
            q["fv"][c] = f
            if in_worklist and c in q["top25"]:
                istruth = q["truth"] is not None and key14(q["cand"][c][1]) == q["truth"]
                if istruth: tr_tot += 1; tr_hit += ok
                else:       de_tot += 1; de_hit += ok

    tr_rate = tr_hit / max(tr_tot, 1); de_rate = de_hit / max(de_tot, 1)
    log(f"COVERAGE GUARD  truth {tr_rate:.1%} ({tr_hit}/{tr_tot})  "
        f"decoy {de_rate:.1%} ({de_hit}/{de_tot})  gap {abs(tr_rate-de_rate):.1%}")
    if abs(tr_rate - de_rate) > GUARD_TOL:
        print(f"\nREFUSING TO SCORE: prediction coverage differs between truths and decoys by "
              f"{abs(tr_rate-de_rate):.1%} (> {GUARD_TOL:.0%}). That asymmetry is what made E48 an "
              f"artifact -- the feature would encode pool membership, not chemistry.")
        sys.exit(1)

    # ---- arms -------------------------------------------------------------------------------
    ARMS = {"baseline": FEATS2, "+ iceberg": FEATS2 + ["iceberg"],
            "+ iceberg_flat": FEATS2 + ["iceberg_flat"]}

    def fit_w(qs, feats):
        X, y = [], []
        for q in qs:
            tk = [c for c in q["top25"] if key14(q["cand"][c][1]) == q["truth"]]
            if not tk: continue
            t = np.array([q["fv"][tk[0]][f] for f in feats])
            for c in q["top25"]:
                if c in tk: continue
                dd = t - np.array([q["fv"][c][f] for f in feats]); X += [dd, -dd]; y += [1, 0]
        if not X: return np.zeros(len(feats))
        X = np.array(X); sd = X.std(0) + 1e-9
        return LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000
                                  ).fit(X / sd, y).coef_[0] / sd

    def rr(q, w, feats):
        s = {c: float(np.dot(w, [q["fv"][c][f] for f in feats])) for c in q["top25"]}
        order = ([q["cand"][c][1] for c in sorted(s, key=lambda c: (-s[c], str(c)))]
                 + [q["cand"][c][1] for c in q["rest"]] + q["tail"])
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

    ids = sorted({q["truth"] or q["k"] for q in ev})
    rng = np.random.default_rng(0); rng.shuffle(ids)
    fold = {i: n % 2 for n, i in enumerate(ids)}
    per = {a: defaultdict(list) for a in ARMS}
    byid = defaultdict(lambda: defaultdict(list))
    for name, feats in ARMS.items():
        for f in (0, 1):
            trq = [q for q in ev if fold[q["truth"] or q["k"]] != f]
            teq = [q for q in ev if fold[q["truth"] or q["k"]] == f]
            w = fit_w(trq, feats)
            for q in teq:
                s = rr(q, w, feats)
                per[name][q["cls"]].append(s)
                byid[q["truth"] or q["k"]][name].append((q["cls"], s))

    def wt(n): return sum(W[c] * float(np.mean(per[n][c])) for c in (1, 2, 3) if per[n][c])

    L = ["E50 — ICEBERG reranking within the baseline top 25 (corrected; split 60)", "",
         f"coverage guard: truth {tr_rate:.1%}, decoy {de_rate:.1%}, gap "
         f"{abs(tr_rate-de_rate):.1%} -- PASSED", "",
         f"{'arm':18s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9} {'vs baseline':>12}"]
    for n in ARMS:
        c = [float(np.mean(per[n][k])) if per[n][k] else float('nan') for k in (1, 2, 3)]
        L.append(f"{n:18s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt(n):>9.4f} "
                 f"{wt(n)-wt('baseline'):>+12.4f}")
    idl = list(byid); rng = np.random.default_rng(1)
    L.append("")
    for n in ARMS:
        if n == "baseline": continue
        boots = []
        for _ in range(3000):
            pick = rng.integers(0, len(idl), len(idl))
            a, b = defaultdict(list), defaultdict(list)
            for j in pick:
                for cls, s in byid[idl[j]]["baseline"]: a[cls].append(s)
                for cls, s in byid[idl[j]][n]: b[cls].append(s)
            boots.append(sum(W[c] * (np.mean(b[c]) - np.mean(a[c]))
                             for c in (1, 2, 3) if a[c] and b[c]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        L.append(f"{n:18s} {wt(n)-wt('baseline'):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                 f"half-width {(hi-lo)/2:.4f}" + ("   <- clears zero" if lo > 0 else ""))
    L += ["", f"intensity effect: (+ iceberg) - (+ iceberg_flat) = "
              f"{wt('+ iceberg')-wt('+ iceberg_flat'):+.4f}",
          f"oracle for perfect reranking within 25 (E49): +0.1479",
          f"ChatGPT's continuation threshold: +0.020",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E50_arms_top25_2026-09-23.txt", "w").write("\n".join(L) + "\n")
    json.dump({a: dict(weighted=wt(a)) for a in ARMS},
              open(f"{RESULTS}/E50_arms_top25.json", "w"), indent=2)


if __name__ == "__main__":
    main()
