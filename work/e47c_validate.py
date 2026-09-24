#!/usr/bin/env python3
"""E47 step 3a — validate the pilot predictions BEFORE spending 4.4 hours on the other two splits.

The pilot cost 88 minutes against my 9-minute estimate (my benchmark was caffeine; real natural
products are 40-80 heavy atoms and the fragment DAG scales badly with size). At that price the
remaining two splits are 3 hours, so the pilot has to earn them.

Three checks, cheapest and most decisive first:

  1. NON-DEGENERACY. Do different candidates get different predicted spectra? This is exactly how
     the panaesthesis field reading died in E42 -- a representation that returns the same thing for
     everything cannot rank. If predicted spectra are near-identical across a pool, stop here.

  2. DOES THE PREDICTION MATCH THE OBSERVATION BETTER FOR THE TRUTH? Cosine between predicted and
     observed spectrum, for the true structure against its pool-mates. This is the whole premise
     in one number.

  3. IS IT NEW INFORMATION? E43 showed our `explain` feature gives the truth only an 18.8% vs
     16.8% edge over its top rival. If the predicted-spectrum cosine ranks the truth above its top
     rival substantially more often than that, it is carrying something our features do not.

PREDICTIONS (before the run, 2026-09-23):
  P1  Predictions are non-degenerate: median pairwise cosine between two candidates' predicted
      spectra is below 0.9.
  P2  The truth's predicted-vs-observed cosine beats the median pool-mate's in 60-75% of queries.
  P3  The truth beats its TOP RIVAL on this cosine in 55-65% of queries -- better than the near
      coin-flip our fragment features manage, but not a solved problem.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
BIN = 0.01          # Da, for comparing predicted against observed peaks
TOP_OBS = 40        # observed peaks compared, same depth our explain feature uses


def sparse_cos(mz1, it1, mz2, it2, bin_=BIN):
    """Cosine between two sparse spectra on a shared mass grid, sqrt-weighted (our ground)."""
    if len(mz1) == 0 or len(mz2) == 0: return 0.0
    b1 = np.rint(np.asarray(mz1) / bin_).astype(np.int64)
    b2 = np.rint(np.asarray(mz2) / bin_).astype(np.int64)
    w1 = np.sqrt(np.asarray(it1, float)); w2 = np.sqrt(np.asarray(it2, float))
    # collapse duplicate bins by max
    def collapse(b, w):
        o = np.lexsort((-w, b)); b, w = b[o], w[o]
        first = np.concatenate(([True], b[1:] != b[:-1]))
        return b[first], w[first]
    b1, w1 = collapse(b1, w1); b2, w2 = collapse(b2, w2)
    n1 = np.linalg.norm(w1); n2 = np.linalg.norm(w2)
    if n1 <= 0 or n2 <= 0: return 0.0
    common, i1, i2 = np.intersect1d(b1, b2, return_indices=True)
    return float(np.dot(w1[i1], w2[i2]) / (n1 * n2))


def main(split="npxmtsseed60_n300", pred="iceberg_pred_seed60.npz"):
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    d = np.load(f"{EXPORTS}/{pred}")
    keys = list(d.keys())
    byq = defaultdict(dict)
    for k in keys:
        q, c, kind = k.split("|")
        byq[q][c] = byq[q].get(c, {})
        byq[q][c][kind] = d[k]
    log(f"{len(byq)} queries with predictions, {sum(len(v) for v in byq.values()):,} candidates")

    pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{split}.pkl", "rb"))}
    sp = json.load(open(f"{SPLITS}/split_{split}.json"))

    # 1. non-degeneracy
    degen = []
    rng = np.random.default_rng(0)
    for q, cands in byq.items():
        cl = [c for c in cands if "mz" in cands[c] and len(cands[c]["mz"])]
        if len(cl) < 2: continue
        pick = rng.choice(len(cl), size=min(6, len(cl)), replace=False)
        for i in range(len(pick)):
            for j in range(i + 1, len(pick)):
                a, b = cands[cl[pick[i]]], cands[cl[pick[j]]]
                degen.append(sparse_cos(a["mz"], a["it"], b["mz"], b["it"]))
    degen = np.array(degen)
    log(f"non-degeneracy: {len(degen):,} candidate pairs, median cosine {np.median(degen):.4f}")

    # 2 & 3. predicted-vs-observed cosine, truth against pool-mates
    rows = []
    for q, cands in byq.items():
        p = pools.get(q);
        if p is None or p["truth"] is None: continue
        obs_mz, obs_it = np.asarray(p["allmz"]), np.asarray(p["allit"])
        if len(obs_mz) == 0: continue
        o = np.argsort(-obs_it)[:TOP_OBS]
        obs_mz, obs_it = obs_mz[o], obs_it[o]
        scores = {}
        for c, v in cands.items():
            if "mz" not in v or not len(v["mz"]): continue
            scores[c] = sparse_cos(v["mz"], v["it"], obs_mz, obs_it)
        if len(scores) < 2: continue
        truth_c = [c for c in scores if scoring.key14(p["mass"].get(c, "")) == p["truth"]]
        if not truth_c: continue
        tc = max(truth_c, key=lambda c: scores[c])
        others = {c: s for c, s in scores.items() if c not in truth_c}
        if not others: continue
        top_rival = max(others, key=lambda c: others[c])
        rows.append(dict(query=q, cls=sp["assign"][q], n_cands=len(scores),
                         truth_cos=scores[tc], median_other=float(np.median(list(others.values()))),
                         top_rival_cos=others[top_rival],
                         beats_median=scores[tc] > float(np.median(list(others.values()))),
                         beats_top_rival=scores[tc] > others[top_rival],
                         rank=1 + sum(1 for s in others.values() if s > scores[tc])))
    log(f"{len(rows)} queries with the truth in the predicted pool")

    L = ["E47c — do ICEBERG's predicted spectra discriminate the true structure?", "",
         f"predictions: {len(byq)} queries, {sum(len(v) for v in byq.values()):,} candidates", "",
         f"1. NON-DEGENERACY  median pairwise cosine between candidates' predicted spectra: "
         f"{np.median(degen):.4f}",
         f"   (share of pairs above 0.9: {np.mean(degen > 0.9):.1%}; above 0.99: "
         f"{np.mean(degen > 0.99):.1%})", ""]
    if rows:
        tc = np.array([r["truth_cos"] for r in rows])
        mo = np.array([r["median_other"] for r in rows])
        tr = np.array([r["top_rival_cos"] for r in rows])
        rk = np.array([r["rank"] for r in rows])
        L += [f"2. PREDICTED-vs-OBSERVED COSINE, n={len(rows)} queries with the truth in pool",
              f"   truth            median {np.median(tc):.4f}  mean {tc.mean():.4f}",
              f"   median pool-mate median {np.median(mo):.4f}  mean {mo.mean():.4f}",
              f"   top rival        median {np.median(tr):.4f}  mean {tr.mean():.4f}",
              f"   truth beats the median pool-mate: {np.mean(tc > mo):.1%}",
              f"   truth beats its TOP RIVAL:        {np.mean(tc > tr):.1%}",
              "",
              f"3. RANK OF THE TRUTH by predicted-spectrum cosine alone",
              f"   rank 1: {np.mean(rk == 1):.1%}   within top 3: {np.mean(rk <= 3):.1%}   "
              f"median rank {np.median(rk):.0f}",
              "",
              "   for reference, E43: our fragment `explain` feature puts the truth ahead of its top",
              "   rival by 18.8% to 16.8% -- a two-point edge. Anything much above 50% here is new",
              "   information, and that is the whole question."]
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E47c_validate_2026-09-23.txt", "w").write("\n".join(L) + "\n")
    json.dump(rows, open(f"{RESULTS}/E47c_validate.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
