#!/usr/bin/env python3
"""E43 — is the evidence degenerate, or is our scoring compressing it? ChatGPT's diagnostic.

I have been claiming the discrimination is information-limited: same precursor mass by
construction, therefore nothing to separate the truth from its look-alikes, therefore rank 2 is
near the floor. ChatGPT's correction is that none of that is established, and he is right:

  "The oracle gap identifies where your pipeline loses points. It does not establish how much of
   that gap a real method could recover. Likewise, median rank 2 is not an estimate of the best
   achievable rank."

And the technical half I had wrong: same precursor mass within 10 ppm does NOT imply the same
fragment list, or even the same molecular formula. Same-formula isomers can still fragment
distinguishably. So there are two different worlds and I have never checked which one we are in:

  DEGENERATE   the truth and its top rival present the SAME evidence under our representation.
               No scoring function of that representation can separate them. Intensity prediction
               or nothing.
  COMPRESSED   their evidence differs and our scoring flattens the difference into a near-tie.
               That is our failure, and it is fixable without importing anything.

His point that near-ties in the final score do NOT establish degeneracy is the whole reason this
needs measuring rather than assuming.

THREE LEVELS, coarse to sharp:
  1. fragment-set collision     the two candidates' quantised fragment mass sets are identical
  2. EXPLANATORY collision      the sets of observed peaks each one explains are identical --
                                the sharper test, because `explain` is the feature that carries
                                our ranking, so this is the evidence our pipeline actually sees
  3. score near-tie             reported for contrast, NOT as evidence of degeneracy

PREDICTIONS (before the run, 2026-09-23):
  P1  Exact fragment-set collisions are rare, under 10%. Our own E20 mass work says different
      structures usually differ somewhere in a 200-fragment list.
  P2  EXPLANATORY collisions are much more common, 30-60%, because the observed spectrum has
      maybe 20-40 peaks worth explaining and many candidates cover the same ones.
  P3  Score near-ties are more common still, and the gap between (2) and (3) is the part our
      scoring throws away by choice rather than by necessity.
  If P2 is high the evidence really is degenerate under this representation and ChatGPT's
  intensity route is the only door. If P2 is low, our scoring is the problem and I have been
  blaming the task for a failure of method.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e39_gate import CACHE, code_hash
from casmi_pipeline import characterise, key14, TOL
import casmi_pipeline as cp

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
NEAR_TIE = 0.02          # score gap counted as a near-tie, for contrast only
QUANT = TOL              # fragment masses agreeing within our own matching tolerance


def explained(peaks_mz, peaks_it, frags, positive, topn=40):
    """Which of the query's strongest peaks this candidate's fragments can account for.

    Deliberately the same arithmetic as the production `explain` feature -- the point is to see
    the evidence OUR PIPELINE sees, not a better version of it.
    """
    if frags is None or len(frags) == 0 or len(peaks_mz) == 0:
        return frozenset()
    o = np.argsort(-np.asarray(peaks_it))[:topn]
    mz = np.asarray(peaks_mz, np.float64)[o]
    shift = cp.PROTON if positive else -cp.PROTON
    f = np.array(sorted(frags), np.float64)   # 'full' is a set, not an array
    out = []
    for i, m in enumerate(mz):
        target = m - shift
        j = np.searchsorted(f, target)
        for jj in (j - 1, j):
            if 0 <= jj < len(f) and abs(f[jj] - target) <= TOL:
                out.append(i); break
    return frozenset(out)


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    got = pickle.load(open(CACHE, "rb"))
    assert got["hash"] == code_hash(), "cache stale -- rerun E39"
    ev = got["ev"]
    w = np.array(json.load(open(os.path.join(SPLITS, "ranker_w_fr.json")))["w"])
    pools = {}
    for split in ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300"):
        for p in pickle.load(open(f"{SPLITS}/pools_fr_{split}.pkl", "rb")):
            pools[p["k"]] = p
    log(f"{len(ev)} eval queries, {len(pools)} pools")

    # for each query: the truth, and the top-ranked candidate that is NOT the truth
    pairs, need = [], set()
    for q in ev:
        if q["truth"] is None or q["k"] not in pools: continue
        sc = {c: float(np.dot(w, v)) for c, (v, _) in q["cand"].items()}
        smi = {c: s for c, (_, s) in q["cand"].items()}
        truth_c = [c for c in sc if key14(smi[c]) == q["truth"]]
        if not truth_c: continue                      # Class 3: nothing to collide with
        tc = max(truth_c, key=lambda c: sc[c])
        rivals = [c for c in sc if key14(smi[c]) != q["truth"]]
        if not rivals: continue
        rc = max(rivals, key=lambda c: sc[c])
        pairs.append((q, tc, rc, sc[tc], sc[rc]))
        need.add(smi[tc]); need.add(smi[rc])
    log(f"{len(pairs)} queries with truth and at least one rival in the pool")
    info = characterise(sorted(need), procs=7)
    log(f"{len(info):,} structures characterised")

    rows, per = [], defaultdict(lambda: defaultdict(int))
    for q, tc, rc, st, sr in pairs:
        p = pools[q["k"]]
        ts, rs = q["cand"][tc][1], q["cand"][rc][1]
        ti, ri = info.get(ts), info.get(rs)
        tf = ti["full"] if ti else None
        rf = ri["full"] if ri else None
        # level 1: quantised fragment sets
        qs = lambda f: frozenset(np.rint(np.asarray(sorted(f), np.float64) / QUANT).astype(np.int64)) \
            if f is not None and len(f) else frozenset()
        frag_same = qs(tf) == qs(rf) and len(qs(tf)) > 0
        # level 2: which observed peaks each explains
        et = explained(p["allmz"], p["allit"], tf, p["positive"])
        er = explained(p["allmz"], p["allit"], rf, p["positive"])
        expl_same = (et == er) and len(et) > 0
        expl_empty = (len(et) == 0 and len(er) == 0)
        # level 3: score near-tie (contrast only)
        tie = abs(st - sr) <= NEAR_TIE
        c = q["cls"]
        per[c]["n"] += 1
        per[c]["frag_same"] += frag_same
        per[c]["expl_same"] += expl_same
        per[c]["expl_empty"] += expl_empty
        per[c]["tie"] += tie
        per[c]["truth_ahead"] += (st > sr)
        rows.append(dict(query=q["k"], cls=c, frag_collision=bool(frag_same),
                         explanatory_collision=bool(expl_same), both_explain_nothing=bool(expl_empty),
                         score_near_tie=bool(tie), truth_ahead=bool(st > sr),
                         score_gap=round(float(st - sr), 5),
                         n_truth_frag=int(len(qs(tf))), n_rival_frag=int(len(qs(rf))),
                         n_peaks_truth=int(len(et)), n_peaks_rival=int(len(er)),
                         jaccard=round(len(et & er) / max(1, len(et | er)), 4)))

    L = ["E43 — truth versus its top rival: is the evidence identical, or do we flatten it?", "",
         f"{'class':6s} {'pairs':>7} {'frag collision':>15} {'EXPL collision':>15} "
         f"{'both explain 0':>15} {'score near-tie':>15} {'truth ahead':>12}"]
    for c in (1, 2):
        v = per[c]; n = max(1, v["n"])
        L.append(f"C{c:<5d} {v['n']:>7} {v['frag_same']/n:>14.1%} {v['expl_same']/n:>14.1%} "
                 f"{v['expl_empty']/n:>14.1%} {v['tie']/n:>14.1%} {v['truth_ahead']/n:>11.1%}")
    tot = {k: sum(per[c][k] for c in (1, 2)) for k in ("n", "frag_same", "expl_same", "expl_empty",
                                                       "tie", "truth_ahead")}
    n = max(1, tot["n"])
    L.append(f"{'all':6s} {tot['n']:>7} {tot['frag_same']/n:>14.1%} {tot['expl_same']/n:>14.1%} "
             f"{tot['expl_empty']/n:>14.1%} {tot['tie']/n:>14.1%} {tot['truth_ahead']/n:>11.1%}")
    jac = np.array([r["jaccard"] for r in rows])
    L += ["", f"Jaccard overlap of the explained-peak sets: median {np.median(jac):.3f}, "
              f"mean {jac.mean():.3f}",
          f"pairs where the truth explains STRICTLY MORE peaks than its rival: "
          f"{np.mean([r['n_peaks_truth'] > r['n_peaks_rival'] for r in rows]):.1%}",
          f"pairs where the rival explains strictly more: "
          f"{np.mean([r['n_peaks_truth'] < r['n_peaks_rival'] for r in rows]):.1%}",
          "",
          "reading: an EXPLANATORY collision means the truth and its best rival account for exactly",
          "the same observed peaks -- our evidence cannot tell them apart and no scoring function of",
          "it could. A score near-tie without an explanatory collision is the opposite: the evidence",
          "differed and we compressed it. The first is a limit of the data; the second is our fault.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E43_collision_2026-09-23.txt", "w").write("\n".join(L) + "\n")
    json.dump({f"C{c}": dict(per[c]) for c in (1, 2)}, open(f"{RESULTS}/E43_collision.json", "w"),
              indent=2, default=int)
    import csv
    with open(f"{RESULTS}/E43_per_pair.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
