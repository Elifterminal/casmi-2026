#!/usr/bin/env python3
"""E36 — re-run the database decision on a query set that can show coverage.

E15 concluded "ship COCONUT alone": the merged database (COCONUT + the training structures)
scored 0.302 against 0.342. We shipped that and Seda endorsed it. E35 shows why the experiment
could not have answered the question it was asked:

    every query in it was REQUIRED to be in COCONUT, so the merged database could only add
    decoys. It measured the dilution cost of a bigger pool and never the coverage benefit.

On a natural-product query set chosen WITHOUT that requirement, only 44.7% of Class 2 answers
are in COCONUT at all. So this repeats the comparison on tsseed40-42, where coverage is a real
variable and the two effects can fight.

  coconut   COCONUT minus this split's Class 3 exclusions            (what we ship)
  union     COCONUT plus the training structures, same exclusions    (E15's loser)

Everything else is production: sqrt-p ground, generation off, ranker retrained per arm on the
usual NP training splits (a shared constant across the two arms, so it cannot favour either).

PREDICTIONS (before the run, 2026-09-21):
  P1  union BEATS coconut here, reversing E15 -- coverage should dominate dilution when 55% of
      answers are missing from the smaller database.
  P2  the gain is concentrated in Class 2, +0.05 or more, since that is the class whose only
      route is mass retrieval.
  P3  Class 1 barely moves: its answers arrive through the spectral channel, which neither arm
      changes.
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from twins import db_exclusions
from e22_ranker_v2 import fit, rank_score, FEATS2, TRAIN, W
from casmi_pipeline import (NB_TOP, PPM, TOPN, MassIndex, neutral_mass, explain, chance,
                            characterise, key14)

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("tsseed40_n300", "tsseed41_n300", "tsseed42_n300")
ARMS = ("coconut", "union")


def build(splits, st, db, excl_keys=frozenset(), prefix="fr_", log=print):
    keys, pmz, add, tr, co, _ = st
    trm, trs = dict(zip(tr.k, tr.m)), dict(zip(tr.k, tr.s))
    com, cos_ = dict(zip(co.k, co.m)), dict(zip(co.k, co.s))
    smi_db = lambda k: trs.get(k) or cos_.get(k, "")
    out, info = [], {}
    for split in splits:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{prefix}{split}.pkl", "rb"))}
        excl = db_exclusions(sp, tr, co)
        src = com if db == "coconut" else {**com, **trm}
        dbm = {k: m for k, m in src.items() if k not in excl}
        midx = MassIndex(np.array(list(dbm)), np.array(list(dbm.values())))
        qs = []
        for k, rows in qrows.items():
            if k in excl_keys: continue
            p = pools[k]; cands, Ms = set(), []
            for r in rows:
                M = neutral_mass(pmz[r], add[r])
                if np.isfinite(M) and M > 0: cands.update(midx.window(M, PPM).tolist()); Ms.append(M)
            qs.append((k, assign[k], p, cands, float(np.median(Ms)) if Ms else np.nan))
        need = ({smi_db(c) for _, _, _, cands, _ in qs for c in cands} |
                {smi_db(h) for _, _, p, _, _ in qs
                 for h, _ in sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]}) - set(info)
        info.update(characterise(sorted(need), procs=7))
        log(f"[{db}/{split}] {len(qs)} queries, {len(info):,} structures, "
            f"median pool {np.median([len(c) for *_, c, _ in qs]):.0f}")
        for k, cls, p, cands, M in qs:
            hits = sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
            good = [(h, c) for h, c in hits if info.get(smi_db(h)) is not None]
            hfp = [info[smi_db(h)]["fp"] for h, _ in good]
            hcos = np.array([c for _, c in good])
            cand = {}
            for c in cands:
                inf = info.get(smi_db(c))
                if inf is None: continue
                e = explain(p["allmz"], p["allit"], inf["full"], p["positive"])
                if hfp:
                    v = hcos * np.array(DataStructs.BulkTanimotoSimilarity(inf["fp"], hfp))
                    nbm, nb10 = float(v.max()), float(v[:10].mean())
                else:
                    nbm = nb10 = 0.0
                f = dict(explain=e, chance_corr=e - chance(inf["full"], M),
                         single=explain(p["allmz"], p["allit"], inf["single"], p["positive"]),
                         nb_max=nbm, nb_mean10=nb10, spec_self=p["spec"].get(c, 0.0),
                         heavy=inf["heavy"], is_generated=0.0)
                cand[c] = (np.array([f[x] for x in FEATS2]), smi_db(c))
            out.append(dict(k=k, cls=cls, truth=p["truth"], cand=cand,
                            tail=[smi_db(c) for c in sorted(set(dict(hits)) - cands)[:TOPN]]))
    return out


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    st = load_structures()
    per, out = {}, {}
    for db in ARMS:
        ev = build(EVAL, st, db, log=log)
        tr_q = build(TRAIN, st, db, frozenset(q["k"] for q in ev), log=log)
        w, n = fit(tr_q)
        p = defaultdict(list)
        for q in ev: p[q["cls"]].append(rank_score(q, w, use_gen=False))
        c = [float(np.mean(p[k])) for k in (1, 2, 3)]
        per[db] = p
        out[db] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=sum(W[k]*c[k-1] for k in (1,2,3)))
        log(f"  {db:8s} weighted {out[db]['weighted']:.4f}  (trained on {n:,} pairs)")
    L = ["E36 — COCONUT vs the merged database, on queries not chosen for COCONUT membership", "",
         f"{'database':10s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}"]
    for db in ARMS:
        o = out[db]
        L.append(f"{db:10s} {o['C1']:>8.4f} {o['C2']:>8.4f} {o['C3']:>8.4f} {o['weighted']:>9.4f}")
    d = out["union"]["weighted"] - out["coconut"]["weighted"]
    rng = np.random.default_rng(0)
    a, b = per["coconut"], per["union"]
    boots = [sum(W[c] * np.mean([b[c][i] - a[c][i] for i in rng.integers(0, len(a[c]), len(a[c]))])
                 for c in (1, 2, 3)) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    L += ["", f"union - coconut: {d:+.4f} weighted  95% CI [{lo:+.4f}, {hi:+.4f}]"
              + ("   <- clears zero" if lo > 0 else ""),
          "", "E15 measured the reverse ordering (coconut 0.342, union 0.302) on queries that were",
          "all in COCONUT by construction, so only dilution could show there.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E36_coverage_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump({"arms": out, "delta": [d, lo, hi]}, open(f"{RESULTS}/E36_coverage.json", "w"), indent=2)


if __name__ == "__main__":
    main()
