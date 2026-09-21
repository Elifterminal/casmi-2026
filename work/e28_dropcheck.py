#!/usr/bin/env python3
"""E28 — why is E27's baseline 0.3881 where E25 measured 0.3682 for the same configuration?

Same pools (sqrt-p ground), same features, same splits, generation off. The one difference I can
see in the code: E25 scored a candidate whose fragment table could not be computed as explain=0
and LEFT IT IN THE POOL; E27 drops it. Large molecules (>60 heavy atoms) and unparseable SMILES
are the cases. If that is the whole difference, then simply not ranking candidates we cannot
fragment is worth ~0.02 -- more than every weighting E27 tested -- and it is free.

Unexplained discrepancies are debts, so this measures rather than assumes: build once, score
twice, differing only in whether un-fragmentable candidates stay in the ranked list.

PREDICTION (before the run): dropping them accounts for most of the 0.02 gap, and dropping is
better -- a candidate we cannot fragment has no evidence for or against it, so ranking it on
heavy-atom count and spectral self-match alone is noise that can displace the truth.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from twins import db_exclusions
from e22_ranker_v2 import fit, rank_score, FEATS2
from casmi_pipeline import (NB_TOP, PPM, TOPN, MassIndex, neutral_mass, explain, chance,
                            characterise, key14)

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
TRAIN = ("nptsseed20_n300", "nptsseed21_n300", "nptsseed22_n300")
W = {1: 0.16, 2: 0.45, 3: 0.39}


def build(splits, st, excl=frozenset(), log=print):
    keys, pmz, add, tr, co, _ = st
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_db = lambda k: trs.get(k) or cos_.get(k, "")
    out, stats = [], defaultdict(int)
    for split in splits:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{split}.pkl", "rb"))}
        keep = ~co.k.isin(db_exclusions(sp, tr, co))
        midx = MassIndex(co.k[keep].to_numpy(), co.m[keep].to_numpy())
        qs = []
        for k, rows in qrows.items():
            if k in excl: continue
            p = pools[k]; cands, Ms = set(), []
            for r in rows:
                M = neutral_mass(pmz[r], add[r])
                if np.isfinite(M) and M > 0: cands.update(midx.window(M, PPM).tolist()); Ms.append(M)
            qs.append((k, assign[k], p, cands, float(np.median(Ms)) if Ms else np.nan))
        need = {smi_db(c) for _, _, _, cands, _ in qs for c in cands}
        need |= {smi_db(h) for _, _, p, _, _ in qs
                 for h, _ in sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]}
        info = characterise(need, procs=7)
        log(f"[{split}] {len(qs)} queries, {len(info):,} structures")
        for k, cls, p, cands, M in qs:
            hits = sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
            good = [(h, c) for h, c in hits if info.get(smi_db(h)) is not None]
            hfp = [info[smi_db(h)]["fp"] for h, _ in good]
            hcos = np.array([c for _, c in good])
            cand = {}
            for c in cands:
                smi = smi_db(c); inf = info.get(smi)
                stats["candidates"] += 1
                if inf is None:
                    stats["unparseable"] += 1
                    continue                      # both arms drop these: no SMILES, no vector
                nofrag = not inf["full"]
                if nofrag: stats["no_fragments"] += 1
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
                cand[c] = (np.array([f[x] for x in FEATS2]), smi, nofrag)
            out.append(dict(k=k, cls=cls, truth=p["truth"], cand=cand,
                            tail=[smi_db(c) for c in sorted(set(dict(hits)) - cands)[:TOPN]]))
    return out, stats


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    st = load_structures()
    ev, s_ev = build(EVAL, st, log=log)
    tr_q, _ = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log)
    strip = lambda qs, drop: [dict(q, cand={c: (v, smi) for c, (v, smi, nf) in q["cand"].items()
                                             if not (drop and nf)}) for q in qs]
    L = [f"E28 — do un-fragmentable candidates belong in the ranked list?", "",
         f"candidates seen: {s_ev['candidates']:,}   unparseable SMILES: {s_ev['unparseable']:,}   "
         f"parse but yield NO fragment table: {s_ev['no_fragments']:,} "
         f"({s_ev['no_fragments']/max(1,s_ev['candidates']):.2%})", "",
         f"{'arm':22s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}"]
    per, out = {}, {}
    for name, drop in (("keep them (E25)", False), ("drop them (E27)", True)):
        w = fit(strip(tr_q, drop))[0] if isinstance(fit(strip(tr_q, drop)), tuple) else fit(strip(tr_q, drop))
        w = w[0] if isinstance(w, tuple) else w
        p = defaultdict(list)
        for q in strip(ev, drop): p[q["cls"]].append(rank_score(q, w, use_gen=False))
        c = [float(np.mean(p[k])) for k in (1, 2, 3)]
        per[name] = p
        out[name] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=sum(W[k]*c[k-1] for k in (1,2,3)))
        L.append(f"{name:22s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {out[name]['weighted']:>9.4f}")
    a, b = per["keep them (E25)"], per["drop them (E27)"]
    d = out["drop them (E27)"]["weighted"] - out["keep them (E25)"]["weighted"]
    rng = np.random.default_rng(0)
    boots = [sum(W[c] * np.mean([b[c][i] - a[c][i] for i in rng.integers(0, len(a[c]), len(a[c]))])
                 for c in (1, 2, 3)) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    L += ["", f"dropping vs keeping: {d:+.4f} weighted  95% CI [{lo:+.4f}, {hi:+.4f}]"
              + ("   <- clears zero" if lo > 0 else ""),
          f"\nE25 measured 0.3682 keeping them; E27 measured 0.3881 dropping them (gap 0.0199)",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E28_dropcheck_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump({"arms": out, "delta": [d, lo, hi], "stats": dict(s_ev)},
              open(f"{RESULTS}/E28_dropcheck.json", "w"), indent=2)


if __name__ == "__main__":
    main()
