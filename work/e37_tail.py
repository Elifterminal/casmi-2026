#!/usr/bin/env python3
"""E37 — rank the spectral-only candidates by their spectral score, not alphabetically.

Our ranked list is: every mass-retrieved candidate scored by the learned ranker, then the
spectral hits that retrieval missed, appended to fill the 25 slots. That tail has always been
built as

    sorted(spectral_hits - mass_candidates)[:25]

which sorts by InChIKey -- alphabetically. The candidates are chosen sensibly and then ordered
arbitrarily. For any molecule whose structure is missing from COCONUT but whose spectrum matched
a reference, the answer is in that tail and we rank it by the alphabet.

E36 made the cost visible by accident: on queries not filtered for COCONUT membership, Class 1
scored 0.3206 with COCONUT-only retrieval. Class 1 molecules have public spectra -- the match is
sitting right there in the hits.

Two arms, same everything else, both tails drawn from the same candidate set:

  by_key    sorted by InChIKey             (what we ship)
  by_spec   sorted by spectral score, best first

Measured on both query sets, because the production set may mask it: a natural product is
usually in COCONUT, so its answer usually arrives through mass retrieval instead.

PREDICTIONS (before the run, 2026-09-21):
  P1  On the unrestricted set the fix is worth more than +0.05 weighted, nearly all of it C1.
  P2  On the production (COCONUT-filtered NP) set it is small, under +0.01 -- those answers are
      in the mass pool already.
  P3  It cannot hurt on either set. The same candidates occupy the same slots; only their order
      within the tail changes, and score order dominates alphabetical order in expectation.
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
from e22_ranker_v2 import fit, FEATS2, TRAIN, W
from casmi_pipeline import (NB_TOP, PPM, TOPN, MassIndex, neutral_mass, explain, chance,
                            characterise, key14)

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
SETS = {"production (NP, in COCONUT)": ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300"),
        "unrestricted (any database)": ("tsseed40_n300", "tsseed41_n300", "tsseed42_n300")}


def build(splits, st, excl_keys=frozenset(), log=print):
    keys, pmz, add, tr, co, _ = st
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    com = dict(zip(co.k, co.m))
    smi_db = lambda k: trs.get(k) or cos_.get(k, "")
    out, info = [], {}
    for split in splits:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{split}.pkl", "rb"))}
        excl = db_exclusions(sp, tr, co)
        dbm = {k: m for k, m in com.items() if k not in excl}
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
        log(f"[{split}] {len(qs)} queries, {len(info):,} structures")
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
            rest = set(dict(hits)) - cands
            tail_key = [smi_db(c) for c in sorted(rest)[:TOPN]]
            tail_spec = [smi_db(c) for c in sorted(rest, key=lambda c: (-p["spec"][c], c))[:TOPN]]
            out.append(dict(k=k, cls=cls, truth=p["truth"], cand=cand,
                            tail_key=tail_key, tail_spec=tail_spec))
    return out


def score(q, w, which):
    sc = {c: float(np.dot(w, v)) for c, (v, _) in q["cand"].items()}
    order = [q["cand"][c][1] for c in sorted(sc, key=lambda c: (-sc[c], c))] + q[which]
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
    L = ["E37 — ordering the spectral-only tail by score instead of by key", ""]
    out = {}
    tr_q = None
    for label, splits in SETS.items():
        ev = build(splits, st, log=log)
        if tr_q is None:
            tr_q = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log)
            w, n = fit([dict(q, tail=q["tail_key"]) for q in tr_q])
            log(f"trained on {n:,} pairs")
        per = {}
        for which in ("tail_key", "tail_spec"):
            p = defaultdict(list)
            for q in ev: p[q["cls"]].append(score(q, w, which))
            per[which] = p
        L.append(f"### {label}")
        L.append(f"{'tail order':12s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}")
        wt = {}
        for which, name in (("tail_key", "by key"), ("tail_spec", "by score")):
            c = [float(np.mean(per[which][k])) for k in (1, 2, 3)]
            wt[which] = sum(W[k]*c[k-1] for k in (1, 2, 3))
            L.append(f"{name:12s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt[which]:>9.4f}")
        d = wt["tail_spec"] - wt["tail_key"]
        rng = np.random.default_rng(0)
        a, b = per["tail_key"], per["tail_spec"]
        boots = [sum(W[c] * np.mean([b[c][i] - a[c][i] for i in rng.integers(0, len(a[c]), len(a[c]))])
                     for c in (1, 2, 3)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        L += [f"by score - by key: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              + ("   <- clears zero" if lo > 0 else ""), ""]
        out[label] = dict(by_key=wt["tail_key"], by_score=wt["tail_spec"], delta=[d, lo, hi])
    L.append(f"runtime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E37_tail_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E37_tail.json", "w"), indent=2)


if __name__ == "__main__":
    main()
