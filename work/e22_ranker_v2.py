#!/usr/bin/env python3
"""E22 — teach the ranker what a generated candidate is worth.

E20/E21 built Class 3 candidates by editing a relative and let them compete with retrieved ones.
Class 3 went 0.000 -> 0.036, but Class 2 paid for it (0.492 -> 0.468) and the net never cleared
zero. Capping the number of generated candidates barely moved anything, which rules out volume
as the cause and points at calibration:

    a generated candidate is built by editing a top spectral hit, so its neighbour-vote score is
    near-maximal BY CONSTRUCTION -- it resembles the molecules whose spectra match the query
    because it is one of them with an atom moved.

The E18 ranker never saw a generated candidate in training, so nothing taught it to discount
that. E22 retrains on both channels with one extra feature, `is_generated`, which in a linear
model is exactly a calibration offset for the manufactured channel.

Training: NP twin-safe splits 20-22, retrieval AND generated candidates, evaluation-split query
structures excluded (as in E17). Evaluation: splits 10-12, unchanged.

PREDICTIONS (before the run, 2026-09-20):
  P1  The learned weight on is_generated is NEGATIVE (generated evidence gets discounted).
  P2  Class 2 recovers to within 0.005 of retrieval-only (0.4920).
  P3  Net vs retrieval-only clears zero: 95% CI lower bound > 0.
  P4  Class 3 stays above 0.02 -- the discount must not simply switch generation off.
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from twins import db_exclusions
from e20_edits import candidates_for
from casmi_pipeline import (FEATS, NB_TOP, PPM, TOPN, MassIndex, neutral_mass, explain, chance,
                            characterise, key14)
import scoring

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
TRAIN = ("nptsseed20_n300", "nptsseed21_n300", "nptsseed22_n300")
W = {1: 0.16, 2: 0.45, 3: 0.39}
FEATS2 = FEATS + ["is_generated"]
MAX_REL, MAX_SITES, GAP_TOL = 10, 20, 0.003


def _gen(args):
    smi, gap = args
    return [s for s, _ in candidates_for(smi, gap, tol=GAP_TOL, max_sites=MAX_SITES)]


def build(splits, st, excl_keys=frozenset(), log=print, pools_prefix="", generate=True):
    """Per query: {candidate_id: (feature vector, smiles)}, plus class and truth."""
    keys, pmz, add, tr, co, _ = st
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_db = lambda k: trs.get(k) or cos_.get(k, "")
    exact = {}
    def emass(smi):
        if smi not in exact:
            m = Chem.MolFromSmiles(smi or "")
            exact[smi] = Descriptors.ExactMolWt(m) if m is not None else np.nan
        return exact[smi]

    out = []
    for split in splits:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{pools_prefix}{split}.pkl", "rb"))}
        keep = ~co.k.isin(db_exclusions(sp, tr, co))
        midx = MassIndex(co.k[keep].to_numpy(), co.m[keep].to_numpy())
        qs, jobs = [], []
        for k, rows in qrows.items():
            if k in excl_keys: continue
            p = pools[k]; cands, Ms = set(), []
            for r in rows:
                M = neutral_mass(pmz[r], add[r])
                if np.isfinite(M) and M > 0: cands.update(midx.window(M, PPM).tolist()); Ms.append(M)
            M = float(np.median(Ms)) if Ms else np.nan
            pend = []
            if generate and np.isfinite(M):
                for h, _ in sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_REL]:
                    s = smi_db(h); rm = emass(s)
                    if np.isfinite(rm) and abs(M - rm) > 1e-6: pend.append((s, M - rm))
            jobs.append(pend); qs.append((k, assign[k], p, cands, M))
        with Pool(7) as pool:
            flat = pool.map(_gen, [j for pend in jobs for j in pend], chunksize=4)
        gen_per_q, i = [], 0
        for pend in jobs:
            got = set()
            for _ in pend: got.update(flat[i]); i += 1
            gen_per_q.append(sorted(got))
        need = {smi_db(c) for _, _, _, cands, _ in qs for c in cands}
        need |= {smi_db(h) for _, _, p, _, _ in qs
                 for h, _ in sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]}
        need |= {s for g in gen_per_q for s in g}
        info = characterise(need, procs=7)
        log(f"[{split}] {len(qs)} queries, {sum(len(g) for g in gen_per_q):,} generated, "
            f"{len(info):,} structures characterised")

        for (k, cls, p, cands, M), gen in zip(qs, gen_per_q):
            hits = sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
            good = [(h, c) for h, c in hits if info.get(smi_db(h)) is not None]
            hfp = [info[smi_db(h)]["fp"] for h, _ in good]
            hcos = np.array([c for _, c in good])
            def vec(smi, spec_self, is_gen):
                inf = info.get(smi)
                if inf is None: return None
                e = explain(p["allmz"], p["allit"], inf["full"], p["positive"])
                if hfp:
                    v = hcos * np.array(DataStructs.BulkTanimotoSimilarity(inf["fp"], hfp))
                    nbm, nb10 = float(v.max()), float(v[:10].mean())
                else:
                    nbm = nb10 = 0.0
                f = dict(explain=e, chance_corr=e - chance(inf["full"], M),
                         single=explain(p["allmz"], p["allit"], inf["single"], p["positive"]),
                         nb_max=nbm, nb_mean10=nb10, spec_self=spec_self, heavy=inf["heavy"],
                         is_generated=float(is_gen))
                return np.array([f[x] for x in FEATS2])
            cand = {}
            for c in cands:
                v = vec(smi_db(c), p["spec"].get(c, 0.0), 0)
                if v is not None: cand[c] = (v, smi_db(c))
            for s in gen:
                v = vec(s, 0.0, 1)
                if v is not None: cand[f"gen:{s}"] = (v, s)
            out.append(dict(k=k, cls=cls, truth=p["truth"], cand=cand,
                            # E37: tail ordered by spectral score, not by InChIKey. Worth
                            # +0.0037 [+0.0004, +0.0077] here and +0.0105 on unrestricted queries.
                            tail=[smi_db(c) for c in sorted(set(dict(hits)) - cands,
                                                            key=lambda c: (-p["spec"][c], c))[:TOPN]]))
    return out


def fit(train_q):
    X, y = [], []
    for q in train_q:
        tk = [c for c, (_, s) in q["cand"].items() if key14(s) == q["truth"]]
        if not tk: continue
        t = q["cand"][tk[0]][0]
        for c, (v, _) in q["cand"].items():
            if c in tk: continue
            d = t - v; X += [d, -d]; y += [1, 0]
    X = np.array(X); sd = X.std(0) + 1e-9
    m = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X / sd, y)
    return m.coef_[0] / sd, len(y) // 2


def rank_score(q, w, use_gen=True):
    pool = {c: (float(np.dot(w, v)), s) for c, (v, s) in q["cand"].items()
            if use_gen or not c.startswith("gen:")}
    order = [pool[c][1] for c in sorted(pool, key=lambda c: (-pool[c][0], c))] + q["tail"]
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
    st = load_structures()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    ev = build(EVAL, st, log=log)
    tr_q = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log)
    w, npairs = fit(tr_q)
    log(f"trained on {npairs:,} truth/decoy pairs")
    old = json.load(open(f"{RESULTS}/E18_ranker_coconut.json"))["weights"]
    w_old = np.array([old[f] for f in FEATS] + [0.0])

    per = defaultdict(lambda: defaultdict(list))
    for q in ev:
        per["E18 weights, retrieval only"][q["cls"]].append(rank_score(q, w_old, use_gen=False))
        per["E18 weights, + generated"][q["cls"]].append(rank_score(q, w_old, use_gen=True))
        per["E22 weights, + generated"][q["cls"]].append(rank_score(q, w, use_gen=True))
        per["E22 weights, retrieval only"][q["cls"]].append(rank_score(q, w, use_gen=False))

    L = ["E22 — ranker retrained with generated candidates in the training data", "",
         "weights: " + ", ".join(f"{f} {v:+.3g}" for f, v in zip(FEATS2, w)), "",
         f"{'variant':30s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}"]
    out, base = {}, "E18 weights, retrieval only"
    for name in ("E18 weights, retrieval only", "E18 weights, + generated",
                 "E22 weights, retrieval only", "E22 weights, + generated"):
        c = [float(np.mean(per[name][k])) for k in (1, 2, 3)]
        wt = sum(W[k] * c[k - 1] for k in (1, 2, 3))
        out[name] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=wt)
        L.append(f"{name:30s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt:>9.4f}")
    rng = np.random.default_rng(0)
    L.append("")
    cis = {}
    for name in out:
        if name == base: continue
        by = {c: list(zip(per[base][c], per[name][c])) for c in (1, 2, 3)}
        d = out[name]["weighted"] - out[base]["weighted"]
        boots = [sum(W[c] * np.mean([b - a for a, b in [by[c][i] for i in rng.integers(0, len(by[c]), len(by[c]))]])
                     for c in by) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cis[name] = [d, lo, hi]
        L.append(f"{name:30s} vs baseline: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                 + ("   <- clears zero" if lo > 0 else ""))
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E22_ranker_v2_2026-09-20.txt", "w").write("\n".join(L) + "\n")
    json.dump({"weights": dict(zip(FEATS2, map(float, w))), "results": out, "vs_baseline": cis},
              open(f"{RESULTS}/E22_ranker_v2.json", "w"), indent=2)


if __name__ == "__main__":
    main()
