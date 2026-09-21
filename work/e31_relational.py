#!/usr/bin/env python3
"""E31 — score a candidate by the pattern of its relations, not by its list of masses.

The premise under the whole Hodos programme: a thing is the pattern of its connections, not its
substrate. Our strongest feature violates that. `explain` asks how many of a candidate's
fragment MASSES appear as peaks -- a list of magnitudes, checked by set overlap. Two candidates
can share a fragment list and differ completely in which losses connect those fragments.

The relational reading of a spectrum is the differences BETWEEN peaks. A difference is a neutral
loss: the chemical group the molecule shed between one fragment and another. And here the
argument is sharper than the general principle, because of how our pools are built:

    every candidate in a 10 ppm mass window has the SAME precursor mass by construction.
    Absolute mass anchoring is therefore shared across the pool and carries no discriminating
    information at all. Whatever separates the true structure from 35 look-alikes has to live
    in the pattern, not the magnitudes.

  explain      fraction of the top peaks whose neutral mass matches a candidate fragment (ours)
  loss_pairs   fraction of observed PEAK-PAIR DIFFERENCES matched by some difference between two
               of that candidate's fragments -- the same evidence read relationally
  both         the ranker gets each as its own feature and decides

Same protocol as E27: sqrt-p pools, generation off, every arm retrained from scratch on the
twin-safe training splits, evaluated on 10-12, paired bootstrap.

PREDICTIONS (before the run, 2026-09-21):
  P1  loss_pairs ALONE is worse than explain. It throws away absolute anchoring, which is weak
      within a pool but not worthless -- it still separates a candidate whose fragments land
      nowhere near the observed peaks.
  P2  both > explain, and the interval clears zero. The relational signal should be complementary
      rather than redundant: it is computed from peak pairs that explain never looks at.
  P3  The gain lands in Class 2, where the pool is crowded with same-mass look-alikes and
      absolute anchoring is most nearly useless.
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from twins import db_exclusions
from casmi_pipeline import (NB_TOP, PPM, TOPN, MassIndex, neutral_mass, explain, chance,
                            characterise, key14, candidate_info, PROTON, TOL)
import scoring

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
TRAIN = ("nptsseed20_n300", "nptsseed21_n300", "nptsseed22_n300")
W = {1: 0.16, 2: 0.45, 3: 0.39}
BASE = ["chance_corr", "single", "nb_max", "nb_mean10", "spec_self", "heavy"]
ARMS = [("explain", BASE + ["explain"]),
        ("loss_pairs", BASE + ["loss_pairs"]),
        ("both", BASE + ["explain", "loss_pairs"])]
NPEAK_PAIRS = 20        # peaks whose pairwise differences we read (20 -> 190 relations)
MAX_FRAG = 200          # fragments per candidate used for differences (200 -> 19,900)
MIN_DIFF = 2.0          # ignore differences below this: noise and isotope spacing, not losses


def diff_set(frag):
    """Sorted array of differences between pairs of a candidate's fragment masses."""
    if not frag: return None
    f = np.array(sorted(frag))
    if len(f) > MAX_FRAG:
        f = f[np.linspace(0, len(f) - 1, MAX_FRAG).round().astype(int)]
    d = f[None, :] - f[:, None]
    d = d[d > MIN_DIFF]
    return np.sort(d) if len(d) else None


def _prep_cand(smi):
    smi_, inf = candidate_info(smi)
    if inf is None: return smi_, None
    return smi_, dict(inf, diffs=diff_set(inf["full"]))


def loss_pairs_score(mz, it, diffs, positive):
    """Intensity-weighted share of observed peak-pair differences this candidate can produce."""
    if diffs is None or len(mz) < 2: return 0.0
    o = np.argsort(-np.asarray(it))[:NPEAK_PAIRS]
    m, w = np.asarray(mz, np.float64)[o], np.asarray(it, np.float64)[o]
    tot = hit = 0.0
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            d = abs(m[i] - m[j])
            if d <= MIN_DIFF: continue
            wt = min(w[i], w[j])          # a relation is only as strong as its weaker end
            tot += wt
            k = np.searchsorted(diffs, d)
            for kk in (k - 1, k):
                if 0 <= kk < len(diffs) and abs(diffs[kk] - d) <= TOL:
                    hit += wt; break
    return hit / tot if tot > 0 else 0.0


def build(splits, st, excl=frozenset(), log=print):
    keys, pmz, add, tr, co, _ = st
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_db = lambda k: trs.get(k) or cos_.get(k, "")
    out, info = [], {}
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
        need = sorted(({smi_db(c) for _, _, _, cands, _ in qs for c in cands} |
                       {smi_db(h) for _, _, p, _, _ in qs
                        for h, _ in sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]})
                      - set(info))
        with Pool(7) as pool:
            for smi, inf in pool.imap_unordered(_prep_cand, need, chunksize=8): info[smi] = inf
        log(f"[{split}] {len(qs)} queries, {len(info):,} structures characterised")
        for k, cls, p, cands, M in qs:
            hits = sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
            good = [(h, c) for h, c in hits if info.get(smi_db(h)) is not None]
            hfp = [info[smi_db(h)]["fp"] for h, _ in good]
            hcos = np.array([c for _, c in good])
            cand = {}
            for c in cands:
                smi = smi_db(c); inf = info.get(smi)
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
                         heavy=inf["heavy"],
                         loss_pairs=loss_pairs_score(p["allmz"], p["allit"], inf["diffs"], p["positive"]))
                cand[c] = (f, smi)
            out.append(dict(k=k, cls=cls, truth=p["truth"], cand=cand,
                            tail=[smi_db(c) for c in sorted(set(dict(hits)) - cands)[:TOPN]]))
    return out


def fit(train_q, feats):
    X, y = [], []
    for q in train_q:
        tk = [c for c, (_, s) in q["cand"].items() if key14(s) == q["truth"]]
        if not tk: continue
        t = np.array([q["cand"][tk[0]][0][f] for f in feats])
        for c, (fv, _) in q["cand"].items():
            if c in tk: continue
            d = t - np.array([fv[f] for f in feats]); X += [d, -d]; y += [1, 0]
    X = np.array(X); sd = X.std(0) + 1e-9
    return LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X / sd, y).coef_[0] / sd


def score(q, w, feats):
    sc = {c: float(np.dot(w, [fv[f] for f in feats])) for c, (fv, _) in q["cand"].items()}
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
    ev = build(EVAL, st, log=log)
    tr = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log)
    log("features built; fitting arms")
    per, out = {}, {}
    for name, feats in ARMS:
        w = fit(tr, feats)
        p = defaultdict(list)
        for q in ev: p[q["cls"]].append(score(q, w, feats))
        c = [float(np.mean(p[k])) for k in (1, 2, 3)]
        per[name] = p
        out[name] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=sum(W[k]*c[k-1] for k in (1,2,3)),
                         weights=dict(zip(feats, map(float, w))))
        log(f"  {name:12s} weighted {out[name]['weighted']:.4f}")
    L = ["E31 — relations between peaks, not the list of masses", "",
         f"{'arm':12s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9} {'vs explain':>11}"]
    base = out["explain"]["weighted"]
    for name, _ in ARMS:
        o = out[name]
        L.append(f"{name:12s} {o['C1']:>8.4f} {o['C2']:>8.4f} {o['C3']:>8.4f} {o['weighted']:>9.4f} "
                 f"{o['weighted']-base:>+11.4f}")
    rng = np.random.default_rng(0)
    L.append("")
    for name, _ in ARMS:
        if name == "explain": continue
        a, b = per["explain"], per[name]
        d = out[name]["weighted"] - base
        boots = [sum(W[c] * np.mean([b[c][i] - a[c][i] for i in rng.integers(0, len(a[c]), len(a[c]))])
                     for c in (1, 2, 3)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        L.append(f"{name:12s} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]" + ("   <- clears zero" if lo > 0 else ""))
    L.append(f"\nlearned weight on loss_pairs in the 'both' arm: "
             f"{out['both']['weights'].get('loss_pairs', float('nan')):+.4g}")
    L.append(f"runtime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E31_relational_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E31_relational.json", "w"), indent=2)


if __name__ == "__main__":
    main()
