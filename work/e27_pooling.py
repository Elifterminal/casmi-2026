#!/usr/bin/env python3
"""E27 — the pooling theorem, applied where it actually bites.

Vanta's A5 says a time-average is QUADRATICALLY blind to a brief event riding on an ongoing
signal (verified here in July: the exponent climbs to 1.999 as the event shortens). Our single
strongest ranking feature is exactly such an average:

    explain = sum over the 30 most intense peaks of (intensity x matched) / sum(intensity)

The sharp form of the argument is not "weak peaks matter". It is that the INTENSE peaks are
explained by nearly every candidate in a pool of 36 look-alike natural products -- they are the
shared skeleton, so they carry almost no ranking information -- while they dominate the weight.
The discriminating evidence is in the peaks only a few candidates can explain, which is precisely
where the intensity weighting is blind.

Five ways to weight the same match data, all computed in one pass so the arms are paired:

  explain        intensity-weighted, over the top 30 peaks            (today)
  ex_sqrt        sqrt(intensity) weighted -- the same variance stabiliser that won in E25
  ex_flat        unweighted: the plain fraction of peaks explained
  ex_idf         each peak weighted by how RARE its explanation is across this query's pool:
                 w_i = -log((1 + candidates explaining peak i) / (1 + pool size)).
                 A peak every candidate explains contributes ~0; a peak one candidate explains
                 dominates. This is the theorem's claim stated as a feature.
  ex_top60       intensity-weighted but over 60 peaks instead of 30 (reach further down)

Arms: swap `explain` for each variant in turn, and one arm with all of them available. Every arm
retrains from scratch on the twin-safe training splits and is evaluated on 10-12, on the sqrt-p
pools (E25's ground), generation off (it costs points on the real board).

PREDICTIONS (before the run, 2026-09-21):
  P1  ex_idf is the best single swap -- it is the theorem stated directly.
  P2  ex_flat beats explain: dropping the weighting entirely should already help, because the
      weighting is actively harmful rather than merely uninformative.
  P3  The all-variants arm is best overall, by <= 0.01 over the best single swap.
  P4  Any gain lands in Class 2, where a crowded pool of near-identical candidates is the norm.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from twins import db_exclusions
from casmi_pipeline import (NB_TOP, PPM, TOPN, MassIndex, neutral_mass, chance, characterise,
                            key14, PROTON, H, TOL)
import scoring

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
TRAIN = ("nptsseed20_n300", "nptsseed21_n300", "nptsseed22_n300")
W = {1: 0.16, 2: 0.45, 3: 0.39}
BASE = ["chance_corr", "single", "nb_max", "nb_mean10", "spec_self", "heavy"]
VARIANTS = ["explain", "ex_sqrt", "ex_flat", "ex_idf", "ex_top60"]
ARMS = ([(v, BASE + [v]) for v in VARIANTS] + [("all variants", BASE + VARIANTS)])


def matches(mz, it, frag, positive, npeaks):
    """Which of the top `npeaks` peaks this candidate's fragments explain (+-1 H), and their
    intensities. Same matching rule as explain(); returned per peak so the weighting is free."""
    if not frag: return None, None
    order = np.argsort(-it)[:npeaks]
    mzz, itt = mz[order].astype(np.float64), it[order].astype(np.float64)
    fm = np.array(sorted(frag))
    neutral = (mzz - PROTON) if positive else (mzz + PROTON)
    hit = np.zeros(len(mzz), bool)
    for j, nt in enumerate(neutral):
        k = np.searchsorted(fm, nt)
        for kk in (k - 1, k, k + 1):
            if 0 <= kk < len(fm) and min(abs(nt - fm[kk]), abs(nt - fm[kk] - H), abs(nt - fm[kk] + H)) <= TOL:
                hit[j] = True; break
    return hit, itt


def build(splits, st, excl=frozenset(), log=print):
    keys, pmz, add, tr, co, _ = st
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_db = lambda k: trs.get(k) or cos_.get(k, "")
    out = []
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
        log(f"[{split}] {len(qs)} queries, {len(info):,} structures characterised")

        for k, cls, p, cands, M in qs:
            hits = sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
            good = [(h, c) for h, c in hits if info.get(smi_db(h)) is not None]
            hfp = [info[smi_db(h)]["fp"] for h, _ in good]
            hcos = np.array([c for _, c in good])
            mz, it = p["allmz"], p["allit"]
            # pass 1: per-candidate match masks, and how many candidates explain each peak
            per, count30, count60, n30, n60 = {}, None, None, 0, 0
            for c in cands:
                inf = info.get(smi_db(c))
                if inf is None: continue
                h30, i30 = matches(mz, it, inf["full"], p["positive"], 30)
                h60, i60 = matches(mz, it, inf["full"], p["positive"], 60)
                if h30 is None: continue
                per[c] = (h30, i30, h60, i60, inf)
                count30 = h30.astype(int) if count30 is None else count30 + h30
                count60 = h60.astype(int) if count60 is None else count60 + h60
                n30 += 1; n60 += 1
            if not per: continue
            idf = -np.log((1.0 + count30) / (1.0 + n30))      # rare explanations weigh most
            cand = {}
            for c, (h30, i30, h60, i60, inf) in per.items():
                s30 = i30.sum(); s60 = i60.sum(); sq = np.sqrt(i30)
                f = {
                    "explain":  float((h30 * i30).sum() / s30) if s30 > 0 else 0.0,
                    "ex_sqrt":  float((h30 * sq).sum() / sq.sum()) if sq.sum() > 0 else 0.0,
                    "ex_flat":  float(h30.mean()),
                    "ex_idf":   float((h30 * idf).sum() / idf.sum()) if idf.sum() > 0 else 0.0,
                    "ex_top60": float((h60 * i60).sum() / s60) if s60 > 0 else 0.0,
                }
                e = f["explain"]
                hh, ii = matches(mz, it, inf["single"], p["positive"], 30)
                f["single"] = float((hh * ii).sum() / ii.sum()) if hh is not None and ii.sum() > 0 else 0.0
                f["chance_corr"] = e - chance(inf["full"], M)
                if hfp:
                    v = hcos * np.array(DataStructs.BulkTanimotoSimilarity(inf["fp"], hfp))
                    f["nb_max"], f["nb_mean10"] = float(v.max()), float(v[:10].mean())
                else:
                    f["nb_max"] = f["nb_mean10"] = 0.0
                f["spec_self"] = p["spec"].get(c, 0.0); f["heavy"] = inf["heavy"]
                cand[c] = (f, smi_db(c))
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
    m = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X / sd, y)
    return m.coef_[0] / sd


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
        out[name] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=sum(W[k] * c[k-1] for k in (1, 2, 3)),
                         weights=dict(zip(feats, map(float, w))))
        log(f"  {name:14s} weighted {out[name]['weighted']:.4f}")
    L = ["E27 — the pooling theorem: five ways to weight the same match data", "",
         f"{'arm':14s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9} {'vs explain':>11}"]
    base = out["explain"]["weighted"]
    for name, _ in ARMS:
        o = out[name]
        L.append(f"{name:14s} {o['C1']:>8.4f} {o['C2']:>8.4f} {o['C3']:>8.4f} {o['weighted']:>9.4f} "
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
        L.append(f"{name:14s} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]" + ("   <- clears zero" if lo > 0 else ""))
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E27_pooling_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E27_pooling.json", "w"), indent=2)


if __name__ == "__main__":
    main()
