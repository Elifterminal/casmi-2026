#!/usr/bin/env python3
"""E78 — the second stage: rerank the fork's top 25 on Class 2 using implied bond energy.

TARGET (L-064). On the faithful Class 2 arm the fork puts the truth at rank 1 in 38.3% of queries and
at rank 2-3 in another 18.3%. Perfect reordering inside the 25 is worth +0.129 board; the rank-2-3
band alone is +0.048. We need +0.02, i.e. 15% of the headroom. On the 100 exported queries the fork
scores 0.5464 with a ceiling of 0.8500.

THE DESIGN PROBLEM. What E66/E70 validated is PAIRWISE: given the truth and a rival, the truth
explains the peaks both account for with less broken bond energy, 0.589-0.605 of the time. A reranker
cannot use that directly, because it does not know which candidate is the truth -- and scoring each
candidate on the peaks IT happens to explain reintroduces exactly the coverage asymmetry that made
E63 uninterpretable (truths explain 9.9 observed peaks, decoys 5.5).

THE FIX: use the fork's own rank-1 candidate as the fixed reference. For every candidate, compute the
implied bond energy over the peaks shared with that front-runner. This is available at inference, it
keeps every comparison on a common peak set, and it encodes precisely the decision worth +0.048:
"does this candidate account for the same peaks with less bond breaking than the current leader?" The
front-runner's own feature is zero by construction, which is the correct neutral value.

FEATURES (deliberately few -- 100 queries is a small training set):
  fork_rank, inv_fork_rank     the prior. The fork is right 38.3% of the time already and the second
                               stage must not throw that away.
  bde_vs_top                   implied bond energy minus the front-runner's, on shared peaks.
                               Negative = breaks less = more plausible. THE feature under test.
  bde_vs_top_rank              its rank within the query, so the scale is comparable across molecules
  shared_peaks                 how many peaks the comparison rested on (low = the feature is noise)
  n_explained                  peaks this candidate explains at all -- lets a tree discount
                               bde_vs_top when coverage is thin, which is the conditional structure
                               E72 showed trees exploit and linear models cannot
  heavy, rings, arom_rings, frac_csp3, tpsa, logp    chemistry, from E72's set

TWO MODELS, because n=100 invites overfitting:
  A  a two-parameter blend: score = -fork_rank + w * (-bde_vs_top_rank), w swept by cross-validation.
     Almost impossible to overfit; if the feature carries anything this should show it.
  B  a small LambdaMART over all the features above.

EVALUATION. Five-fold cross-validation grouped by Murcko scaffold, so no fold shares a scaffold with
its training data. Every query is scored by a model that never saw its scaffold. Reported as a paired
bootstrap against the fork's own order, because paired is the only way 100 queries can resolve the
+0.044 of Class 2 that +0.02 board requires.

PREDICTED, before the run: model A gains between 0 and +0.03 of Class 2 MRR; model B does not beat A,
because 100 queries cannot support eleven features. If both intervals include zero I will say the
feature does not survive contact with a real ranker, rather than hunting for a variant that works.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS, RESULTS = os.path.join(WORK, "splits"), os.path.join(WORK, "results")
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
WORKLIST = f"{EXPORTS}/c2_rerank_coconut.json"
PRED = f"{EXPORTS}/iceberg_bde_c2_rerank.npz"
TOL, E_TOL, TOP_OBS, BOOT, FOLDS = 0.01, 0.05, 40, 4000, 5
FEATS = ["fork_rank", "inv_fork_rank", "bde_vs_top", "bde_vs_top_rank", "shared_peaks",
         "n_explained", "heavy", "rings", "arom_rings", "frac_csp3", "tpsa", "logp"]


def explained(obs_mz, pmz, pval):
    o = np.argsort(pmz); pm, pv = pmz[o], np.asarray(pval)[o]
    out = {}
    for i, m in enumerate(obs_mz):
        j = np.searchsorted(pm, m); best = None
        for k in (j - 1, j):
            if 0 <= k < len(pm) and abs(pm[k] - m) <= TOL:
                if best is None or abs(pm[k] - m) < abs(pm[best] - m): best = k
        if best is not None: out[i] = float(pv[best])
    return out


def chem(smi, _c={}):
    if smi in _c: return _c[smi]
    m = Chem.MolFromSmiles(smi or "")
    v = (dict(heavy=float(m.GetNumHeavyAtoms()), rings=float(rdMolDescriptors.CalcNumRings(m)),
              arom_rings=float(rdMolDescriptors.CalcNumAromaticRings(m)),
              frac_csp3=float(rdMolDescriptors.CalcFractionCSP3(m)),
              tpsa=float(rdMolDescriptors.CalcTPSA(m)), logp=float(Crippen.MolLogP(m)))
         if m is not None else None)
    _c[smi] = v
    return v


def scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi or "")
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else ""
    except Exception:
        return ""


def build_rows(log):
    d = np.load(PRED)
    pred = defaultdict(lambda: defaultdict(dict))
    for key in d.keys():
        q, rest, kind = key.split("|")
        cand, _, e = rest.rpartition("@")
        pred[q][cand].setdefault(float(e), {})[kind] = d[key]
    wl = json.load(open(WORKLIST))["queries"]
    vs = json.load(open(f"{SPLITS}/valset_fork_queries.json"))
    keys = {k for a in vs["queries"].values() for k in a}

    f = pq.ParquetFile(os.path.join(WORK, "data", "train.parquet"))
    spec = defaultdict(dict); base = 0
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["inchikey14", "collision_energy_ev", "ms2_mzs",
                                         "ms2_normalized_intensities"])
        ik = t.column("inchikey14").to_pylist()
        for j, k in enumerate(ik):
            if k not in keys: continue
            ce = t.column("collision_energy_ev")[j].as_py()
            vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
            vals = [float(x) for x in vals if x is not None and np.isfinite(x)]
            if len(vals) != 1: continue
            mz = np.asarray(t.column("ms2_mzs")[j].as_py(), float)
            it = np.asarray(t.column("ms2_normalized_intensities")[j].as_py(), float)
            if not len(mz): continue
            o = np.argsort(-it)[:TOP_OBS]
            spec[k].setdefault(vals[0], (mz[o], it[o]))
        base += t.num_rows
        del t
    log(f"observed spectra for {len(spec)} molecules")

    rows = []
    for q in wl:
        qk = q["key"]
        cands = pred.get(qk)
        obs = spec.get(qk)
        if not cands or not obs: continue
        obs_es = np.array(sorted(obs))
        # implied summed bond energy per candidate per energy, over the peaks it explains
        expl = defaultdict(dict)
        for c, byE in cands.items():
            for pe in sorted(byE):
                v = byE[pe]
                if "mz" not in v or "bd" not in v or len(v["mz"]) == 0: continue
                j = int(np.argmin(np.abs(obs_es - pe)))
                if abs(obs_es[j] - pe) > E_TOL: continue
                oe = float(obs_es[j])
                expl[c][oe] = explained(obs[oe][0], v["mz"], v["bd"])
        by_key = {c["key"]: c for c in q["candidates"]}
        top = min((c["key"] for c in q["candidates"]), key=lambda k: by_key[k]["fork_rank"])
        if top not in expl: continue

        per = {}
        for c in expl:
            diffs, shared, nexp = [], 0, 0
            for e in sorted(set(expl[c]) & set(expl[top])):
                a, b = expl[c][e], expl[top][e]
                both = set(a) & set(b)
                nexp += len(a)
                if not both: continue
                w = obs[e][1]; den = sum(w[i] for i in both)
                if den <= 0: continue
                diffs.append(sum(w[i] * a[i] for i in both) / den
                             - sum(w[i] * b[i] for i in both) / den)
                shared += len(both)
            if not diffs and c != top: continue
            per[c] = dict(bde_vs_top=float(np.mean(diffs)) if diffs else 0.0,
                          shared_peaks=float(shared), n_explained=float(nexp))
        if len(per) < 2: continue
        order = sorted(per, key=lambda c: per[c]["bde_vs_top"])
        rank_of = {c: i + 1 for i, c in enumerate(order)}
        for c, v in per.items():
            ch = chem(by_key[c]["smiles"])
            if ch is None: continue
            fr = float(by_key[c]["fork_rank"])
            rows.append(dict(q=qk, cand=c, smiles=by_key[c]["smiles"],
                             y=int(scoring.key14(by_key[c]["smiles"]) == q["truth"]),
                             fork_rank=fr, inv_fork_rank=1.0 / fr,
                             bde_vs_top=v["bde_vs_top"],
                             bde_vs_top_rank=float(rank_of[c]),
                             shared_peaks=v["shared_peaks"], n_explained=v["n_explained"], **ch))
    log(f"{len(rows):,} candidate rows over {len({r['q'] for r in rows})} queries")
    return rows


def mrr_from_order(rows_by_q, score_of):
    """MRR@25 where each query's candidates are sorted by score_of, ties broken by fork rank."""
    tot = 0.0
    for qk, rs in rows_by_q.items():
        ranked = sorted(rs, key=lambda r: (-score_of(r), r["fork_rank"]))
        for i, r in enumerate(ranked[:25], 1):
            if r["y"]:
                tot += 1.0 / i
                break
    return tot / len(rows_by_q)


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    rows = build_rows(log)
    by_q = defaultdict(list)
    for r in rows: by_q[r["q"]].append(r)
    qs = sorted(by_q)
    truth_smi = {}
    for qk, rs in by_q.items():
        t = [r["smiles"] for r in rs if r["y"]]
        truth_smi[qk] = t[0] if t else rs[0]["smiles"]
    scaf = {qk: scaffold(truth_smi[qk]) for qk in qs}
    groups = {}
    for i, s in enumerate(sorted(set(scaf.values()))): groups[s] = i % FOLDS
    fold = {qk: groups[scaf[qk]] for qk in qs}
    log(f"{len(qs)} queries, {len(set(scaf.values()))} scaffolds, folds "
        f"{[sum(1 for q in qs if fold[q]==k) for k in range(FOLDS)]}")

    base = mrr_from_order(by_q, lambda r: -r["fork_rank"])
    ceiling = np.mean([any(r["y"] for r in rs) for rs in by_q.values()])
    log(f"fork order MRR@25 {base:.4f}   ceiling {ceiling:.4f}   headroom {ceiling-base:+.4f}")

    # ---- model A: two-parameter blend, weight swept by cross-validation -------------------------
    import lightgbm as lgb
    ws = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
    perq = {"fork": {}, "A": {}, "B": {}}
    for qk, rs in by_q.items():
        ranked = sorted(rs, key=lambda r: r["fork_rank"])
        perq["fork"][qk] = next((1.0 / i for i, r in enumerate(ranked[:25], 1) if r["y"]), 0.0)

    for k in range(FOLDS):
        tr_q = [q for q in qs if fold[q] != k]
        te_q = [q for q in qs if fold[q] == k]
        if not te_q or not tr_q: continue
        # A: choose w on the training folds only
        best_w, best_v = 0.0, -1.0
        for w in ws:
            v = mrr_from_order({q: by_q[q] for q in tr_q},
                               lambda r, w=w: -r["fork_rank"] - w * r["bde_vs_top_rank"])
            if v > best_v: best_v, best_w = v, w
        for q in te_q:
            ranked = sorted(by_q[q], key=lambda r: (r["fork_rank"] + best_w * r["bde_vs_top_rank"],
                                                    r["fork_rank"]))
            perq["A"][q] = next((1.0 / i for i, r in enumerate(ranked[:25], 1) if r["y"]), 0.0)
        # B: small LambdaMART
        Xtr = np.array([[by_q[q][i][f] for f in FEATS] for q in tr_q for i in range(len(by_q[q]))])
        ytr = np.array([r["y"] for q in tr_q for r in by_q[q]])
        gtr = np.array([len(by_q[q]) for q in tr_q])
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=120, learning_rate=0.05,
                           num_leaves=7, min_child_samples=40, reg_lambda=5.0,
                           label_gain=[0, 1], verbose=-1, random_state=78)
        m.fit(Xtr, ytr, group=gtr)
        for q in te_q:
            X = np.array([[r[f] for f in FEATS] for r in by_q[q]])
            s = m.predict(X)
            ranked = [by_q[q][i] for i in sorted(range(len(s)), key=lambda i: (-s[i],
                                                 by_q[q][i]["fork_rank"]))]
            perq["B"][q] = next((1.0 / i for i, r in enumerate(ranked[:25], 1) if r["y"]), 0.0)
        log(f"  fold {k}: w={best_w}, {len(te_q)} test queries")

    rng = np.random.default_rng(78)
    F = np.array([perq["fork"][q] for q in qs])
    L = [f"E78 — second-stage rerank of the fork's top 25, Class 2 arm", "",
         f"{len(qs)} queries, {FOLDS}-fold scaffold-grouped cross-validation", "",
         f"  fork's own order          {F.mean():.4f}",
         f"  ceiling (truth in the 25) {ceiling:.4f}   headroom {ceiling-F.mean():+.4f}", ""]
    for name, label in (("A", "blend: -rank - w*bde_rank"), ("B", "LambdaMART, 12 features")):
        if len(perq[name]) != len(qs):
            L.append(f"  {label:28s} incomplete ({len(perq[name])}/{len(qs)})"); continue
        V = np.array([perq[name][q] for q in qs])
        d = V - F
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L.append(f"  {label:28s} {V.mean():.4f}   vs fork {d.mean():+.4f} "
                 f"[{lo:+.4f}, {hi:+.4f}]  {'CLEARS ZERO' if lo > 0 else 'does not clear zero'}")
        L.append(f"     -> board equivalent {0.45*d.mean():+.4f} "
                 f"[{0.45*lo:+.4f}, {0.45*hi:+.4f}]   (Class 2 is 45% of the score)")
        L.append(f"     queries improved {int((d>0).sum())}, worsened {int((d<0).sum())}, "
                 f"unchanged {int((d==0).sum())}")
    L += ["", "+0.02 board needs +0.044 of Class 2 MRR here.", "",
          "CAVEAT: 100 queries. The comparison is paired, which is the only reason an effect this",
          "size could be visible at all, and the interval is still wide. A result that clears zero",
          "here is a reason to build the board version, not a measurement of what it will score.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E78_second_stage.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
