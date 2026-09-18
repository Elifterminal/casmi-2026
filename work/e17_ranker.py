#!/usr/bin/env python3
"""E17 — a ranker with more to go on than "fraction of peaks explained".

E15/E16: in real natural-product pools (median ~20-40 look-alike candidates) the E07 score
can't separate the answer from its neighbours. E11 showed why adding fragments backfires:
every extra fragment mass is another chance to explain a peak by luck. E17 tests features that
attack that directly, each alone first, then a learned combination:

  explain       E07 fraction of top peaks explained (the baseline, E10's ranker)
  chance_corr   explain minus the fraction a RANDOM spectrum would get: the share of the mass
                axis the candidate's fragment windows cover. Penalises fragment-rich candidates.
  single        explain using single-bond cleavages only (the most plausible fragments)
  nb_max        neighbour vote: max over the query's top-50 spectral hits of
                cosine(query, hit) x Tanimoto(candidate, hit structure). A candidate that looks
                like the molecules whose spectra look like ours gets credit -- no spectrum of the
                candidate itself needed, which is the Class 2 situation.
  nb_mean10     the same, averaged over the top-10 hits (less noisy, less sharp)
  spec_self     the candidate's own best spectral cosine (non-zero only if it has spectra)
  learned       pairwise logistic regression over all of the above + heavy-atom count, trained
                on NP-only splits 20-22 (queries overlapping the eval splits removed), evaluated
                on NP-only splits 10-12.

All evaluation: NP-only splits 10-12, one retrieval DB (CLI, default coconut), mass candidates
ranked by the feature, then spectral-only candidates by key. Paired bootstrap vs explain.

PREDICTIONS (before the run, 2026-09-18):
  P1  chance_corr beats explain on C2 (it removes exactly the bias E11 exposed).
  P2  nb_max is the strongest single feature on C2.
  P3  learned beats every single feature, by >= +0.02 weighted over explain, CI clear of zero.
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import neutral_mass, MassIndex
from e07_fragment import fragment_masses, explain, MONO, H, TOL, MAX_HEAVY
from e08_unified import PPM, TOPN
from e11_ringbreak import _components
from e15_coconut_pools import load_structures
import scoring

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
TRAIN = ("npseed20_n300", "npseed21_n300", "npseed22_n300")
EVAL = ("npseed10_n300", "npseed11_n300", "npseed12_n300")
W = {1: 0.16, 2: 0.45, 3: 0.39}
NB_TOP = 50
FEATS = ["explain", "chance_corr", "single", "nb_max", "nb_mean10", "spec_self", "heavy"]
SINGLE = ["explain", "chance_corr", "single", "nb_max", "nb_mean10", "spec_self"]
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _cand_info(smi):
    """Everything per-candidate that doesn't depend on the query."""
    mol = Chem.MolFromSmiles(smi or "")
    if mol is None:
        return smi, None
    full = fragment_masses(mol)
    single = None
    if mol.GetNumHeavyAtoms() <= MAX_HEAVY:
        n = mol.GetNumAtoms()
        am = [MONO.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H for a in mol.GetAtoms()]
        adj = [[] for _ in range(n)]
        for b in mol.GetBonds():
            adj[b.GetBeginAtomIdx()].append((b.GetEndAtomIdx(), b.GetIdx()))
            adj[b.GetEndAtomIdx()].append((b.GetBeginAtomIdx(), b.GetIdx()))
        single = {sum(am)}
        for b in mol.GetBonds():
            if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing():
                for c in _components(n, adj, {b.GetIdx()}): single.add(sum(am[i] for i in c))
    return smi, dict(full=full, single=single, heavy=mol.GetNumHeavyAtoms(),
                     fp=DataStructs.BitVectToText(FPG.GetFingerprint(mol)))


def chance(frag, precursor_neutral):
    """Share of [0, M] covered by the candidate's +-1H match windows (capped at 1)."""
    if not frag or not np.isfinite(precursor_neutral) or precursor_neutral <= 0: return 0.0
    return min(1.0, 3 * len(frag) * 2 * TOL / precursor_neutral)


def build_queries(splits, db, st, excl=frozenset()):
    """Per query: candidate keys + features. Returns list of dicts."""
    keys, pmz, add, tr, co, _ = st
    trm, trs = dict(zip(tr.k, tr.m)), dict(zip(tr.k, tr.s))
    com, cos_ = dict(zip(co.k, co.m)), dict(zip(co.k, co.s))
    smi = {**cos_, **trs}
    out, info, fpo = [], {}, {}
    for split in splits:
        sp = json.load(open(f"{SPLITS}/split_{split}.json")); assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))}
        c3 = {k for k, c in assign.items() if c == 3}
        src = {"coconut": com, "union": {**com, **trm}, "train": trm}[db]
        dbm = {k: m for k, m in src.items() if k not in c3}
        idx = MassIndex(np.array(list(dbm)), np.array(list(dbm.values())))
        qs = []
        for k, rows in qrows.items():
            if k in excl: continue
            s = set(); Ms = []
            for r in rows:
                M = neutral_mass(pmz[r], add[r])
                if np.isfinite(M) and M > 0: s.update(idx.window(M, PPM).tolist()); Ms.append(M)
            qs.append((split, k, assign[k], s, float(np.median(Ms)) if Ms else np.nan, pools[k]))
        need = sorted({smi[c] for *_, s, _, _ in qs for c in s} | {smi.get(c, "") for *_, p in qs
                      for c in sorted(p["spec"], key=lambda c: -p["spec"][c])[:NB_TOP]} - set(info))
        with Pool(7) as pool:
            for sm, inf in pool.imap_unordered(_cand_info, need, chunksize=16): info[sm] = inf
        for split_, k, cls, s, M, p in qs:
            hits = sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
            hfp = [DataStructs.CreateFromBitString(info[smi.get(h, "")]["fp"]) for h, _ in hits
                   if info.get(smi.get(h, "")) is not None]
            hcos = np.array([c for h, c in hits if info.get(smi.get(h, "")) is not None])
            feats = {}
            for c in s:
                inf = info[smi[c]]
                if inf is None: continue
                e = explain(p["allmz"], p["allit"], inf["full"], p["positive"])
                if hfp:
                    tan = np.array(DataStructs.BulkTanimotoSimilarity(DataStructs.CreateFromBitString(inf["fp"]), hfp))
                    v = hcos * tan; nbm, nb10 = float(v.max()), float(v[:10].mean())
                else:
                    nbm = nb10 = 0.0
                feats[c] = dict(explain=e, chance_corr=e - chance(inf["full"], M),
                                single=explain(p["allmz"], p["allit"], inf["single"], p["positive"]),
                                nb_max=nbm, nb_mean10=nb10, spec_self=p["spec"].get(c, 0.0), heavy=inf["heavy"])
            spec_only = sorted(set(p["spec"]) - s)[:TOPN]
            out.append(dict(split=split_, k=k, cls=cls, truth=p["truth"], feats=feats, spec_only=spec_only,
                            smi={c: smi.get(c, "") for c in list(feats) + spec_only}))
        print(f"  [{split}] {len(qs)} queries, {len(info):,} candidates characterised", flush=True)
    return out


def rr(q, score):
    ranked = sorted(q["feats"], key=lambda c: (-score(q["feats"][c]), c)) + q["spec_only"]
    for i, c in enumerate(ranked[:TOPN], 1):
        if q["truth"] is not None and scoring.key14(q["smi"][c]) == q["truth"]: return 1.0 / i
    return 0.0


def fit(train_q):
    """Pairwise logistic regression: truth-minus-decoy feature differences."""
    X, y = [], []
    for q in train_q:
        tk = [c for c in q["feats"] if scoring.key14(q["smi"][c]) == q["truth"]]
        if not tk: continue
        t = np.array([q["feats"][tk[0]][f] for f in FEATS])
        for c, f in q["feats"].items():
            if c in tk: continue
            d = t - np.array([f[x] for x in FEATS]); X += [d, -d]; y += [1, 0]
    X = np.array(X); sd = X.std(0) + 1e-9
    m = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X / sd, y)
    w = m.coef_[0] / sd
    return w, len(y) // 2


def main(db="coconut"):
    t0 = time.time()
    st = load_structures()
    print(f"loaded {time.time()-t0:.0f}s", flush=True)
    ev = build_queries(EVAL, db, st)
    excl = frozenset(q["k"] for q in ev)
    tr = build_queries(TRAIN, db, st, excl)
    w, npairs = fit(tr)
    print(f"characterised {time.time()-t0:.0f}s; learned on {npairs:,} truth/decoy pairs", flush=True)

    scorers = {f: (lambda x, f=f: x[f]) for f in SINGLE}
    scorers["learned"] = lambda x: float(np.dot(w, [x[f] for f in FEATS]))
    per = [{"split": q["split"], "cls": q["cls"], **{n: rr(q, s) for n, s in scorers.items()}} for q in ev]

    L = [f"E17 ranker features — eval NP splits 10-12, DB={db}, learned on NP splits 20-22 "
         f"({len(excl)} eval query structures excluded from training)", "",
         "learned weights (raw feature units): " + ", ".join(f"{f} {v:+.3g}" for f, v in zip(FEATS, w)), "",
         f"{'ranker':12s} {'C1':>7} {'C2':>7} {'C3':>7} {'weighted':>9}  per split"]
    by = {c: [r for r in per if r["cls"] == c] for c in (1, 2, 3)}
    res = {}
    for n in scorers:
        c = [np.mean([r[n] for r in by[k]]) for k in (1, 2, 3)]
        wt = sum(W[k] * c[k - 1] for k in (1, 2, 3))
        splits = [sum(W[k] * np.mean([r[n] for r in by[k] if r["split"] == s]) for k in (1, 2, 3)) for s in EVAL]
        res[n] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=wt, splits=splits)
        L.append(f"{n:12s} {c[0]:>7.4f} {c[1]:>7.4f} {c[2]:>7.4f} {wt:>9.4f}  " + " / ".join(f"{x:.3f}" for x in splits))
    L += ["", "paired Δweighted vs explain (95% bootstrap CI, 2000 resamples within class):"]
    rng = np.random.default_rng(0)
    for n in scorers:
        if n == "explain": continue
        diff = lambda idx: sum(W[c] * np.mean([by[c][i][n] - by[c][i]["explain"] for i in idx[c]]) for c in by)
        full = diff({c: np.arange(len(by[c])) for c in by})
        b = [diff({c: rng.integers(0, len(by[c]), len(by[c])) for c in by}) for _ in range(2000)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        L.append(f"  {n:12s} {full:+.4f}  [{lo:+.4f}, {hi:+.4f}]{'   <- clears zero' if lo > 0 else ''}")
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E17_ranker_{db}_2026-09-18.txt", "w").write("\n".join(L) + "\n")
    json.dump(dict(db=db, weights=dict(zip(FEATS, map(float, w))), results=res),
              open(f"{RESULTS}/E17_ranker_{db}.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "coconut")
