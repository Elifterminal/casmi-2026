#!/usr/bin/env python3
"""E60 — hold the ranker out at SCAFFOLD level, and see how much of our local score was borrowed.

Our local yardstick has read about 0.04-0.08 high all project. The direction of our measurements
has mostly been right -- local said the union would hurt and it did, local said per-frame would
hurt and it did -- but the LEVEL has never matched the board, which makes it useless for comparing
against other teams' numbers and dangerous for deciding when something is "good enough".

senanuretin published the fix, having measured that his own validation set tracks his board score
within 0.01: hold out at SCAFFOLD level, not InChIKey, "or V-real's structures leak into training
through their tautomers". Our harness is already tautomer-safe -- we hold out at the scorer's key,
which canonicalises tautomers -- but not scaffold-safe.

MEASURED FIRST, because a fix needs a leak to fix: 224 of our 864 evaluation queries (25.9%) share
a Murcko scaffold with a query the ranker trains on. For a quarter of our evaluation the ranker has
already seen the skeleton.

WHAT THIS DOES AND DOES NOT DO. It holds out RANKER TRAINING, not the spectral library. Removing
same-scaffold relatives from the library would make the task harder than reality -- a real Class 2
molecule usually does have relatives with published spectra, and exploiting them is exactly what
analog propagation legitimately does. The leak is that the ranker learns scaffold-specific
patterns, not that relatives exist.

  arm A  current: training queries whose IDENTITY matches an eval query are dropped (E17)
  arm B  scaffold: additionally drop training queries whose MURCKO SCAFFOLD matches an eval query's

LIMITATION, stated rather than hidden: 215 of 864 eval queries (24.9%) are acyclic and have no
Murcko scaffold. Scaffold holdout cannot protect those, so arm B is an improvement on three
quarters of the set, not all of it.

PREDICTIONS (before the run, 2026-09-26):
  P1  Arm B scores LOWER than arm A. We are giving up an advantage we should never have had.
  P2  The drop is 0.01-0.03 weighted. Smaller than the 0.04 level gap to the board, because
      scaffold leakage is one contributor and famousness (E46, worth 0.064) is another.
  P3  The drop is concentrated in Class 2, where the neighbour vote does the work and a familiar
      scaffold is worth most.
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import build, FEATS2, W
from casmi_pipeline import key14, TOPN

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
EVAL = ("npxmtsseed60_n300", "npxmtsseed61_n300", "npxmtsseed62_n300")
TRAIN = ("nptsseed20_n300", "nptsseed21_n300", "nptsseed22_n300")
SCAF_CACHE = os.path.join(SPLITS, "murcko_scaffolds.json")


def _scaf(s):
    try:
        m = Chem.MolFromSmiles(s or "")
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else None
    except Exception:
        return None


def scaffolds_for(smis, log):
    cache = json.load(open(SCAF_CACHE)) if os.path.exists(SCAF_CACHE) else {}
    need = sorted(set(smis) - set(cache))
    if need:
        with Pool(7) as p:
            for s, v in zip(need, p.map(_scaf, need, chunksize=64)): cache[s] = v
        json.dump(cache, open(SCAF_CACHE, "w"))
        log(f"scaffolds computed for {len(need):,} new structures ({len(cache):,} cached)")
    return cache


def fit_w(qs, feats):
    X, y = [], []
    for q in qs:
        tk = [c for c, (_, s) in q["cand"].items() if key14(s) == q["truth"]]
        if not tk: continue
        t = np.array([q["fv"][tk[0]][f] for f in feats])
        for c in q["cand"]:
            if c in tk: continue
            d = t - np.array([q["fv"][c][f] for f in feats]); X += [d, -d]; y += [1, 0]
    if not X: return np.zeros(len(feats)), 0
    X = np.array(X); sd = X.std(0) + 1e-9
    m = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X / sd, y)
    return m.coef_[0] / sd, len(y) // 2


def rr(q, w, feats):
    sc = {c: float(np.dot(w, [q["fv"][c][f] for f in feats])) for c in q["cand"]}
    order = [q["cand"][c][1] for c in sorted(sc, key=lambda c: (-sc[c], str(c)))] + q["tail"]
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
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["inchikey14", "normalized_smiles"])
    smi_of_key = {}
    for k, s in zip(np.asarray(t.column("inchikey14")), np.asarray(t.column("normalized_smiles"))):
        smi_of_key.setdefault(k, s)
    del t
    st = load_structures()
    ev = build(EVAL, st, log=log, pools_prefix="fr_", generate=False)
    tr = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log, generate=False)
    log(f"{len(ev)} eval, {len(tr)} train queries (identity holdout already applied)")

    for q in ev + tr:
        q["fv"] = {c: {n: float(v) for n, v in zip(FEATS2, vec)} for c, (vec, _) in q["cand"].items()}

    cache = scaffolds_for([smi_of_key.get(q["k"], "") for q in ev + tr], log)
    ev_scaf = {cache.get(smi_of_key.get(q["k"], "")) for q in ev}
    ev_scaf.discard(None); ev_scaf.discard("")
    tr_clean = [q for q in tr if cache.get(smi_of_key.get(q["k"], "")) not in ev_scaf]
    dropped = len(tr) - len(tr_clean)
    log(f"scaffold holdout drops {dropped}/{len(tr)} training queries ({dropped/len(tr):.1%})")

    arms = {"A: identity holdout (current)": tr, "B: + scaffold holdout": tr_clean}
    per, byid, out = {}, {}, {}
    for name, trq in arms.items():
        w, npairs = fit_w(trq, FEATS2)
        p = defaultdict(list); bid = defaultdict(list)
        for q in ev:
            s = rr(q, w, FEATS2)
            p[q["cls"]].append(s); bid[q["truth"] or q["k"]].append((q["cls"], s))
        per[name] = p; byid[name] = bid
        wt = sum(W[c] * float(np.mean(p[c])) for c in (1, 2, 3) if p[c])
        out[name] = dict(weighted=wt, n_train=len(trq), pairs=npairs,
                         **{f"C{c}": float(np.mean(p[c])) if p[c] else None for c in (1, 2, 3)})
        log(f"  {name}: {len(trq)} train queries, {npairs:,} pairs -> weighted {wt:.4f}")

    a, b = list(arms)
    L = ["E60 — scaffold-level ranker holdout: how much of our local score was borrowed?", "",
         f"scaffold leakage before the fix: 224/864 eval queries (25.9%) shared a Murcko scaffold "
         f"with a training query", f"this arm drops {dropped}/{len(tr)} training queries "
         f"({dropped/len(tr):.1%})", "",
         f"{'arm':32s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9} {'vs A':>9}"]
    for n in arms:
        o = out[n]
        L.append(f"{n:32s} {o['C1']:>8.4f} {o['C2']:>8.4f} {o['C3']:>8.4f} {o['weighted']:>9.4f} "
                 f"{o['weighted']-out[a]['weighted']:>+9.4f}")
    rng = np.random.default_rng(0)
    idl = sorted(set(byid[a]) | set(byid[b]))
    boots = []
    for _ in range(3000):
        pick = rng.integers(0, len(idl), len(idl))
        x, y = defaultdict(list), defaultdict(list)
        for j in pick:
            for cls, s in byid[a].get(idl[j], []): x[cls].append(s)
            for cls, s in byid[b].get(idl[j], []): y[cls].append(s)
        boots.append(sum(W[c] * (np.mean(y[c]) - np.mean(x[c])) for c in (1, 2, 3) if x[c] and y[c]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    L += ["", f"scaffold holdout effect: {out[b]['weighted']-out[a]['weighted']:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]",
          "", "reading: our board score for this pipeline family is 0.288. Arm B is the honest local",
          "number -- if it lands closer to 0.288 than arm A does, scaffold leakage was part of the",
          "level gap, and what remains is attributable to famousness (E46) and to whatever else we",
          "have not found.",
          "", "LIMITATION: 215 of 864 eval queries (24.9%) are acyclic and have no Murcko scaffold,",
          "so this protects three quarters of the set, not all of it.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E60_scaffold_2026-09-26.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E60_scaffold.json", "w"), indent=2)


if __name__ == "__main__":
    main()
