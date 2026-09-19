#!/usr/bin/env python3
"""E17b — why does E17's 'explain' baseline (0.2439) differ from E16's coconut number (0.2305)?

Same ranker, same splits, same DB. Two deliberate differences in E17:
  (a) ordering: E17 ranks ALL mass candidates before any spectral-only one; E10/E16 sort
      zero-score mass candidates and spectral-only candidates together by key.
  (b) E17 drops mass candidates whose SMILES RDKit can't parse; E16 keeps them (score 0).
Recompute E16's explain ranking under each combination to attribute the gap.
"""
import json, os, pickle, sys
from multiprocessing import Pool
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import neutral_mass, MassIndex
from e07_fragment import fragment_masses, explain
from e08_unified import PPM, TOPN
from e15_coconut_pools import load_structures
import scoring

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
W = {1: 0.16, 2: 0.45, 3: 0.39}


def _f(s):
    m = Chem.MolFromSmiles(s or "")
    return s, (m is not None), fragment_masses(m)


def main():
    keys, pmz, add, tr, co, _ = load_structures()
    trs = dict(zip(tr.k, tr.s)); com, cos_ = dict(zip(co.k, co.m)), dict(zip(co.k, co.s))
    smi = {**cos_, **trs}
    rows = []
    for split in ("npseed10_n300", "npseed11_n300", "npseed12_n300"):
        sp = json.load(open(f"{SPLITS}/split_{split}.json")); assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))}
        c3 = {k for k, c in assign.items() if c == 3}
        dbm = {k: m for k, m in com.items() if k not in c3}
        idx = MassIndex(np.array(list(dbm)), np.array(list(dbm.values())))
        cands = {}
        for k, rr in qrows.items():
            s = set()
            for r in rr:
                M = neutral_mass(pmz[r], add[r])
                if np.isfinite(M) and M > 0: s.update(idx.window(M, PPM).tolist())
            cands[k] = s
        need = sorted({smi[c] for s in cands.values() for c in s})
        with Pool(7) as pool: info = {s: (ok, f) for s, ok, f in pool.imap_unordered(_f, need, chunksize=16)}
        for k in qrows:
            p = pools[k]; sc = {c: explain(p["allmz"], p["allit"], info[smi[c]][1], p["positive"]) for c in cands[k]}
            ok = {c for c in sc if info[smi[c]][0]}
            so = sorted(set(p["spec"]) - cands[k])[:TOPN]
            def rr_of(lst):
                for i, c in enumerate(lst[:TOPN], 1):
                    if p["truth"] is not None and scoring.key14(smi.get(c, "")) == p["truth"]: return 1.0 / i
                return 0.0
            out = {"cls": assign[k]}
            for drop in (False, True):
                mc = {c: v for c, v in sc.items() if (c in ok or not drop)}
                inter = {**mc, **{c: 0.0 for c in so if c not in mc}}
                out[f"interleave,drop={drop}"] = rr_of(sorted(inter, key=lambda c: (-inter[c], c)))
                out[f"massfirst,drop={drop}"] = rr_of(sorted(mc, key=lambda c: (-mc[c], c)) + so)
            rows.append(out)
        print(f"[{split}] done", flush=True)
    for v in [x for x in rows[0] if x != "cls"]:
        w = sum(W[c] * np.mean([r[v] for r in rows if r["cls"] == c]) for c in (1, 2, 3))
        print(f"  {v:26s} weighted {w:.4f}")


if __name__ == "__main__":
    main()
