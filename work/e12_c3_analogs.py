#!/usr/bin/env python3
"""E12 — diagnostic: can Class 3 be reached by editing a known relative?

Class 3 molecules are in no database, so retrieval can't return them. The proposed route
is analog editing: take the structures behind the best spectral hits, apply the edit that
explains the precursor-mass gap, and rank the edited candidates. That only works if
(a) the best hits are structurally CLOSE to the truth, and (b) the mass gap is a SIMPLE
edit. This measures both before anything is built. No ranking, no score — a ceiling check.

PREDICTIONS (before the run, 2026-09-18):
  P1  Median Tanimoto(truth, best of top-5 spectral hits) >= 0.6 — the 0.96 cosines mean
      close relatives, not look-alikes.
  P2  >= 30% of C3 queries have a top-5 hit whose mass gap is one of the common edits below
      (incl. 0 = isomer).
  RESULT (2026-09-18, 351 queries): P1 FAIL — median best-of-5 Tanimoto 0.452; top-1 only
      0.260 despite ~0.96 cosine (high cosine != close structure). P2 PASS — 49.0% have a
      common-edit gap in the top 5. Reachable set (common edit AND Tanimoto >= 0.6): 24.2%.
"""
import json, os, pickle, sys
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import formula_mass

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
TOPK = 5
EDIT_TOL = 0.003     # Da
EDITS = {"isomer (0)": 0.0, "CH2": 14.01565, "O": 15.99491, "H2": 2.01565, "H2O": 18.01056,
         "CH2O": 30.01056, "C2H2O (acetyl)": 42.01056, "CO2": 43.98983, "CH2O2": 46.00548,
         "C5H8 (prenyl)": 68.06260, "C6H10O5 (hexose)": 162.05282, "C5H8O4 (pentose)": 132.04226,
         "C6H10O4 (deoxyhexose)": 146.05791, "C6H8O6 (glucuronide)": 176.03209,
         "SO3": 79.95682, "NH": 15.01090, "C2H4": 28.03130, "O2": 31.98983}
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def classify(delta):
    a = abs(delta)
    for name, m in EDITS.items():
        if abs(a - m) <= EDIT_TOL: return name
    return None


def main():
    t = pq.read_table(DATA, columns=["inchikey14", "normalized_smiles", "molecular_formula"])
    df = pd.DataFrame({"k": np.asarray(t.column("inchikey14")), "s": np.asarray(t.column("normalized_smiles")),
                       "f": np.asarray(t.column("molecular_formula"))}).drop_duplicates("k")
    smi = dict(zip(df.k, df.s)); mass = {k: formula_mass(f) for k, f in zip(df.k, df.f)}
    fpc = {}
    def fp(k):
        if k not in fpc:
            m = Chem.MolFromSmiles(smi.get(k, "") or "")
            fpc[k] = FPG.GetFingerprint(m) if m is not None else None
        return fpc[k]

    rows = []
    for split in ("seed0_n300", "seed1_n300", "seed2_n300"):
        for q in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb")):
            if q["cls"] != 3 or not q["spec"]: continue
            top = sorted(q["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:TOPK]
            ft = fp(q["k"])
            best_tan, best_edit, edits = 0.0, None, []
            for k, cos in top:
                f = fp(k)
                tan = DataStructs.TanimotoSimilarity(ft, f) if ft is not None and f is not None else 0.0
                e = classify(mass[q["k"]] - mass[k])
                edits.append(e)
                if tan > best_tan: best_tan = tan
            e1 = classify(mass[q["k"]] - mass[top[0][0]])
            rows.append(dict(split=split, k=q["k"], cos1=top[0][1], best_tan=best_tan,
                             tan1=(DataStructs.TanimotoSimilarity(ft, fp(top[0][0]))
                                   if ft is not None and fp(top[0][0]) is not None else 0.0),
                             edit1=e1, any_edit=any(e is not None for e in edits),
                             any_edit_close=any(e is not None and DataStructs.TanimotoSimilarity(ft, fp(k)) >= 0.6
                                                for (k, _), e in zip(top, edits) if ft is not None and fp(k) is not None)))
    n = len(rows)
    bt = np.array([r["best_tan"] for r in rows]); t1 = np.array([r["tan1"] for r in rows])
    lines = [f"E12 Class 3 analog diagnostic — {n} C3 queries over 3 splits, top-{TOPK} spectral hits", "",
             f"Tanimoto(truth, top-1 hit):        median {np.median(t1):.3f}   mean {t1.mean():.3f}",
             f"Tanimoto(truth, best of top-{TOPK}):   median {np.median(bt):.3f}   mean {bt.mean():.3f}",
             f"  share with best >= 0.8: {np.mean(bt>=0.8):.1%}   >= 0.6: {np.mean(bt>=0.6):.1%}   < 0.4: {np.mean(bt<0.4):.1%}",
             "", f"mass gap of top-1 hit is a common edit: {np.mean([r['edit1'] is not None for r in rows]):.1%}",
             f"any top-{TOPK} hit's gap is a common edit: {np.mean([r['any_edit'] for r in rows]):.1%}",
             f"  ...AND that hit has Tanimoto >= 0.6:  {np.mean([r['any_edit_close'] for r in rows]):.1%}   <- the reachable set",
             "", "top-1 edit breakdown:"]
    for e, c in Counter(r["edit1"] or "(none)" for r in rows).most_common(): lines.append(f"  {e:24s} {c:4d}  {c/n:.1%}")
    print("\n".join(lines))
    open(f"{RESULTS}/E12_c3_analogs_2026-09-18.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
