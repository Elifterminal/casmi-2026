#!/usr/bin/env python3
"""E14b — how much bigger do candidate pools get if we ship COCONUT instead of the training DB?

E10's 0.377 used pools drawn from the training structures (median 9 at 10 ppm). A real
submission retrieves from COCONUT (and/or training structures). E04/E05 showed bigger pools
hurt ranking, so this sizes the pools, for the true Class 2 masses, under each database.
"""
import json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import formula_mass

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
COCO = os.path.expanduser("~/casmi-2026/work/data/coconut/coconut_csv_lite-09-2026.csv")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
PPM = 10

co = pd.read_csv(COCO, usecols=["standard_inchi_key", "molecular_formula"], dtype=str).dropna()
co["k"] = co.standard_inchi_key.str.split("-").str[0]; co = co.drop_duplicates("k")
co["m"] = co.molecular_formula.map(formula_mass); co = co[np.isfinite(co.m)]
t = pq.read_table(DATA, columns=["inchikey14", "molecular_formula"]).to_pandas().drop_duplicates("inchikey14")
t["m"] = t.molecular_formula.map(formula_mass); t = t[np.isfinite(t.m)]
dbs = {"training": dict(zip(t.inchikey14, t.m)), "coconut": dict(zip(co.k, co.m))}
dbs["union"] = {**dbs["coconut"], **dbs["training"]}
sorted_m = {n: np.sort(np.fromiter(d.values(), float)) for n, d in dbs.items()}
tm = dict(zip(t.inchikey14, t.m))
lines = [f"unique structures: training {len(dbs['training']):,}  coconut {len(dbs['coconut']):,}  union {len(dbs['union']):,}",
         "(pool = structures within 10 ppm of the TRUE structure's mass; truth counted when present)", "",
         f"{'class':6s} {'db':9s} {'median pool':>12} {'mean':>8} {'p90':>8}"]
for cls in (1, 2):
    ms = []
    for s in ("seed0_n300", "seed1_n300", "seed2_n300"):
        sp = json.load(open(f"{SPLITS}/split_{s}.json"))
        ms += [tm[k] for k, c in sp["assign"].items() if c == cls and k in tm]
    ms = np.array(ms)
    for n, arr in sorted_m.items():
        tol = ms * PPM / 1e6
        cnt = np.searchsorted(arr, ms + tol, "right") - np.searchsorted(arr, ms - tol, "left")
        lines.append(f"C{cls:<5d} {n:9s} {np.median(cnt):>12.0f} {cnt.mean():>8.1f} {np.percentile(cnt, 90):>8.0f}")
print("\n".join(lines))
open(os.path.expanduser("~/casmi-2026/work/results/E14b_pool_size_2026-09-18.txt"), "w").write("\n".join(lines) + "\n")
