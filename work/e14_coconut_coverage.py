#!/usr/bin/env python3
"""E14 — COCONUT-2026-09 coverage of the competition's OWN training structures.

Seda's 99.6% used NPAtlas + KNApSAcK as the proxy, but COCONUT is partly built from those,
so the number is partly circular. The training structures weren't chosen by anyone building
COCONUT and include non-NP libraries, so this is a harder, more honest bar. Run locally
because the training data can't be shared (rules bar redistribution).

Match key: InChIKey first block (connectivity), the same key the data ships with. The scorer
additionally tautomer-canonicalises; this is noted, not applied (it could only raise coverage
where a tautomer pair was split, and costs ~1M canonicalisations).

PREDICTION (before the run, 2026-09-18): enveda-np-examples coverage >= 90% (Class 2 is by
definition "in a database", and those are the organisers' own NP examples); whole-training
coverage much lower (< 60%) because of the non-NP libraries.
RESULT: PASS both. enveda-np-examples 249/250 = 99.6%; all training 9.48% (enveda-180, 183k
structures, 0.03% in COCONUT, dominates the total).
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
COCO = os.path.expanduser("~/casmi-2026/work/data/coconut/coconut_csv_lite-09-2026.csv")
OUT = os.path.expanduser("~/casmi-2026/work/results/E14_coconut_coverage_2026-09-18.txt")

co = pd.read_csv(COCO, usecols=["standard_inchi_key"], dtype=str)
ck = set(co["standard_inchi_key"].dropna().str.split("-").str[0])
t = pq.read_table(DATA, columns=["inchikey14", "ingest_lib"]).to_pandas()
per = t.drop_duplicates(["inchikey14", "ingest_lib"])
per = per.assign(hit=per["inchikey14"].isin(ck))
uniq = t.drop_duplicates("inchikey14").assign(hit=lambda d: d["inchikey14"].isin(ck))
lines = [f"COCONUT 2026-09 lite: {len(co):,} rows, {len(ck):,} unique InChIKey14", "",
         f"ALL training structures: {uniq.hit.sum():,} / {len(uniq):,} = {uniq.hit.mean():.2%}", "",
         f"{'library':34s} {'structures':>11} {'in COCONUT':>11} {'share':>8}"]
for lib, g in per.groupby("ingest_lib"):
    lines.append(f"{lib:34s} {len(g):>11,} {g.hit.sum():>11,} {g.hit.mean():>8.2%}")
print("\n".join(lines)); open(OUT, "w").write("\n".join(lines) + "\n")
