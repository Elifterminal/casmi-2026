#!/usr/bin/env python3
"""E04 — can mass retrieval reach the answer at all?

For each held-out query we recover the neutral mass from the precursor m/z and the
adduct, pull every database structure inside a ppm window, and ask two questions:

  RECALL     is the true structure in that pool?
  POOL SIZE  how many decoys must a re-ranker beat?

Built-in control: Class 3 structures were deleted from the candidate database by the
harness, so their recall MUST be 0. If it isn't, the experiment is lying.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import formula_mass, neutral_mass, MassIndex, ADDUCT_SHIFT
import scoring

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "seed0_n300"
PPMS = [5, 10, 20, 50]

t0 = time.time()
sp = json.load(open(f"{SPLITS}/split_{SPLIT}.json"))
assign, qrows = sp["assign"], sp["query_rows"]

df = pq.read_table(DATA, columns=["inchikey14", "normalized_smiles", "molecular_formula",
                                  "precursor_mz", "adduct", "precursor_error_ppm",
                                  "ingest_lib"]).to_pandas()
print(f"loaded {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)

# --- how accurate are precursor masses in this data? picks the window -------
err = df["precursor_error_ppm"].dropna().to_numpy()
err = err[np.isfinite(err)]
print("\nprecursor_error_ppm (label-quality signal, ships uncleaned):")
for p in (50, 80, 90, 95, 99):
    print(f"  |err| p{p}: {np.percentile(np.abs(err), p):8.2f} ppm")

# --- structure database: one row per structure ------------------------------
struct = df.drop_duplicates("inchikey14")[["inchikey14", "molecular_formula", "normalized_smiles"]]
mass = struct["molecular_formula"].map(formula_mass).to_numpy()
print(f"\nstructures: {len(struct):,}, formula parsed for {np.isfinite(mass).sum():,} "
      f"({100*np.isfinite(mass).mean():.1f}%)")

c3 = {k for k, c in assign.items() if c == 3}
in_db = ~struct["inchikey14"].isin(c3).to_numpy()      # harness deleted class 3
idx = MassIndex(struct["inchikey14"].to_numpy()[in_db], mass[in_db])
print(f"candidate DB: {len(idx.keys):,} structures with a usable mass")

key2smiles = dict(zip(struct["inchikey14"], struct["normalized_smiles"]))

# Ranking prior. MUST come from the reference index, not from df.
# df still contains every held-out spectrum, so a class 2 molecule -- whose spectra
# the harness deleted -- would otherwise carry its full count and get ranked high
# for free. That is the leak this whole project exists to avoid, and it was sitting
# in the first draft of this file.
ref_rows = np.load(f"{SPLITS}/ref_rows_{SPLIT}.npy")
pop = df["inchikey14"].iloc[ref_rows].value_counts()
_c2 = [k for k, c in assign.items() if c == 2]
_leaked = sum(1 for k in _c2 if pop.get(k, 0) > 0)
print(f"\nleak check: class 2 structures with a non-zero reference-index prior: "
      f"{_leaked} of {len(_c2)} (must be 0)")
assert _leaked == 0, "class 2 structures still carry reference-index mass -- prior is leaking"

# The dataset's `inchikey14` is the PLAIN InChIKey14 of normalized_smiles. The
# competition additionally applies RDKit tautomer canonicalisation to BOTH sides
# before comparing, which shifts the key for the tautomer-ambiguous ~1.5%. So the
# two keys are different objects and our scorer's is the one that counts.
#
# Ranking on dataset keys is therefore CONSERVATIVE: a candidate whose plain key
# differs from the truth's but whose CANONICAL key matches would be scored correct
# by the contest and missed here. It can under-credit, never over-credit.
#
# Check the explanation rather than a bare count: every disagreement must be a case
# where the plain key equals the dataset key (i.e. it is canonicalisation, not rot).
from rdkit import Chem as _Chem
from rdkit.Chem import inchi as _inchi
_rng = np.random.default_rng(0)
_samp = struct.iloc[_rng.choice(len(struct), 400, replace=False)]
_dis, _explained = 0, 0
for kk, ss in zip(_samp["inchikey14"], _samp["normalized_smiles"]):
    if scoring.key14(ss) != kk:
        _dis += 1
        _m = _Chem.MolFromSmiles(ss)
        if _m and _inchi.MolToInchiKey(_m).split("-")[0] == kk:
            _explained += 1
print(f"scorer vs dataset key: {400-_dis}/400 agree; {_dis} differ, "
      f"{_explained} of those explained by tautomer canonicalisation")
assert _dis == _explained, "a key disagreement is NOT explained by canonicalisation"

pmz = df["precursor_mz"].to_numpy()
add = df["adduct"].to_numpy()
keys = df["inchikey14"].to_numpy()

# --- per query ---------------------------------------------------------------
rec = {p: defaultdict(list) for p in PPMS}
pool = {p: defaultdict(list) for p in PPMS}
mrr = {p: defaultdict(list) for p in PPMS}
no_adduct = 0

for k, rows in qrows.items():
    cls = assign[k]
    truth_smiles = key2smiles.get(k)
    for p in PPMS:
        cand = set()
        for r in rows:
            M = neutral_mass(pmz[r], add[r])
            if not np.isfinite(M) or M <= 0:
                continue
            cand.update(idx.window(M, p).tolist())
        if not cand:
            no_adduct += 1 if p == PPMS[0] else 0
        rec[p][cls].append(1.0 if k in cand else 0.0)
        pool[p][cls].append(len(cand))
        ranked = sorted(cand, key=lambda x: -pop.get(x, 0))[:25]
        rr_ = 0.0
        for i, c in enumerate(ranked, 1):
            if c == k:
                rr_ = 1.0 / i
                break
        mrr[p][cls].append(rr_)

print(f"\nqueries with no usable adduct/mass at the tightest window: {no_adduct}")
print("\n" + "=" * 74)
print("E04 — mass-window candidate retrieval")
print("=" * 74)
print(f"{'ppm':>5} {'class':>6} {'n':>5} {'recall':>8} {'median pool':>12} "
      f"{'mean pool':>10} {'MRR (popularity rank)':>22}")
for p in PPMS:
    for c in (1, 2, 3):
        n = len(rec[p][c])
        if not n:
            continue
        print(f"{p:>5} {c:>6} {n:>5} {np.mean(rec[p][c]):>8.3f} "
              f"{np.median(pool[p][c]):>12.0f} {np.mean(pool[p][c]):>10.0f} "
              f"{np.mean(mrr[p][c]):>22.4f}")
    print()

print("CONTROL — class 3 recall must be 0.000 at every window (the harness deleted them):")
bad = [p for p in PPMS if np.mean(rec[p][3]) > 0]
print("  ok  every window" if not bad else f"  !! FAIL at ppm {bad} — the experiment is lying")

out = {"split": SPLIT,
       "recall": {p: {c: float(np.mean(v)) for c, v in rec[p].items()} for p in PPMS},
       "median_pool": {p: {c: float(np.median(v)) for c, v in pool[p].items()} for p in PPMS},
       "mrr_popularity": {p: {c: float(np.mean(v)) for c, v in mrr[p].items()} for p in PPMS}}
json.dump(out, open(f"{RESULTS}/E04_recall_{SPLIT}.json", "w"), indent=2)
print(f"\ntotal {time.time()-t0:.0f}s")
