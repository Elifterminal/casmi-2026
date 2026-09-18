#!/usr/bin/env python3
"""Phase 1, step 0 — verify the premise before building anything on it.

The whole plan rests on two claims taken from a competitor's forum post that I
have never checked:

  A. The 250 `enveda-np-examples` structures ALSO appear in other training
     libraries, so holding out by `ingest_lib` does not hold out the structures.
     -> if false, the "contaminated CV" opening evaporates.
  B. The test set is roughly 16% / 45% / 39% Class 1 / 2 / 3.
     -> not checkable from here; the labels are hidden. Recorded as unverified.

Reads only the metadata columns. The peak arrays are the bulk of the file and
are not needed for this.
"""
import os, sys
import pyarrow.parquet as pq
import pandas as pd

DATA = os.path.expanduser("~/casmi-2026/work/data")
TRAIN = os.path.join(DATA, "train.parquet")
COLS = ["inchikey14", "inchikey", "normalized_smiles", "molecular_formula",
        "ingest_lib", "adduct", "precursor_mz", "precursor_error_ppm"]

print("=" * 72)
sch = pq.read_schema(TRAIN)
print(f"train.parquet: {os.path.getsize(TRAIN)/1e9:.2f} GB, {len(sch.names)} columns")
print("columns:", ", ".join(sch.names))
missing = [c for c in COLS if c not in sch.names]
if missing:
    print(f"!! expected columns absent: {missing}")
use = [c for c in COLS if c in sch.names]

df = pq.read_table(TRAIN, columns=use).to_pandas()
print(f"\nrows: {len(df):,}")
print(f"unique structures (inchikey14): {df['inchikey14'].nunique():,}")

print("\n--- null rates on the fields the harness needs ---")
for c in use:
    n = df[c].isna().sum()
    print(f"  {c:22s} {n:>9,} null  ({100*n/len(df):5.2f}%)")

print("\n--- ingest_lib distribution (vs the documented table) ---")
g = df.groupby("ingest_lib").agg(spectra=("inchikey14", "size"),
                                 structures=("inchikey14", "nunique"))
g = g.sort_values("spectra", ascending=False)
print(g.to_string())

# ---------------- CLAIM A ----------------
print("\n" + "=" * 72)
print("CLAIM A — do enveda-np-examples structures also live in other libraries?")
print("=" * 72)
NP = "enveda-np-examples"
if NP not in set(df["ingest_lib"].dropna()):
    sys.exit(f"!! '{NP}' not present in ingest_lib; cannot test the claim")

np_keys = set(df.loc[df["ingest_lib"] == NP, "inchikey14"].dropna())
other = df[df["ingest_lib"] != NP]
other_keys = set(other["inchikey14"].dropna())
overlap = np_keys & other_keys

print(f"\nstructures in {NP}: {len(np_keys)}")
print(f"of those, ALSO present under another ingest_lib: {len(overlap)} "
      f"({100*len(overlap)/len(np_keys):.1f}%)")
print(f"unique to {NP}: {len(np_keys - other_keys)}")

sub = other[other["inchikey14"].isin(overlap)]
by_lib = (sub.groupby("ingest_lib")["inchikey14"].nunique()
            .sort_values(ascending=False))
print("\nwhich libraries the leaked structures turn up in:")
print(by_lib.to_string())

spec = sub.groupby("inchikey14").size()
print(f"\nspectra available elsewhere for those structures: {len(sub):,}")
print(f"  median per structure: {spec.median():.0f}   max: {spec.max()}")

print("\nVERDICT ON CLAIM A:", end=" ")
frac = len(overlap) / len(np_keys)
if frac > 0.9:
    print(f"CONFIRMED — {100*frac:.1f}% leak. Holding out by ingest_lib is useless.")
elif frac > 0.5:
    print(f"PARTIALLY CONFIRMED — {100*frac:.1f}% leak. Weaker than claimed but real.")
else:
    print(f"NOT CONFIRMED — only {100*frac:.1f}% leak. The opening may not exist.")

# ---------------- general leakage ----------------
print("\n" + "=" * 72)
print("How much do libraries overlap in general? (bears on holdout design)")
per = df.groupby("inchikey14")["ingest_lib"].nunique()
print(f"  structures in exactly 1 library : {(per == 1).sum():,} ({100*(per==1).mean():.1f}%)")
print(f"  structures in 2+ libraries      : {(per >= 2).sum():,} ({100*(per>=2).mean():.1f}%)")
print(f"  max libraries for one structure : {per.max()}")
print("\n=> a holdout must remove a structure by inchikey14 across ALL libraries at once.")
