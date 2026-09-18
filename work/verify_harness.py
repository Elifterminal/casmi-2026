#!/usr/bin/env python3
"""Positive control for harness.verify().

One of its five checks has already earned trust the hard way -- it caught class 1
queries that had consumed every spectrum of their own structure, silently demoting
them to class 2. The other four have never failed, so they are not yet evidence of
anything. Corrupt each one deliberately and confirm it is seen.
"""
import numpy as np, harness

df = harness.load_meta().reset_index(drop=True)
assign, qrows = harness.build(df, n_query=200, seed=1)
ref_rows, cand, _ = harness.apply_split(df, assign, qrows)

base = harness.verify(df, assign, qrows, ref_rows, cand)
print(f"clean split: {'PASSES' if not base else 'ALREADY BROKEN: ' + base[0]}")
assert not base
keys = df["inchikey14"].to_numpy()
groups = df.groupby("inchikey14").indices
ok = True

def check(name, fails, expect):
    global ok
    hit = any(expect in f for f in fails)
    ok &= hit
    print(f"  {'CAUGHT ' if hit else '!! MISSED'} {name}")
    if hit: print(f"            -> {fails[0][:95]}")

# 1. a class 2 structure's spectra sneak back into the reference index
k2 = next(k for k, c in assign.items() if c == 2)
bad = np.union1d(ref_rows, groups[k2])
check("class 2 spectra left in the reference index",
      harness.verify(df, assign, qrows, bad, cand), "still has spectra")

# 2. a class 3 structure left in the candidate database
k3 = next(k for k, c in assign.items() if c == 3)
check("class 3 structure left in the candidate DB",
      harness.verify(df, assign, qrows, ref_rows, cand | {k3}), "still in the candidate")

# 3. a class 2 structure missing from the candidate database
check("class 2 structure missing from the candidate DB",
      harness.verify(df, assign, qrows, ref_rows, cand - {k2}), "missing from the candidate")

# 4. a query spectrum itself left in the reference index
kq = next(iter(qrows))
leak = np.union1d(ref_rows, np.array(qrows[kq][:1]))
fails = harness.verify(df, assign, qrows, leak, cand)
hit = any("leaked" in f or "still has spectra" in f for f in fails)
ok &= hit
print(f"  {'CAUGHT ' if hit else '!! MISSED'} a query spectrum left in the reference index")
if hit: print(f"            -> {fails[0][:95]}")

# 5. the one that already caught a real bug, reproduced deliberately
k1 = next(k for k, c in assign.items() if c == 1)
strip = np.setdiff1d(ref_rows, groups[k1])
check("class 1 structure with no reference spectra left (the real bug)",
      harness.verify(df, assign, qrows, strip, cand), "NO reference spectra")

print("\nRESULT:", "every split check fires" if ok else "SOME SPLIT CHECKS ARE DEAD")
