#!/usr/bin/env python3
"""E02 — the honest validation harness.

E01 established that holding out by `ingest_lib` does not hold out the structures:
250/250 of the enveda-np-examples set also live in 8 other libraries, with a median
of 168 spectra each. So every holdout here is by **inchikey14, across all libraries
at once**.

The three novelty classes are then MANUFACTURED rather than hoped for:

  Class 1  the molecule has other public spectra. Query spectra are held out, but
           OTHER spectra of the same structure stay in the reference index -- this
           mimics a fresh measurement of a known compound, not an exact-row match.
  Class 2  no public spectrum exists. EVERY spectrum of the structure is removed
           from the reference index; the structure stays in the candidate database.
  Class 3  not in any database. Spectra removed AND the structure removed from the
           candidate database.

Everything is seeded and written to disk so experiments are comparable.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
OUT = os.path.expanduser("~/casmi-2026/work/splits")
META = ["inchikey14", "normalized_smiles", "molecular_formula", "ingest_lib",
        "adduct", "precursor_mz", "instrument_type"]

# The test set is natural products on a Bruker timsTOF. enveda-180 is the wrong
# chemistry (synthetic drug-like screening compounds), so query molecules are drawn
# from the natural-product libraries.
NP_LIBS = {"riken", "gnps", "mona", "massbank", "msdial", "spectraverse",
           "enveda-np-examples", "masaryk"}
MASS_LO, MASS_HI = 157.0, 1159.0     # observed range of the real test set
MAX_Q_SPECTRA = 16                   # test has 1-16 spectra per molecule, median 3
CLASS_SHARES = (0.16, 0.45, 0.39)    # UNVERIFIED -- see scoring.weighted_mrr


def load_meta():
    return pq.read_table(DATA, columns=META).to_pandas()


def build(df, n_query=900, seed=0):
    rng = np.random.default_rng(seed)

    # candidate query structures: NP-ish, in the test's mass range, not absurdly
    # over-represented, and with enough spectra to split query-vs-reference for class 1.
    # Vectorised: a per-row bool column beats a python lambda over every group.
    df = df.reset_index(drop=True)
    df["_np"] = df["ingest_lib"].isin(NP_LIBS)
    per = df.groupby("inchikey14").agg(
        n=("ingest_lib", "size"),
        np_hits=("_np", "sum"),
        mz=("precursor_mz", "median"))
    eligible = per[(per.np_hits >= 1) & (per.mz.between(MASS_LO, MASS_HI)) & (per.n >= 2)]
    print(f"eligible query structures: {len(eligible):,} of {len(per):,}")

    keys = rng.permutation(eligible.index.values)[:n_query]
    n1 = int(round(n_query * CLASS_SHARES[0]))
    n2 = int(round(n_query * CLASS_SHARES[1]))
    cls = np.array([1] * n1 + [2] * n2 + [3] * (n_query - n1 - n2))
    rng.shuffle(cls)
    assign = dict(zip(keys, cls.tolist()))

    # pick which spectra form each query (the rest of that structure's spectra are
    # only ever visible for class 1).
    # groupby(...).indices gives key -> row positions in ONE pass; the previous
    # version rebuilt a 2.5m-row boolean mask per structure, which is quadratic.
    groups = df.groupby("inchikey14").indices
    query_rows = {}
    for k in keys:
        rows = groups[k]
        cap = min(MAX_Q_SPECTRA, len(rows))
        if assign[k] == 1:
            # A class 1 molecule MUST keep at least one spectrum in the reference
            # index -- that is what makes it class 1. Taking all of them silently
            # demotes it to class 2 and the harness would then report class 1
            # performance while measuring class 2. Caught by verify() on the first
            # run: 7 of 48 class 1 queries had been fully consumed.
            cap = min(cap, len(rows) - 1)
        take = int(rng.integers(1, cap + 1))
        query_rows[k] = rng.choice(rows, size=take, replace=False).tolist()

    return assign, query_rows


def apply_split(df, assign, query_rows):
    """Return (reference_index_rows, candidate_structures, queries)."""
    q_all = {r for rows in query_rows.values() for r in rows}
    c23 = {k for k, c in assign.items() if c in (2, 3)}
    c3 = {k for k, c in assign.items() if c == 3}

    in_q = np.zeros(len(df), dtype=bool)
    if q_all:
        in_q[np.fromiter(q_all, dtype=np.int64)] = True
    is_c23 = df["inchikey14"].isin(c23).to_numpy()
    ref_mask = (~in_q) & (~is_c23)            # class 1 keeps its OTHER spectra

    cand = set(df["inchikey14"].unique()) - c3
    queries = [{"key": k, "cls": assign[k], "rows": query_rows[k]} for k in assign]
    return np.flatnonzero(ref_mask), cand, queries


def verify(df, assign, query_rows, ref_rows, cand):
    """Positive control. A split that cannot be wrong proves nothing, so check it."""
    fails = []
    ref_keys = set(df["inchikey14"].to_numpy()[ref_rows])
    q_all = {r for rows in query_rows.values() for r in rows}

    for k, c in assign.items():
        present = k in ref_keys
        if c in (2, 3) and present:
            fails.append(f"class {c} structure {k} still has spectra in the reference index")
        if c == 1 and not present:
            fails.append(f"class 1 structure {k} has NO reference spectra (should keep its others)")
        if c == 3 and k in cand:
            fails.append(f"class 3 structure {k} is still in the candidate database")
        if c in (1, 2) and k not in cand:
            fails.append(f"class {c} structure {k} is missing from the candidate database")
    if q_all & set(ref_rows.tolist()):
        fails.append("a query spectrum leaked into the reference index")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    print("loading metadata ...", flush=True)
    df = load_meta().reset_index(drop=True)
    print(f"  {len(df):,} spectra, {df['inchikey14'].nunique():,} structures")

    assign, query_rows = build(df, a.n_query, a.seed)
    ref_rows, cand, queries = apply_split(df, assign, query_rows)

    counts = pd.Series([c for c in assign.values()]).value_counts().sort_index()
    print("\nclass assignment:")
    for c, n in counts.items():
        print(f"  class {c}: {n:>4} queries ({100*n/len(assign):.1f}%)")
    print(f"\nreference index : {len(ref_rows):,} spectra "
          f"({100*len(ref_rows)/len(df):.1f}% of train)")
    print(f"candidate DB    : {len(cand):,} structures")
    nq = sum(len(v) for v in query_rows.values())
    print(f"query spectra   : {nq:,} over {len(query_rows)} molecules "
          f"(median {int(np.median([len(v) for v in query_rows.values()]))} each)")

    print("\n--- verifying the split does what it claims ---")
    fails = verify(df, assign, query_rows, ref_rows, cand)
    if fails:
        print(f"  {len(fails)} PROBLEM(S):")
        for f in fails[:10]:
            print("   x", f)
        sys.exit(1)
    print("  ok  class 2/3 structures have ZERO spectra anywhere in the reference index")
    print("  ok  class 1 structures kept their other spectra")
    print("  ok  class 3 structures removed from the candidate database")
    print("  ok  class 1/2 structures present in the candidate database")
    print("  ok  no query spectrum leaked into the reference index")

    os.makedirs(OUT, exist_ok=True)
    tag = f"seed{a.seed}_n{a.n_query}"
    with open(f"{OUT}/split_{tag}.json", "w") as fh:
        json.dump({"seed": a.seed, "n_query": a.n_query,
                   "class_shares_UNVERIFIED": CLASS_SHARES,
                   "assign": {k: int(v) for k, v in assign.items()},
                   "query_rows": {k: [int(r) for r in v] for k, v in query_rows.items()}},
                  fh)
    np.save(f"{OUT}/ref_rows_{tag}.npy", np.asarray(ref_rows))
    print(f"\nwrote {OUT}/split_{tag}.json and ref_rows_{tag}.npy")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
