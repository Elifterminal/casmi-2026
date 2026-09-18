#!/usr/bin/env python3
"""E03 — a public-baseline-class pipeline, measured on our own harness.

WHAT THIS IS: entropy-weighted spectral cosine against a reference index, plus
mass-shifted analog propagation, plus a popularity prior to fill empty slots.

WHAT THIS IS NOT, corrected after the first run:
  * NOT a reproduction of haideptry's published notebook. That has a trained torch
    spectrum->fingerprint reranker we have not trained.
  * NOT a Class 2 pipeline at all. An earlier version of this docstring claimed
    "mass-window retrieval from a candidate database". THAT WAS FALSE -- the
    candidate DB produced by the harness is never loaded or consulted here. The
    only `cand` in this file is a local variable for inverted-index postings.
    Class 2 and Class 3 queries are therefore treated IDENTICALLY by this code:
    both have had their spectra stripped from the reference index, and the
    candidate-DB distinction between them is never used.

So this measures what pure spectral retrieval achieves. It is a floor, and it is a
Class 1 machine with an analog tail.

Retrieval is done with an inverted index over binned peaks: comparing 1,559 queries
against 2.53m reference spectra all-pairs is ~4bn cosines, so candidates are first
gathered by shared high-intensity peak bins and only those are scored exactly.
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")

BIN = 0.02            # Da; peaks within ~0.01 Da co-bin
INT_FLOOR = 0.002     # drop peaks below 0.2% of base peak (as the public baseline does)
MAX_PEAKS = 256
IDX_PEAKS = 20        # peaks per spectrum entered into the inverted index
PPM = 10.0            # precursor window for the exact-match channel
ANALOG_DA = 200.0     # mass-shift window for analog propagation
N_RESCORE = 1500      # candidates scored exactly per query spectrum
TOPN = 25


def binize(mz):
    return np.rint(np.asarray(mz) / BIN).astype(np.int32)


def prep(mzs, ints):
    """Intensity floor, top-N cap, entropy-ish weighting. Returns (bins, weights)."""
    if len(mzs) == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    ints = np.asarray(ints, dtype=np.float32)
    mzs = np.asarray(mzs, dtype=np.float64)
    m = ints.max()
    if m <= 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    keep = ints >= INT_FLOOR * m
    mzs, ints = mzs[keep], ints[keep]
    if len(ints) > MAX_PEAKS:
        top = np.argpartition(-ints, MAX_PEAKS)[:MAX_PEAKS]
        mzs, ints = mzs[top], ints[top]
    b = binize(mzs)
    w = ints / (np.linalg.norm(ints) + 1e-12)
    return b, w.astype(np.float32)


def cosine(qb, qw, rb, rw):
    """Cosine over shared bins. Both sides are L2-normalised."""
    if len(qb) == 0 or len(rb) == 0:
        return 0.0
    qo = np.argsort(qb); ro = np.argsort(rb)
    qb, qw = qb[qo], qw[qo]; rb, rw = rb[ro], rw[ro]
    i = j = 0; s = 0.0
    while i < len(qb) and j < len(rb):
        if qb[i] == rb[j]:
            s += qw[i] * rw[j]; i += 1; j += 1
        elif qb[i] < rb[j]:
            i += 1
        else:
            j += 1
    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="seed0_n300")
    a = ap.parse_args()
    t0 = time.time()

    sp = json.load(open(f"{SPLITS}/split_{a.split}.json"))
    assign = sp["assign"]; qrows = {k: v for k, v in sp["query_rows"].items()}
    ref_rows = np.load(f"{SPLITS}/ref_rows_{a.split}.npy")
    print(f"split {a.split}: {len(assign)} queries, {len(ref_rows):,} reference spectra", flush=True)

    print("loading spectra ...", flush=True)
    t = pq.read_table(DATA, columns=["inchikey14", "normalized_smiles", "precursor_mz",
                                     "ms2_mzs", "ms2_normalized_intensities", "ingest_lib"])
    keys = np.asarray(t.column("inchikey14"))
    smiles = np.asarray(t.column("normalized_smiles"))
    pmz = t.column("precursor_mz").to_numpy()

    # Flat values + offsets, not to_pylist(). to_pylist() on 401m peaks materialises
    # a Python float per peak and blew past 21 GB before being killed.
    def flat(col):
        arr = t.column(col).combine_chunks()
        return (arr.values.to_numpy(zero_copy_only=False).astype(np.float32),
                arr.offsets.to_numpy().astype(np.int64))
    mz_flat, mz_off = flat("ms2_mzs")
    it_flat, it_off = flat("ms2_normalized_intensities")
    del t
    print(f"  {len(mz_flat):,} peaks held as flat float32 "
          f"({(mz_flat.nbytes + it_flat.nbytes)/1e9:.2f} GB)", flush=True)

    def peaks(i):
        return mz_flat[mz_off[i]:mz_off[i+1]], it_flat[it_off[i]:it_off[i+1]]
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    ref_set = ref_rows
    q_all = sorted({r for v in qrows.values() for r in v})

    # ---- preprocess only what we touch -------------------------------------
    print("preprocessing reference spectra ...", flush=True)
    prepped = {}
    for r in ref_set:
        prepped[r] = prep(*peaks(r))
    for r in q_all:
        prepped[r] = prep(*peaks(r))
    print(f"  {len(prepped):,} spectra prepped, {time.time()-t0:.0f}s", flush=True)

    # ---- inverted index over the most intense bins --------------------------
    print("building inverted index ...", flush=True)
    inv = defaultdict(list)
    for r in ref_set:
        b, w = prepped[r]
        if len(b) == 0:
            continue
        top = np.argsort(-w)[:IDX_PEAKS]
        for bb in np.unique(b[top]):
            inv[int(bb)].append(r)
    inv = {k: np.asarray(v, dtype=np.int64) for k, v in inv.items()}
    print(f"  {len(inv):,} bins, {sum(len(v) for v in inv.values()):,} postings, "
          f"{time.time()-t0:.0f}s", flush=True)

    ref_pmz = pmz[ref_set]
    order = np.argsort(ref_pmz)
    ref_sorted = ref_set[order]; pmz_sorted = ref_pmz[order]

    # popularity prior: how often a structure shows up (a crude stand-in for the
    # public baselines' natural-product prior)
    pop = pd.Series(keys[ref_set]).value_counts()

    # key -> a representative SMILES, built once. Looking this up with
    # np.flatnonzero(keys == k) per guess is a 2.5m-row scan, ~19bn comparisons.
    key2smiles = {}
    for kk, ss in zip(keys, smiles):
        if kk not in key2smiles:
            key2smiles[kk] = ss

    # ---- per query ----------------------------------------------------------
    per_class = defaultdict(list)
    per_query = []          # raw reciprocal ranks -- E03 could only estimate the
                            # class 2 vs 3 comparison from means without these
    for n, (k, rows) in enumerate(qrows.items(), 1):
        cls = assign[k]
        truth = smiles[rows[0]]
        scores = defaultdict(float)

        for r in rows:
            qb, qw = prepped[r]
            if len(qb) == 0:
                continue
            # gather candidates sharing high-intensity bins
            top = np.argsort(-qw)[:IDX_PEAKS]
            cand = [inv[int(bb)] for bb in np.unique(qb[top]) if int(bb) in inv]
            if not cand:
                continue
            cand = np.concatenate(cand)
            uniq, cnt = np.unique(cand, return_counts=True)
            take = uniq[np.argsort(-cnt)[:N_RESCORE]]

            qmz = pmz[r]
            for c in take:
                rb, rw = prepped[c]
                sim = cosine(qb, qw, rb, rw)
                if sim <= 0:
                    continue
                dm = abs(pmz[c] - qmz)
                if dm <= qmz * PPM / 1e6:
                    scores[keys[c]] += sim ** 3          # exact-precursor channel
                elif dm <= ANALOG_DA:
                    scores[keys[c]] += 0.35 * sim ** 3   # mass-shifted analog channel

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:TOPN]
        guesses = []
        seen = set()
        for kk, _ in ranked:
            if kk not in seen and kk in key2smiles:
                guesses.append(key2smiles[kk]); seen.add(kk)

        # pad with the popularity prior (a wrong guess costs nothing but its slot)
        for kk in pop.index:
            if len(guesses) >= TOPN:
                break
            if kk not in seen and kk in key2smiles:
                guesses.append(key2smiles[kk]); seen.add(kk)

        r_ = scoring.rr(truth, guesses)
        per_class[cls].append(r_)
        per_query.append({"key": k, "cls": cls, "rr": r_})
        if n % 25 == 0:
            print(f"  {n}/{len(qrows)} queries, {time.time()-t0:.0f}s", flush=True)

    print("\n" + "=" * 60)
    print("E03 — baseline-class pipeline, measured on our harness")
    print("=" * 60)
    out = {}
    for c in (1, 2, 3):
        rs = per_class.get(c, [])
        m = float(np.mean(rs)) if rs else 0.0
        out[c] = m
        print(f"  Class {c}: MRR {m:.4f}   (n={len(rs)})")
    w = scoring.weighted_mrr(out)
    print(f"\n  weighted (16/45/39, UNVERIFIED): {w:.4f}")
    print(f"  published baseline public LB      : 0.339")
    print(f"  total runtime {time.time()-t0:.0f}s")
    json.dump({"per_class": out, "weighted": w, "split": a.split,
               "per_query": per_query},
              open(f"{os.path.dirname(SPLITS)}/results/E03_baseline_{a.split}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
