#!/usr/bin/env python3
"""E05 — rank the ~9 mass-window candidates for Class 2.

E04 established the true structure is in a ~9-candidate pool 92% of the time, and
that popularity-only ranking already scores C2 MRR 0.2566. E05 asks: does cheap,
non-neural chemistry beat that popularity baseline?

Features scored per candidate (no model yet -- a transparent weighted sum, so any
lift is attributable to a specific signal, not a black box):
  * popularity   reference-index frequency (the E04 baseline, kept as one channel)
  * fragexplain  fraction of the query's intense peaks whose mass matches a
                 substructure/loss of THIS candidate -- the one real spectral signal
  * nplike       ring count + oxygen density, a crude natural-product prior
  * massfit      closeness of candidate mass to the adduct-corrected precursor

Reported against the E04 popularity floor (0.2566) on the same split. If nothing
beats popularity, that is a real (negative) result and gets recorded as one.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import formula_mass, neutral_mass, MassIndex

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "seed0_n300"
PPM = 10
PROTON = 1.00727646688
# common neutral losses (Da) seen in NP fragmentation
LOSSES = [0.0, 18.0106, 28.0313, 44.0262, 46.0055, 42.0106, 162.0528,
          146.0579, 132.0423, 176.0321, 15.9949, 17.0265]


def frag_explain(qmz, qint, cand_mass, tol=0.02):
    """Fraction of the query's top peaks explainable as a fragment of a molecule of
    this mass. A weak proxy: real fragment enumeration needs the structure, but a
    peak above the candidate's own mass cannot come from it, and losses from the
    precursor are the commonest explained peaks."""
    if len(qmz) == 0:
        return 0.0
    order = np.argsort(-qint)[:20]
    mz, it = qmz[order], qint[order]
    prec = cand_mass + PROTON
    explained = 0.0
    wsum = 0.0
    for m, i in zip(mz, it):
        wsum += i
        if m > prec + 2 * tol:          # heavier than precursor -> not from this
            continue
        if any(abs(m - (prec - L)) <= tol for L in LOSSES):
            explained += i
    return explained / wsum if wsum > 0 else 0.0


def np_like(mol):
    if mol is None:
        return 0.0
    rings = rdMolDescriptors.CalcNumRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    return min(rings, 6) / 6.0 * 0.5 + min(o / heavy * 4, 1.0) * 0.5


def main():
    t0 = time.time()
    sp = json.load(open(f"{SPLITS}/split_{SPLIT}.json"))
    assign, qrows = sp["assign"], sp["query_rows"]

    cols = ["inchikey14", "normalized_smiles", "molecular_formula", "precursor_mz",
            "adduct", "ms2_mzs", "ms2_normalized_intensities"]
    t = pq.read_table(DATA, columns=cols)
    keys = np.asarray(t.column("inchikey14"))
    smiles = np.asarray(t.column("normalized_smiles"))
    formula = np.asarray(t.column("molecular_formula"))
    pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct"))

    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities")
    del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    print(f"loaded, {time.time()-t0:.0f}s", flush=True)

    ref_rows = np.load(f"{SPLITS}/ref_rows_{SPLIT}.npy")
    pop = pd.Series(keys[ref_rows]).value_counts()

    struct = pd.DataFrame({"k": keys, "f": formula, "s": smiles}).drop_duplicates("k")
    mass_by_key = {}
    smiles_by_key = {}
    for k_, f_, s_ in zip(struct["k"], struct["f"], struct["s"]):
        mass_by_key[k_] = formula_mass(f_)
        smiles_by_key[k_] = s_
    c3 = {k for k, c in assign.items() if c == 3}
    dbk = np.array([k for k in struct["k"] if k not in c3 and np.isfinite(mass_by_key[k])])
    idx = MassIndex(dbk, np.array([mass_by_key[k] for k in dbk]))
    mol_cache = {}
    def mol(k):
        if k not in mol_cache:
            mol_cache[k] = Chem.MolFromSmiles(smiles_by_key.get(k, "")) 
        return mol_cache[k]
    print(f"index ready, {time.time()-t0:.0f}s", flush=True)

    # weight sets to compare. popularity-only reproduces the E04 baseline.
    SCHEMES = {
        "popularity_only": dict(pop=1.0, frag=0.0, nplike=0.0, mass=0.0),
        "frag_only":       dict(pop=0.0, frag=1.0, nplike=0.0, mass=0.0),
        "pop+frag":        dict(pop=0.5, frag=1.0, nplike=0.0, mass=0.0),
        "pop+frag+np":     dict(pop=0.5, frag=1.0, nplike=0.3, mass=0.0),
        "all":             dict(pop=0.5, frag=1.0, nplike=0.3, mass=0.5),
    }
    rr = {s: defaultdict(list) for s in SCHEMES}
    rr["RANDOM"] = defaultdict(list)      # control: shuffle the pool, average 20 draws
    exp_random = defaultdict(list)        # analytic E[RR] for a uniform-random rank

    import random as _random
    _rng = _random.Random(0)

    for k, rows in qrows.items():
        cls = assign[k]
        # merge the query's peaks (evidence fusion across its spectra)
        allmz = np.concatenate([peaks(r)[0] for r in rows]) if rows else np.empty(0)
        allit = np.concatenate([peaks(r)[1] for r in rows]) if rows else np.empty(0)
        cand = set()
        for r in rows:
            M = neutral_mass(pmz[r], add[r])
            if np.isfinite(M) and M > 0:
                cand.update(idx.window(M, PPM).tolist())
        cand = list(cand)
        if not cand:
            for s in SCHEMES: rr[s][cls].append(0.0)
            continue
        Mq = np.nanmedian([neutral_mass(pmz[r], add[r]) for r in rows])
        feats = {}
        for c in cand:
            cm = mass_by_key[c]
            feats[c] = dict(
                pop=np.log1p(pop.get(c, 0)),
                frag=frag_explain(allmz, allit, cm),
                nplike=np_like(mol(c)),
                mass=1.0 - min(abs(cm - Mq) / (Mq * PPM / 1e6 + 1e-9), 1.0),
            )
        # min-max each feature across THIS pool so a 0..7 log-count doesn't swamp a
        # 0..1 fraction. Without this, mixed schemes silently reduce to popularity.
        for f in ("pop", "frag", "nplike", "mass"):
            vals = [feats[c][f] for c in cand]
            lo, hi = min(vals), max(vals)
            rng_ = (hi - lo) or 1.0
            for c in cand:
                feats[c][f] = (feats[c][f] - lo) / rng_
        for s, w in SCHEMES.items():
            ranked = sorted(cand, key=lambda c: -sum(w[f] * feats[c][f] for f in w))
            r_ = 0.0
            for i, c in enumerate(ranked[:25], 1):
                if c == k: r_ = 1.0 / i; break
            rr[s][cls].append(r_)
        # RANDOM control
        draws = []
        for _ in range(20):
            perm = cand[:]; _rng.shuffle(perm)
            rr_ = 0.0
            for i, c in enumerate(perm[:25], 1):
                if c == k: rr_ = 1.0 / i; break
            draws.append(rr_)
        rr["RANDOM"][cls].append(float(np.mean(draws)))
        # analytic E[RR] if the true answer is uniformly placed among the pool
        if k in cand:
            npool = len(cand)
            exp_random[cls].append(float(np.mean([1.0/i for i in range(1, min(npool,25)+1)])))
        else:
            exp_random[cls].append(0.0)

    print("\n" + "=" * 60)
    print("E05 — ranking the candidate pool (Class 2 is the target)")
    print("=" * 60)
    print(f"{'scheme':18} {'C1':>8} {'C2':>8} {'C3':>8}")
    out = {}
    for s in list(SCHEMES) + ["RANDOM"]:
        c1 = float(np.mean(rr[s][1])); c2 = float(np.mean(rr[s][2])); c3 = float(np.mean(rr[s][3]))
        out[s] = dict(C1=c1, C2=c2, C3=c3)
        print(f"{s:18} {c1:>8.4f} {c2:>8.4f} {c3:>8.4f}")
    print(f"\n  {'expected-random':18} {'':>8} "
          f"{np.mean(exp_random[2]):>8.4f} {'':>8}   <- E[RR] of a uniform-random rank")
    print(f"\nE04 popularity floor for C2: 0.2566 (now suspect -- see RANDOM/expected-random)")
    best = max(out, key=lambda s: out[s]["C2"])
    print(f"best C2 scheme: {best} @ {out[best]['C2']:.4f}  "
          f"(delta {out[best]['C2']-0.2566:+.4f})")
    print(f"runtime {time.time()-t0:.0f}s")
    json.dump({"split": SPLIT, "schemes": out, "e04_floor_c2": 0.2566},
              open(f"{RESULTS}/E05_rank_{SPLIT}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
