#!/usr/bin/env python3
"""E07 — a Class 2 ranker that actually reads each candidate's STRUCTURE.

E05 proved the mass-only 'frag' proxy loses to a random shuffle of the pool (0.312
vs 0.356). E07 replaces it with real in-silico fragmentation: break the candidate's
own bonds, enumerate fragment masses, and score how much of the observed spectrum
that specific structure can explain. It must beat RANDOM (0.356) to count -- E06
proved beating random on Class 2 is the only path to the 0.339 gate.

Per GPT's review, the truth match uses the competition's tautomer-canonical
InChIKey14 (scoring.key14), computed only for the ~9 pool members + truth per query.
"""
import json, os, sys, time, random
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import formula_mass, neutral_mass, MassIndex
import scoring

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "seed0_n300"
PPM = 10
PROTON = 1.00727646688
H = 1.007825032
TOL = 0.01           # Da, fragment-peak match tolerance
MAX_HEAVY = 60       # skip pathological molecules (treat as no structural signal)
TOPPEAKS = 30        # score against the query's most intense peaks

MONO = {"C":12.0,"H":1.00782503207,"N":14.0030740048,"O":15.9949146196,"S":31.97207100,
        "P":30.97376163,"Cl":34.96885268,"Br":78.9183371,"F":18.99840322,"I":126.904473,
        "Si":27.9769265325,"Se":79.9165213,"B":11.0093054,"Na":22.9897692809,"K":38.96370668}


def fragment_masses(mol, max_break=2):
    """Neutral monoisotopic masses of substructure fragments from breaking 1-2
    acyclic single bonds. Fragment mass = sum of its heavy atoms + their H's
    (the broken position is treated as H-capped, the standard MS approximation)."""
    if mol is None or mol.GetNumHeavyAtoms() > MAX_HEAVY:
        return None
    acyclic = [b.GetIdx() for b in mol.GetBonds()
               if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
    masses = set()

    # Precompute per-atom mass (heavy + its H's) from the ORIGINAL sanitised mol.
    # Fragments from FragmentOnBonds are unsanitised, so GetTotalNumHs() throws on
    # them (this silently discarded every fragment in the first version). Mapping
    # fragment atom indices back to the original mol avoids that entirely.
    N = mol.GetNumAtoms()
    atom_mass = [MONO.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H for a in mol.GetAtoms()]

    masses.add(sum(atom_mass))                      # whole molecule (no cleavage)
    combos = [(b,) for b in acyclic]
    if max_break >= 2 and len(acyclic) <= 40:
        combos += [(a, b) for i, a in enumerate(acyclic) for b in acyclic[i+1:]]
    for bonds in combos:
        try:
            frg = Chem.FragmentOnBonds(mol, list(bonds), addDummies=True)
            for idxs in Chem.GetMolFrags(frg, asMols=False):   # tuples of atom indices
                masses.add(sum(atom_mass[i] for i in idxs if i < N))
        except Exception:
            continue
    return masses


def explain(mz, it, frag_masses, positive):
    """Intensity-weighted fraction of the query's top peaks whose neutral mass
    matches some candidate fragment (allowing a proton and +-1 H rearrangement)."""
    if not frag_masses:
        return 0.0
    order = np.argsort(-it)[:TOPPEAKS]
    mz, it = mz[order], it[order]
    fm = np.array(sorted(frag_masses))
    matched = wsum = 0.0
    for m, i in zip(mz, it):
        wsum += i
        neutral = (m - PROTON) if positive else (m + PROTON)
        # nearest fragment mass, allow +-1 H
        j = np.searchsorted(fm, neutral)
        ok = False
        for k in (j-1, j, j+1):
            if 0 <= k < len(fm) and min(
                abs(neutral - fm[k]), abs(neutral - fm[k] - H), abs(neutral - fm[k] + H)
            ) <= TOL:
                ok = True; break
        if ok:
            matched += i
    return matched / wsum if wsum > 0 else 0.0


def main():
    t0 = time.time()
    sp = json.load(open(f"{SPLITS}/split_{SPLIT}.json"))
    assign, qrows = sp["assign"], sp["query_rows"]

    t = pq.read_table(DATA, columns=["inchikey14","normalized_smiles","molecular_formula",
                                     "precursor_mz","adduct","ionization_mode",
                                     "ms2_mzs","ms2_normalized_intensities"])
    keys = np.asarray(t.column("inchikey14"))
    smiles = np.asarray(t.column("normalized_smiles"))
    formula = np.asarray(t.column("molecular_formula"))
    pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct"))
    mode = np.asarray(t.column("ionization_mode"))
    def flat(c):
        a=t.column(c).combine_chunks(); return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf,mzo=flat("ms2_mzs"); itf,ito=flat("ms2_normalized_intensities")
    del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    struct = pd.DataFrame({"k":keys,"f":formula,"s":smiles}).drop_duplicates("k")
    massk = {k: formula_mass(f) for k,f in zip(struct["k"],struct["f"])}
    smik = dict(zip(struct["k"],struct["s"]))
    c3 = {k for k,c in assign.items() if c==3}
    dbk = np.array([k for k in struct["k"] if k not in c3 and np.isfinite(massk[k])])
    idx = MassIndex(dbk, np.array([massk[k] for k in dbk]))
    print(f"index ready {time.time()-t0:.0f}s", flush=True)

    molc, fragc, ckc = {}, {}, {}
    def mol(k):
        if k not in molc: molc[k]=Chem.MolFromSmiles(smik.get(k,""))
        return molc[k]
    def frags(k):
        if k not in fragc: fragc[k]=fragment_masses(mol(k))
        return fragc[k]
    def ckey(k):
        if k not in ckc: ckc[k]=scoring.key14(smik.get(k,""))
        return ckc[k]

    rng = random.Random(0)
    rr = {"random": defaultdict(list), "frag_structural": defaultdict(list)}
    only2 = [k for k,c in assign.items() if c==2]

    for n,(k,rows) in enumerate(qrows.items(),1):
        cls = assign[k]
        if cls != 2:   # E07 targets Class 2; skip others for speed
            continue
        allmz = np.concatenate([peaks(r)[0] for r in rows])
        allit = np.concatenate([peaks(r)[1] for r in rows])
        positive = (str(mode[rows[0]]).lower().startswith("pos"))
        cand = set()
        for r in rows:
            M = neutral_mass(pmz[r], add[r])
            if np.isfinite(M) and M>0: cand.update(idx.window(M,PPM).tolist())
        cand = list(cand)
        truth_ck = ckey(k)
        if not cand:
            for s in rr: rr[s][cls].append(0.0)
            continue
        # random baseline
        draws=[]
        for _ in range(20):
            perm=cand[:]; rng.shuffle(perm)
            rrv=0.0
            for i,c in enumerate(perm[:25],1):
                if ckey(c)==truth_ck: rrv=1.0/i; break
            draws.append(rrv)
        rr["random"][cls].append(float(np.mean(draws)))
        # structural fragmentation ranker
        sc = {c: explain(allmz, allit, frags(c), positive) for c in cand}
        ranked = sorted(cand, key=lambda c: -sc[c])
        rrv=0.0
        for i,c in enumerate(ranked[:25],1):
            if ckey(c)==truth_ck: rrv=1.0/i; break
        rr["frag_structural"][cls].append(rrv)
        if n % 50 == 0:
            print(f"  {n}/{len(qrows)} ({time.time()-t0:.0f}s)  "
                  f"frag so far {np.mean(rr['frag_structural'][2]):.3f} "
                  f"vs random {np.mean(rr['random'][2]):.3f}", flush=True)

    print("\n"+"="*56)
    print("E07 — structural fragmentation ranker, Class 2")
    print("="*56)
    out={}
    for s in ("random","frag_structural"):
        v=rr[s][2]; out[s]=float(np.mean(v))
        print(f"  {s:18s} C2 MRR {out[s]:.4f}  (n={len(v)})")
    delta = out["frag_structural"]-out["random"]
    print(f"\n  frag_structural - random = {delta:+.4f}")
    print(f"  VERDICT: {'BEATS random -> the ranker has real signal' if delta>0 else 'does NOT beat random -> kill-gate territory'}")
    print(f"  runtime {time.time()-t0:.0f}s")
    json.dump({"split":SPLIT,"class2":out,"delta_vs_random":delta},
              open(f"{RESULTS}/E07_fragment_{SPLIT}.json","w"), indent=2)


if __name__ == "__main__":
    main()
