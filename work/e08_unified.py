#!/usr/bin/env python3
"""E08 — one pipeline, NO class oracle, measured across all classes at once.

Every prior experiment routed by the known class (E07 only ran on Class 2 queries).
A real submission gets a spectrum and does not know if it is Class 1/2/3. E08 builds
the single pipeline a submission would use and measures it honestly.

For each query, with no knowledge of its class:
  1. SPECTRAL MATCH (the E03 mechanism): cosine of the query against the reference
     index -> a spectral score per structure. High only when public spectra exist
     (i.e. Class 1).
  2. MASS RETRIEVAL + FRAGMENT RANK (the E04/E07 mechanism): mass-window candidates,
     each scored by how well its own fragments explain the spectrum. Works without
     any reference spectrum (i.e. Class 2).
  3. MERGE into one ranked list of 25. The routing is implicit: a Class 1 answer
     wins on spectral score, a Class 2 answer wins on fragment score. No class is
     predicted; the scores decide. (This is GPT's slot-allocation point.)

Per-class MRR is REPORTED using the (hidden) labels, but never USED in ranking.
"""
import json, os, sys, time, random
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import formula_mass, neutral_mass, MassIndex
from e07_fragment import fragment_masses, explain, MONO
import scoring

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "seed0_n300"
BIN, INT_FLOOR, MAX_PEAKS, IDX_PEAKS = 0.02, 0.002, 256, 20
PPM, N_RESCORE, TOPN = 10, 800, 25


def prep(mz, it):
    if len(mz) == 0: return np.empty(0, np.int32), np.empty(0, np.float32)
    it = np.asarray(it, np.float32); mz = np.asarray(mz, np.float64)
    mx = it.max()
    if mx <= 0: return np.empty(0, np.int32), np.empty(0, np.float32)
    keep = it >= INT_FLOOR * mx; mz, it = mz[keep], it[keep]
    if len(it) > MAX_PEAKS:
        top = np.argpartition(-it, MAX_PEAKS)[:MAX_PEAKS]; mz, it = mz[top], it[top]
    b = np.rint(mz / BIN).astype(np.int32)
    return b, (it / (np.linalg.norm(it) + 1e-12)).astype(np.float32)


def cosine(qb, qw, rb, rw):
    if len(qb) == 0 or len(rb) == 0: return 0.0
    i = j = 0; s = 0.0
    while i < len(qb) and j < len(rb):
        if qb[i] == rb[j]: s += qw[i]*rw[j]; i += 1; j += 1
        elif qb[i] < rb[j]: i += 1
        else: j += 1
    return float(s)


def main():
    t0 = time.time()
    sp = json.load(open(f"{SPLITS}/split_{SPLIT}.json"))
    assign, qrows = sp["assign"], sp["query_rows"]
    ref_rows = np.load(f"{SPLITS}/ref_rows_{SPLIT}.npy")

    t = pq.read_table(DATA, columns=["inchikey14","normalized_smiles","molecular_formula",
                                     "precursor_mz","adduct","ionization_mode",
                                     "ms2_mzs","ms2_normalized_intensities"])
    keys = np.asarray(t.column("inchikey14")); smiles = np.asarray(t.column("normalized_smiles"))
    formula = np.asarray(t.column("molecular_formula")); pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct")); mode = np.asarray(t.column("ionization_mode"))
    def flat(c):
        a=t.column(c).combine_chunks(); return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf,mzo=flat("ms2_mzs"); itf,ito=flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    # reference spectral index (sorted for the pointer cosine)
    prepped = {}
    for r in ref_rows:
        b,w = prep(*peaks(r)); o=np.argsort(b); prepped[r]=(b[o],w[o])
    inv = defaultdict(list)
    for r in ref_rows:
        b,w = prepped[r]
        if len(b)==0: continue
        for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]): inv[int(bb)].append(r)
    inv = {k:np.asarray(v) for k,v in inv.items()}
    print(f"spectral index {time.time()-t0:.0f}s", flush=True)

    struct = pd.DataFrame({"k":keys,"f":formula,"s":smiles}).drop_duplicates("k")
    massk = {k:formula_mass(f) for k,f in zip(struct["k"],struct["f"])}
    smik = dict(zip(struct["k"],struct["s"]))
    c3 = {k for k,c in assign.items() if c==3}
    dbk = np.array([k for k in struct["k"] if k not in c3 and np.isfinite(massk[k])])
    midx = MassIndex(dbk, np.array([massk[k] for k in dbk]))
    fragc, ckc = {}, {}
    def frags(k):
        if k not in fragc: fragc[k]=fragment_masses(Chem.MolFromSmiles(smik.get(k,"")))
        return fragc[k]
    def ckey(k):
        if k not in ckc: ckc[k]=scoring.key14(smik.get(k,""))
        return ckc[k]
    print(f"mass index {time.time()-t0:.0f}s", flush=True)

    RULES = {"sum":lambda s,f:s+f, "spec_priority":lambda s,f:2*s+f, "max":lambda s,f:max(s,f)}
    rr = {rule:defaultdict(list) for rule in RULES}

    for n,(k,rows) in enumerate(qrows.items(),1):
        cls = assign[k]; truth = ckey(k)
        qb=[]; qmz=pmz[rows[0]]; positive=str(mode[rows[0]]).lower().startswith("pos")
        allmz=np.concatenate([peaks(r)[0] for r in rows]); allit=np.concatenate([peaks(r)[1] for r in rows])
        # 1. spectral: gather + cosine -> per-structure spectral score
        spec=defaultdict(float)
        for r in rows:
            b,w = prep(*peaks(r)); o=np.argsort(b); b,w=b[o],w[o]
            if len(b)==0: continue
            cand=[inv[int(bb)] for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]) if int(bb) in inv]
            if not cand: continue
            cc=np.concatenate(cand); u,ct=np.unique(cc,return_counts=True); take=u[np.argsort(-ct)[:N_RESCORE]]
            for c in take:
                sim=cosine(b,w,*prepped[c])
                if sim>0: spec[keys[c]]=max(spec[keys[c]], sim)   # best spectral hit per structure
        # 2. mass + fragment
        frag=defaultdict(float); mcand=set()
        for r in rows:
            M=neutral_mass(pmz[r],add[r])
            if np.isfinite(M) and M>0: mcand.update(midx.window(M,PPM).tolist())
        for c in mcand:
            frag[c]=explain(allmz,allit,frags(c),positive)
        # 3. merge: union of candidates, normalise each score within-query, combine
        allc=set(spec)|set(mcand)
        if not allc:
            for rule in RULES: rr[rule][cls].append(0.0); 
            continue
        smax=max(spec.values()) if spec else 1.0; fmax=max(frag.values()) if frag else 1.0
        smax=smax or 1.0; fmax=fmax or 1.0
        feats={c:(spec.get(c,0.0)/smax, frag.get(c,0.0)/fmax) for c in allc}
        for rule,fn in RULES.items():
            ranked=sorted(allc, key=lambda c:-fn(*feats[c]))[:TOPN]
            v=0.0
            for i,c in enumerate(ranked,1):
                if ckey(c)==truth: v=1.0/i; break
            rr[rule][cls].append(v)
        if n%75==0: print(f"  {n}/{len(qrows)} {time.time()-t0:.0f}s", flush=True)

    W=(0.16,0.45,0.39)
    print("\n"+"="*64); print("E08 — unified pipeline, no class oracle"); print("="*64)
    print(f"{'merge rule':14s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted16/45/39':>18}")
    out={}
    for rule in RULES:
        c1=np.mean(rr[rule][1]); c2=np.mean(rr[rule][2]); c3=np.mean(rr[rule][3])
        w=W[0]*c1+W[1]*c2+W[2]*c3; out[rule]=dict(C1=float(c1),C2=float(c2),C3=float(c3),weighted=float(w))
        print(f"{rule:14s} {c1:>8.4f} {c2:>8.4f} {c3:>8.4f} {w:>18.4f}")
    best=max(out,key=lambda r:out[r]['weighted'])
    print(f"\nbest rule: {best} weighted {out[best]['weighted']:.4f}  (gate 0.339, on TRAINING-db coverage)")
    json.dump({"split":SPLIT,"rules":out}, open(f"{RESULTS}/E08_unified_{SPLIT}.json","w"), indent=2)
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
