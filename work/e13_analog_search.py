#!/usr/bin/env python3
"""E13 — can a better relative-search raise the Class 3 ceiling E12 measured?

E12: plain-cosine spectral hits look identical (~0.96) but are structurally far (top-1
Tanimoto 0.26, best-of-5 0.45); only 24% of C3 queries had a close (Tan>=0.6) relative a
common edit away in the top 5. That 24% is the ceiling for analog editing. Four searches:

  A  cosine          E08's binned cosine over the gathered references (E12 baseline)
  B  modcos          modified cosine: a peak also matches at +delta (the precursor gap)
  C  nlcos           cosine over neutral losses (precursor - fragment)
  D  edit-first      DB structures exactly one common edit from the query mass (+-), ranked
                     by fragment explain where each fragment may appear unshifted OR +delta
                     (the truth = relative + edit, so its fragments are one or the other)
  U  union           best of A-D at each K (an upper bound on combining them)

Metrics per method, top-K (K=5, 25): best Tanimoto(truth, hit); share >= 0.6; and REACHABLE =
some hit with Tanimoto >= 0.6 whose mass gap is a common edit. No ranking score -- ceiling only.

PREDICTIONS (before the run, 2026-09-18):
  P1  A reproduces E12's top-5 reachable 24% within +-2 points (different code path, same idea).
  P2  B beats A on top-5 reachable (it's designed for exactly the analog case).
  P3  D gives the highest top-25 reachable of A-D (it only returns common-edit candidates).
  RESULT (2026-09-18, 351 queries): P1 PASS (A reach@5 0.242 == E12). P2 PASS (B 0.313 vs
      0.242). P3 FAIL (D reach@25 0.348 < B 0.413). Union of A-D: reach@5 0.464, reach@25 0.510
      -- nearly double the plain-cosine ceiling; B and C find different relatives.
      Budget caveat: B/C used only 3 query spectra x 300 refs and still beat A.
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import formula_mass, neutral_mass, MassIndex
from e08_unified import prep, cosine, IDX_PEAKS, N_RESCORE, PPM
from e11_ringbreak import fragment_table
from e07_fragment import PROTON, H, TOL, TOPPEAKS
from e12_c3_analogs import EDITS, classify

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
KS = (5, 25)
MZ_TOL = 0.01
NPK = 50                      # peaks per spectrum for modcos / nlcos
SLOW_ROWS, SLOW_REFS = 3, 300 # B/C budget: first 3 query spectra, top 300 gathered refs (A uses all)
EDIT_MASSES = sorted({m for m in EDITS.values() if m > 0})
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def top_peaks(mz, it, n=NPK):
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    if len(it) == 0 or it.max() <= 0: return np.empty(0), np.empty(0)
    o = np.argsort(-it)[:n]; mz, it = mz[o], it[o]
    w = np.sqrt(it); w /= np.linalg.norm(w) + 1e-12      # sqrt-intensity, matchms convention
    s = np.argsort(mz)
    return mz[s], w[s]


def greedy_cos(qm, qw, rm, rw, shift=None):
    """Greedy peak-matching cosine; with shift, a query peak may also match ref+shift."""
    if len(qm) == 0 or len(rm) == 0: return 0.0
    pairs = []
    for off in ((0.0,) if shift is None else (0.0, shift)):
        lo = np.searchsorted(rm + off, qm - MZ_TOL, "left"); hi = np.searchsorted(rm + off, qm + MZ_TOL, "right")
        for i, (a, b) in enumerate(zip(lo, hi)):
            for j in range(a, b): pairs.append((qw[i] * rw[j], i, j))
    if not pairs: return 0.0
    pairs.sort(reverse=True); ui, uj = set(), set(); s = 0.0
    for sc, i, j in pairs:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s += sc
    return s


def explain_shift(mz, it, table, delta, positive):
    """Fraction of top peaks explained by an analog's fragments, unshifted or +delta, ±1H."""
    if table is None: return 0.0
    base = np.array(sorted({m for m, _ in table["acyc"]}))
    fm = np.sort(np.concatenate([base + j * H for j in (-1, 0, 1)] + [base + delta + j * H for j in (-1, 0, 1)]))
    o = np.argsort(-it)[:TOPPEAKS]; mz, it = mz[o].astype(np.float64), it[o]
    neutral = (mz - PROTON) if positive else (mz + PROTON)
    lo = np.searchsorted(fm, neutral - TOL, "left"); hi = np.searchsorted(fm, neutral + TOL, "right")
    s = it.sum()
    return float(it[hi > lo].sum() / s) if s > 0 else 0.0


def main():
    t0 = time.time()
    t = pq.read_table(DATA, columns=["inchikey14", "normalized_smiles", "molecular_formula",
                                     "precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities"])
    keys = np.asarray(t.column("inchikey14")); smiles = np.asarray(t.column("normalized_smiles"))
    formula = np.asarray(t.column("molecular_formula")); pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct"))
    def flat(c):
        a = t.column(c).combine_chunks(); return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    struct = pd.DataFrame({"k": keys, "f": formula, "s": smiles}).drop_duplicates("k")
    massk = {k: formula_mass(f) for k, f in zip(struct["k"], struct["f"])}
    smik = dict(zip(struct["k"], struct["s"]))
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    fpc = {}
    def fp(k):
        if k not in fpc:
            m = Chem.MolFromSmiles(smik.get(k, "") or ""); fpc[k] = FPG.GetFingerprint(m) if m is not None else None
        return fpc[k]
    def tan(a, b):
        fa, fb = fp(a), fp(b)
        return DataStructs.TanimotoSimilarity(fa, fb) if fa is not None and fb is not None else 0.0

    per = defaultdict(list)       # method -> list of per-query dicts
    tabs = {}
    for split in ("seed0_n300", "seed1_n300", "seed2_n300"):
        sp = json.load(open(f"{SPLITS}/split_{split}.json")); assign, qrows = sp["assign"], sp["query_rows"]
        ref_rows = np.load(f"{SPLITS}/ref_rows_{split}.npy")
        prepped = {}; inv = defaultdict(list)
        for r in ref_rows:
            b, w = prep(*peaks(r)); o = np.argsort(b); prepped[r] = (b[o], w[o])
            if len(b):
                for bb in np.unique(b[o][np.argsort(-w[o])[:IDX_PEAKS]]): inv[int(bb)].append(r)
        inv = {k: np.asarray(v) for k, v in inv.items()}
        c3 = {k for k, c in assign.items() if c == 3}
        dbk = np.array([k for k in struct["k"] if k not in c3 and np.isfinite(massk[k])])
        midx = MassIndex(dbk, np.array([massk[k] for k in dbk]))
        print(f"[{split}] indexes {time.time()-t0:.0f}s", flush=True)

        tpc = {}
        qs = [(k, rows) for k, rows in qrows.items() if assign[k] == 3]
        # D needs fragment tables for every edit-window candidate: gather, then fragment in parallel
        edit_cands = {}
        for k, rows in qs:
            M = np.nanmedian([neutral_mass(pmz[r], add[r]) for r in rows])
            cs = {}
            for e in [0.0] + EDIT_MASSES:            # 0 = same-mass isomer (22% of E12 top-1 gaps)
                for sgn in ((1,) if e == 0 else (1, -1)):
                    for c in midx.window(M - sgn * e, PPM).tolist(): cs[c] = sgn * e
            edit_cands[k] = (M, cs)
        need = sorted({smik[c] for _, cs in edit_cands.values() for c in cs} - set(tabs))
        with Pool(7) as pool:
            for smi, tb in zip(need, pool.map(fragment_table, need, chunksize=16)): tabs[smi] = tb
        print(f"[{split}] {len(qs)} C3 queries, fragmented {len(need)} edit candidates {time.time()-t0:.0f}s", flush=True)

        for n, (k, rows) in enumerate(qs, 1):
            M, cs = edit_cands[k]
            allmz = np.concatenate([peaks(r)[0] for r in rows]); allit = np.concatenate([peaks(r)[1] for r in rows])
            positive = str(add[rows[0]]).endswith("+")
            A = defaultdict(float); B = defaultdict(float); C = defaultdict(float)
            for ri, r in enumerate(rows):
                b, w = prep(*peaks(r)); o = np.argsort(b); b, w = b[o], w[o]
                if len(b) == 0: continue
                g = [inv[int(bb)] for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]) if int(bb) in inv]
                if not g: continue
                u, ct = np.unique(np.concatenate(g), return_counts=True)
                qm, qw = top_peaks(*peaks(r)); qnl = top_peaks(pmz[r] - peaks(r)[0], peaks(r)[1])
                slow = ri < SLOW_ROWS          # B/C are python-greedy: first 3 spectra, top 300 refs
                for rank, c in enumerate(u[np.argsort(-ct)[:N_RESCORE]]):
                    kc = keys[c]
                    A[kc] = max(A[kc], cosine(b, w, *prepped[c]))
                    if not slow or rank >= SLOW_REFS: continue
                    if c not in tpc:
                        tpc[c] = (top_peaks(*peaks(c)), top_peaks(pmz[c] - peaks(c)[0], peaks(c)[1]))
                    (rm, rw), rnl = tpc[c]
                    B[kc] = max(B[kc], greedy_cos(qm, qw, rm, rw, shift=pmz[r] - pmz[c]))
                    C[kc] = max(C[kc], greedy_cos(*qnl, *rnl))
            D = {c: explain_shift(allmz, allit, tabs[smik[c]], d, positive) for c, d in cs.items()}
            ranked = {m: [c for c, _ in sorted(S.items(), key=lambda kv: (-kv[1], kv[0]))[:max(KS)]]
                      for m, S in (("A_cosine", A), ("B_modcos", B), ("C_nlcos", C), ("D_editfirst", D))}
            for m, lst in ranked.items():
                tans = [tan(k, c) for c in lst]
                edit = [classify(massk[k] - massk[c]) is not None for c in lst]
                per[m].append({f"best{K}": max(tans[:K], default=0.0) for K in KS} |
                              {f"reach{K}": any(t >= 0.6 and e for t, e in zip(tans[:K], edit[:K])) for K in KS})
            per["U_union"].append({f: (max if f.startswith("best") else any)(per[m][-1][f] for m in ranked)
                                   for f in per["A_cosine"][-1]})
            if n % 40 == 0: print(f"[{split}]  {n}/{len(qs)} {time.time()-t0:.0f}s", flush=True)

    lines = [f"E13 Class 3 relative-search — {len(per['A_cosine'])} C3 queries, 3 splits", "",
             f"{'method':12s} " + " ".join(f"{h:>11}" for K in KS for h in (f'medTan@{K}', f'Tan>=.6@{K}', f'reach@{K}'))]
    out = {}
    for m in ("A_cosine", "B_modcos", "C_nlcos", "D_editfirst", "U_union"):
        v = per[m]; row = {}
        for K in KS:
            bt = np.array([x[f"best{K}"] for x in v])
            row |= {f"medTan@{K}": float(np.median(bt)), f"tan06@{K}": float(np.mean(bt >= 0.6)),
                    f"reach@{K}": float(np.mean([x[f"reach{K}"] for x in v]))}
        out[m] = row
        lines.append(f"{m:12s} " + " ".join(f"{row[f'{h}@{K}']:>11.3f}" for K in KS for h in ("medTan", "tan06", "reach")))
    lines.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(lines))
    open(f"{RESULTS}/E13_analog_search_2026-09-18.txt", "w").write("\n".join(lines) + "\n")
    json.dump(out, open(f"{RESULTS}/E13_analog_search_3seeds.json", "w"), indent=2)


if __name__ == "__main__":
    main()
