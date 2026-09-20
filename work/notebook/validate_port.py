#!/usr/bin/env python3
"""Port-equivalence check: the notebook pipeline must reproduce E18's learned ranker EXACTLY.

Each twin-safe eval split is dressed up as a test set -- its query spectra become the test
molecules, its reference rows the spectral library, COCONUT minus the split's Class 3
exclusions the database -- and run through casmi_pipeline.predict, the same call the Kaggle
notebook makes. Pass = every split's weighted MRR matches E18 to 1e-9.
"""
import json, os, sys, time
import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, WORK)
import casmi_pipeline as cp
from e15_coconut_pools import load_structures
from twins import db_exclusions
import scoring

SPLITS = os.path.join(WORK, "splits")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
W = {1: 0.16, 2: 0.45, 3: 0.39}


def main():
    t0 = time.time()
    e18 = json.load(open(os.path.join(WORK, "results", "E18_ranker_coconut.json")))
    weights, expect = e18["weights"], e18["results"]["learned"]["splits"]
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["inchikey14", "normalized_smiles", "ionization_mode", "ms2_mzs", "ms2_normalized_intensities"])
    keys = np.asarray(t.column("inchikey14")); mode = np.asarray(t.column("ionization_mode"))
    truth_smi = dict(zip(keys, np.asarray(t.column("normalized_smiles"))))   # E17: truth from the FULL table
    def flat(c):
        a = t.column(c).combine_chunks(); return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    _, pmz, add, tr, co, _ = load_structures()
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_of = lambda k: trs.get(k) or cos_.get(k, "")          # training SMILES win, as in E17
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    ok = True; got_all, ded_all = [], []
    for split, want in zip(EVAL, expect):
        sp = json.load(open(os.path.join(SPLITS, f"split_{split}.json")))
        ref_rows = np.load(os.path.join(SPLITS, f"ref_rows_{split}.npy"))
        sindex = cp.SpectralIndex(keys, peaks, ref_rows)
        excl = db_exclusions(sp, tr, co)
        keep = ~co.k.isin(excl)
        midx = cp.MassIndex(co.k[keep].to_numpy(), co.m[keep].to_numpy())
        mols = [dict(id=k, spectra=[(*peaks(r), add[r], pmz[r], mode[r]) for r in rows])
                for k, rows in sp["query_rows"].items()]
        # limit=50 so the submission layer can drop scorer-duplicate guesses and still fill 25
        pred = cp.predict(mols, sindex, midx, smi_of, weights, procs=7, limit=50,
                          log=lambda m: print(f"[{split}]{m}", flush=True))
        rr = {1: [], 2: [], 3: []}; rrd = {1: [], 2: [], 3: []}
        for k, c in sp["assign"].items():
            truth = scoring.key14(truth_smi[k])
            cand_keys = [scoring.key14(smi_of(cand)) for cand in pred[k]]   # NOT `keys`: that's the
            plain = cand_keys[:cp.TOPN]                                     # training key array
            seen, dedup = set(), []                      # keep first of each scorer-identical guess
            for kk in cand_keys:
                if kk in seen: continue
                seen.add(kk); dedup.append(kk)
                if len(dedup) == cp.TOPN: break
            for lst, acc in ((plain, rr), (dedup, rrd)):
                r = 0.0
                for i, kk in enumerate(lst, 1):
                    if truth is not None and kk == truth: r = 1.0 / i; break
                acc[c].append(r)
        got = sum(W[c] * np.mean(rr[c]) for c in (1, 2, 3))
        gotd = sum(W[c] * np.mean(rrd[c]) for c in (1, 2, 3))
        match = abs(got - want) < 1e-9; ok &= match
        got_all.append(got); ded_all.append(gotd)
        print(f"[{split}] weighted {got:.6f}   E18 {want:.6f}   {'MATCH' if match else 'MISMATCH'}"
              f"   | dedup {gotd:.6f} ({gotd-got:+.6f})   {time.time()-t0:.0f}s", flush=True)
    print(f"\n3-split mean: as-ranked {np.mean(got_all):.4f}   deduplicated {np.mean(ded_all):.4f} "
          f"({np.mean(ded_all)-np.mean(got_all):+.4f})")
    print("RESULT:", "PORT EQUIVALENT to E18" if ok else "PORT DIFFERS from E18")


if __name__ == "__main__":
    main()
