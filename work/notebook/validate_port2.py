#!/usr/bin/env python3
"""Port check for the generation pipeline: casmi_pipeline.predict2 must reproduce E23 exactly.

Same shape as validate_port.py, which held predict() to E18. Each twin-safe eval split is
dressed as a test set and run through predict2 -- the same call the notebook makes -- and the
pooled weighted MRR over all three splits must equal E23's 0.3663 to 1e-9.
"""
import json, os, sys, time
import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(HERE)
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
    e23 = json.load(open(os.path.join(WORK, "results", "E23_better_relatives.json")))
    weights, want = e23["weights"], e23["results"]["retrieval + generated"]
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["inchikey14", "normalized_smiles", "ionization_mode", "precursor_mz",
                               "adduct", "ms2_mzs", "ms2_normalized_intensities"])
    keys = np.asarray(t.column("inchikey14")); mode = np.asarray(t.column("ionization_mode"))
    pmz = t.column("precursor_mz").to_numpy(); add = np.asarray(t.column("adduct"))
    truth_smi = dict(zip(keys, np.asarray(t.column("normalized_smiles"))))
    def flat(c):
        a = t.column(c).combine_chunks(); return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    _, _, _, tr, co, _ = load_structures()
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_of = lambda k: trs.get(k) or cos_.get(k, "")
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    rr = {1: [], 2: [], 3: []}
    for split in EVAL:
        sp = json.load(open(os.path.join(SPLITS, f"split_{split}.json")))
        ref_rows = np.load(os.path.join(SPLITS, f"ref_rows_{split}.npy"))
        sindex = cp.SpectralIndex(keys, peaks, ref_rows)
        keep = ~co.k.isin(db_exclusions(sp, tr, co))
        midx = cp.MassIndex(co.k[keep].to_numpy(), co.m[keep].to_numpy())
        mols = [dict(id=k, spectra=[(*peaks(r), add[r], pmz[r], mode[r]) for r in rows])
                for k, rows in sp["query_rows"].items()]
        pred = cp.predict2(mols, sindex, midx, smi_of, weights, peaks, pmz, procs=7,
                           log=lambda m: print(f"[{split}]{m}", flush=True))
        for k, c in sp["assign"].items():
            truth = scoring.key14(truth_smi[k]); r = 0.0
            for i, smi in enumerate(pred[k], 1):
                if truth is not None and scoring.key14(smi) == truth: r = 1.0 / i; break
            rr[c].append(r)
        print(f"[{split}] done {time.time()-t0:.0f}s", flush=True)

    got = {f"C{c}": float(np.mean(rr[c])) for c in (1, 2, 3)}
    gotw = sum(W[c] * np.mean(rr[c]) for c in (1, 2, 3))
    print(f"\n{'':10s} {'C1':>9} {'C2':>9} {'C3':>9} {'weighted':>10}")
    print(f"{'port':10s} {got['C1']:>9.4f} {got['C2']:>9.4f} {got['C3']:>9.4f} {gotw:>10.4f}")
    print(f"{'E23':10s} {want['C1']:>9.4f} {want['C2']:>9.4f} {want['C3']:>9.4f} {want['weighted']:>10.4f}")
    ok = abs(gotw - want["weighted"]) < 1e-9
    print("\nRESULT:", "PORT EQUIVALENT to E23" if ok else f"PORT DIFFERS ({gotw-want['weighted']:+.6f})")


if __name__ == "__main__":
    main()
