#!/usr/bin/env python3
"""E67 — the contaminated companion to E66: energy ladders on molecules ICEBERG MEMORISED.

E66 tests break depth on molecules deliberately kept out of ICEBERG's training, where its
predicted-versus-observed cosine is 0.147. A null there has two readings that demand opposite
responses:

  (a) the mechanism does not exist -- breaking order carries no usable information, so stop;
  (b) the mechanism exists but the predictor cannot see it on novel chemistry -- so the answer is a
      better fragment predictor, not a different feature.

The only way to separate them is to run the identical test where the predictor is accurate, which
means molecules INSIDE MassSpecGym -- ICEBERG's own training set. That is contamination on purpose:
these numbers are an upper bound on the mechanism, never an estimate of competition performance,
and every report of them has to say so.

This builds the query key list only. Same energy-ladder requirement as E64 (>=2 distinct
single-valued timsTOF energies in 15-65 eV) and the same mass and spectra-count gates, but control
1 is INVERTED: every structure must be in MassSpecGym rather than absent from it.

Read together:
  E66 clean       novel chemistry, poor predictions  -> can the feature work where we need it?
  E67 contaminated  memorised chemistry, good predictions -> does the mechanism exist at all?
"""
import json, os, pickle, sys, time
from collections import Counter, defaultdict
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
N_TOTAL = 600
E_LO, E_HI = 15.0, 65.0
MASS_LO, MASS_HI = 100.0, 1300.0


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    f = pq.ParquetFile(os.path.join(WORK, "data", "train.parquet"))
    per = defaultdict(lambda: dict(n=0, smi=None, mass=[]))
    ladder = defaultdict(set)
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["inchikey14", "normalized_smiles", "precursor_mz",
                                         "collision_energy_ev", "instrument_type"])
        for k, s, m, v, ins in zip(t.column("inchikey14").to_pylist(),
                                   t.column("normalized_smiles").to_pylist(),
                                   t.column("precursor_mz").to_pylist(),
                                   t.column("collision_energy_ev").to_pylist(),
                                   t.column("instrument_type").to_pylist()):
            e = per[k]; e["n"] += 1
            if e["smi"] is None: e["smi"] = s
            if m is not None: e["mass"].append(m)
            if "timstof" in str(ins).lower() and isinstance(v, list) and len(v) == 1:
                ev = float(v[0])
                if E_LO <= ev <= E_HI: ladder[k].add(ev)
        del t
    msg = pickle.load(open(f"{SPLITS}/massspecgym_keys.pkl", "rb"))["keys"]
    log(f"{len(per):,} structures | MassSpecGym {len(msg):,}")

    elig, drop = {}, Counter()
    for k, e in per.items():
        if e["n"] < 2 or not e["smi"]: drop["too few spectra / no smiles"] += 1; continue
        if not e["mass"]: drop["no precursor mass"] += 1; continue
        if not (MASS_LO <= float(np.median(e["mass"])) <= MASS_HI):
            drop["mass out of range"] += 1; continue
        if len(ladder.get(k, ())) < 2: drop["fewer than 2 timsTOF energies"] += 1; continue
        sk = scoring.key14(e["smi"])
        if sk is None: drop["unscorable smiles"] += 1; continue
        if sk not in msg: drop["NOT in MassSpecGym (control 1 inverted)"] += 1; continue
        elig[k] = sk
    log(f"eligible (energy-rich AND memorised by ICEBERG): {len(elig):,}")
    for kk, v in drop.most_common(): log(f"    dropped {v:7,d}  {kk}")

    rng = np.random.default_rng(2026)
    chosen, seen = [], set()
    for k in rng.permutation(sorted(elig)):
        if len(chosen) >= N_TOTAL: break
        if elig[k] in seen: continue
        seen.add(elig[k]); chosen.append(k)

    path = f"{SPLITS}/valset_energy_contaminated.json"
    json.dump(dict(n_total=N_TOTAL, energy_band=[E_LO, E_HI],
                   note="CONTAMINATED BY DESIGN: every truth is inside MassSpecGym, which is "
                        "ICEBERG's training set. Upper bound on the mechanism, never an estimate "
                        "of competition performance.",
                   queries=dict(contaminated=sorted(chosen))), open(path, "w"))
    print(f"\nselected {len(chosen)} contaminated energy-rich queries -> {path}")
    print(f"pool available: {len(elig):,} structures")
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
