#!/usr/bin/env python3
"""E47 step 1 — export the prediction work list for ICEBERG, from the project venv.

Deliberately split across two environments. ICEBERG needs numpy<2, torch 2.2.1, dgl 2.1 and
rdkit 2025.03; our pipeline runs numpy 2 and rdkit 2026.3. Six version conflicts went into making
ICEBERG importable at all, and the fastest way to undo that is to let the two dependency sets
touch. So this writes a plain JSON work list, the predictor reads it in its own venv, and scoring
happens back here.

Scope: EVERY candidate in each query's 10 ppm mass pool, not the top-k. The clean splits have a
mean pool of 21 (median 6), so the whole pool costs about the same as reranking the top 25 would
have, and it removes any dependence on the current ranker's ordering -- which matters because the
ranker is the thing under test.

INSTRUMENT MAPPING, and the trap it avoids. ICEBERG accepts exactly
['Orbitrap', 'QTOF', 'IT-FT', 'Unknown'] and its own normalize_instrument() silently maps anything
else to 'Orbitrap'. Our largest instrument class is timsTOF -- a QTOF-family instrument, and 100%
of the hidden test. Passing the raw string would have embedded the wrong instrument for every
query with no error to notice.

ADDUCTS: supported ones are passed through; unsupported (the dimers, [2M+H]+ and friends) are
flagged so scoring can retain baseline behaviour for those queries and keep them in the
denominator rather than dropping them, which is what ChatGPT specified.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
OUT = os.path.expanduser("~/casmi-2026/exports")
MAX_ENERGIES = 4          # median is 3; cap so one odd query cannot dominate the run
# test.parquet carries a collision energy for EVERY spectrum, and they are 20/40/60 eV. Half our
# own query rows have none recorded, so this is the honest default for those: the ladder the
# deployment condition actually uses, rather than a number I picked.
TEST_LADDER = [20.0, 40.0, 60.0]

def map_instrument(raw):
    """Our instrument strings -> ICEBERG's four categories, case-insensitively.

    Half the unmapped values were case variants ('qtof', 'orbitrap', 'ToF'). Pure ion traps
    ('iontrap', 'Linear Ion Trap') are deliberately NOT mapped to IT-FT: that category means an
    ion-trap/FT hybrid, and calling a pure trap IT-FT would be inventing a capability. Those get
    'Unknown', which is a real category the model was trained with.
    """
    v = (raw or "").strip().lower()
    if not v or v in ("none", "nan"): return "Unknown"
    if "qft" in v or "orbitrap" in v: return "Orbitrap"
    if "itft" in v or ("it" == v) or "ion trap-ft" in v: return "IT-FT"
    if "tof" in v: return "QTOF"                     # timsTOF, qtof, LC-ESI-QTOF, ToF
    if "trap" in v: return "Unknown"                 # pure ion trap is not IT-FT
    return "Unknown"
SUPPORTED_ADDUCTS = {"[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M]+", "[M-H]-", "[M+Cl]-",
                     "[M+CHO2]-", "[M+CH2O2-H]-", "[M-H2O+H]+", "[M+H-H2O]+"}
ADDUCT_FIX = {"[M+CH2O2-H]-": "[M+CHO2]-"}      # same formate adduct, their spelling


def main(splits):
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["inchikey14", "normalized_smiles", "precursor_mz", "adduct",
                               "instrument_type", "ionization_mode", "collision_energy_ev"])
    pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct")); inst = np.asarray(t.column("instrument_type"))
    mode = np.asarray(t.column("ionization_mode"))
    ce_raw = t.column("collision_energy_ev").to_pylist()
    del t
    log("metadata loaded")

    out, stats = [], defaultdict(int)
    for split in splits:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{split}.pkl", "rb"))}
        for k, rows in sp["query_rows"].items():
            p = pools.get(k)
            if p is None: continue
            energies, adducts, insts = [], [], []
            for r in rows:
                ce = ce_raw[r]
                vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
                for v in vals:                       # take EVERY recorded energy, not just the first
                    if v is not None and np.isfinite(v): energies.append(float(v))
                adducts.append(str(add[r])); insts.append(str(inst[r]))
            raw_adduct = adducts[0] if adducts else ""
            adduct = ADDUCT_FIX.get(raw_adduct, raw_adduct)
            supported = adduct in SUPPORTED_ADDUCTS
            instrument = map_instrument(insts[0] if insts else "")
            stats[f"adduct_{'ok' if supported else 'UNSUPPORTED'}"] += 1
            stats[f"instrument_{instrument}"] += 1
            energy_recorded = bool(energies)
            if not energy_recorded:
                energies = list(TEST_LADDER); stats["energy_from_test_ladder"] += 1
            else:
                stats["energy_recorded"] += 1
            energies = sorted(set(energies))[:MAX_ENERGIES]
            cands = [{"key": c, "smiles": s} for c, s in p["mass"].items() if s]
            if not cands: stats["empty_pool"] += 1
            out.append(dict(split=split, key=k, cls=sp["assign"][k], truth=p["truth"],
                            precursor=float(np.median([pmz[r] for r in rows])),
                            adduct=adduct, adduct_supported=bool(supported),
                            instrument=instrument, positive=bool(p["positive"]),
                            energy_recorded=bool(energy_recorded),
                            energies=energies, candidates=cands))
            stats["queries"] += 1
            stats["predictions"] += len(cands) * len(energies)
        log(f"[{split}] cumulative {stats['queries']} queries, "
            f"{stats['predictions']:,} predictions queued")

    path = f"{OUT}/iceberg_work_{'_'.join(s[-8:] for s in splits)}.json"
    json.dump(dict(max_energies=MAX_ENERGIES, test_ladder=TEST_LADDER,
                   supported_adducts=sorted(SUPPORTED_ADDUCTS), queries=out),
              open(path, "w"))
    log(f"wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    print("\nwork list summary:")
    for k in sorted(stats): print(f"  {k:28s} {stats[k]:,}")
    est = stats["predictions"] * 0.19 / 7
    print(f"\nestimated wall time at 0.19 s/prediction on 7 processes: {est/60:.0f} min")


if __name__ == "__main__":
    main(sys.argv[1:] or ["npxmtsseed60_n300"])
