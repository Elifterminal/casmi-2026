#!/usr/bin/env python3
"""E65 — export the bond-depth work list from the MASS-WINDOW POOL, not the scorer's top 25.

WHY THE TOP-25 ROUTE IS A DEAD END. The scorer draws its candidates from COCONUT. A truth that is
not in COCONUT can never be ranked at all, so it can never appear in a top-25 work list. Counting
the structures in all of training that satisfy every requirement the energy test needs at once:

    in COCONUT (so the scorer can rank it)  +  >=2 single-valued timsTOF collision energies
    +  absent from MassSpecGym (so ICEBERG has not memorised it)  +  sane mass  +  >=2 spectra
    = 49 structures.   With scaffold rarity as well: 28.

So E63's n=28 was not an accident of split 60. It is approximately the ceiling this dataset allows
for that design, and no amount of re-splitting raises it.

WHAT CHANGES HERE. The question "does implied break depth separate the truth from same-mass
decoys?" does not need the scorer's database. It needs a candidate set containing the truth and
plausible decoys, and the mass-window pool is exactly that: it holds the truth for 93.4% of Class
1+2 queries, with a median of 71 candidates drawn from the training structures. That gives ~340
measurable queries instead of 28.

THE CLAIM THIS CAN AND CANNOT SUPPORT, stated before the run. It can support "break depth does /
does not separate the truth from its same-mass rivals". It cannot support "adding break depth
improves MRR", because these candidates are not the ones the shipping ranker sees. The first has to
hold for the second to be possible, which is why it is worth measuring first and separately.

DECOY CHOICE. The 24 pool candidates NEAREST IN MASS to the truth, excluding scorer-key twins of
the truth. Nearest-in-mass is the hardest honest choice: these are the near-isomers whose mass
evidence is least distinguishable, which is the regime E43 showed bounds us. Random pool decoys
would be easier and would flatter the result.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
OUT = os.path.expanduser("~/casmi-2026/exports")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "e64energy_n600"
N_DECOY = 24
MAX_ENERGIES = 3
SUPPORTED = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+H-H2O]+"}
ADDUCT_FIX = {"[M+H]1+": "[M+H]+", "[M-H]1-": "[M-H]-"}


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    sp = json.load(open(f"{SPLITS}/split_{SPLIT}.json"))
    pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{SPLIT}.pkl", "rb"))}
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["precursor_mz", "adduct", "instrument_type", "collision_energy_ev"])
    pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct")); inst = np.asarray(t.column("instrument_type"))
    ce_raw = t.column("collision_energy_ev").to_pylist(); del t
    log("metadata loaded")

    mw = {}
    def exact(smi):
        if smi not in mw:
            m = Chem.MolFromSmiles(smi or "")
            mw[smi] = Descriptors.ExactMolWt(m) if m is not None else np.nan
        return mw[smi]

    out, stats = [], defaultdict(int)
    for k, rows in sp["query_rows"].items():
        cls = sp["assign"][k]
        p = pools.get(k)
        if p is None or cls == 3:
            stats["skipped class 3 (truth absent from the pool by design)" if cls == 3
                  else "no pool"] += 1
            continue
        truth = p["truth"]
        if not truth: stats["no truth key"] += 1; continue

        # the pool's mass-window candidates, keyed by training InChIKey14 -> smiles
        cand_smi = {c: s for c, s in p["mass"].items() if s}
        tk = [c for c, s in cand_smi.items() if scoring.key14(s) == truth]
        if not tk: stats["truth not in the mass-window pool"] += 1; continue
        tc = tk[0]
        tmass = exact(cand_smi[tc])
        if not np.isfinite(tmass): stats["truth mass unparseable"] += 1; continue

        # decoys: nearest in mass, never a scorer-key twin of the truth
        pairs = []
        for c, s in cand_smi.items():
            if c == tc: continue
            if scoring.key14(s) == truth: continue        # twin of the answer, not a decoy
            m = exact(s)
            if np.isfinite(m): pairs.append((abs(m - tmass), c, s))
        if len(pairs) < 4: stats["fewer than 4 usable decoys"] += 1; continue
        pairs.sort(key=lambda x: (x[0], x[1]))
        chosen = [(tc, cand_smi[tc])] + [(c, s) for _, c, s in pairs[:N_DECOY]]

        # energies: only rows carrying exactly ONE recorded value can be placed on the ladder
        energies = []
        for r in rows:
            ce = ce_raw[r]
            vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
            vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
            if len(vals) == 1: energies.append(vals[0])
        energies = sorted(set(energies))[:MAX_ENERGIES]
        if len(energies) < 2: stats["fewer than 2 single-valued energies"] += 1; continue

        raw = str(add[rows[0]]); adduct = ADDUCT_FIX.get(raw, raw)
        if adduct not in SUPPORTED: stats[f"adduct unsupported ({adduct})"] += 1; continue
        ins = str(inst[rows[0]])
        instrument = "QTOF" if "tof" in ins.lower() else ("Orbitrap" if "orbi" in ins.lower()
                                                          else "Unknown")
        out.append(dict(split=SPLIT, key=k, cls=cls, truth=truth,
                        precursor=float(np.median([pmz[r] for r in rows])),
                        adduct=adduct, adduct_supported=True, instrument=instrument,
                        positive=bool(p["positive"]), energy_recorded=True, energies=energies,
                        n_pool=len(cand_smi),
                        candidates=[{"key": str(c), "smiles": s} for c, s in chosen]))
        stats["exported"] += 1
        stats[f"cls{cls}"] += 1
        stats["predictions"] += len(chosen) * len(energies)

    path = f"{OUT}/iceberg_bonds_pool_{SPLIT}.json"
    json.dump(dict(n_decoy=N_DECOY, candidate_source="mass-window pool, nearest-in-mass decoys",
                   truth_always_present=True, queries=out), open(path, "w"))
    log(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    print("\nwork list summary:")
    for k in sorted(stats): print(f"  {k:52s} {stats[k]:,}")
    print(f"\n{stats['exported']} queries, every one with the truth in its candidate set "
          f"by construction")
    print(f"estimated wall time at E62's measured 2.66 predictions/s on 7 processes: "
          f"{stats['predictions']/2.66/60:.0f} min")


if __name__ == "__main__":
    main()
