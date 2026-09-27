#!/usr/bin/env python3
"""E64 — a validation split that can actually carry an energy-ladder test.

WHY. E52, E54 and E63 all concluded that collision-energy ordering carries no usable signal. All
three were run on query sets where the energy axis was mostly absent. In split 60, 682 of 1,357
query rows had NO usable collision energy and 173 of 300 queries had none at all, so 111 of the
164 queries with predictions were discarded before anything could be measured. What survived was
n=28.

The energy metadata is not scarce in the data; it is scarce in the slice our novelty controls
select. Cross-tabulated by instrument:

    timsTOF      1,154,969 rows    0.0% unusable energy   <- what the test set is, 100% of it
    Orbitrap       745,562 rows   16.3%
    qTof/ToF/None   93,365 rows  100.0%                   <- GNPS-style NP depositions

Our splits ask for obscure non-MassSpecGym natural products, and that request lands on the
depositions that never recorded an energy. Meanwhile the scored test set is the richest data in the
project: 400 molecules, 3.03 spectra each, every row carrying collision_energy_ev, mostly clean
single values on a 20/40/60 eV grid, every spectrum timsTOF. 341 of its 400 molecules (85%) have
two or more distinct energies.

So the energy line of attack has never been tested on data resembling the board. This split fixes
that by adding a fifth control, which happens to pull local and board TOWARDS each other rather
than apart:

  5. ENERGY LADDER (controlled, new). Every query has >=2 distinct single-valued collision energies
     in the 15-65 eV band, measured on timsTOF. 180,272 fame-controlled structures qualify, 134,217
     of them with exactly three energies -- the same shape as the test set.

Controls 1-4 are inherited from E61 unchanged (famousness, COCONUT balance, scaffold rarity,
twin-safety), as is the one thing we cannot control and therefore quote: the fork's rankers train
on rows carrying no structure identity, so local stays an UPPER BOUND on the board.

A note on what control 5 costs. Requiring timsTOF makes the set MORE like the test set on
instrument, and no less novel -- famousness is controlled independently, at the scorer's key. What
it does change is provenance: these structures come from the libraries that record metadata
properly, which may correlate with being better studied in ways MassSpecGym membership does not
capture. That residual runs in the familiar direction, making local read high, and is quoted with
the set rather than assumed away.
"""
import json, os, pickle, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
COCO = os.path.join(WORK, "data", "coconut", "coconut_csv_lite-09-2026.csv")
N_TOTAL = 600        # one pool, not two arms -- see CONTROL 2 IS UNACHIEVABLE HERE below
SCAF_MAX = 3
CACHE = "e64_eligible_cache.pkl"
MASS_LO, MASS_HI = 100.0, 1300.0
E_LO, E_HI = 15.0, 65.0      # the test grid is 20/40/60
MIN_ENERGIES = 2


def _scaf(s):
    try:
        m = Chem.MolFromSmiles(s or "")
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else None
    except Exception:
        return None


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    import pandas as pd

    cache_path = os.path.join(SPLITS, CACHE)
    if os.path.exists(cache_path):
        elig, ladder, scafs, msg, co = pickle.load(open(cache_path, "rb"))
        log(f"loaded cache: {len(elig):,} eligible, {len(scafs):,} scaffolds")
        return finish(log, t0, elig, ladder, scafs, co)

    f = pq.ParquetFile(os.path.join(WORK, "data", "train.parquet"))
    per = defaultdict(lambda: dict(n=0, smi=None, mass=[]))
    # ladder[k] = {energy: [row, ...]} over timsTOF rows with ONE recorded energy
    ladder = defaultdict(lambda: defaultdict(list))
    row = 0
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
                if E_LO <= ev <= E_HI: ladder[k][ev].append(row)
            row += 1
        del t
    log(f"{len(per):,} structures; {len(ladder):,} have >=1 single-energy timsTOF row in band")

    msg = pickle.load(open(f"{SPLITS}/massspecgym_keys.pkl", "rb"))["keys"]
    co = set(pd.read_csv(COCO, usecols=["standard_inchi_key"], dtype=str)
             .dropna().standard_inchi_key.str.split("-").str[0])
    log(f"MassSpecGym {len(msg):,} structures | COCONUT {len(co):,}")

    elig = {}
    drop = Counter()
    for k, e in per.items():
        if e["n"] < 2 or not e["smi"]: drop["too few spectra / no smiles"] += 1; continue
        if not e["mass"]: drop["no precursor mass"] += 1; continue
        med = float(np.median(e["mass"]))
        if not (MASS_LO <= med <= MASS_HI): drop["mass out of range"] += 1; continue
        if len(ladder.get(k, {})) < MIN_ENERGIES:
            drop[f"fewer than {MIN_ENERGIES} timsTOF energies (control 5)"] += 1; continue
        sk = scoring.key14(e["smi"])
        if sk is None: drop["unscorable smiles"] += 1; continue
        if sk in msg: drop["in MassSpecGym (control 1)"] += 1; continue
        elig[k] = (e["smi"], sk)
    log(f"eligible: {len(elig):,}")
    for kk, v in drop.most_common(): log(f"    dropped {v:7,d}  {kk}")

    with Pool(7) as p:
        scafs = dict(zip(elig, p.map(_scaf, [v[0] for v in elig.values()], chunksize=64)))
    pickle.dump((elig, dict(ladder), scafs, msg, co), open(cache_path, "wb"))
    log(f"cached eligibility to {CACHE} (re-selection is now cheap)")
    return finish(log, t0, elig, ladder, scafs, co)


def finish(log, t0, elig, ladder, scafs, co):
    """Selection and reporting, split out so variants re-run in seconds off the cache."""
    freq = Counter(v for v in scafs.values() if v)
    rare = {k for k, v in scafs.items() if v and freq[v] <= SCAF_MAX}
    log(f"{len(freq):,} distinct scaffolds; scaffold-rare (<={SCAF_MAX}): {len(rare):,}")

    # CONTROL 2 IS UNACHIEVABLE ALONGSIDE CONTROL 5, and the numbers say why rather than my
    # judgement: COCONUT membership runs at 9.5% of all training structures but 0.11% of the
    # energy-rich ones (194 of 180,426). Natural-product spectra in our libraries sit on Orbitrap,
    # qTof and unlabelled instruments -- the ones that never recorded an energy. So a set cannot be
    # both COCONUT-balanced and energy-laddered from this data. The test set is both, because
    # Enveda acquired it themselves on one instrument; we cannot reconstruct that from training.
    # Rather than fake a balance, take every COCONUT member available and fill the rest, then
    # report the share honestly. COCONUT balance exists to let an MRR estimate see the BENEFIT of
    # adding a candidate database as well as its cost; the bond-depth test asks whether a physical
    # quantity separates truth from decoys inside a fixed pool, which that balance does not affect.
    rng = np.random.default_rng(2026)
    chosen, seen = [], set()
    order = ([k for k in rng.permutation(sorted(rare & co))] +
             [k for k in rng.permutation(sorted(rare - co))])
    for k in order:
        if len(chosen) >= N_TOTAL: break
        sk = elig[k][1]
        if sk in seen: continue
        seen.add(sk); chosen.append(k)
    n_co = sum(1 for k in chosen if k in co)
    log(f"selected {len(chosen)} queries; {n_co} COCONUT ({n_co/max(len(chosen),1):.1%})")
    chosen = sorted(chosen)
    arms = {"coconut": sorted(k for k in chosen if k in co),
            "non_coconut": sorted(k for k in chosen if k not in co)}
    rows = {k: {f"{e:g}": ladder[k][e] for e in sorted(ladder[k])} for k in chosen}
    nE = Counter(len(rows[k]) for k in chosen)
    path = f"{SPLITS}/valset_energy_queries.json"
    json.dump(dict(n_total=N_TOTAL, scaffold_max_freq=SCAF_MAX,
                   energy_band=[E_LO, E_HI], min_energies=MIN_ENERGIES,
                   instrument="timsTOF only, matching the test set",
                   controls=dict(famousness="MassSpecGym excluded",
                                 coconut_balance="ABANDONED -- incompatible with control 5; "
                                                 "COCONUT is 9.5% of training structures but "
                                                 "0.11% of energy-rich ones",
                                 scaffold_rarity=f"Murcko scaffold seen <= {SCAF_MAX} times",
                                 twin_safety="one query per scorer key14",
                                 energy_ladder=f">={MIN_ENERGIES} distinct single-valued timsTOF "
                                               f"energies in {E_LO}-{E_HI} eV",
                                 ranker_training_leakage="UNCONTROLLED -- fork training rows carry "
                                                         "no structure identity"),
                   queries={a: sorted(v) for a, v in arms.items()},
                   energy_rows=rows), open(path, "w"))

    L = ["E64 — validation split carrying a real energy ladder", "",
         f"{'arm':16s} {'selected':>9}",
         f"{'COCONUT':16s} {len(arms['coconut']):>9}",
         f"{'non-COCONUT':16s} {len(arms['non_coconut']):>9}",
         f"{'total':16s} {len(chosen):>9}", "",
         "queries by number of distinct timsTOF energies:"]
    for k in sorted(nE): L.append(f"  {nE[k]:4d} queries with {k} energies")
    L += ["",
          "CONTROL 2 (COCONUT BALANCE) IS ABANDONED, and not by preference. COCONUT membership is",
          "9.5% of all 275,810 training structures but 0.11% of the 180,426 energy-rich ones -- 194",
          "structures in total. Natural-product spectra in our libraries sit on Orbitrap, qTof and",
          "unlabelled instruments, which are exactly the ones that recorded no collision energy. A",
          "set cannot be both COCONUT-balanced and energy-laddered from this data.",
          "",
          "That is itself worth writing down: the test set IS all three at once -- natural-product-",
          "like, timsTOF, and energy-rich -- because Enveda acquired it themselves on one instrument.",
          "No validation set drawn from the training libraries can reproduce that combination.",
          "",
          "Balance was there so an MRR estimate could see the BENEFIT of adding a candidate database",
          "and not only its cost. The bond-depth test asks whether a physical quantity separates",
          "truth from decoys inside a fixed pool, which balance does not bear on. It must be",
          "restored before this split is used for any MRR comparison.",
          "",
          "controls: 1 famousness (MassSpecGym excluded) | 2 COCONUT balance ABANDONED (see above) |",
          f"          3 scaffold rarity (<= {SCAF_MAX}) | 4 twin-safety |",
          f"          5 energy ladder (>= {MIN_ENERGIES} single-valued timsTOF energies, "
          f"{E_LO:.0f}-{E_HI:.0f} eV)  <- NEW",
          "",
          "WHY CONTROL 5 EXISTS: in split 60, 173 of 300 queries had no usable collision energy and",
          "the bond-depth test (E63) reduced to n=28. The scarcity is a property of the slice our",
          "novelty controls select -- qTof/ToF/unlabelled NP depositions record no energy -- not of",
          "the data. The scored test set is 100% timsTOF with 85% of molecules carrying >=2 energies,",
          "so the energy line of attack has never been tested on data resembling the board.",
          "",
          "STILL NOT CONTROLLED, quoted with the set: the fork's rankers train on rows with no",
          "structure identity, so these queries cannot be held out of their training. The residual",
          "makes local read HIGH. Local is an upper bound, not an estimate.",
          "",
          "A COST OF CONTROL 5, stated rather than assumed: requiring timsTOF selects the libraries",
          "that record metadata properly, which may correlate with being better studied in ways",
          "MassSpecGym membership does not capture. Same direction as every other residual we have.",
          f"\nwrote {path}",
          f"runtime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E64_energy_split_2026-09-27.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
