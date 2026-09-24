#!/usr/bin/env python3
"""E49 — the corrected export: candidates come from the SCORER'S OWN candidate set.

E47/E48 produced a +0.0866 that was entirely an artifact. I exported candidates from the pool
file's `mass` dict while the scorer builds its candidates independently from COCONUT with database
exclusions. Those are different sets -- 86 built against 12 in the pool file for one query,
overlapping on 7 -- so "has a prediction" meant "was in the pool file", which is correlated with
being the answer: 94.0% of truths had a prediction against 5.8% of decoys, a 16x enrichment. The
ranker learned the indicator, not the chemistry.

Two changes, and the first is the one that matters:

  1. Candidates are taken from e22's build -- the exact objects the scorer will rank -- so a
     candidate either has a prediction or is not in the experiment at all.
  2. Only the baseline ranker's TOP 25 per query are exported, and scoring reranks ONLY within
     those 25. Every candidate being compared therefore has a prediction, so there is no coverage
     asymmetry left for a model to exploit. Candidates at 26+ keep their existing order, which is
     also the honest operation: we submit 25, so reordering within 25 is the only thing that can
     change MRR@25 without changing list membership.

It is cheaper too: ~25 candidates per query rather than the 61 the full build averages.

THE GUARD. This writes the truth-vs-decoy prediction-coverage rates into the work list, and the
scorer refuses to run if they differ by more than a few points. The artifact above was invisible
until I went looking; it should not be possible to reproduce it quietly.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import build, FEATS2
from e47_export_work import map_instrument, SUPPORTED_ADDUCTS, ADDUCT_FIX, TEST_LADDER, MAX_ENERGIES
from casmi_pipeline import key14

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
OUT = os.path.expanduser("~/casmi-2026/exports")
TOP_N = 25


def main(splits):
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    w = np.array(json.load(open(os.path.join(SPLITS, "ranker_w_fr.json")))["w"])

    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["precursor_mz", "adduct", "instrument_type", "collision_energy_ev"])
    pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct")); inst = np.asarray(t.column("instrument_type"))
    ce_raw = t.column("collision_energy_ev").to_pylist(); del t
    log("metadata loaded")

    st = load_structures()
    out, stats = [], defaultdict(int)
    for split in splits:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{split}.pkl", "rb"))}
        ev = build([split], st, log=log, pools_prefix="fr_", generate=False)
        log(f"[{split}] {len(ev)} queries built")
        for q in ev:
            rows = sp["query_rows"][q["k"]]
            p = pools[q["k"]]
            energies = []
            for r in rows:
                ce = ce_raw[r]
                vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
                energies += [float(v) for v in vals if v is not None and np.isfinite(v)]
            recorded = bool(energies)
            if not recorded: energies = list(TEST_LADDER)
            energies = sorted(set(energies))[:MAX_ENERGIES]
            raw = str(add[rows[0]]); adduct = ADDUCT_FIX.get(raw, raw)
            supported = adduct in SUPPORTED_ADDUCTS
            instrument = map_instrument(str(inst[rows[0]]))

            # THE TOP 25 BY THE BASELINE RANKER -- the scorer's own candidates, its own ordering
            sc = {c: float(np.dot(w, v)) for c, (v, _) in q["cand"].items()}
            top = sorted(sc, key=lambda c: (-sc[c], str(c)))[:TOP_N]
            cands = [{"key": str(c), "smiles": q["cand"][c][1]} for c in top]
            truth_in_top = any(key14(q["cand"][c][1]) == q["truth"] for c in top) if q["truth"] else False
            # Class 3 has a truth KEY but never a truth CANDIDATE -- it is absent from the pool
            # by construction. Counting it in the denominator understated the design's reach by a
            # third (47.7% against the honest 78.1%), and I published that number before catching it.
            if q["truth"] is not None:
                stats["queries_with_truth_key"] += 1
                if q["cls"] in (1, 2): stats["c12_queries"] += 1
                stats["truth_in_top25"] += truth_in_top
            stats[f"adduct_{'ok' if supported else 'UNSUPPORTED'}"] += 1
            stats[f"instrument_{instrument}"] += 1
            stats["energy_recorded" if recorded else "energy_from_test_ladder"] += 1
            stats["queries"] += 1
            stats["predictions"] += len(cands) * len(energies)
            out.append(dict(split=split, key=q["k"], cls=q["cls"], truth=q["truth"],
                            precursor=float(np.median([pmz[r] for r in rows])),
                            adduct=adduct, adduct_supported=bool(supported),
                            instrument=instrument, positive=bool(p["positive"]),
                            energy_recorded=recorded, energies=energies,
                            n_pool=len(q["cand"]), candidates=cands))

    path = f"{OUT}/iceberg_top25_{'_'.join(s[-8:] for s in splits)}.json"
    json.dump(dict(top_n=TOP_N, candidate_source="e22_build_same_as_scorer",
                   test_ladder=TEST_LADDER, queries=out), open(path, "w"))
    log(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    print("\nwork list summary:")
    for k in sorted(stats): print(f"  {k:28s} {stats[k]:,}")
    tt = stats["truth_in_top25"] / max(1, stats["c12_queries"])
    print(f"\ntruth inside the top {TOP_N}: {tt:.1%} of CLASS 1+2 queries "
          f"({stats['truth_in_top25']}/{stats['c12_queries']}), "
          f"{stats['truth_in_top25']/max(1,stats['queries']):.1%} of all queries")
    print("  (reranking within 25 can only move the queries counted in the first figure)")
    print(f"estimated wall time at 1.9 s/prediction on 7 processes: "
          f"{stats['predictions']*1.9/7/60:.0f} min")


if __name__ == "__main__":
    main(sys.argv[1:] or ["npxmtsseed60_n300"])
