#!/usr/bin/env python3
"""E56 — the control: run Lee's ordering test on molecules ICEBERG ACTUALLY KNOWS.

E54 found the ordering signal at 54.2% (chance) across 96 queries, failing both pre-registered
thresholds. But that test inherits ICEBERG's accuracy, and on our clean splits that accuracy is
poor BY CONSTRUCTION -- we deliberately selected molecules absent from its training data, and the
truth's predicted-vs-observed cosine there has a median of 0.147.

So the null has two readings and they demand different responses:

  THE AXIS IS EMPTY      the order in which fragments appear carries no candidate-discriminating
                         information, and the idea closes
  THE INSTRUMENT IS BLIND on unfamiliar chemistry ICEBERG cannot predict energy behaviour well
                          enough to reveal a signal that is really there

This separates them. Take molecules from the ORIGINAL splits whose structures ARE in MassSpecGym's
training fold -- the ones ICEBERG has effectively memorised -- and run the identical test. If the
ordering signal appears there, the mechanism is real and our problem is predictor accuracy on novel
chemistry. If it is absent there too, the axis genuinely does not carry the information and we can
close it honestly rather than leaving it as a maybe.

This is deliberately a contaminated set. That contamination is the POINT: it is the only way to
observe the mechanism through a predictor operating inside its own domain. Nothing measured here
transfers to a leaderboard claim, and it is not meant to.

EFFICIENCY, learned from E53/E54: filter to MEASURABLE queries before spending prediction time.
Of the 342 queries in the last run only 96 survived -- the rest lacked a truth in the top 25 or a
ladder of two distinct energies. Requiring both upfront costs nothing and saves hours.

PREDICTIONS (before the run, 2026-09-24):
  P1  Per-frame agreement is far higher here than on the clean splits (truth cosine well above the
      0.147 median), confirming the domain difference is real and large.
  P2  If the ordering mechanism exists, truth beats median decoy on diagonal advantage in over 65%
      of queries. Under 58% and I call the axis closed.
  P3  The eliminator ratio exceeds 1.5x here if the mechanism is real; at 1.25x, as measured on the
      clean set, it stays a curiosity.
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
from e22_ranker_v2 import build
from e47_export_work import map_instrument, SUPPORTED_ADDUCTS, ADDUCT_FIX
from casmi_pipeline import key14
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
OUT = os.path.expanduser("~/casmi-2026/exports")
CONTAM = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
TOP_N = 25
TARGET = 160          # enough for a ~+/-8% margin; more is wasted prediction time


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    msg = pickle.load(open(f"{SPLITS}/massspecgym_keys.pkl", "rb"))["keys"]
    log(f"MassSpecGym structures: {len(msg):,}")
    w = np.array(json.load(open(os.path.join(SPLITS, "ranker_w_fr.json")))["w"])

    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["precursor_mz", "adduct", "instrument_type", "collision_energy_ev"])
    pmz = t.column("precursor_mz").to_numpy(); add = np.asarray(t.column("adduct"))
    inst = np.asarray(t.column("instrument_type")); ce_raw = t.column("collision_energy_ev").to_pylist()
    del t
    st = load_structures()

    out, stats = [], defaultdict(int)
    for split in CONTAM:
        if len(out) >= TARGET: break
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{split}.pkl", "rb"))}
        ev = build([split], st, log=lambda m: None, pools_prefix="fr_", generate=False)
        log(f"[{split}] {len(ev)} queries built")
        for q in ev:
            if len(out) >= TARGET: break
            if q["cls"] not in (1, 2) or not q["truth"]: continue
            p = pools.get(q["k"])
            if p is None: continue
            stats["c12"] += 1
            # 1. ICEBERG must KNOW this molecule -- that is the whole point of the control
            if q["truth"] not in msg: continue
            stats["known_to_iceberg"] += 1
            # 2. a real ladder: at least two distinct SINGLE-energy frames
            rows = sp["query_rows"][q["k"]]
            energies = []
            for r in rows:
                ce = ce_raw[r]
                vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
                vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
                if len(vals) == 1: energies.append(vals[0])
            energies = sorted(set(energies))
            if len(energies) < 2: continue
            stats["has_ladder"] += 1
            # 3. the truth must be inside the top 25, or there is nothing to separate
            sc = {c: float(np.dot(w, v)) for c, (v, _) in q["cand"].items()}
            top = sorted(sc, key=lambda c: (-sc[c], str(c)))[:TOP_N]
            if not any(key14(q["cand"][c][1]) == q["truth"] for c in top): continue
            stats["truth_in_top25"] += 1
            raw = str(add[rows[0]]); adduct = ADDUCT_FIX.get(raw, raw)
            if adduct not in SUPPORTED_ADDUCTS: continue
            stats["adduct_ok"] += 1
            out.append(dict(split=split, key=q["k"], cls=q["cls"], truth=q["truth"],
                            precursor=float(np.median([pmz[r] for r in rows])),
                            adduct=adduct, adduct_supported=True,
                            instrument=map_instrument(str(inst[rows[0]])),
                            positive=bool(p["positive"]), energy_recorded=True,
                            energies=energies[:4],
                            candidates=[{"key": str(c), "smiles": q["cand"][c][1]} for c in top]))
            stats["exported"] += 1
            stats["predictions"] += len(top) * len(energies[:4])

    path = f"{OUT}/iceberg_known_control.json"
    json.dump(dict(top_n=TOP_N, note="CONTAMINATED BY DESIGN: every truth is in MassSpecGym's "
                                     "training fold, so ICEBERG is inside its own domain. Measures "
                                     "the mechanism, not transferable performance.",
                   queries=out), open(path, "w"))
    log(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    print("\nfunnel:")
    for k in ("c12", "known_to_iceberg", "has_ladder", "truth_in_top25", "adduct_ok", "exported"):
        print(f"  {k:20s} {stats[k]:,}")
    print(f"  predictions          {stats['predictions']:,}")
    print(f"estimated {stats['predictions']*1.9/7/60:.0f} min on 7 processes")


if __name__ == "__main__":
    main()
