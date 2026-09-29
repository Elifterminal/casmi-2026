#!/usr/bin/env python3
"""E76 — export the fork's top-25 for the CLASS 2 arm, as the work list for a second-stage rerank.

THE TARGET (L-064). On the COCONUT arm -- the faithful Class 2 simulation -- the fork puts the truth
at rank 1 in only 38.3% of queries, at rank 2-3 in another 18.3%, and below 25 in 22.7%. Perfect
reordering of what is already inside the 25 is worth +0.129 on the board, and the rank-2-3 band alone
is +0.048. We need +0.02, which is 15% of that headroom.

WHY THE FORK'S OWN RANKER CANNOT BE EXTENDED. Both of its ensembles train on shipped row matrices
(rank_train.npz, sim_rank_rows_nofp.npz) that carry no structure identity, so we cannot regenerate
them with an extra column. A second stage over their output is the only place a new feature can enter.

NO NEW FORK RUN IS NEEDED. The completed v5 validation run already produced the fork's top 25 per
query, in its own order, as submission.csv -- 600 molecules, 25 SMILES each. Rank is the right scale
anyway: the fork's own blend_scores works on within-molecule rank, not raw probability.

WHAT THIS WRITES. For each Class 2 query with at least one single-valued collision energy, the 25
candidate SMILES with their fork rank, plus the query's energies. E75 showed the break-depth level
needs only ONE energy and works at 20, 40 and 60 eV alike (0.585 / 0.625 / 0.586), so the ladder
requirement that limited E66 is gone -- which matters because it takes local coverage of this arm from
48% to 53%, and board coverage to essentially all of it.

HONEST LIMIT ON WHAT FOLLOWS. 300 Class 2 queries, of which about 158 carry a usable energy. That is
a small set to train a reranker on, so the evaluation has to be cross-validated and scaffold-grouped
rather than a single holdout, and the result will carry a wide interval. Reported as such.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
OUT = os.path.expanduser("~/casmi-2026/exports")
SUB = ("/tmp/claude-1000/-home-lee/c046c9a5-0d6b-4917-ae00-430a3bba9009/scratchpad/nbout2/"
       "submission.csv")
MAX_ENERGIES = 3
SUPPORTED = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+H-H2O]+"}
ADDUCT_FIX = {"[M+H]1+": "[M+H]+", "[M-H]1-": "[M-H]-"}


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    vs = json.load(open(f"{SPLITS}/valset_fork_queries.json"))
    arm = {k: a for a, ks in vs["queries"].items() for k in ks}
    co = set(vs["queries"]["coconut"])
    # Export BOTH arms. Evaluation stays Class-2-only (the COCONUT arm), but 100 queries is thin to
    # train on, and the non-COCONUT arm supplies ~300 more where the feature relationships should be
    # identical even though the difficulty is not (0.785 vs 0.486). Each row is labelled with its arm
    # so the split can train on both and score only the faithful one.
    both = set(arm)

    # the query molecules' own rows: energies, adduct, instrument, precursor, and truth SMILES
    f = pq.ParquetFile(os.path.join(WORK, "data", "train.parquet"))
    rows = defaultdict(list); smi_of = {}
    base = 0
    cols = ["inchikey14", "normalized_smiles", "collision_energy_ev", "adduct",
            "instrument_type", "precursor_mz"]
    meta = {}
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=cols)
        ik = t.column("inchikey14").to_pylist()
        for j, k in enumerate(ik):
            if k not in both: continue
            smi_of.setdefault(k, t.column("normalized_smiles")[j].as_py())
            meta[base + j] = (t.column("collision_energy_ev")[j].as_py(),
                              str(t.column("adduct")[j].as_py()),
                              str(t.column("instrument_type")[j].as_py()),
                              t.column("precursor_mz")[j].as_py())
            rows[k].append(base + j)
        base += t.num_rows
        del t
    log(f"{len(rows)} Class 2 (COCONUT arm) molecules with rows")

    sub = pd.read_csv(SUB)
    order = {m: str(s).split(";")[:25] for m, s in zip(sub.molecule_id, sub.smiles)}
    log(f"fork top-25 loaded for {len(order)} molecules")

    out, stats = [], defaultdict(int)
    for k in sorted(both):
        cands = order.get(k)
        if not cands: stats["no fork ranking"] += 1; continue
        if set(cands) == {"CCO"}: stats["fork produced only filler"] += 1; continue
        energies, add_s, inst_s, pmz = [], None, None, []
        for r in rows.get(k, []):
            ce, a, ins, p = meta[r]
            vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
            vals = [float(x) for x in vals if x is not None and np.isfinite(x)]
            if len(vals) == 1: energies.append(vals[0])
            if add_s is None: add_s, inst_s = a, ins
            if p is not None: pmz.append(float(p))
        energies = sorted(set(energies))[:MAX_ENERGIES]
        if not energies: stats["no single-valued energy"] += 1; continue
        adduct = ADDUCT_FIX.get(add_s, add_s)
        if adduct not in SUPPORTED: stats[f"adduct unsupported"] += 1; continue
        truth = scoring.key14(smi_of.get(k, ""))
        if truth is None: stats["truth unscorable"] += 1; continue
        # rank is the feature; index is the key so duplicate SMILES stay distinct candidates
        cl = [{"key": f"r{i}", "smiles": s, "fork_rank": i}
              for i, s in enumerate(cands, 1) if s and s != "CCO"]
        if len(cl) < 2: stats["fewer than 2 real candidates"] += 1; continue
        tr = next((c["fork_rank"] for c in cl
                   if scoring.key14(c["smiles"]) == truth), None)
        out.append(dict(split="valset_fork", arm=arm[k], key=k,
                        cls=2 if arm[k] == "coconut" else 0, truth=truth,
                        precursor=float(np.median(pmz)) if pmz else float("nan"),
                        adduct=adduct, adduct_supported=True,
                        instrument="QTOF" if "tof" in inst_s.lower() else "Orbitrap",
                        positive=not adduct.endswith("-"), energy_recorded=True,
                        energies=energies, n_pool=len(cl),
                        truth_fork_rank=tr, candidates=cl))
        stats[f"exported ({arm[k]})"] += 1
        stats["exported"] += 1
        stats["truth inside the 25"] += (tr is not None)
        stats["predictions"] += len(cl) * len(energies)

    path = f"{OUT}/c2_rerank_worklist.json"
    json.dump(dict(source="W088 fork v5 validation submission, COCONUT arm",
                   note="candidates are the fork's top 25 IN ITS OWN ORDER; fork_rank is the prior",
                   queries=out), open(path, "w"))
    log(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    print("\nexport summary:")
    for kk in sorted(stats): print(f"  {kk:34s} {stats[kk]:,}")
    n = stats["exported"]
    if n:
        print(f"\n{n} Class 2 queries exported; truth inside the 25 for "
              f"{stats['truth inside the 25']} ({stats['truth inside the 25']/n:.1%})")
        print(f"ICEBERG cost at 2.66 predictions/s: {stats['predictions']/2.66/60:.0f} min")
        for a in ("coconut", "non_coconut"):
            qs = [q for q in out if q["arm"] == a]
            rk = np.array([q["truth_fork_rank"] for q in qs if q["truth_fork_rank"]], float)
            if not len(qs) or not len(rk): continue
            mrr = float(np.sum(1.0 / rk) / len(qs))
            print(f"  {a:12s} n={len(qs):4d}  fork MRR@25 {mrr:.4f}  "
                  f"ceiling {len(rk)/len(qs):.4f}  headroom {len(rk)/len(qs) - mrr:+.4f}")


if __name__ == "__main__":
    main()
