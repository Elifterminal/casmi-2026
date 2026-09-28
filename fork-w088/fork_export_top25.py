#!/usr/bin/env python3
"""Run the W088 fork on our energy-rich split and export ITS top 25 per query, for bond-energy rerank.

WHY THIS EXISTS. E70 showed that summed bond dissociation energy at the lowest collision energy
separates the truth from its same-mass rivals (0.605, label-controlled). E72 showed why a linear
ranker could never have used it -- its value is conditional -- and that a tree ensemble can. The fork
already carries a tree ensemble: prvsiyan's 31-feature GBMs plus our own blocks, blended. So the
feature belongs on the fork's candidates, not on ours.

WHY IT CANNOT BE AN ORDINARY FEATURE BLOCK. Blocks are evaluated for EVERY candidate, and the fork
ranks hundreds per molecule. ICEBERG at that scale is roughly 60 hours for the test set. The feature
is only affordable over a shortlist, so the design is a SECOND STAGE over the fork's top 25:
400 molecules x 25 candidates x 3 energies is about 30,000 predictions, near 3 hours, which is the
number worth deciding on before spending it.

WHAT THAT BOUNDS. Reordering 25 can lift a truth that is already in the list and can never add one.
So the ceiling is set by how often the truth is inside the fork's top 25 at all, and the gain by how
far down it sits. This script reports both, because they decide whether the rerank is worth running
at all -- if the truth is nearly always at rank 1 already, there is nothing to win.

HOW THE SCORES ARE OBTAINED. casmi_engine.main() does not return per-candidate scores, only
submissions and records. Rather than edit a file that carries another author's code, make_submissions
is wrapped to capture the score sets on their way through. Non-invasive, and it captures exactly what
the submission was built from.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.expanduser("~/casmi-2026/work")
ASSETS = os.path.expanduser("~/casmi-2026/local-assets")
DATA = os.path.join(WORK, "data")
SPLITS = os.path.join(WORK, "splits")
OUT = os.path.expanduser("~/casmi-2026/exports")
WORKDIR = os.path.join(HERE, "localrun_bde")
SUPPORTED = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+H-H2O]+"}
ADDUCT_FIX = {"[M+H]1+": "[M+H]+", "[M-H]1-": "[M-H]-"}
MAX_ENERGIES = 3


def build_inputs(split, limit, log):
    """Query spectra are the SPLIT's own rows, so the collision-energy ladder survives intact.

    run_local.py picks query spectra itself (prefer timsTOF, cap at three). That is right for an MRR
    estimate and wrong here: the whole point of split e64energy is that its query rows were chosen to
    carry >=2 distinct single-valued energies, and re-picking them would throw the ladder away.
    """
    os.makedirs(WORKDIR, exist_ok=True)
    sp = json.load(open(f"{SPLITS}/split_{split}.json"))
    qrows = {k: v for k, v in sp["query_rows"].items()}
    keys = sorted(qrows)
    if limit: keys = keys[:limit]
    keep_rows = sorted({r for k in keys for r in qrows[k]})
    log(f"{len(keys)} query molecules, {len(keep_rows)} query spectra from split {split}")

    src = pq.ParquetFile(f"{DATA}/train.parquet")
    qp, lp = f"{WORKDIR}/val_queries.parquet", f"{WORKDIR}/train_minus_val.parquet"
    wq = wl = None
    smi_of, nq, nl, base, sid = {}, 0, 0, 0, 0
    want_rows = set(keep_rows)
    hold = set(keys)                     # every spectrum of a held-out STRUCTURE leaves the library
    for i in range(src.num_row_groups):
        t = src.read_row_group(i)
        n = t.num_rows
        ik = np.asarray(t.column("inchikey14"))
        sm = np.asarray(t.column("normalized_smiles"))
        idx = np.arange(base, base + n)
        qmask = np.isin(idx, list(want_rows))
        if qmask.any():
            for k, s in zip(ik[qmask], sm[qmask]): smi_of.setdefault(k, s)
            q = t.filter(pa.array(qmask))
            q = q.append_column("molecule_id", q.column("inchikey14"))
            if "spectrum_id" not in q.schema.names:
                q = q.append_column("spectrum_id", pa.array(np.arange(sid, sid + q.num_rows)))
            sid += q.num_rows
            if wq is None: wq = pq.ParquetWriter(qp, q.schema)
            wq.write_table(q); nq += q.num_rows
        lmask = ~np.isin(ik, list(hold))
        if lmask.any():
            keep = t.filter(pa.array(lmask))
            if wl is None: wl = pq.ParquetWriter(lp, keep.schema)
            wl.write_table(keep); nl += keep.num_rows
        base += n
        del t
    if wq: wq.close()
    if wl: wl.close()
    log(f"held out {nq} query spectra over {len(smi_of)} structures; library keeps {nl:,}")
    sample = f"{WORKDIR}/val_sample_submission.csv"
    pd.DataFrame({"molecule_id": sorted(smi_of), "smiles": "CCO"}).to_csv(sample, index=False)
    return qp, lp, sample, smi_of, qrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="e64energy_n600")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--w-pv", type=float, default=0.88)
    ap.add_argument("--topn", type=int, default=25)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    os.environ["CASMI_LOCAL"] = "1"
    sys.path.insert(0, HERE)
    sys.path.insert(0, WORK); sys.path.insert(0, os.path.join(WORK, "notebook"))
    from casmi_pipeline import key14
    globals()["key14"] = key14
    import casmi_engine as E
    E.ROOTS = [ASSETS, DATA]
    log("engine loaded")

    qp, lp, sample, smi_of, qrows = build_inputs(a.split, a.limit, log)

    # The pool is built from the LIBRARY, and build_inputs strips every held-out structure's
    # spectra from it -- so without this the answers are not candidates at all and the fork scores
    # zero by construction. My first run measured 0 of 35 truths in the top 25 for exactly that
    # reason; run_local.py already solved it and I failed to carry it over. Restoring the structure
    # (not its spectra) is the same thing the real setting offers: a molecule present in some
    # searchable database whose spectrum nobody has measured.
    orig_build_pool = E.build_pool
    def build_pool_with_val(L, workers):
        P = orig_build_pool(L, workers)
        have = set(P["keys"])
        add = [(k, sm) for k, sm in smi_of.items() if k not in have]
        res = [(k, E.fp_and_mass(sm)) for k, sm in add]
        ok = [(k, r) for k, r in res if r is not None]
        if ok:
            fp = np.vstack([P["fp"]] + [r[0][None, :] for _, r in ok])
            mass = np.concatenate([P["mass"], [r[1] for _, r in ok]])
            keys = np.concatenate([P["keys"], [k for k, _ in ok]])
            smis = np.concatenate([P["smiles"], [smi_of[k] for k, _ in ok]])
            o = np.argsort(mass, kind="mergesort")
            P = dict(fp=fp[o], mass=mass[o], keys=keys[o], smiles=smis[o])
            P["k2i"] = pd.Series(np.arange(len(o)), index=P["keys"])
            P["k2i"] = P["k2i"][~P["k2i"].index.duplicated()]
            log(f"restored {len(ok)} held-out structures to the pool (pool {len(o):,})")
        return P
    E.build_pool = build_pool_with_val

    captured = {}
    orig = E.make_submissions
    def capture(mols, recs, score_sets, *args, **kw):
        captured.update(score_sets)            # {name: {molecule_id: per-candidate scores}}
        return orig(mols, recs, score_sets, *args, **kw)
    E.make_submissions = capture

    import glob
    fp_models = sorted(glob.glob(f"{ASSETS}/**/fp_*.pt", recursive=True))
    subs, recs = E.main(qp, lp, sample, E.find("sim_rank_rows_nofp.npz"), E.find("rank_train.npz"),
                        fp_models, workers=a.workers, w_pv=a.w_pv)
    log(f"engine done; captured score sets: {sorted(captured)}")

    # observed energies per query, from the rows the split designated
    t = pq.read_table(f"{DATA}/train.parquet",
                      columns=["precursor_mz", "adduct", "instrument_type", "collision_energy_ev"])
    pmz = t.column("precursor_mz").to_numpy(); add = np.asarray(t.column("adduct"))
    inst = np.asarray(t.column("instrument_type")); ce = t.column("collision_energy_ev").to_pylist()
    del t
    pools = {p["k"]: p for p in __import__("pickle").load(
        open(f"{SPLITS}/pools_fr_{a.split}.pkl", "rb"))}

    score = captured.get("blend") or captured[sorted(captured)[0]]
    rec_of = {r["mid"]: r for r in recs}
    smiles_pool = E.POOL["smiles"]
    queries, stats = [], {"no score": 0, "no energies": 0, "adduct": 0, "exported": 0,
                          "truth in top25": 0, "truth ranks": []}
    for k in sorted(set(qrows) & set(rec_of)):
        r = rec_of[k]
        s = score.get(k)
        if s is None or not len(s): stats["no score"] += 1; continue
        rows = qrows[k]
        energies = []
        for rw in rows:
            v = ce[rw]
            vals = v if isinstance(v, list) else ([v] if isinstance(v, (int, float)) else [])
            vals = [float(x) for x in vals if x is not None and np.isfinite(x)]
            if len(vals) == 1: energies.append(vals[0])
        energies = sorted(set(energies))[:MAX_ENERGIES]
        if len(energies) < 2: stats["no energies"] += 1; continue
        raw = str(add[rows[0]]); adduct = ADDUCT_FIX.get(raw, raw)
        if adduct not in SUPPORTED: stats["adduct"] += 1; continue

        order = np.argsort(-np.asarray(s))[: a.topn]
        cand_idx = np.asarray(r["cand"])[order]
        cands = [{"key": str(int(ci)), "smiles": str(smiles_pool[ci])} for ci in cand_idx]
        truth = pools[k]["truth"] if k in pools else None
        # where does the answer sit -- in the shortlist, or merely in the pool? The first version of
        # this script reported a counter it never incremented, which read as 0% and looked like a
        # finding. Compute it here, and separate "not in the top 25" from "not a candidate at all".
        tr_rank = next((i for i, c in enumerate(cands, 1)
                        if c["smiles"] and key14(c["smiles"]) == truth), None)
        if tr_rank:
            stats["truth in top25"] += 1
            stats["truth ranks"].append(tr_rank)
        else:
            full = np.asarray(r["cand"])
            in_pool = any(key14(str(smiles_pool[ci])) == truth for ci in full)
            stats["truth in pool but below 25" if in_pool else "truth NOT a candidate"] = \
                stats.get("truth in pool but below 25" if in_pool else "truth NOT a candidate", 0) + 1
        ins = str(inst[rows[0]])
        queries.append(dict(split=a.split, key=k, cls=pools[k]["cls"] if k in pools else 0,
                            truth=truth, precursor=float(np.median([pmz[rw] for rw in rows])),
                            adduct=adduct, adduct_supported=True,
                            instrument="QTOF" if "tof" in ins.lower() else "Orbitrap",
                            positive=bool(pools[k]["positive"]) if k in pools else True,
                            energy_recorded=True, energies=energies,
                            n_pool=int(len(s)), fork_rank=list(range(1, len(cands) + 1)),
                            fork_score=[float(s[i]) for i in order],
                            candidates=cands))
        stats["exported"] += 1

    path = f"{OUT}/fork_top25_{a.split}.json"
    json.dump(dict(source="W088 fork blend score", topn=a.topn, queries=queries), open(path, "w"))
    log(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    rk = np.array(stats["truth ranks"], dtype=float)
    n = max(stats["exported"], 1)
    print("\nexport summary:")
    for kk, v in stats.items():
        if kk != "truth ranks": print(f"  {kk:22s} {v}")
    print(f"\n{stats['exported']} queries exported with the fork's top {a.topn}")
    if len(rk):
        as_is = float(np.sum(1.0 / rk) / n)
        ceiling = len(rk) / n
        print(f"  truth inside the top {a.topn}: {len(rk)}/{n} = {len(rk)/n:.1%}; "
              f"rank median {int(np.median(rk))}, at rank 1 {np.mean(rk == 1):.1%}")
        print(f"  MRR@25 in the fork's own order      {as_is:.4f}")
        print(f"  MRR@25 if a PERFECT rerank ran      {ceiling:.4f}")
        print(f"  -> headroom available to any rerank {ceiling - as_is:+.4f}")
    print(f"ICEBERG cost at E62's measured 2.66 predictions/s: "
          f"{sum(len(q['candidates']) * len(q['energies']) for q in queries) / 2.66 / 60:.0f} min")
    print("\nNote: whether the rerank CAN help is decided by how often the truth is inside these 25")
    print("and how far down it sits -- e73 measures that from this file before any ICEBERG time is")
    print("spent, because a truth already at rank 1 has nothing to gain.")


if __name__ == "__main__":
    main()
