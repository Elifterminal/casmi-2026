#!/usr/bin/env python3
"""Run the forked W088 engine locally against our held-out validation set.

WHY LOCAL. Two Kaggle runs died with no traceback -- an OOM kill leaves none -- and each cost 47
to 65 minutes to discover. The engine ships with CASMI_LOCAL=1 support, so we can watch memory
directly, start small, and iterate in minutes.

Its LOCAL root is hardcoded to the author's Windows desktop
(C:\\Users\\HW-LEE\\Desktop\\CASMI), so ROOTS is repointed at our asset directory after import.

WHAT IT MEASURES. MRR@25 over our 600 held-out queries (E61), reported separately for the two arms:

  COCONUT arm      structure stays in the candidate pool, spectra removed -> a faithful Class 2
  non-COCONUT arm  same, with the structure restored to the pool by this script

Every number carries the same caveat as the Kaggle version: the fork's rankers train on
precomputed rows with no structure identity, so we cannot hold these queries out of THEIR
training. The bias runs HIGH. This is an UPPER BOUND on the board, not an estimate of it.

--limit is the whole point of running locally: start at 50 molecules, confirm the pipeline
completes and watch peak RSS, then raise it until either the full 600 run or we find the ceiling
honestly instead of guessing at it.
"""
import argparse, glob, json, os, resource, sys, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ASSETS = os.path.expanduser("~/casmi-2026/local-assets")
WORKDIR = os.path.expanduser("~/casmi-2026/fork-w088/localrun")
DATA = os.path.expanduser("~/casmi-2026/work/data")
VALSET = os.path.expanduser("~/casmi-2026/work/splits/valset_fork_queries.json")
SPECTRA_PER_MOL = 3


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def build_inputs(limit, log):
    """Held-out queries (capped) + the library with every one of their spectra removed."""
    os.makedirs(WORKDIR, exist_ok=True)
    vs = json.load(open(VALSET))
    arm = {k: a for a, ks in vs["queries"].items() for k in ks}
    keys = set(arm)
    if limit:
        # keep the COCONUT / non-COCONUT balance while shrinking
        co = [k for k in vs["queries"]["coconut"]][: limit // 2]
        nc = [k for k in vs["queries"]["non_coconut"]][: limit - len(co)]
        keys = set(co) | set(nc)
    log(f"validation molecules: {len(keys)} "
        f"({sum(1 for k in keys if arm[k]=='coconut')} COCONUT)")

    src = pq.ParquetFile(f"{DATA}/train.parquet")
    qp, lp = f"{WORKDIR}/val_queries.parquet", f"{WORKDIR}/train_minus_val.parquet"
    wq = wl = None
    smi_of, nq, nl, sid = {}, 0, 0, 0
    for i in range(src.num_row_groups):
        t = src.read_row_group(i)
        ik = np.asarray(t.column("inchikey14"))
        mask = np.isin(ik, list(keys))
        if mask.any():
            sm = np.asarray(t.column("normalized_smiles"))
            for k, s in zip(ik[mask], sm[mask]): smi_of.setdefault(k, s)
            inst = (np.asarray(t.column("instrument_type"))
                    if "instrument_type" in t.schema.names else np.array([""] * t.num_rows, object))
            idx = np.where(mask)[0]
            # prefer timsTOF: the test set is 100% timsTOF, so the query side should look like it
            order = sorted(idx, key=lambda r: 0 if "tof" in str(inst[r]).lower() else 1)
            take, cnt = [], {}
            for r in order:
                k = ik[r]
                if cnt.get(k, 0) >= SPECTRA_PER_MOL: continue
                cnt[k] = cnt.get(k, 0) + 1; take.append(r)
            if take:
                sel = np.zeros(t.num_rows, bool); sel[np.array(take)] = True
                q = t.filter(pa.array(sel))
                q = q.append_column("molecule_id", q.column("inchikey14"))
                if "spectrum_id" not in q.schema.names:
                    q = q.append_column("spectrum_id", pa.array(np.arange(sid, sid + q.num_rows)))
                sid += q.num_rows
                if wq is None: wq = pq.ParquetWriter(qp, q.schema)
                wq.write_table(q); nq += q.num_rows
        # the LIBRARY drops every spectrum of a held-out structure, capped or not
        keep = t.filter(pa.array(~mask))
        if keep.num_rows:
            if wl is None: wl = pq.ParquetWriter(lp, keep.schema)
            wl.write_table(keep); nl += keep.num_rows
        del t, keep
    if wq: wq.close()
    if wl: wl.close()
    log(f"held out {nq:,} query spectra over {len(smi_of)} structures "
        f"({nq/max(len(smi_of),1):.1f} each); library keeps {nl:,}")
    # make_submissions LEFT-MERGES its output onto sample_submission.csv, so a real sample file
    # would keep the 400 TEST ids and silently discard every validation molecule -- which is
    # exactly what the first local run did: 400 rows of "CCO" filler and zero scored queries.
    # Hand it a sample containing OUR ids instead.
    sp = f"{WORKDIR}/val_sample_submission.csv"
    pd.DataFrame({"molecule_id": sorted(smi_of), "smiles": "CCO"}).to_csv(sp, index=False)
    log(f"validation sample file: {len(smi_of)} molecule ids -> {os.path.basename(sp)}")
    return qp, lp, smi_of, arm, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="validation molecules (0 = all 600)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--w-pv", type=float, default=0.88)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.0f}s  {rss_gb():5.1f}GB] {m}", flush=True)

    os.environ["CASMI_LOCAL"] = "1"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import casmi_engine as E
    E.ROOTS = [ASSETS, DATA]                    # the shipped LOCAL root is the author's desktop
    log(f"engine loaded; assets at {ASSETS}")

    qp, lp, smi_of, arm, sample_path = build_inputs(a.limit, log)

    orig_build_pool = E.build_pool
    def build_pool_with_val(L, workers):
        P = orig_build_pool(L, workers)
        have = set(P["keys"])
        add = [(k, s) for k, s in smi_of.items() if k not in have]
        if add:
            res = [(k, E.fp_and_mass(s)) for k, s in add]
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

    fp_models = sorted(glob.glob(f"{ASSETS}/**/fp_*.pt", recursive=True))
    log(f"fp models: {[os.path.basename(f) for f in fp_models]}")
    subs, recs = E.main(qp, lp, sample_path,
                        E.find("sim_rank_rows_nofp.npz"), E.find("rank_train.npz"),
                        fp_models, workers=a.workers, w_pv=a.w_pv)
    log("engine finished")

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    def k14(s):
        try:
            m = Chem.MolFromSmiles(s or "")
            return Chem.MolToInchiKey(m).split("-")[0] if m is not None else None
        except Exception: return None
    truth = {k: k14(s) for k, s in smi_of.items()}
    rows = []
    for name, df in subs.items():
        per = {"coconut": [], "non_coconut": []}
        for mid, smis in zip(df.molecule_id, df.smiles):
            t = truth.get(mid)
            if t is None: continue
            rr = 0.0
            for i, s in enumerate(str(smis).split(";")[:25], 1):
                if k14(s) == t: rr = 1.0 / i; break
            per[arm.get(mid, "coconut")].append(rr)
        rows.append(dict(arm=name,
                         coconut_MRR=float(np.mean(per["coconut"])) if per["coconut"] else None,
                         n_co=len(per["coconut"]),
                         non_coconut_MRR=float(np.mean(per["non_coconut"])) if per["non_coconut"] else None,
                         n_non=len(per["non_coconut"])))
    V = pd.DataFrame(rows)
    print("\n" + V.to_string(index=False))
    V.to_csv(f"{WORKDIR}/validation_scores.csv", index=False)
    print(f"\npeak RSS {rss_gb():.1f} GB, {time.time()-t0:.0f}s")
    print("UPPER BOUND on the board, not an estimate: these queries cannot be held out of the")
    print("rankers' own training rows, which carry no structure identity. The bias runs HIGH.")


if __name__ == "__main__":
    main()
