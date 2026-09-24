#!/usr/bin/env python3
"""E51 — export all 900 MassSpecGym-free query pools for Vanta.

She measured a Symploke coupling AUC of 0.7500 on the Class-2-vs-3 axis where our own gate managed
0.6152, then priced it honestly: 11 pools, 7 present against 4 absent, 28 pairs, AUC moving in
steps of 1/28, and a magnitude that wanders 0.6071-0.7857 across a tuning knob. Her words: a lead
that clears the existing axis, not a result, and "what would settle it is more pools, which is a
data request, not a method claim."

This is that data request answered: every query from the three clean splits, 900 of them, drawn
from structures absent from MassSpecGym by construction so nothing here is inside the open ICEBERG
weights either.

Per query: each collision-energy frame (m/z, intensity, energy, adduct, instrument, precursor),
and per candidate the SMILES, its full fragment mass list, our spectral score, and an is_truth
flag. Fragment lists are ragged, so they are stored flat with offsets -- nothing padded, nothing
truncated.

ALSO REPORTED, because it came out of her noticing one empty pool and it bounds every
fragment-based method including hers: 101 of 800 non-empty pools (12.6%) have NO fragment masses
across ANY candidate. Those are flagged per query rather than silently averaged in.
"""
import json, os, pickle, sys, time
import numpy as np
import pyarrow.parquet as pq
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e47_export_work import map_instrument
from casmi_pipeline import characterise, key14

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
OUT = os.path.expanduser("~/casmi-2026/exports")
CLEAN = ("npxmtsseed60_n300", "npxmtsseed61_n300", "npxmtsseed62_n300")


def main(splits):
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    os.makedirs(OUT, exist_ok=True)

    qrows, assign, pools = {}, {}, {}
    for s in splits:
        sp = json.load(open(f"{SPLITS}/split_{s}.json"))
        qrows.update(sp["query_rows"]); assign.update(sp["assign"])
        for p in pickle.load(open(f"{SPLITS}/pools_fr_{s}.pkl", "rb")):
            pools[p["k"]] = (s, p)
    keys = sorted(k for k in qrows if k in pools)
    log(f"{len(keys)} queries across {len(splits)} clean splits")

    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["precursor_mz", "adduct", "instrument_type", "ionization_mode",
                               "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"])
    pmz = t.column("precursor_mz").to_numpy(); add = np.asarray(t.column("adduct"))
    inst = np.asarray(t.column("instrument_type")); mode = np.asarray(t.column("ionization_mode"))
    ce_raw = t.column("collision_energy_ev").to_pylist()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    log("train.parquet loaded")

    smis = sorted({s for k in keys for s in pools[k][1]["mass"].values() if s})
    info = characterise(smis, procs=7)
    log(f"{len(smis):,} candidate structures characterised")

    arrays, meta, n_empty = {}, [], 0
    for n, k in enumerate(keys, 1):
        split, p = pools[k]
        rows = qrows[k]
        frames = []
        for r in rows:
            ce = ce_raw[r]
            vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
            energies = [float(v) for v in vals if v is not None and np.isfinite(v)]
            frames.append(dict(energies=energies, adduct=str(add[r]), precursor=float(pmz[r]),
                               instrument=map_instrument(str(inst[r])), mode=str(mode[r])))
            j = len(frames) - 1
            arrays[f"{k}/spec{j}_mz"] = mzf[mzo[r]:mzo[r+1]]
            arrays[f"{k}/spec{j}_intensity"] = itf[ito[r]:ito[r+1]]

        cands = sorted(((c, s) for c, s in p["mass"].items() if s),
                       key=lambda cs: -p["spec"].get(cs[0], 0.0))
        rows_out, frags = [], []
        for rank, (cid, smi) in enumerate(cands, 1):
            inf = info.get(smi)
            fm = np.asarray(sorted(inf["full"]), np.float64) if inf and inf.get("full") else np.empty(0)
            frags.append(fm)
            rows_out.append(dict(rank=rank, smiles=smi, n_fragments=int(len(fm)),
                                 spec_score=round(float(p["spec"].get(cid, 0.0)), 6),
                                 is_truth=bool(p["truth"] and key14(smi) == p["truth"])))
        arrays[f"{k}/fragments"] = np.concatenate(frags) if frags else np.empty(0)
        arrays[f"{k}/fragment_offsets"] = np.cumsum([0] + [len(f) for f in frags]).astype(np.int64)
        arrays[f"{k}/is_truth"] = np.array([r["is_truth"] for r in rows_out], bool)
        no_frag = bool(rows_out) and all(r["n_fragments"] == 0 for r in rows_out)
        n_empty += no_frag
        meta.append(dict(query=k, split=split, cls=assign[k], truth_key=p["truth"],
                         truth_in_pool=any(r["is_truth"] for r in rows_out),
                         truth_rank=next((r["rank"] for r in rows_out if r["is_truth"]), 0),
                         n_candidates=len(rows_out), n_frames=len(frames),
                         no_fragments_anywhere=no_frag, frames=frames, candidates=rows_out))
        if n % 200 == 0: log(f"  {n}/{len(keys)}")

    npz = f"{OUT}/casmi_clean_pools.npz"; js = f"{OUT}/casmi_clean_pools.json"
    np.savez_compressed(npz, **arrays)
    json.dump(dict(splits=list(splits), n_queries=len(meta),
                   note=("query structures absent from MassSpecGym by construction, so nothing "
                         "here is inside the open ICEBERG weights; candidates ordered by spectral "
                         "score; fragment lists flat with offsets"),
                   pools_with_no_fragments=n_empty, queries=meta), open(js, "w"))
    log(f"wrote {npz} ({os.path.getsize(npz)/1e6:.1f} MB) and {js} "
        f"({os.path.getsize(js)/1e6:.1f} MB)")
    byc = {}
    for m in meta: byc.setdefault(m["cls"], []).append(m)
    print("\nper class:")
    for c in sorted(byc):
        v = byc[c]
        print(f"  C{c}: {len(v):>4} queries, truth in pool {sum(m['truth_in_pool'] for m in v):>4}, "
              f"pools with no fragments anywhere {sum(m['no_fragments_anywhere'] for m in v):>3}")
    print(f"\npools with NO fragment masses on ANY candidate: {n_empty}/{len(meta)} "
          f"({n_empty/len(meta):.1%}) -- no fragment-based method can speak to these")


if __name__ == "__main__":
    main(sys.argv[1:] or CLEAN)
