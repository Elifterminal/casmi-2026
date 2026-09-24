#!/usr/bin/env python3
"""E41 — export real query pools for Vanta to shape into a panaesthesis field.

Vanta's proposal, from the CASMI ranker thread: connect the query AND its whole candidate pool as
the PARTS of one field, so each candidate's relation to the query and to the rest of the pool is
read at once, rather than as 65 independent pairwise scores. That is a representation change, not
a feature -- which puts it in the only bucket that has ever shipped a gain on this project.

She offered to shape the .npz herself from a stand-in. Lee's call, and the right one: give her a
REAL pool. Our pools are the thing in question, and a stand-in would answer a question about a
stand-in.

WHAT A POOL IS, precisely, because the shaping depends on it:

  query      one unknown molecule, observed as 1..n MS/MS spectra at different collision energies.
             Each spectrum is a set of (m/z, intensity) peaks. This is ALL the evidence we get.
  candidates every structure in COCONUT whose neutral mass is within 10 ppm of the query's. Median
             65 of them. They are NOT spectra -- they are structures. What we can compute for each
             is the set of fragment masses it could produce if it broke apart, which is what makes
             a candidate commensurate with a spectrum: both become vectors over the same mass axis.
  truth      exactly one candidate is the right answer (except in Class 3, where the right answer
             is absent from the pool entirely -- that is what Class 3 MEANS).

THE ASYMMETRY THAT MAKES THIS HARD, and the thing worth her attention: the query has intensities
and the candidates do not. A candidate offers a LIST of masses it could make, with no prediction
of how much of each. Our scoring throws that asymmetry away by reducing each candidate to a
scalar overlap. If a field reading over the pool can use it instead, that is where the gain lives.

Exported per query, all arrays raw so she can reshape freely:
  spectra      per collision-energy frame: m/z, intensity, energy, adduct, precursor m/z
  candidates   SMILES, fragment mass list, our 8 feature values, our score, is_truth flag
  labels       class (1/2/3), truth key, and the rank we currently give the truth
"""
import json, os, pickle, sys, time
import numpy as np
import pyarrow.parquet as pq
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import FEATS2
from e39_gate import CACHE, code_hash
from casmi_pipeline import characterise, key14
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
OUT = os.path.expanduser("~/casmi-2026/exports")
SPLIT = "nptsseed10_n300"
PER_CLASS = 4   # overridden by --all


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    got = pickle.load(open(CACHE, "rb"))
    assert got["hash"] == code_hash(), "cache stale -- rerun E39"
    w = np.array(json.load(open(os.path.join(SPLITS, "ranker_w_fr.json")))["w"])

    # queries from one split, balanced across classes
    splits = sys.argv[1:] or [SPLIT]
    qrows, assign, pools_all = {}, {}, {}
    for sname in splits:
        sp = json.load(open(f"{SPLITS}/split_{sname}.json"))
        qrows.update(sp["query_rows"]); assign.update(sp["assign"])
        for pp in pickle.load(open(f"{SPLITS}/pools_fr_{sname}.pkl", "rb")):
            pools_all[pp["k"]] = pp
    byq = pools_all
    pick = sorted(k for k in qrows if k in byq)
    log(f"{len(pick)} queries from {len(splits)} split(s): {', '.join(splits)}")

    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["inchikey14", "normalized_smiles", "precursor_mz", "adduct",
                               "ionization_mode", "collision_energy_ev", "ms2_mzs",
                               "ms2_normalized_intensities"])
    pmz = t.column("precursor_mz").to_numpy(); add = np.asarray(t.column("adduct"))
    mode = np.asarray(t.column("ionization_mode"))
    ce_raw = t.column("collision_energy_ev").to_pylist()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    log("train.parquet loaded")

    # fragment masses for every candidate we are about to export
    smis = sorted({s for k in pick for _, (_, s) in byq[k]["cand"].items()})
    info = characterise(smis, procs=7)
    log(f"{len(smis):,} candidate structures characterised")

    arrays, meta = {}, []
    for k in pick:
        q = byq[k]; rows = qrows[k]
        frames = []
        for r in rows:
            ce = ce_raw[r]
            ce = (ce[0] if isinstance(ce, list) and ce else
                  (ce if isinstance(ce, (int, float)) else float("nan")))
            frames.append(dict(mz=mzf[mzo[r]:mzo[r+1]], it=itf[ito[r]:ito[r+1]],
                               energy=float(ce) if ce is not None else float("nan"),
                               adduct=str(add[r]), precursor=float(pmz[r]),
                               mode=str(mode[r])))
        for j, f in enumerate(frames):
            arrays[f"{k}/spec{j}_mz"] = f["mz"]; arrays[f"{k}/spec{j}_intensity"] = f["it"]

        cands = sorted(((c, (None, sm)) for c, sm in byq[k]["mass"].items() if sm),
                       key=lambda kv: -byq[k]["spec"].get(kv[0], 0.0))
        rows_out, frags = [], []
        for rank, (cid, (vec, smi)) in enumerate(cands, 1):
            inf = info.get(smi)
            fm = np.asarray(sorted(inf["full"]), np.float64) if inf and inf.get("full") else np.empty(0)
            frags.append(fm)
            rows_out.append(dict(rank=rank, smiles=smi, n_fragments=int(len(fm)),
                                 spec_score=float(byq[k]["spec"].get(cid, 0.0)),
                                 is_truth=bool(key14(smi) == q["truth"])))
        # ragged fragment lists -> one flat array plus offsets, so nothing is padded or truncated
        arrays[f"{k}/fragments"] = np.concatenate(frags) if frags else np.empty(0)
        arrays[f"{k}/fragment_offsets"] = np.cumsum([0] + [len(f) for f in frags])
        arrays[f"{k}/features"] = np.array([[r[f] for f in FEATS2] for r in rows_out], np.float32)
        arrays[f"{k}/scores"] = np.array([r["score"] for r in rows_out], np.float32)
        arrays[f"{k}/is_truth"] = np.array([r["is_truth"] for r in rows_out], bool)
        truth_rank = next((r["rank"] for r in rows_out if r["is_truth"]), 0)
        meta.append(dict(query=k, cls=assign[k], truth_key=q["truth"],
                         truth_in_pool=bool(truth_rank), truth_rank=truth_rank,
                         n_candidates=len(rows_out), n_frames=len(frames),
                         frames=[{kk: vv for kk, vv in f.items() if kk not in ("mz", "it")}
                                 for f in frames],
                         candidates=rows_out))
        log(f"  {k} class {assign[k]}  {len(rows_out)} candidates  truth rank {truth_rank or '-'}")

    np.savez_compressed(f"{OUT}/casmi_pools.npz", **arrays)
    json.dump(dict(splits=splits, note="candidates ordered by spectral score; clean splits = query structures absent from MassSpecGym by construction",
                   queries=meta),
              open(f"{OUT}/casmi_pools.json", "w"), indent=1)
    mb = os.path.getsize(f"{OUT}/casmi_pools.npz") / 1e6
    log(f"wrote {OUT}/casmi_pools.npz ({mb:.1f} MB) + casmi_pools.json")


if __name__ == "__main__":
    main()
