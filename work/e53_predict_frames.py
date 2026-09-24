#!/usr/bin/env python3
"""E53 step 2 — run ICEBERG over the work list, KEEPING ENERGY FRAMES SEPARATE.

E47 merged the per-energy predictions by taking the strongest value per mass. That
destroyed the ordering signal Lee identified: which pieces appear gently and which
need force. This variant writes one array per (candidate, energy) instead.

Original: Runs in ~/.iceberg-venv, NOT the project venv.

Reads the JSON work list, predicts a spectrum for every (candidate, collision energy) pair, merges
the energies per candidate the same way our `explain` feature merges observed energies, and writes
a sparse .npz. Nothing about our pipeline is imported here -- the two dependency sets are kept
apart on purpose (ICEBERG needs numpy<2/torch 2.2.1/dgl 2.1; ours runs numpy 2).

SPARSIFICATION: each raw prediction is ~2600 peaks; keeping all of them for 19k predictions would
be most of a gigabyte. Top-100 by intensity per merged candidate spectrum matches ICEBERG's own
--sparse-k default and is far more than the ~40 observed peaks we ever compare against.

Queries whose adduct the model does not support are skipped and recorded as skipped, so scoring
can retain baseline behaviour for them and keep them in the denominator rather than dropping them.
"""
import json, os, sys, time
import numpy as np
import torch
from multiprocessing import Pool

CK = "/tmp/claude-1000/-home-lee/c046c9a5-0d6b-4917-ae00-430a3bba9009/scratchpad/iceberg_ckpt"
TOP_K = 100
MODEL = {}


def _init():
    from ms_pred.iceberg import inten_model, gen_model, joint_model
    im = inten_model.IntenGNN.load_from_checkpoint(f"{CK}/inten_contr/best.ckpt", map_location="cpu")
    gm = gen_model.FragGNN.load_from_checkpoint(f"{CK}/gen/best.ckpt", map_location="cpu")
    jm = joint_model.JointModel(gen_model_obj=gm, inten_model_obj=im)
    jm.eval(); jm.freeze()
    MODEL["jm"] = jm
    torch.set_num_threads(1)          # 7 workers each grabbing every core is slower, not faster


def _one_query(q):
    """All candidates for one query -> {candidate_key: (mz[K], intensity[K])}."""
    jm = MODEL["jm"]
    out, fails = {}, 0
    if not q["adduct_supported"]:
        return q["key"], {}, 0, "adduct_unsupported"
    with torch.no_grad():
        for c in q["candidates"]:
            acc_mz, acc_it = [], []
            for e in q["energies"]:
                try:
                    r = jm.predict_mol(smi=c["smiles"], collision_eng=float(e),
                                       precursor_mz=float(q["precursor"]), adduct=q["adduct"],
                                       threshold=0.0, device="cpu", max_nodes=100,
                                       instrument=q["instrument"], binned_out=False)
                except Exception:
                    fails += 1
                    continue
                spec = np.asarray(r["spec"], dtype=np.float64)
                if spec.ndim != 2 or spec.shape[0] == 0:
                    acc_mz.append(np.empty(0)); acc_it.append(np.empty(0)); continue
                acc_mz.append(spec[:, 0]); acc_it.append(spec[:, 1])
            if not acc_mz: continue
            # E53: no cross-energy merge -- each frame is kept as its own array
            for e_val, (m_, i_) in zip(q["energies"], zip(acc_mz, acc_it)):
                if len(i_) > TOP_K:
                    keep = np.argpartition(-i_, TOP_K)[:TOP_K]; m_, i_ = m_[keep], i_[keep]
                o_ = np.argsort(-i_)
                out[f'{c["key"]}@{e_val:g}'] = (m_[o_].astype(np.float32), i_[o_].astype(np.float32))
            continue
            mz = np.concatenate(acc_mz); it = np.concatenate(acc_it)
            # merge energies: strongest prediction per 0.01 Da bin, mirroring how we merge the
            # observed spectra (max over frames) rather than summing, so a peak strong at one
            # energy is not diluted by energies where it is absent
            b = np.rint(mz / 0.01).astype(np.int64)
            order = np.lexsort((-it, b))
            b, mz, it = b[order], mz[order], it[order]
            first = np.concatenate(([True], b[1:] != b[:-1]))
            mz, it = mz[first], it[first]
            if len(it) > TOP_K:
                keep = np.argpartition(-it, TOP_K)[:TOP_K]
                mz, it = mz[keep], it[keep]
            o = np.argsort(-it)
            out[c["key"]] = (mz[o].astype(np.float32), it[o].astype(np.float32))
    return q["key"], out, fails, None


def main(work_path, out_path, procs=7):
    t0 = time.time()
    w = json.load(open(work_path))
    queries = w["queries"]
    todo = [q for q in queries if q["candidates"]]
    print(f"{len(queries)} queries, {len(todo)} with a non-empty pool, "
          f"{sum(len(q['candidates'])*len(q['energies']) for q in todo):,} predictions", flush=True)

    arrays, meta, fails = {}, [], 0
    with Pool(procs, initializer=_init) as pool:
        for n, (k, res, f, skip) in enumerate(pool.imap_unordered(_one_query, todo, chunksize=2), 1):
            fails += f
            for ck, (mz, it) in res.items():
                arrays[f"{k}|{ck}|mz"] = mz
                arrays[f"{k}|{ck}|it"] = it
            meta.append(dict(query=k, n_predicted=len(res), skipped=skip))
            if n % 25 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)} queries  {time.time()-t0:.0f}s  "
                      f"eta {(len(todo)-n)/max(rate,1e-9)/60:.1f} min  fails {fails}", flush=True)

    np.savez_compressed(out_path, **arrays)
    json.dump(meta, open(out_path.replace(".npz", "_meta.json"), "w"))
    got = sum(m["n_predicted"] for m in meta)
    print(f"\nwrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"predicted spectra for {got:,} (query, candidate) pairs; "
          f"{sum(1 for m in meta if m['skipped'])} queries skipped; {fails} individual failures")
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 7)
