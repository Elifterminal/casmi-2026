#!/usr/bin/env python3
"""E62 — regenerate ICEBERG predictions keeping BOND PROVENANCE, not just the spectrum.

Lee's correction, which was right and which I had fudged. I had argued the energy-ordering idea
was weak because bond strength is a property of bond TYPE, and same-formula isomers share their
types. He pointed out that is not how bonds work: strength depends on what sits on either side,
so T-S can be stronger than T-N. Adjacency IS arrangement. The ordering information genuinely
exists in the chemistry, and my analogy let the idea off the hook.

The failure was therefore in observation and prediction, not in the chemistry -- and I made it
worse. ICEBERG returns TWO arrays per prediction:

    spec (n, 2)   m/z and intensity          <- I saved this
    frag (n, A)   atom-membership mask       <- I discarded this

`frag` says WHICH ATOMS of the candidate are in each predicted peak's fragment, so the bonds that
had to break are exactly the bonds crossing the mask boundary. That is the bond-level quantity his
argument calls for, and I built the whole of E52/E54 on the mass-collapsed shadow of it.

Verified on caffeine before spending the run: 14 mask columns for 14 heavy atoms, the intact
precursor at 0 broken bonds carrying intensity 0.83 at 20 eV, deeper fragments (4-5 bonds) at
0.05-0.51. Intensity falls as depth rises, which is the physics.

WHAT THIS STORES, per (query, candidate, collision energy): the top peaks as
(m/z, intensity, bonds_broken). Storing the count rather than the mask keeps it compact and is
all the downstream test needs.

THE TEST IT ENABLES (E63): for each OBSERVED peak at energy e, find the candidate's predicted
fragment at that mass and read its break depth. Gentle energy should be explained by shallow
breaks. A candidate forced to explain a 20 eV peak by snapping five bonds is implausible in a way
that owes nothing to mass -- which is what E52 tried to measure with a fragmenter that matched
four peaks per molecule, and what E54 could not see at all.
"""
import json, os, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

CK = "/tmp/claude-1000/-home-lee/c046c9a5-0d6b-4917-ae00-430a3bba9009/scratchpad/iceberg_ckpt"
TOP_K = 60
MODEL = {}


def _init():
    from ms_pred.iceberg import inten_model, gen_model, joint_model
    im = inten_model.IntenGNN.load_from_checkpoint(f"{CK}/inten_contr/best.ckpt", map_location="cpu")
    gm = gen_model.FragGNN.load_from_checkpoint(f"{CK}/gen/best.ckpt", map_location="cpu")
    jm = joint_model.JointModel(gen_model_obj=gm, inten_model_obj=im)
    jm.eval(); jm.freeze()
    MODEL["jm"] = jm
    torch.set_num_threads(1)


def _broken(mol_bonds, mask):
    """Bonds crossing the fragment boundary = the bonds that had to break for this fragment."""
    n = len(mask)
    return sum(1 for a, b in mol_bonds if a < n and b < n and mask[a] != mask[b])


def _one_query(q):
    jm = MODEL["jm"]
    out, fails = {}, 0
    if not q["adduct_supported"]:
        return q["key"], {}, 0
    with torch.no_grad():
        for c in q["candidates"]:
            mol = Chem.MolFromSmiles(c["smiles"] or "")
            if mol is None: continue
            bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
            for e in q["energies"]:
                try:
                    r = jm.predict_mol(smi=c["smiles"], collision_eng=float(e),
                                       precursor_mz=float(q["precursor"]), adduct=q["adduct"],
                                       threshold=0.0, device="cpu", max_nodes=100,
                                       instrument=q["instrument"], binned_out=False)
                except Exception:
                    fails += 1; continue
                spec = np.asarray(r["spec"], dtype=np.float64)
                frag = np.asarray(r["frag"])
                if spec.ndim != 2 or spec.shape[0] == 0: continue
                keep = np.argsort(-spec[:, 1])[:TOP_K]
                mz = spec[keep, 0].astype(np.float32)
                it = spec[keep, 1].astype(np.float32)
                nb = np.array([_broken(bonds, frag[i]) for i in keep], dtype=np.uint8)
                out[f'{c["key"]}@{e:g}'] = (mz, it, nb)
    return q["key"], out, fails


def main(work_path, out_path, procs=7):
    t0 = time.time()
    w = json.load(open(work_path))
    todo = [q for q in w["queries"] if q["candidates"]]
    n_pred = sum(len(q["candidates"]) * len(q["energies"]) for q in todo)
    print(f"{len(todo)} queries, {n_pred:,} predictions with bond provenance", flush=True)

    arrays, meta, fails = {}, [], 0
    with Pool(procs, initializer=_init) as pool:
        for n, (k, res, f) in enumerate(pool.imap_unordered(_one_query, todo, chunksize=2), 1):
            fails += f
            for ck, (mz, it, nb) in res.items():
                arrays[f"{k}|{ck}|mz"] = mz
                arrays[f"{k}|{ck}|it"] = it
                arrays[f"{k}|{ck}|nb"] = nb
            meta.append(dict(query=k, n=len(res)))
            if n % 25 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  {time.time()-t0:.0f}s  "
                      f"eta {(len(todo)-n)/max(rate,1e-9)/60:.1f} min  fails {fails}", flush=True)
    np.savez_compressed(out_path, **arrays)
    json.dump(meta, open(out_path.replace(".npz", "_meta.json"), "w"))
    got = sum(m["n"] for m in meta)
    print(f"\nwrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"{got:,} (candidate, energy) predictions with bond counts; {fails} failures")
    print(f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 7)
