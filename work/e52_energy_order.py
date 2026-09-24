#!/usr/bin/env python3
"""E52 — do fragments appear in the order their bonds should break? Lee's elimination idea.

THE OBSERVATION, which came from Lee and not from any of the models working on this. He asked
whether the machine gives us a snapshot of the decay or a slide show. It gives a slide show:
293 of the 400 test molecules (73.2%) are measured at more than one distinct collision energy,
132 of them across the full 20/40/60 ladder.

His argument, which is chemically sound and which none of us had made: two molecules built from
the same atoms differ only in ARRANGEMENT -- but arrangement determines which bonds are weakest,
and weak bonds break first. So the ORDER in which pieces appear as the energy rises carries
structural information that the list of piece-masses does not.

WHY THIS IS A DIFFERENT LEVER FROM EVERYTHING WE HAVE TRIED. E43 measured that for 60% of queries
the true structure and its best rival explain EXACTLY the same observed peaks. No scoring function
of that evidence can separate them, which is why twenty ranking experiments returned noise. But an
ordering constraint does not score -- it ELIMINATES. A candidate whose only route to an observed
low-energy peak is snapping a bond that should survive to high energy is not a worse candidate; it
is an impossible one. Removing it lifts the truth mechanically, because fewer decoys sit above it.

WHAT WE ALREADY HAVE. Our fragmenter breaks 1-2 acyclic single bonds, so it carries a depth ladder
for free -- depth 1 is one bond broken, depth 2 is two. We throw the depth away and score the
whole set against the MERGED spectrum. This keeps both: fragments tagged by depth, frames kept
separate.

THE TEST. For each candidate, find the lowest-energy frame in which each of its fragments appears.
If the structure is right, depth-1 fragments should surface EARLIER (lower energy) than depth-2
fragments -- easy cleavages first. If the candidate merely shares masses by coincidence, that
ordering should be scrambled.

  gap = mean(first-appearance energy | depth 2) - mean(first-appearance energy | depth 1)

  positive gap  = consistent with chemistry
  negative gap  = deep fragments appearing BEFORE easy ones, which should not happen

PREDICTIONS (before the run, 2026-09-24):
  P1  The truth's mean gap is positive and larger than the median decoy's. I expect the separation
      to be real but modest -- AUC 0.55-0.65, in the same band as every other single feature here.
  P2  As an ELIMINATOR it is worth more than as a ranker: the share of decoys with a clearly
      negative gap should exceed the share of truths by enough to delete a meaningful slice of the
      pool. If decoys show impossible ordering at 2x the rate of truths, that is the result.
  P3  It works better on queries with the full 20/40/60 ladder than on two-frame queries, because
      an ordering needs something to order.
  RISK, stated plainly: our depth ladder ignores ring bonds entirely and stops at two breaks, so a
  molecule whose real chemistry runs through ring cleavage will look scrambled no matter what. If
  P1 and P2 both fail, that is the first thing to blame rather than the idea.
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import casmi_pipeline as cp
from casmi_pipeline import key14, TOL, PROTON, H, MAX_HEAVY, _MONO
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
CLEAN = ("npxmtsseed60_n300", "npxmtsseed61_n300", "npxmtsseed62_n300")
TOPPEAK = 60          # observed peaks per frame considered


def depth_fragments(smi):
    """{mass: shallowest depth that produces it}, depth 1 = one acyclic single bond, 2 = two."""
    mol = Chem.MolFromSmiles(smi or "")
    if mol is None or mol.GetNumHeavyAtoms() > MAX_HEAVY: return None
    acyclic = [b.GetIdx() for b in mol.GetBonds()
               if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
    if not acyclic: return None
    N = mol.GetNumAtoms()
    am = [_MONO.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H for a in mol.GetAtoms()]
    out = {}
    def add(m, d):
        if m not in out or d < out[m]: out[m] = d
    for b in acyclic:
        try:
            frg = Chem.FragmentOnBonds(mol, [b], addDummies=True)
            for idxs in Chem.GetMolFrags(frg, asMols=False):
                add(round(sum(am[i] for i in idxs if i < N), 4), 1)
        except Exception: continue
    if len(acyclic) <= 40:
        for i, a in enumerate(acyclic):
            for b in acyclic[i + 1:]:
                try:
                    frg = Chem.FragmentOnBonds(mol, [a, b], addDummies=True)
                    for idxs in Chem.GetMolFrags(frg, asMols=False):
                        add(round(sum(am[i2] for i2 in idxs if i2 < N), 4), 2)
                except Exception: continue
    return out


def _work(smi): return smi, depth_fragments(smi)


def first_energy(frag_depth, frames, positive):
    """For each fragment mass, the LOWEST frame energy at which it is observed."""
    seen = {}
    for energy, mz, it in frames:
        if len(mz) == 0: continue
        o = np.argsort(-it)[:TOPPEAK]
        obs = np.sort(np.asarray(mz, float)[o])
        shift = PROTON if positive else -PROTON
        for m in frag_depth:
            if m in seen: continue
            target = m + shift
            j = np.searchsorted(obs, target)
            for k in (j - 1, j):
                if 0 <= k < len(obs) and abs(obs[k] - target) <= TOL:
                    seen[m] = energy; break
    return seen


def gap_for(frag_depth, frames, positive):
    """mean first-appearance energy of depth-2 fragments minus that of depth-1 fragments."""
    seen = first_energy(frag_depth, frames, positive)
    if not seen: return None, 0, 0
    d1 = [e for m, e in seen.items() if frag_depth[m] == 1]
    d2 = [e for m, e in seen.items() if frag_depth[m] == 2]
    if not d1 or not d2: return None, len(d1), len(d2)
    return float(np.mean(d2) - np.mean(d1)), len(d1), len(d2)


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"])
    ce_raw = t.column("collision_energy_ev").to_pylist()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    log("spectra loaded")

    queries, need = [], set()
    for split in CLEAN:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{split}.pkl", "rb"))}
        for k, rows in sp["query_rows"].items():
            p = pools.get(k)
            if p is None or not p["mass"]: continue
            frames = []
            for r in rows:
                ce = ce_raw[r]
                vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
                vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
                if len(vals) != 1: continue          # skip blended/stepped frames: no ordering in them
                frames.append((vals[0], mzf[mzo[r]:mzo[r+1]], itf[ito[r]:ito[r+1]]))
            energies = sorted({f[0] for f in frames})
            if len(energies) < 2: continue           # a slide show needs at least two slides
            queries.append(dict(k=k, cls=sp["assign"][k], truth=p["truth"], frames=frames,
                                n_energies=len(energies), positive=bool(p["positive"]),
                                cand={c: s for c, s in p["mass"].items() if s}))
            need.update(queries[-1]["cand"].values())
    log(f"{len(queries)} queries with >=2 distinct single-energy frames; "
        f"{len(need):,} candidate structures")

    with Pool(7) as pool:
        frag = dict(pool.imap_unordered(_work, sorted(need), chunksize=16))
    log("fragments tagged by depth")

    rows = []
    for q in queries:
        vals = {}
        for c, smi in q["cand"].items():
            fd = frag.get(smi)
            if not fd: continue
            g, n1, n2 = gap_for(fd, q["frames"], q["positive"])
            if g is None: continue
            vals[c] = (g, n1, n2, smi)
        if len(vals) < 2: continue
        tk = [c for c, v in vals.items() if q["truth"] and key14(v[3]) == q["truth"]]
        if not tk: continue
        tg = vals[tk[0]][0]
        others = [v[0] for c, v in vals.items() if c not in tk]
        if not others: continue
        rows.append(dict(query=q["k"], cls=q["cls"], n_energies=q["n_energies"],
                         truth_gap=tg, median_decoy_gap=float(np.median(others)),
                         n_decoys=len(others),
                         beats_median=bool(tg > np.median(others)),
                         truth_negative=bool(tg < 0),
                         decoy_negative_share=float(np.mean([o < 0 for o in others]))))
    log(f"{len(rows)} queries with the truth measurable")

    L = ["E52 — do fragments appear in the order their bonds should break?", "",
         f"{len(queries)} queries with a genuine slide show (>=2 distinct energies); "
         f"{len(rows)} with the truth measurable", ""]
    if rows:
        tg = np.array([r["truth_gap"] for r in rows])
        mg = np.array([r["median_decoy_gap"] for r in rows])
        L += [f"mean gap, truth          {tg.mean():+.3f}  (median {np.median(tg):+.3f})",
              f"mean gap, median decoy   {mg.mean():+.3f}  (median {np.median(mg):+.3f})",
              f"truth beats median decoy {np.mean(tg > mg):.1%}", "",
              "AS AN ELIMINATOR -- the share showing IMPOSSIBLE ordering (deep before easy):",
              f"  truths with a negative gap {np.mean([r['truth_negative'] for r in rows]):.1%}",
              f"  decoys with a negative gap {np.mean([r['decoy_negative_share'] for r in rows]):.1%}"]
        tn = np.mean([r["truth_negative"] for r in rows])
        dn = np.mean([r["decoy_negative_share"] for r in rows])
        L.append(f"  -> deleting every negative-gap candidate would remove {dn:.1%} of decoys "
                 f"and {tn:.1%} of truths")
        if tn > 0: L.append(f"  -> decoys eliminated per truth lost: {dn/max(tn,1e-9):.2f}x")
        L.append("")
        for ne in sorted({r["n_energies"] for r in rows}):
            s = [r for r in rows if r["n_energies"] == ne]
            L.append(f"  {ne} energy frames: n={len(s):>4}  truth beats median decoy "
                     f"{np.mean([r['beats_median'] for r in s]):.1%}")
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E52_energy_order_2026-09-24.txt", "w").write("\n".join(L) + "\n")
    json.dump(rows, open(f"{RESULTS}/E52_energy_order.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
