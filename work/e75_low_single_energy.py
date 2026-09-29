#!/usr/bin/env python3
"""E75 — does the break-depth feature survive on ONE collision energy?

E66/E70 required two or more energies per query, because the implementation fitted a slope of implied
break depth against energy and needed two points for a line. But the slope turned out to be null
(0.471, chance) and the measure that actually worked was the LEVEL at the lowest energy: the truth
explains gentle-energy peaks with less broken bond than its same-mass rivals. That is a
single-energy quantity. The two-energy requirement was an artifact of measuring it alongside the
slope, and it is expensive: only 52.7% of the Class-2 COCONUT arm has two, against ~100% of the test
set, where every row carries a recorded energy on a 20/40/60 grid.

So this re-runs the winning comparison with the slope dropped and the requirement relaxed to ONE
energy, on the predictions already on disk. If it holds, the feature applies to roughly twice as many
local queries and essentially the whole board.

Also reported: whether using the LOWEST available energy matters, or whether any single energy does.
If the effect is specific to gentle collision energies, that is a physical claim worth stating; if it
holds at 40 and 60 eV too, then it is about break depth generally and the "gentle" framing is wrong.

PREDICTION (before the run): the single-energy rate lands within 0.02 of E70's two-energy 0.605,
because the slope contributed nothing and the level was always doing the work. A large drop would mean
the two-energy queries are an easier population, which would matter more than the feature.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS, RESULTS = os.path.join(WORK, "splits"), os.path.join(WORK, "results")
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
SPLIT = "e64energy_n600"
TOL, E_TOL, TOP_OBS, BOOT = 0.01, 0.05, 40, 4000


def explained(obs_mz, pmz, pval):
    o = np.argsort(pmz); pm, pv = pmz[o], np.asarray(pval)[o]
    out = {}
    for i, m in enumerate(obs_mz):
        j = np.searchsorted(pm, m); best = None
        for k in (j - 1, j):
            if 0 <= k < len(pm) and abs(pm[k] - m) <= TOL:
                if best is None or abs(pm[k] - m) < abs(pm[best] - m): best = k
        if best is not None: out[i] = float(pv[best])
    return out


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    d = np.load(f"{EXPORTS}/iceberg_bde_{SPLIT}.npz")
    pred = defaultdict(lambda: defaultdict(dict))
    for key in d.keys():
        q, rest, kind = key.split("|")
        cand, _, e = rest.rpartition("@")
        pred[q][cand].setdefault(float(e), {})[kind] = d[key]
    wl = {q["key"]: q for q in json.load(open(f"{EXPORTS}/iceberg_bonds_pool_{SPLIT}.json"))["queries"]}
    sp = json.load(open(f"{SPLITS}/split_{SPLIT}.json"))
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"])
    ce_raw = t.column("collision_energy_ev").to_pylist()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    log("loaded")

    rng = np.random.default_rng(75)
    # per query: {"lowest": rate, "all": [(energy, rate)]} for the real truth and a pseudo-truth
    real, pseudo, by_energy = [], [], defaultdict(list)
    n_one_energy = 0
    for qk, cands in pred.items():
        q = wl.get(qk)
        if q is None: continue
        tkey = next((c["key"] for c in q["candidates"]
                     if c["smiles"] and scoring.key14(c["smiles"]) == q["truth"]), None)
        if tkey is None or tkey not in cands: continue
        obs = {}
        for r in sp["query_rows"][qk]:
            ce = ce_raw[r]
            vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
            vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
            if len(vals) != 1: continue
            mz, it = mzf[mzo[r]:mzo[r+1]], itf[ito[r]:ito[r+1]]
            if len(mz) == 0: continue
            o = np.argsort(-it)[:TOP_OBS]
            obs.setdefault(vals[0], (np.asarray(mz[o], float), np.asarray(it[o], float)))
        if not obs: continue                      # ONE energy is now enough
        if len(obs) == 1: n_one_energy += 1
        obs_es = np.array(sorted(obs))

        # implied summed bond energy per candidate, per observed energy
        expl = defaultdict(dict)
        for c, byE in cands.items():
            for pe in sorted(byE):
                v = byE[pe]
                if "mz" not in v or "bd" not in v or len(v["mz"]) == 0: continue
                j = int(np.argmin(np.abs(obs_es - pe)))
                if abs(obs_es[j] - pe) > E_TOL: continue
                oe = float(obs_es[j])
                expl[c][oe] = explained(obs[oe][0], v["mz"], v["bd"])
        if tkey not in expl: continue

        def level(anchor_m, rival_m, e):
            """Intensity-weighted implied bond energy on the peaks BOTH explain at energy e."""
            both = set(anchor_m.get(e, {})) & set(rival_m.get(e, {}))
            if not both: return None
            w = obs[e][1]; den = sum(w[i] for i in both)
            if den <= 0: return None
            return (sum(w[i] * anchor_m[e][i] for i in both) / den,
                    sum(w[i] * rival_m[e][i] for i in both) / den)

        def rate(anchor, rivals, energy):
            wins = []
            for c in rivals:
                r = level(expl[anchor], expl[c], energy)
                if r is None: continue
                wins.append(1.0 if r[0] < r[1] else (0.5 if r[0] == r[1] else 0.0))
            return float(np.mean(wins)) if wins else None

        lo_e = float(obs_es[0])
        others = [c for c in expl if c != tkey]
        if not others: continue
        v = rate(tkey, others, lo_e)
        if v is not None: real.append(v)
        pk = others[int(rng.integers(0, len(others)))]
        pv_ = rate(pk, [c for c in others if c != pk], lo_e)
        if pv_ is not None: pseudo.append(pv_)
        for e in obs_es:                          # is it specific to the LOWEST energy?
            r = rate(tkey, others, float(e))
            if r is not None: by_energy[float(e)].append(r)

    def ci(v):
        v = np.asarray(v, float)
        bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT)])
        return v.mean(), *np.percentile(bs, [2.5, 97.5])

    L = ["E75 — break-depth level on a SINGLE collision energy (slope dropped)", "",
         f"{len(real)} queries measurable with >=1 energy ({n_one_energy} of them have exactly one)",
         f"E70 required >=2 energies and measured 0.605 [0.581, 0.627] on 191 queries", ""]
    if real:
        m, lo, hi = ci(real)
        L += [f"  truth lower implied bond energy at its LOWEST energy: {m:.3f} [{lo:.3f}, {hi:.3f}]"]
    if pseudo:
        m2, lo2, hi2 = ci(pseudo)
        L += [f"  a promoted decoy, same measure (label control):        {m2:.3f} [{lo2:.3f}, {hi2:.3f}]"]
    if real and pseudo:
        L += [f"  real minus control: {np.mean(real) - np.mean(pseudo):+.3f}"]
    L += ["", "IS IT SPECIFIC TO GENTLE ENERGY? rate at each observed collision energy:"]
    for e in sorted(by_energy):
        v = by_energy[e]
        if len(v) >= 15:
            L.append(f"  {e:6.1f} eV   n={len(v):4d}   {float(np.mean(v)):.3f}")
    L += ["", "Reading it: a rate near E70's 0.605 means the two-energy requirement was an artifact of",
          "measuring the level alongside a slope that carried nothing -- and the feature then applies",
          "to ~100% of the test set rather than the 52.7% of the Class-2 arm that has two energies.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E75_low_single_energy.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
