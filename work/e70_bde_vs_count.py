#!/usr/bin/env python3
"""E70 — does pricing the broken bonds beat counting them?

E66 found the one real signal in this family by counting bonds: at the lowest collision energy the
correct molecule explains the observed peaks with FEWER broken bonds than its same-mass rivals,
0.574 against 0.500 chance, and a label shuffle put both impostor arms exactly on chance.

Lee's objection to the count is that it is the wrong unit. An amide C-N, an aromatic ring bond and a
thioether C-S are not interchangeable, and "how much had to break" should mean energy. E68 re-ran
the predictions recording three quantities per predicted peak instead of one:

    nb   count of bonds crossing the fragment boundary        (what E66 used)
    bd   summed bond dissociation energy of those bonds, kJ/mol
    rb   how many of them were ring bonds

Each is tested identically: paired against every same-mass rival, restricted to the peaks both
candidates explain, compared at the LOWEST shared energy (the measure that worked) and on the slope
(the measure that did not). Every arm gets the label shuffle, because a quantity that separates
truth from decoys is worthless if it separates any promoted candidate equally well.

WHY bd MIGHT BEAT nb: a count cannot see that opening a ring costs two aromatic bonds at about
1036 kJ/mol while snipping a chain costs two at 696. Natural products are ring-dense, so that is
exactly the distinction that should matter here.
WHY IT MIGHT NOT: tabulated dissociation energies are averages over molecular contexts. The real
energy of a bond depends on what surrounds it -- resonance, strain, neighbouring electronegativity --
which is Lee's own argument turned against the lookup table. If the tables are too coarse, bd is a
noisier version of nb and will score the same or worse.

PRE-REGISTERED, before the run: bd beats nb by at least 2 points on the low-energy measure. Under
that, the extra physics is not reaching the signal and the count stands as the better feature
because it is simpler and needs no table.
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
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "e64energy_n600"
PRED = f"{EXPORTS}/iceberg_bde_{SPLIT}.npz"
TOL, E_TOL, TOP_OBS, BOOT = 0.01, 0.05, 40, 4000
QUANTS = [("nb", "count of bonds"), ("bd", "summed kJ/mol"), ("rb", "ring bonds")]


def explained(obs_mz, pmz, pval):
    """{observed peak index: value} using whichever per-peak quantity was passed in."""
    o = np.argsort(pmz); pm, pv = pmz[o], np.asarray(pval)[o]
    out = {}
    for i, m in enumerate(obs_mz):
        j = np.searchsorted(pm, m); best = None
        for k in (j - 1, j):
            if 0 <= k < len(pm) and abs(pm[k] - m) <= TOL:
                if best is None or abs(pm[k] - m) < abs(pm[best] - m): best = k
        if best is not None: out[i] = float(pv[best])
    return out


def compare(anchor, rival, obs):
    """(anchor beats rival on slope, anchor lower at lowest shared energy) or None."""
    xs, ya, yr = [], [], []
    for e in sorted(set(anchor) & set(rival)):
        both = set(anchor[e]) & set(rival[e])
        if not both: continue
        w = obs[e][1]; den = sum(w[i] for i in both)
        if den <= 0: continue
        xs.append(e)
        ya.append(sum(w[i] * anchor[e][i] for i in both) / den)
        yr.append(sum(w[i] * rival[e][i] for i in both) / den)
    if len(xs) < 2: return None
    sa = float(np.polyfit(xs, ya, 1)[0]); sr = float(np.polyfit(xs, yr, 1)[0])
    win = 1.0 if sa > sr else (0.5 if sa == sr else 0.0)
    low = 1.0 if ya[0] < yr[0] else (0.5 if ya[0] == yr[0] else 0.0)
    return win, low


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    d = np.load(PRED)
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

    rng = np.random.default_rng(70)
    # per quantity, per arm: one value per query
    res = {q: {"real": {"win": [], "low": []}, "pseudo": {"win": [], "low": []}}
           for q, _ in QUANTS}
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
        if len(obs) < 2: continue
        obs_es = np.array(sorted(obs))

        # one pseudo-truth per query, shared across quantities so the arms stay comparable
        avail = [c for c in cands if c != tkey]
        if not avail: continue
        pkey = avail[int(rng.integers(0, len(avail)))]

        for qn, _ in QUANTS:
            expl = {}
            for c, byE in cands.items():
                per_e = {}
                for pe in sorted(byE):
                    v = byE[pe]
                    if "mz" not in v or qn not in v or len(v["mz"]) == 0: continue
                    j = int(np.argmin(np.abs(obs_es - pe)))
                    if abs(obs_es[j] - pe) > E_TOL: continue
                    oe = float(obs_es[j])
                    per_e[oe] = explained(obs[oe][0], v["mz"], v[qn])
                if len(per_e) >= 2: expl[c] = per_e
            if tkey not in expl or len(expl) < 2: continue
            for arm, anchor in (("real", tkey), ("pseudo", pkey)):
                if anchor not in expl: continue
                rivals = [c for c in expl if c != anchor and not (arm == "pseudo" and c == tkey)]
                got = [compare(expl[anchor], expl[c], obs) for c in rivals]
                got = [g for g in got if g]
                if not got: continue
                res[qn][arm]["win"].append(float(np.mean([g[0] for g in got])))
                res[qn][arm]["low"].append(float(np.mean([g[1] for g in got])))

    def ci(v):
        v = np.asarray(v, float)
        if len(v) == 0: return (float("nan"),) * 3
        bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT)])
        return v.mean(), *np.percentile(bs, [2.5, 97.5])

    L = [f"E70 — pricing the broken bonds against counting them  (split {SPLIT})", "",
         f"{'quantity':18s} {'arm':8s} {'n':>4} {'slope win':>22} {'lower at low E':>24}"]
    for qn, label in QUANTS:
        for arm in ("real", "pseudo"):
            v = res[qn][arm]
            if not v["low"]: continue
            w, wlo, whi = ci(v["win"]); lo, llo, lhi = ci(v["low"])
            L.append(f"{(label if arm=='real' else ''):18s} {arm:8s} {len(v['low']):>4}   "
                     f"{w:.3f} [{wlo:.3f},{whi:.3f}]   {lo:.3f} [{llo:.3f},{lhi:.3f}]")
        L.append("")
    nb = res["nb"]["real"]["low"]; bd = res["bd"]["real"]["low"]
    if nb and bd and len(nb) == len(bd):
        diff = np.asarray(bd, float) - np.asarray(nb, float)
        m, lo, hi = ci(diff)
        L += ["PAIRED WITHIN QUERY, energy-sum minus count on the low-energy measure:",
              f"  {m:+.4f}  95% CI {lo:+.4f} to {hi:+.4f}",
              f"  pre-registered threshold was +0.02; "
              f"{'MET' if m >= 0.02 else 'NOT MET -- the count stands, being simpler'}",
              "",
              "If the energy sum does not beat the count, the reading is that the tabulated",
              "dissociation energies are too coarse to add information here -- they are averages over",
              "molecular contexts, and context is precisely what distinguishes two isomers. That is",
              "Lee's own argument applied to the lookup table rather than to the count."]
    L += [f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E70_bde_vs_count_{SPLIT}.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
