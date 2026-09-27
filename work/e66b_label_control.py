#!/usr/bin/env python3
"""E66b — label control for the one measure in E66 that worked.

E66 found the truth explains the lowest-energy peaks with SHALLOWER breaks than its same-mass
rivals in 57.4% of pairs (95% CI 55.0-59.8%, chance 50%). That is Lee's argument showing up in the
data: a candidate that has to snap five bonds to account for a gentle peak is implausible.

Before that goes on the page as a real effect it has to survive the control that E48 taught me to
run first. The question is whether the number depends on the candidate actually BEING the answer, or
whether it would appear for any candidate promoted to the role. So: pick a random decoy, call it the
truth, drop the real truth from the comparison set entirely, and recompute. Three arms:

  real         the truth against its rivals                      -> expect 0.574, as E66 measured
  pseudo       a random decoy against the other decoys           -> expect 0.500 if the effect is real
  mass-nearest the decoy CLOSEST in mass to the truth, promoted  -> guards against the possibility
               that the effect is about sitting at the centre of the mass window rather than about
               being right

If pseudo comes back near 0.574, the statistic is measuring something structural about how the
work list is built and the E66 result is an artifact. If pseudo sits at 0.500 and real does not,
the effect is about being the correct molecule.
"""
import json, os, sys, time
from collections import Counter, defaultdict
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS, RESULTS = os.path.join(WORK, "splits"), os.path.join(WORK, "results")
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "e64energy_n600"
TOL, E_TOL, TOP_OBS, BOOT = 0.01, 0.05, 40, 4000


def explained(obs_mz, pmz, pnb):
    o = np.argsort(pmz); pm, pn = pmz[o], pnb[o]
    out = {}
    for i, m in enumerate(obs_mz):
        j = np.searchsorted(pm, m); best = None
        for k in (j - 1, j):
            if 0 <= k < len(pm) and abs(pm[k] - m) <= TOL:
                if best is None or abs(pm[k] - m) < abs(pm[best] - m): best = k
        if best is not None: out[i] = int(pn[best])
    return out


def pair_stats(anchor, rivals, obs):
    """Share of rivals the anchor beats on slope, and on shallowness at the lowest shared energy."""
    wins, lows = [], []
    for cm in rivals:
        xs, ya, yr = [], [], []
        for e in sorted(set(anchor) & set(cm)):
            both = set(anchor[e]) & set(cm[e])
            if not both: continue
            w = obs[e][1]; den = sum(w[i] for i in both)
            if den <= 0: continue
            xs.append(e)
            ya.append(sum(w[i] * anchor[e][i] for i in both) / den)
            yr.append(sum(w[i] * cm[e][i] for i in both) / den)
        if len(xs) < 2: continue
        sa = float(np.polyfit(xs, ya, 1)[0]); sr = float(np.polyfit(xs, yr, 1)[0])
        wins.append(1.0 if sa > sr else (0.5 if sa == sr else 0.0))
        lows.append(1.0 if ya[0] < yr[0] else (0.5 if ya[0] == yr[0] else 0.0))
    if not wins: return None
    return float(np.mean(wins)), float(np.mean(lows)), len(wins)


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    d = np.load(f"{EXPORTS}/iceberg_bonds_pool_{SPLIT}.npz")
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

    rng = np.random.default_rng(662)
    arms = {"real": [], "pseudo": [], "mass_nearest": []}
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

        expl = {}
        for c, byE in cands.items():
            per_e = {}
            for pe in sorted(byE):
                v = byE[pe]
                if "mz" not in v or len(v["mz"]) == 0: continue
                j = int(np.argmin(np.abs(obs_es - pe)))
                if abs(obs_es[j] - pe) > E_TOL: continue
                oe = float(obs_es[j])
                per_e[oe] = explained(obs[oe][0], v["mz"], v["nb"])
            if len(per_e) >= 2: expl[c] = per_e
        if tkey not in expl or len(expl) < 3: continue

        decoys = [c for c in expl if c != tkey]
        r = pair_stats(expl[tkey], [expl[c] for c in decoys], obs)
        if r: arms["real"].append(r)
        # pseudo-truth: a random decoy, with the REAL truth removed from the rivals so the
        # comparison never contains the actual answer
        pk = decoys[int(rng.integers(0, len(decoys)))]
        others = [expl[c] for c in decoys if c != pk]
        if others:
            r = pair_stats(expl[pk], others, obs)
            if r: arms["pseudo"].append(r)
        # the work list writes decoys in nearest-in-mass order, so candidates[1] is the closest
        order = [c["key"] for c in q["candidates"][1:] if c["key"] in expl and c["key"] != tkey]
        if order:
            nk = order[0]
            others = [expl[c] for c in decoys if c != nk]
            if others:
                r = pair_stats(expl[nk], others, obs)
                if r: arms["mass_nearest"].append(r)

    def ci(v):
        v = np.asarray(v, float)
        rg = np.random.default_rng(7)
        bs = np.array([v[rg.integers(0, len(v), len(v))].mean() for _ in range(BOOT)])
        return v.mean(), *np.percentile(bs, [2.5, 97.5])

    L = ["E66b — label control: is the low-energy shallowness effect about being the ANSWER?", "",
         f"{'arm':14s} {'n':>5} {'slope win':>22} {'shallower at low E':>26}"]
    for name in ("real", "pseudo", "mass_nearest"):
        v = arms[name]
        if not v: L.append(f"{name:14s} {'--':>5}"); continue
        s, slo, shi = ci([x[0] for x in v]); l, llo, lhi = ci([x[1] for x in v])
        L.append(f"{name:14s} {len(v):>5}   {s:.3f} [{slo:.3f},{shi:.3f}]   "
                 f"{l:.3f} [{llo:.3f},{lhi:.3f}]")
    if arms["real"] and arms["pseudo"]:
        lr = np.mean([x[1] for x in arms["real"]]); lp = np.mean([x[1] for x in arms["pseudo"]])
        L += ["", f"real minus pseudo on the low-energy measure: {lr-lp:+.3f}",
              "  A pseudo arm near 0.500 means the effect tracks being the correct molecule.",
              "  A pseudo arm near the real arm means the statistic is an artifact of how the",
              "  work list is built, and E66's 0.574 does not survive."]
    L += ["", f"runtime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E66b_label_control_{SPLIT}_2026-09-27.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
