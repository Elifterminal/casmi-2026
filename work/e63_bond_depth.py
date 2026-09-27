#!/usr/bin/env python3
"""E63 — Lee's idea at bond level: do a candidate's implied break depths match the energy ladder?

E52 asked this with our own fragmenter, which matched a median of FOUR observed peaks -- nothing
to measure. E54 asked a mass-collapsed version through ICEBERG's spectra and got 54.2% (chance).
Lee's objection to my explanation was correct: bond strength depends on what sits either side of
the bond, so adjacency -- arrangement -- does determine breaking order, and the information is
genuinely in the chemistry. What I had tested was the shadow of it.

E62 regenerated the predictions keeping the atom-membership masks ICEBERG returns alongside each
peak, so we can now recover THE NUMBER OF BONDS that had to break to produce any predicted
fragment. This is the first quantity we have tested that can separate candidates whose MASS
evidence is identical: two isomers can yield a fragment of the same weight via genuinely different
bond sets, one shallow and one deep.

THE MEASURE. For each observed peak at collision energy e, find the candidate's predicted fragment
at that mass and read its break depth. Intensity-weight across the observed peaks:

    implied_depth(e) = sum(obs_intensity * bonds_broken) / sum(obs_intensity)

Physics says gentle energy is explained by shallow breaks, so implied_depth should RISE with e.

  slope         least squares fit of implied_depth against energy. Should be positive.
  low_depth     implied_depth at the lowest energy. Should be small.
  ELIMINATOR    a candidate with a NEGATIVE slope claims deep fragments appear before shallow ones,
                which is chemically implausible regardless of how well the masses line up.

PREDICTIONS (before the run, 2026-09-27):
  P1  The truth's slope beats the median decoy's in 55-70% of queries. Better than E54's 54.2%
      because this reads a quantity that mass cannot express -- but not decisive, because it still
      inherits ICEBERG's accuracy on molecules it has never seen (cosine 0.147).
  P2  As an eliminator it clears 1.5x decoys-removed per truth-lost, against E54's 1.25x and E52's
      1.03x. This is the number that would make the idea useful rather than merely real.
  P3  Truths show a LOWER implied depth at the lowest energy than decoys. If a decoy has to invoke
      deep fragmentation to explain a gentle spectrum, that is exactly the implausibility Lee's
      argument predicts.
  If all three fail on the contaminated control as well, the idea is closed at bond level too and
  I will say so.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
SPLIT = "npxmtsseed60_n300"
PRED = "iceberg_bonds_seed60.npz"
WORKLIST = "iceberg_top25_d60_n300_c12.json"
TOL = 0.01          # m/z window for calling an observed peak explained by a predicted fragment
E_TOL = 0.05        # energies survive a "%g" round trip through the npz key; match by tolerance
TOP_OBS = 40
BOOT = 2000


def implied_depth(obs_mz, obs_it, pmz, pit, pnb):
    """Intensity-weighted mean bonds-broken over the observed peaks this candidate can explain."""
    if len(obs_mz) == 0 or len(pmz) == 0: return None, 0
    o = np.argsort(pmz); pm, pn = pmz[o], pnb[o]
    num = den = 0.0; hit = 0
    for m, w in zip(obs_mz, obs_it):
        j = np.searchsorted(pm, m)
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(pm) and abs(pm[k] - m) <= TOL:
                if best is None or abs(pm[k] - m) < abs(pm[best] - m): best = k
        if best is not None:
            num += w * float(pn[best]); den += w; hit += 1
    return (num / den if den > 0 else None), hit


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    d = np.load(f"{EXPORTS}/{PRED}")
    pred = defaultdict(lambda: defaultdict(dict))
    for k in d.keys():
        q, rest, kind = k.split("|")
        cand, _, e = rest.rpartition("@")
        pred[q][cand].setdefault(float(e), {})[kind] = d[k]
    log(f"{len(pred)} queries with bond-level predictions")

    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"])
    ce_raw = t.column("collision_energy_ev").to_pylist()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    sp = json.load(open(f"{SPLITS}/split_{SPLIT}.json"))
    pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{SPLIT}.pkl", "rb"))}
    log("observed spectra loaded")

    rows = []
    for qk, cands in pred.items():
        p = pools.get(qk)
        if p is None or not p["truth"]: continue
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
        per = {}
        for c, byE in cands.items():
            xs, ys, hits = [], [], 0
            for pe in sorted(byE):
                v = byE[pe]
                if "mz" not in v or len(v["mz"]) == 0: continue
                j = int(np.argmin(np.abs(obs_es - pe)))          # %g round trip, not exact equality
                if abs(obs_es[j] - pe) > E_TOL: continue
                oe = float(obs_es[j])
                dpt, h = implied_depth(obs[oe][0], obs[oe][1], v["mz"], v["it"], v["nb"])
                if dpt is None: continue
                xs.append(oe); ys.append(dpt); hits += h
            if len(xs) < 2 or hits == 0: continue
            per[c] = dict(slope=float(np.polyfit(xs, ys, 1)[0]), low=float(ys[0]), hits=hits)
        if len(per) < 2: continue
        # candidate keys ARE scorer InChIKey14s, so identity is a direct comparison -- no re-derivation
        tc = p["truth"] if p["truth"] in per else None
        if tc is None: continue
        others = [v for c, v in per.items() if c != tc]
        if not others: continue
        rows.append(dict(query=qk, cls=sp["assign"][qk], n_cands=len(per),
                         truth_slope=per[tc]["slope"], truth_low=per[tc]["low"],
                         med_slope=float(np.median([o["slope"] for o in others])),
                         med_low=float(np.median([o["low"] for o in others])),
                         truth_implausible=bool(per[tc]["slope"] < 0),
                         decoy_implausible=float(np.mean([o["slope"] < 0 for o in others]))))
    log(f"{len(rows)} queries measurable")

    L = ["E63 — bond-level break depth against the energy ladder (Lee's idea, properly tested)", "",
         f"{len(rows)} queries with >=2 energies and the truth among the candidates", ""]
    if rows:
        ts = np.array([r["truth_slope"] for r in rows]); ms = np.array([r["med_slope"] for r in rows])
        tl = np.array([r["truth_low"] for r in rows]); ml = np.array([r["med_low"] for r in rows])
        ti = np.mean([r["truth_implausible"] for r in rows])
        di = np.mean([r["decoy_implausible"] for r in rows])
        win = (ts > ms).astype(float)
        rng = np.random.default_rng(63)
        bs = np.array([win[rng.integers(0, len(win), len(win))].mean() for _ in range(BOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L += ["SLOPE of implied break depth against collision energy (physics says positive):",
              f"  truth        mean {ts.mean():+.4f}  median {np.median(ts):+.4f}",
              f"  median decoy mean {ms.mean():+.4f}  median {np.median(ms):+.4f}",
              f"  truth beats median decoy: {win.mean():.1%}  (95% CI {lo:.1%} to {hi:.1%})",
              f"  chance is 50.0%; CI excludes chance: {'YES' if lo > 0.5 else 'NO'}"]
        for cl in sorted(set(r["cls"] for r in rows)):
            g = [r for r in rows if r["cls"] == cl]
            L.append(f"    class {cl}: n={len(g)}  "
                     f"truth wins {np.mean([r['truth_slope'] > r['med_slope'] for r in g]):.1%}")
        L += ["",
              "IMPLIED DEPTH AT THE LOWEST ENERGY (physics says truths should be shallower):",
              f"  truth {tl.mean():.3f} vs median decoy {ml.mean():.3f}; "
              f"truth shallower in {np.mean(tl < ml):.1%}", "",
              "AS AN ELIMINATOR (negative slope = deep fragments before shallow ones):",
              f"  truths flagged implausible {ti:.1%}",
              f"  decoys flagged implausible {di:.1%}",
              "  -> decoys removed per truth lost: "
              + (f"{di/ti:.2f}x" if ti > 0 else "undefined (no truth was flagged, so no cost "
                                                "to measure against -- n is too small to trust)"),
              f"     (E52 with our fragmenter 1.03x; E54 mass-collapsed 1.25x)"]
    L += ["", "CAVEAT, unchanged: this inherits ICEBERG's accuracy, and on molecules outside its",
          "training its predicted-vs-observed cosine is 0.147. A claim about which bonds broke is",
          "only as good as the prediction of which fragments appear.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E63_bond_depth_2026-09-27.txt", "w").write("\n".join(L) + "\n")
    json.dump(rows, open(f"{RESULTS}/E63_bond_depth.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
