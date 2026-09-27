#!/usr/bin/env python3
"""E66 — the bond-depth test with enough queries to answer, paired so the answer means something.

Two things were wrong with E63 and both are fixed here.

1. IT WAS CONFOUNDED. The statistic averaged break depth over the peaks a candidate CAN EXPLAIN,
   and candidates explain different peaks: E63b measured the truth explaining 9.9 observed peaks
   against a decoy's 5.5, with the truth explaining more in 68% of pairs. Comparing those two
   averages is the E48 failure mode -- a quantity computed on a truth-favouring subset. Here every
   truth-decoy comparison uses ONLY the peaks both of them explain.

2. IT WAS UNDERPOWERED, for a reason worth recording. E63 measured 28 queries. That was not bad
   luck: requiring the truth in COCONUT (or the scorer cannot rank it) AND >=2 single-valued
   timsTOF energies AND absence from MassSpecGym leaves 49 structures in all of training, 28 with
   scaffold rarity. The top-25 design cannot be powered on this data at all.
   So the candidates here come from the MASS-WINDOW POOL instead -- the truth plus its 24
   nearest-in-mass rivals -- which gives 197 queries. What that buys and costs is stated below.

WHAT THIS MEASURES: whether implied break depth separates the truth from its same-mass rivals.
WHAT IT DOES NOT: whether adding break depth raises MRR inside the shipping ranker's top 25. These
candidates are not the ones the ranker sees. The first has to be true for the second to be
possible, which is the only reason to measure it on its own.

PRE-REGISTERED PREDICTIONS (written before the run, 2026-09-27):
  P1  Paired, the truth's depth-vs-energy slope beats a decoy's in 50-60% of pairs, with a
      bootstrap CI over QUERIES (not pairs -- pairs inside a query are correlated) that EXCLUDES
      50%. E63b saw 33.2% on the crippled slice, which if it survives means the effect is real and
      INVERTED; I am predicting against my own earlier reading here, because that reading came from
      a set where the energy axis barely existed.
  P2  Whichever direction it takes, |rate - 50%| > 5 points. Below that I will call it absent
      rather than argue about a couple of points.
  P3  The eliminator (drop candidates whose depth falls as energy rises) clears >1.3x decoys per
      truth lost. Under 1.3 it cannot pay for itself inside a 25-slot list.
  If P1 and P2 both fail at n=197, the energy axis is closed for this competition and I will say
  so plainly rather than look for a fourth slice to try it on.

CAVEAT THAT DOES NOT GO AWAY: this inherits ICEBERG's accuracy. On molecules outside its training
its predicted-versus-observed cosine is 0.147. A claim about which bonds broke is only as good as
the prediction of which fragments appear, which is why a null here still needs the contaminated
control before the mechanism itself is written off.
"""
import json, os, pickle, sys, time
from collections import Counter, defaultdict
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "e64energy_n600"
PRED = f"{EXPORTS}/iceberg_bonds_pool_{SPLIT}.npz"
WORKLIST = f"{EXPORTS}/iceberg_bonds_pool_{SPLIT}.json"
TOL, E_TOL, TOP_OBS, BOOT = 0.01, 0.05, 40, 4000


def explained(obs_mz, pmz, pnb):
    """{observed peak index: bonds broken} for the peaks this candidate accounts for."""
    o = np.argsort(pmz); pm, pn = pmz[o], pnb[o]
    out = {}
    for i, m in enumerate(obs_mz):
        j = np.searchsorted(pm, m); best = None
        for k in (j - 1, j):
            if 0 <= k < len(pm) and abs(pm[k] - m) <= TOL:
                if best is None or abs(pm[k] - m) < abs(pm[best] - m): best = k
        if best is not None: out[i] = int(pn[best])
    return out


def slope_low(xs, ys):
    return float(np.polyfit(xs, ys, 1)[0]), float(ys[0])


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    d = np.load(PRED)
    pred = defaultdict(lambda: defaultdict(dict))
    for key in d.keys():
        q, rest, kind = key.split("|")
        cand, _, e = rest.rpartition("@")
        pred[q][cand].setdefault(float(e), {})[kind] = d[key]
    wl = {q["key"]: q for q in json.load(open(WORKLIST))["queries"]}
    log(f"{len(pred)} queries predicted, {len(wl)} in the work list")

    sp = json.load(open(f"{SPLITS}/split_{SPLIT}.json"))
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"])
    ce_raw = t.column("collision_energy_ev").to_pylist()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    log("observed spectra loaded")

    rows, why = [], Counter()
    unpaired = []
    for qk, cands in pred.items():
        q = wl.get(qk)
        if q is None: why["not in work list"] += 1; continue
        # the exporter writes the truth first, but position is not identity -- confirm it at the
        # scorer's key, the same way the competition decides whether an answer is right
        tkey = next((c["key"] for c in q["candidates"]
                     if c["smiles"] and scoring.key14(c["smiles"]) == q["truth"]), None)
        if tkey is None: why["truth not identifiable among candidates"] += 1; continue
        if tkey not in cands: why["truth has no predictions"] += 1; continue

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
        if len(obs) < 2: why["fewer than 2 usable observed energies"] += 1; continue
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
        if tkey not in expl: why["truth lacks 2 predicted energies"] += 1; continue
        if len(expl) < 2: why["fewer than 2 candidates measurable"] += 1; continue

        tm = expl[tkey]
        # UNPAIRED, kept only to show what E63's statistic would have said on this data
        def own(m):
            xs, ys = [], []
            for e in sorted(m):
                w = obs[e][1]; den = sum(w[i] for i in m[e])
                if den > 0: xs.append(e); ys.append(sum(w[i] * m[e][i] for i in m[e]) / den)
            return slope_low(xs, ys) if len(xs) >= 2 else None
        ot = own(tm)
        od = [own(expl[c]) for c in expl if c != tkey]
        od = [x for x in od if x]
        if ot and od:
            unpaired.append((ot[0], float(np.median([x[0] for x in od]))))

        # PAIRED: every comparison restricted to the peaks both candidates explain
        wins, lows, npair, shared_tot = [], [], 0, 0
        t_neg = d_neg = d_tot = 0
        for c, cm in expl.items():
            if c == tkey: continue
            xs, yt, yd, ns = [], [], [], 0
            for e in sorted(set(tm) & set(cm)):
                both = set(tm[e]) & set(cm[e])
                if not both: continue
                w = obs[e][1]; den = sum(w[i] for i in both)
                if den <= 0: continue
                xs.append(e)
                yt.append(sum(w[i] * tm[e][i] for i in both) / den)
                yd.append(sum(w[i] * cm[e][i] for i in both) / den)
                ns += len(both)
            if len(xs) < 2 or ns == 0: continue
            ts, tl = slope_low(xs, yt); ds, dl = slope_low(xs, yd)
            wins.append(1.0 if ts > ds else (0.5 if ts == ds else 0.0))
            lows.append(1.0 if tl < dl else (0.5 if tl == dl else 0.0))
            t_neg += (ts < 0); d_neg += (ds < 0); d_tot += 1
            npair += 1; shared_tot += ns
        if npair == 0: why["no pair shared 2 energies of peaks"] += 1; continue
        why["MEASURED"] += 1
        rows.append(dict(q=qk, cls=sp["assign"][qk], npair=npair,
                         shared=shared_tot / npair,
                         win=float(np.mean(wins)), low=float(np.mean(lows)),
                         truth_neg=float(t_neg / npair), decoy_neg=float(d_neg / max(d_tot, 1))))

    log(f"{len(rows)} queries measurable")
    rng = np.random.default_rng(66)
    def ci(v):
        v = np.asarray(v, float)
        bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT)])
        return v.mean(), *np.percentile(bs, [2.5, 97.5])

    L = ["E66 — implied break depth against the energy ladder, paired and powered", "",
         f"{len(rows)} measurable queries (E63 managed 28), "
         f"{sum(r['npair'] for r in rows):,} truth-decoy pairs", ""]
    for k, v in why.most_common(): L.append(f"  {v:5d}  {k}")
    if rows:
        w, wlo, whi = ci([r["win"] for r in rows])
        lo_, llo, lhi = ci([r["low"] for r in rows])
        tn = float(np.mean([r["truth_neg"] for r in rows]))
        dn = float(np.mean([r["decoy_neg"] for r in rows]))
        L += ["",
              "PAIRED ON A COMMON PEAK SET — share of same-mass rivals whose depth-vs-energy",
              "slope the truth beats (chance 0.500, CI bootstrapped over QUERIES):",
              f"  {w:.3f}   95% CI {wlo:.3f} to {whi:.3f}",
              f"  excludes chance: {'YES' if (wlo > 0.5 or whi < 0.5) else 'NO'}"
              f"   |effect| {abs(w-0.5)*100:.1f} points (P2 wanted > 5)",
              "",
              "TRUTH SHALLOWER AT THE LOWEST ENERGY than the rival (chance 0.500):",
              f"  {lo_:.3f}   95% CI {llo:.3f} to {lhi:.3f}",
              "",
              "AS AN ELIMINATOR (drop candidates whose implied depth FALLS as energy rises):",
              f"  truths flagged {tn:.1%} | decoys flagged {dn:.1%}",
              "  -> decoys removed per truth lost: "
              + (f"{dn/tn:.2f}x  (P3 wanted > 1.30)" if tn > 0 else "undefined, no truth flagged"),
              "",
              f"mean shared peaks per pair: {np.mean([r['shared'] for r in rows]):.1f}"]
        for cl in sorted(set(r["cls"] for r in rows)):
            g = [r for r in rows if r["cls"] == cl]
            L.append(f"  class {cl}: n={len(g)}  paired win rate {np.mean([r['win'] for r in g]):.3f}")
        if unpaired:
            u = np.array(unpaired)
            L += ["", "FOR CONTRAST, E63's unpaired statistic on this same data:",
                  f"  truth slope beats median decoy in {np.mean(u[:,0] > u[:,1]):.1%} of "
                  f"{len(u)} queries",
                  "  (it compares depths computed over different peak subsets, so it is reported",
                  "   only to show the size of the distortion, never as a result)"]
    L += ["", "SCOPE: candidates are the truth plus its 24 nearest-in-mass rivals from the",
          "mass-window pool, monomer adducts only -- dimer adducts ([2M+Na]+ and friends) are",
          "excluded because the neutral-mass arithmetic that builds the window assumes a monomer.",
          "This measures separation from same-mass rivals, NOT an MRR gain in the shipping ranker.",
          "",
          "ICEBERG's cosine on molecules outside its training is 0.147, so a null here is not yet",
          "a verdict on the chemistry -- the contaminated control separates 'no mechanism' from",
          "'the predictor cannot see it'.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E66_bond_depth_{SPLIT}_2026-09-27.txt", "w").write("\n".join(L) + "\n")
    json.dump(rows, open(f"{RESULTS}/E66_bond_depth_{SPLIT}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
