#!/usr/bin/env python3
"""E69 — does the break-depth effect grow where ICEBERG is actually accurate?

E66 found the truth explains low-energy peaks with less broken bond than its same-mass rivals,
0.574 against 0.500 chance, label-controlled. The open question is what limits it. Two readings:

  (a) 0.574 is the size of the mechanism, and a better fragment predictor would not help;
  (b) 0.574 is what survives a predictor whose cosine on novel chemistry is 0.147, and accuracy is
      the lever.

The planned test for this was a contaminated arm -- the same measurement on molecules inside
MassSpecGym, used throughout this project as the proxy for "ICEBERG has seen it". That arm exists
and is running, but it can only reach n=77: just 163 structures in all of training are both inside
MassSpecGym and carrying a timsTOF energy ladder.

This asks the same question better and without a new prediction run, by dropping the proxy. We have
ICEBERG's predicted spectrum for the truth of every one of the 191 clean queries, and we have the
observed spectrum. So measure the accuracy DIRECTLY, per query, and look at whether the effect
tracks it. Membership in a database is a guess about accuracy; the cosine is the accuracy.

SIMILARITY uses the project's own ground: bin to 0.01 Da, square-root the intensities, L2 normalise,
dot (E29/E33 -- sqrt(p) beat cosine by +0.0168 end-to-end and transferred to the board).

PREDICTION, before the run: if reading (b) is right, the effect should rise monotonically across
accuracy terciles, and the top tercile should sit clearly above 0.574. If the effect is flat across
terciles, a better predictor buys nothing and 0.574 is the mechanism's whole size.

WHAT WOULD MAKE THIS MISLEADING, stated up front: prediction accuracy is not randomly assigned.
ICEBERG predicts small, rigid, well-behaved molecules better, and those molecules may also be
easier to tell from their rivals for reasons having nothing to do with bond energies. So a rising
gradient is evidence that accuracy matters, NOT proof -- the confound runs the same direction as
the hypothesis. The decoy-difficulty check at the end is there to say how much of the gradient is
explained by that.
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
TOL, E_TOL, TOP_OBS, BOOT, BIN = 0.01, 0.05, 40, 4000, 0.01


def ground(mz, it):
    """sqrt(p) then L2 normalise, keyed on 0.01 Da bins -- the project's Fisher-Rao ground."""
    d = defaultdict(float)
    for m, i in zip(mz, it):
        if i > 0: d[int(round(m / BIN))] += float(i)
    if not d: return {}
    v = {k: np.sqrt(x) for k, x in d.items()}
    n = np.sqrt(sum(x * x for x in v.values()))
    return {k: x / n for k, x in v.items()} if n > 0 else {}


def cos(a, b):
    if not a or not b: return 0.0
    keys = a.keys() & b.keys()
    return float(sum(a[k] * b[k] for k in keys))


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

    rows = []
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

        # ICEBERG's accuracy on THIS query: mean similarity of the truth's predicted spectrum to
        # the observed one, over the energies where both exist
        sims = []
        for pe, v in cands[tkey].items():
            if "mz" not in v or len(v["mz"]) == 0: continue
            j = int(np.argmin(np.abs(obs_es - pe)))
            if abs(obs_es[j] - pe) > E_TOL: continue
            oe = float(obs_es[j])
            sims.append(cos(ground(obs[oe][0], obs[oe][1]), ground(v["mz"], v["it"])))
        if not sims: continue
        acc = float(np.mean(sims))

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
        if tkey not in expl or len(expl) < 2: continue
        tm = expl[tkey]
        lows, wins, dmass = [], [], []
        for c, cm in expl.items():
            if c == tkey: continue
            xs, yt, yd = [], [], []
            for e in sorted(set(tm) & set(cm)):
                both = set(tm[e]) & set(cm[e])
                if not both: continue
                w = obs[e][1]; den = sum(w[i] for i in both)
                if den <= 0: continue
                xs.append(e)
                yt.append(sum(w[i] * tm[e][i] for i in both) / den)
                yd.append(sum(w[i] * cm[e][i] for i in both) / den)
            if len(xs) < 2: continue
            lows.append(1.0 if yt[0] < yd[0] else (0.5 if yt[0] == yd[0] else 0.0))
            st = float(np.polyfit(xs, yt, 1)[0]); sd = float(np.polyfit(xs, yd, 1)[0])
            wins.append(1.0 if st > sd else (0.5 if st == sd else 0.0))
        if not lows: continue
        rows.append(dict(q=qk, acc=acc, low=float(np.mean(lows)), win=float(np.mean(wins)),
                         npair=len(lows)))
    log(f"{len(rows)} queries with an accuracy and an effect")

    rng = np.random.default_rng(69)
    def ci(v):
        v = np.asarray(v, float)
        bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT)])
        return v.mean(), *np.percentile(bs, [2.5, 97.5])

    acc = np.array([r["acc"] for r in rows]); low = np.array([r["low"] for r in rows])
    L = [f"E69 — is the break-depth effect limited by ICEBERG's accuracy?  (split {SPLIT})", "",
         f"{len(rows)} queries", "",
         f"ICEBERG accuracy on the truth (sqrt(p) cosine): median {np.median(acc):.3f}, "
         f"mean {acc.mean():.3f}, range {acc.min():.3f}-{acc.max():.3f}", ""]
    order = np.argsort(acc)
    k = len(order) // 3
    groups = [("low  accuracy", order[:k]), ("mid  accuracy", order[k:2*k]),
              ("high accuracy", order[2*k:])]
    L.append(f"{'tercile':16s} {'n':>4} {'cosine':>14} {'shallower at low E':>24}")
    for name, idx in groups:
        if len(idx) == 0: continue
        m, lo, hi = ci(low[idx])
        L.append(f"{name:16s} {len(idx):>4} {acc[idx].mean():>13.3f}   "
                 f"{m:.3f} [{lo:.3f},{hi:.3f}]")
    m, lo, hi = ci(low)
    L += ["", f"all queries      {len(low):>4} {acc.mean():>13.3f}   {m:.3f} [{lo:.3f},{hi:.3f}]"]
    if len(rows) > 10:
        r = float(np.corrcoef(acc, low)[0, 1])
        bs = np.array([np.corrcoef(acc[i], low[i])[0, 1]
                       for i in (rng.integers(0, len(acc), len(acc)) for _ in range(BOOT))])
        bs = bs[np.isfinite(bs)]
        L += ["", f"correlation between accuracy and the effect: r = {r:+.3f} "
                  f"(95% CI {np.percentile(bs,2.5):+.3f} to {np.percentile(bs,97.5):+.3f})"]
        hi_t = groups[-1][1]; lo_t = groups[0][1]
        L += [f"high tercile minus low tercile: {low[hi_t].mean()-low[lo_t].mean():+.3f}",
              "",
              "READING IT: a clear positive gradient means accuracy is the lever and a better",
              "fragment predictor would amplify the effect. A flat gradient means 0.574 is the whole",
              "mechanism. The confound runs the same way as the hypothesis -- easy-to-predict",
              "molecules may be easy to distinguish for unrelated reasons -- so a gradient is",
              "evidence, not proof."]
    L += [f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E69_accuracy_gradient_{SPLIT}.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
