#!/usr/bin/env python3
"""E63b — where did 164 queries become 28, and is the depth statistic confounded?

E63 returned a null on all three pre-registered predictions. Before recording that as an answer
about the chemistry, two things have to be ruled out, because a null from a broken instrument
looks exactly like a null from an absent effect.

ATTRITION. 164 queries carried predictions; 28 survived to be measured. Every filter is counted
here so the loss is attributable rather than assumed.

CONFOUND. implied_depth averages bonds-broken over the observed peaks THAT CANDIDATE CAN EXPLAIN.
Each candidate is therefore scored on a different subset of the spectrum. The truth generally
explains more peaks than a decoy, so truth and decoy depths are not comparable quantities -- the
same conditioning that produced the E48 artifact, where a feature present for truths and absent
for decoys became a truth detector by construction. This measures the overlap directly: how many
peaks each side explains, and what the depth comparison looks like when BOTH sides are restricted
to the peaks they can both explain.
"""
import json, os, pickle, sys, time
from collections import Counter, defaultdict
import numpy as np
import pyarrow.parquet as pq

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
SPLIT = "npxmtsseed60_n300"
TOL, E_TOL, TOP_OBS = 0.01, 0.05, 40


def matched(obs_mz, pmz, pnb):
    """Return {index of observed peak: bonds_broken} for the peaks this candidate explains."""
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
    d = np.load(f"{EXPORTS}/iceberg_bonds_seed60.npz")
    pred = defaultdict(lambda: defaultdict(dict))
    for k in d.keys():
        q, rest, kind = k.split("|")
        cand, _, e = rest.rpartition("@")
        pred[q][cand].setdefault(float(e), {})[kind] = d[k]

    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"])
    ce_raw = t.column("collision_energy_ev").to_pylist()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    sp = json.load(open(f"{SPLITS}/split_{SPLIT}.json"))
    pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_fr_{SPLIT}.pkl", "rb"))}
    log(f"{len(pred)} queries with predictions")

    why = Counter()
    pair_rows, cover = [], []
    for qk, cands in pred.items():
        p = pools.get(qk)
        if p is None or not p["truth"]: why["no pool or no truth"] += 1; continue
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
        if p["truth"] not in cands: why["TRUTH NOT AMONG THE PREDICTED TOP-25"] += 1; continue
        obs_es = np.array(sorted(obs))

        # per candidate: which observed peaks it explains, at which depth, per energy
        expl = {}
        for c, byE in cands.items():
            per_e = {}
            for pe in sorted(byE):
                v = byE[pe]
                if "mz" not in v or len(v["mz"]) == 0: continue
                j = int(np.argmin(np.abs(obs_es - pe)))
                if abs(obs_es[j] - pe) > E_TOL: continue
                oe = float(obs_es[j])
                per_e[oe] = matched(obs[oe][0], v["mz"], v["nb"])
            if len(per_e) >= 2: expl[c] = per_e
        if p["truth"] not in expl: why["truth had <2 energies with predictions"] += 1; continue
        if len(expl) < 2: why["fewer than 2 candidates measurable"] += 1; continue
        why["MEASURED"] += 1

        tm = expl[p["truth"]]
        n_obs = sum(len(obs[e][0]) for e in tm)
        t_hits = sum(len(v) for v in tm.values())
        for c, cm in expl.items():
            if c == p["truth"]: continue
            shared_e = [e for e in tm if e in cm]
            if len(shared_e) < 2: continue
            c_hits = sum(len(cm[e]) for e in shared_e)
            # PAIRED: only the peaks BOTH explain, so both sides are scored on one peak set
            xs, yt, yd, n_shared = [], [], [], 0
            for e in sorted(shared_e):
                both = set(tm[e]) & set(cm[e])
                if not both: continue
                w = obs[e][1]
                den = sum(w[i] for i in both)
                if den <= 0: continue
                xs.append(e)
                yt.append(sum(w[i] * tm[e][i] for i in both) / den)
                yd.append(sum(w[i] * cm[e][i] for i in both) / den)
                n_shared += len(both)
            cover.append(dict(q=qk, t_hits=t_hits, c_hits=sum(len(v) for v in cm.values()),
                              n_obs=n_obs))
            if len(xs) >= 2 and n_shared > 0:
                pair_rows.append(dict(q=qk, cls=sp["assign"][qk], n_shared=n_shared,
                                      t_slope=float(np.polyfit(xs, yt, 1)[0]),
                                      d_slope=float(np.polyfit(xs, yd, 1)[0]),
                                      t_low=yt[0], d_low=yd[0]))

    L = ["E63b — attrition and confound audit for the bond-depth test", "",
         "WHERE THE QUERIES WENT (164 with predictions):"]
    for k, v in why.most_common():
        L.append(f"  {v:4d}  {k}")
    if cover:
        th = np.array([c["t_hits"] for c in cover]); ch = np.array([c["c_hits"] for c in cover])
        L += ["", "COVERAGE ASYMMETRY (the E48 failure mode, checked not assumed):",
              f"  observed peaks the TRUTH explains, mean   {th.mean():.1f}",
              f"  observed peaks a DECOY explains, mean     {ch.mean():.1f}",
              f"  truth explains MORE than the decoy in     {np.mean(th > ch):.1%} of pairs",
              "  -> if this is far from 50%, the unpaired E63 statistic compared two different",
              "     subsets of the spectrum and its null is uninterpretable either way."]
    if pair_rows:
        ts = np.array([r["t_slope"] for r in pair_rows]); ds = np.array([r["d_slope"] for r in pair_rows])
        tl = np.array([r["t_low"] for r in pair_rows]); dl = np.array([r["d_low"] for r in pair_rows])
        win = (ts > ds).astype(float)
        rng = np.random.default_rng(631)
        bs = np.array([win[rng.integers(0, len(win), len(win))].mean() for _ in range(2000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L += ["", f"PAIRED ON A COMMON PEAK SET ({len(pair_rows)} truth-decoy pairs, "
                  f"{len(set(r['q'] for r in pair_rows))} queries):",
              f"  truth slope steeper than decoy: {win.mean():.1%} (95% CI {lo:.1%}-{hi:.1%})",
              f"  truth shallower at lowest energy: {np.mean(tl < dl):.1%}",
              f"  mean shared peaks per pair: {np.mean([r['n_shared'] for r in pair_rows]):.1f}",
              "  NOTE: the CI here is over PAIRS, which are correlated within a query, so it is",
              "  optimistic. It bounds the effect's size, it does not license a significance claim."]
    L += ["", f"runtime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E63b_attrition_2026-09-27.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
