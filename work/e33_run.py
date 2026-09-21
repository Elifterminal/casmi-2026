#!/usr/bin/env python3
"""E33 part 2 — chi2 and the edit-shift quotient, end to end.

Four points of comparison, because building these pools exposed a latent quirk in production
that has to be priced separately from the grounds themselves:

  fr        production: sqrt(p) Bhattacharyya, peaks NOT aggregated within a bin  (known 0.3682)
  fr_agg    the same ground with same-bin peaks summed before normalising
  chi2      their engine's default divergence, as 1 - chi2, on aggregated bins
  eshift    sqrt(p) Bhattacharyya maximised over the structural-edit shift group

Production's prep() leaves duplicate bin indices; the pointer-walk cosine pairs them off, which
is self-consistent but is not "a distribution over bins" and cannot be written densely (it gave a
Bhattacharyya coefficient of 1.2 when I tried, which is how it was found). fr_agg prices that fix
on its own so the two ground arms are not confounded by it.

Every arm retrains from scratch on the twin-safe training splits, generation off, evaluated on
10-12 -- identical to E25.

PREDICTIONS (before the run, 2026-09-21):
  P1  fr_agg lands within 0.005 of production fr. Same-bin duplicates are rare and the pointer
      walk mostly pairs them correctly; this should be a tidy-up, not a result.
  P2  chi2 does NOT beat fr_agg by a clear margin -- E29's paired mean crossed zero even though
      its median looked better, and a median gain that a paired test cannot see usually means a
      reshuffle among near-ties rather than a real lift.
  P3  eshift beats fr_agg, CI clear of zero. It had the best mean and the best share above 0.6
      in E29, and it is the only arm carrying information about what an analogue IS.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import build, fit, rank_score, FEATS2, EVAL, TRAIN, W

RESULTS = os.path.expanduser("~/casmi-2026/work/results")
ARMS = [("fr (production)", "fr_"), ("fr_agg", "fr_agg_"), ("chi2", "chi2_"), ("eshift", "eshift_")]


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    st = load_structures()
    out, per = {}, {}
    for name, prefix in ARMS:
        ev = build(EVAL, st, log=log, pools_prefix=prefix, generate=False)
        tr = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log, pools_prefix=prefix, generate=False)
        w, n = fit(tr)
        p = defaultdict(list)
        for q in ev: p[q["cls"]].append(rank_score(q, w, use_gen=False))
        c = [float(np.mean(p[k])) for k in (1, 2, 3)]
        per[name] = p
        out[name] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=sum(W[k]*c[k-1] for k in (1,2,3)),
                         weights=dict(zip(FEATS2, map(float, w))))
        log(f"  {name:16s} weighted {out[name]['weighted']:.4f}  (trained on {n:,} pairs)")
    L = ["E33 — chi2 and the edit-shift quotient, end to end", "",
         f"{'arm':16s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9} {'vs fr_agg':>10}"]
    base = out["fr_agg"]["weighted"]
    for name, _ in ARMS:
        o = out[name]
        L.append(f"{name:16s} {o['C1']:>8.4f} {o['C2']:>8.4f} {o['C3']:>8.4f} {o['weighted']:>9.4f} "
                 f"{o['weighted']-base:>+10.4f}")
    rng = np.random.default_rng(0)
    L.append("")
    for name, _ in ARMS:
        if name == "fr_agg": continue
        a, b = per["fr_agg"], per[name]
        d = out[name]["weighted"] - base
        boots = [sum(W[c] * np.mean([b[c][i] - a[c][i] for i in rng.integers(0, len(a[c]), len(a[c]))])
                     for c in (1, 2, 3)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        L.append(f"{name:16s} vs fr_agg: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                 + ("   <- clears zero" if lo > 0 else ""))
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E33_grounds_e2e_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E33_grounds_e2e.json", "w"), indent=2)


if __name__ == "__main__":
    main()
