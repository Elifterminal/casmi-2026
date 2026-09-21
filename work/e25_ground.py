#!/usr/bin/env python3
"""E25 — does Vanta's ground pay end-to-end, or only when measured in isolation?

E24 measured relative QUALITY: ranking reference structures by Fisher-Rao on sqrt(p) instead of
our binned cosine puts a structurally closer relative in the top 5 (+0.0285 mean Tanimoto,
[+0.0156, +0.0420]). That is upstream evidence. What matters is whether it moves the score.

Because ||sqrt(p)|| = 1, the change is literally our own cosine applied to square-rooted
intensities -- one line in prep(). It flattens the base peak's dominance and gives weak peaks a
say, which is also what her pooling theorem argues for: the diagnostic fragment is usually weak,
and the intense peaks are shared by every wrong candidate.

Both arms retrain from scratch on the twin-safe training splits and are evaluated on 10-12, with
the analog-generation channel OFF (submission 2 showed it costs points on the real leaderboard).

  cosine ground   pools_nptsseed*.pkl        our convention
  sqrt-p ground   pools_fr_nptsseed*.pkl     hers

PREDICTIONS (before the run, 2026-09-21):
  P1  The sqrt-p ground raises the weighted score; CI against the cosine ground clears zero.
  P2  The gain is smaller than E24's isolated measurement suggests -- better relatives help the
      neighbour vote, but the ranker has other features and Class 1 barely depends on relatives.
      I expect +0.005 to +0.02 weighted.
  P3  Class 2 gains most (it is what the neighbour vote carries).

  RESULT (2026-09-21): ALL THREE HELD. cosine 0.3513 -> sqrt-p 0.3682, +0.0168 weighted
  [+0.0086, +0.0254]. C2 0.4920 -> 0.5250 (+0.033, the biggest move); C1 0.8122 -> 0.8245.
  The gain is real end-to-end and smaller than E24's isolated +0.0285 Tanimoto implied, as
  predicted. This is a one-line change to prep() -- square-root the intensities before
  normalising -- and unlike analog generation it is not exploiting anything about our
  manufactured Class 3, so it should transfer to the leaderboard.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import build, fit, rank_score, FEATS2, EVAL, TRAIN, W

RESULTS = os.path.expanduser("~/casmi-2026/work/results")


def arm(name, prefix, st, log):
    ev = build(EVAL, st, log=log, pools_prefix=prefix, generate=False)
    tr = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log, pools_prefix=prefix, generate=False)
    w, n = fit(tr)
    log(f"[{name}] trained on {n:,} pairs")
    per = defaultdict(list)
    for q in ev:
        per[q["cls"]].append(rank_score(q, w, use_gen=False))
    return w, per, ev


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    st = load_structures()
    out, pers = {}, {}
    for name, prefix in (("cosine ground", ""), ("sqrt-p ground", "fr_")):
        w, per, _ = arm(name, prefix, st, log)
        c = [float(np.mean(per[k])) for k in (1, 2, 3)]
        out[name] = dict(C1=c[0], C2=c[1], C3=c[2],
                         weighted=sum(W[k] * c[k - 1] for k in (1, 2, 3)),
                         weights=dict(zip(FEATS2, map(float, w))))
        pers[name] = per
    L = ["E25 — spectral ground: our cosine vs Fisher-Rao on sqrt(p), end to end", "",
         f"{'arm':16s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}"]
    for name in out:
        o = out[name]
        L.append(f"{name:16s} {o['C1']:>8.4f} {o['C2']:>8.4f} {o['C3']:>8.4f} {o['weighted']:>9.4f}")
    rng = np.random.default_rng(0)
    a, b = pers["cosine ground"], pers["sqrt-p ground"]
    d = out["sqrt-p ground"]["weighted"] - out["cosine ground"]["weighted"]
    boots = [sum(W[c] * np.mean([b[c][i] - a[c][i] for i in rng.integers(0, len(a[c]), len(a[c]))])
                 for c in (1, 2, 3)) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    L += ["", f"sqrt-p vs cosine: {d:+.4f} weighted  95% CI [{lo:+.4f}, {hi:+.4f}]"
              + ("   <- clears zero" if lo > 0 else ""),
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E25_ground_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump({"arms": out, "delta": [d, lo, hi]}, open(f"{RESULTS}/E25_ground.json", "w"), indent=2)


if __name__ == "__main__":
    main()
