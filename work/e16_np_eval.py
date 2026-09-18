#!/usr/bin/env python3
"""E16 — the honest number: NP-only queries, real databases.

E15 showed our manufactured queries flatter us: Class 2 truths that are natural products
(natively in COCONUT) scored 0.389 on the shipping database, the rest 0.565. The real test is
natural products. So E16 rebuilds the splits from COCONUT-native structures only
(harness.py --np-coconut, seeds 10-12) and re-runs the E10 pipeline on three databases:

  coconut   COCONUT-2026-09 minus Class 3 -- no injection needed, NP truths are native
  union     COCONUT + training structures (E15's shipping DB)
  train     training structures minus Class 3 (the old, easy pools -- for reference)

Paired differences are reported against 'coconut' (the candidate shipping DB).

PREDICTIONS (before the run, 2026-09-18):
  P1  On NP-only queries, 'union' weighted is within 0.03 of E15's C2-native-driven estimate,
      i.e. lands 0.24-0.30 -- below the gate.
  P2  'coconut' beats 'union' (the training structures mostly add decoys; E15: 0.342 vs 0.302).
  P3  Class 1 does NOT collapse on 'coconut' even though training structures are dropped:
      NP-only C1 truths are in COCONUT natively.
  RESULT (2026-09-18): coconut 0.2305, union 0.2086, train 0.3069 (3-split means).
      P1 FAIL -- worse than predicted: union 0.209, not 0.24-0.30. P2 PASS: coconut beats union
      by +0.022 [+0.014,+0.030]. P3 PASS (partial): C1 on coconut 0.392 -- down from 0.512 on
      the small training pools, but above union's 0.348; the drop is pool size, not missing
      answers. Pools on NP-only queries: C2 median 36 (coconut) / 51 (union) / 10 (train).
      Note: 4 C2 truths count as non-native because COCONUT's formula for them doesn't parse,
      so they're missing from the coconut DB -- negligible (4/405).
  ** Honest realistic score: ~0.23 weighted. Leader 0.353, gate 0.339. **
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e15_coconut_pools import main

if __name__ == "__main__":
    main(("npseed10_n300", "npseed11_n300", "npseed12_n300"), ("coconut", "union", "train"), "E16")
