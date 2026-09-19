#!/usr/bin/env python3
"""E18 — E16 + E17 re-run on the twin-safe harness.

E17 found the harness holding out by InChIKey14 while the scorer tautomer-canonicalises, so a
Class 2 answer's twin could keep its spectra in the reference index (19/405 NP C2 queries), and
a Class 3 answer's twin could stay in the candidate database (6-14 of 117 per split). The
harness now holds out whole twin groups, and retrieval drops Class 3 twins in both training
and COCONUT (twins.db_exclusions). Positive controls: re-inserting a twin makes verify() fail;
the legacy verify() stays silent on the same input.

Same code as E16/E17, new splits (nptsseed10-12 eval, 20-22 ranker training):
  1. baseline: pure fragment, COCONUT and union      (E16's measurement)
  2. rankers:  every E17 feature + learned, COCONUT  (E17's measurement), clean=False -- the
               harness fix alone must remove the leak, no post-hoc cleaning.

PREDICTIONS (before the run, 2026-09-18):
  P1  Baseline on COCONUT within +-0.02 of E16's 0.2305 (the leak mostly fed the spectral side).
  P2  Learned ranker drops from E17's 0.3647 but by <= 0.02 (the leak's ceiling was ~0.013),
      and still clears 0.339 on the 3-split mean.
  P3  nb_max's Class 2 falls more than explain's (it's the feature that read the twins).
  RESULT (2026-09-18):
      baseline COCONUT 0.2272 (E16 0.2305), union 0.2012; Class 3 now EXACTLY 0.0000 on every
      split -- the small non-zero C3 scores in every earlier experiment were the twin leak.
      learned 0.3511 (0.367 / 0.340 / 0.346, all three >= 0.339), +0.1105 [+0.092,+0.129] over
      explain; nb_max 0.3271; single +0.012 (clears zero); chance_corr n.s.; spec_self hurts.
      P1 PASS (-0.003). P2 PASS (learned -0.0136 vs E17, clears the gate on the mean -- but
      split 11 only by 0.001). P3 PASS (nb_max C2 0.437 -> 0.423; explain C2 unchanged 0.388).
      Cross-check: E17c (old splits, twins removed post hoc) 0.3566 -- agrees within the C3 leak.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e15_coconut_pools, e17_ranker

EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
TRAIN = ("nptsseed20_n300", "nptsseed21_n300", "nptsseed22_n300")

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("all", "baseline"):
        e15_coconut_pools.main(EVAL, ("coconut", "union"), "E18")
    if step in ("all", "ranker"):
        e17_ranker.main("coconut", False, TRAIN, EVAL, "E18")
