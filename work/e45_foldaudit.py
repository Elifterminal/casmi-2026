#!/usr/bin/env python3
"""E45 — how much of the E44 overlap is actually in the weights? And is a clean subset left?

E44 found 68.3% of our evaluation queries sitting in MassSpecGym, 67.9% with the same adduct.
That is the ceiling on contamination, not the contamination: MassSpecGym ships train/val/test
folds and the open ICEBERG weights are fitted on the TRAIN fold. A query whose structure sits in
MassSpecGym's val or test fold was never trained on and is therefore still usable.

So this asks the two questions that decide whether ChatGPT's 48-hour experiment can be evaluated
honestly at all:

  1. Of our queries, how many are in MassSpecGym's TRAIN fold -- the ones whose spectra are
     genuinely inside the weights?
  2. What is left after removing them, per class? A clean subset is only useful if it is large
     enough to measure a +0.020 effect with an interval that clears zero.

AND THE REASON A CLEAN SUBSET IS THE RIGHT INSTRUMENT RATHER THAN A PATCH. In the real
competition a Class 2 molecule has no public spectrum, so it cannot be in MassSpecGym -- but its
COCONUT decoys are well-known compounds that often can be. Restricting our evaluation to queries
whose TRUTH is outside the predictor's training data, while leaving the decoys alone, reproduces
exactly that asymmetry. It is not a convenience; it is the only configuration that matches the
board.

PREDICTIONS (before the run, 2026-09-23):
  P1  About 84% of the overlap is train-fold, since train is 84% of MassSpecGym's rows, leaving
      roughly 57% of our queries genuinely contaminated.
  P2  The clean subset is 350-450 queries, with over 100 in each of Class 2 and Class 3 -- enough
      to run the experiment, at roughly 1.5x the interval width of the full set.
  P3  Class balance in the clean subset stays close to the full set, because contamination is a
      property of the molecule's fame rather than of our class labels.
"""
import csv, json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
EVAL_SPLITS = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
MSG = ("/tmp/claude-1000/-home-lee/c046c9a5-0d6b-4917-ae00-430a3bba9009/scratchpad/msg15.tsv")
FCACHE = os.path.join(SPLITS, "massspecgym_folds.pkl")
csv.field_size_limit(10 ** 9)


def msg_folds(log):
    """scorer key -> set of MassSpecGym folds it appears in."""
    if os.path.exists(FCACHE):
        d = pickle.load(open(FCACHE, "rb"))
        log(f"folds from cache: {len(d):,} structures")
        return d
    folds = defaultdict(set)
    n = 0
    with open(MSG, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            n += 1
            k = scoring.key14(row.get("smiles", ""))
            if k: folds[k].add((row.get("fold") or "?").strip())
            if n % 100000 == 0: log(f"  rows {n:,}, {len(folds):,} structures")
    folds = dict(folds)
    pickle.dump(folds, open(FCACHE, "wb"))
    log(f"parsed {n:,} rows -> {len(folds):,} structures")
    return folds


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    folds = msg_folds(log)
    in_train = {k for k, f in folds.items() if "train" in f}
    log(f"{len(in_train):,} structures in MassSpecGym's TRAIN fold "
        f"({len(in_train)/max(1,len(folds)):.1%} of its structures)")

    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["inchikey14", "normalized_smiles"])
    smi_of_key = {}
    for k, s in zip(np.asarray(t.column("inchikey14")), np.asarray(t.column("normalized_smiles"))):
        smi_of_key.setdefault(k, s)
    del t
    log("train.parquet indexed")

    per = defaultdict(lambda: defaultdict(int))
    clean = defaultdict(list)
    for sp_name in EVAL_SPLITS:
        sp = json.load(open(f"{SPLITS}/split_{sp_name}.json"))
        for k in sp["query_rows"]:
            c = sp["assign"][k]
            sk = scoring.key14(smi_of_key.get(k, ""))
            if sk is None: continue
            f = folds.get(sk, set())
            per[c]["n"] += 1
            per[c]["any"] += bool(f)
            per[c]["train"] += ("train" in f)
            per[c]["valtest_only"] += (bool(f) and "train" not in f)
            if "train" not in f:
                per[c]["clean"] += 1
                clean[c].append(dict(split=sp_name, query=k))

    L = ["E45 — how much of the MassSpecGym overlap is inside the ICEBERG weights?", "",
         f"MassSpecGym structures: {len(folds):,}; in the TRAIN fold: {len(in_train):,}", "",
         f"{'class':6s} {'queries':>8} {'in MSG':>8} {'TRAIN fold':>11} {'val/test only':>14} "
         f"{'CLEAN (usable)':>15}"]
    for c in (1, 2, 3):
        v = per[c]; n = max(1, v["n"])
        L.append(f"C{c:<5d} {v['n']:>8} {v['any']/n:>7.1%} {v['train']/n:>10.1%} "
                 f"{v['valtest_only']/n:>13.1%} {v['clean']:>8} ({v['clean']/n:>4.1%})")
    tot = {k: sum(per[c][k] for c in (1, 2, 3))
           for k in ("n", "any", "train", "valtest_only", "clean")}
    n = max(1, tot["n"])
    L.append(f"{'all':6s} {tot['n']:>8} {tot['any']/n:>7.1%} {tot['train']/n:>10.1%} "
             f"{tot['valtest_only']/n:>13.1%} {tot['clean']:>8} ({tot['clean']/n:>4.1%})")

    # what a clean-subset evaluation costs in precision
    import math
    L += ["", "what the clean subset costs in resolution:",
          f"  full set     {tot['n']:>4} queries",
          f"  clean subset {tot['clean']:>4} queries  -> intervals about "
          f"{math.sqrt(tot['n']/max(1,tot['clean'])):.2f}x wider"]
    L += ["", "reading: 'TRAIN fold' is the real contamination -- those spectra are inside the open",
          "ICEBERG weights. 'val/test only' molecules are in MassSpecGym but were not trained on, so",
          "they stay usable. The CLEAN column is the evaluation set for the intensity experiment:",
          "queries whose true structure the predictor has never been fitted on. Decoys are left",
          "alone deliberately -- in the real competition a Class 2 truth has no public spectrum while",
          "its COCONUT decoys often do, and this reproduces that asymmetry rather than erasing it.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E45_foldaudit_2026-09-23.txt", "w").write("\n".join(L) + "\n")
    json.dump({f"C{c}": dict(per[c]) for c in (1, 2, 3)},
              open(f"{RESULTS}/E45_foldaudit.json", "w"), indent=2, default=int)
    json.dump({str(c): v for c, v in clean.items()},
              open(f"{SPLITS}/clean_eval_queries.json", "w"), indent=1)
    log(f"clean evaluation set written -> {SPLITS}/clean_eval_queries.json")


if __name__ == "__main__":
    main()
