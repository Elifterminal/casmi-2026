#!/usr/bin/env python3
"""E35 — the retrieval ceiling: is ranking even the binding constraint?

Twenty experiments have gone into ranking and almost all of them measured as noise. Seda asked
the question none of them answered: does the candidate list even CONTAIN the true structure?
ChatGPT sharpened it into a cascade, separating four different failures that the weighted score
cannot tell apart:

  1. database absence   the answer is not in COCONUT at all -> unreachable, forever
  2. mass window        it is in COCONUT but falls outside 10 ppm of the neutral mass we compute
  3. pruning            it survives retrieval but we drop it (unparseable, or no fragment table)
  4. ranking            it is in the scored pool and we fail to put it in the top 25

Only stage 4 is a ranking problem. If stages 1-3 leak badly, every ranker experiment we have run
was chasing a ceiling we had never measured.

TWO QUERY SETS, and the difference between them is the point:

  nptsseed10-12   our production yardstick. Query structures were REQUIRED to be in COCONUT.
  tsseed40-42     the same construction WITHOUT that requirement -- natural-product library
                  structures, twin-safe, whatever database they happen to be in.

ChatGPT's circularity point is that the first set cannot measure stage 1 by construction: we
choose queries that are in COCONUT and then congratulate ourselves that COCONUT contains them.
The official competition definition is Class 2 = in PubChem OR COCONUT, and Class 3 = absent
from PubChem -- so a molecule can be legitimately Class 2 and permanently unreachable for us.

PREDICTIONS (before the run, 2026-09-21):
  P1  On the production set, stage 1 is ~100% by construction and stage 2 is the largest leak.
  P2  On the unrestricted set, stage 1 loses 10-30% of Class 2 -- real, and invisible until now.
  P3  Stage 3 (pruning) is small, under 5%.
  P4  Stage 4 is the largest single loss on both sets: ranking IS the binding constraint, and
      the last twenty experiments were not chasing a phantom.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from twins import db_exclusions
from casmi_pipeline import MassIndex, neutral_mass, PPM, characterise, key14
import scoring

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
SETS = {"production (COCONUT-filtered)": ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300"),
        "unrestricted (any database)": ("tsseed40_n300", "tsseed41_n300", "tsseed42_n300")}


def main():
    t0 = time.time()
    keys, pmz, add, tr, co, _ = load_structures()
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_of = lambda k: trs.get(k) or cos_.get(k, "")
    # COCONUT membership judged at the SCORER's identity, not the raw key: a tautomer of the
    # answer in COCONUT counts as reachable, because the scorer would accept it.
    print(f"loaded {time.time()-t0:.0f}s", flush=True)
    # formula -> COCONUT keys, so stage 1 only canonicalises the group that could match
    co_by_f = defaultdict(list)
    for k, f in zip(co.k, co.f): co_by_f[f].append(k)
    trf = dict(zip(tr.k, tr.f))

    def in_coconut(qkey, truth):
        """Is any COCONUT structure the same molecule as the answer, at the scorer's identity?"""
        f = trf.get(qkey)
        if f is None: return False
        for c in co_by_f.get(f, ()):
            if scoring.key14(cos_[c]) == truth: return True
        return False

    L = ["E35 — where the true structure is lost, before ranking is even asked", ""]
    out = {}
    for label, splits in SETS.items():
        stage = defaultdict(lambda: defaultdict(int))     # cls -> stage -> count
        for split in splits:
            sp = json.load(open(f"{SPLITS}/split_{split}.json"))
            assign, qrows = sp["assign"], sp["query_rows"]
            excl = db_exclusions(sp, tr, co)
            keep = ~co.k.isin(excl)
            midx = MassIndex(co.k[keep].to_numpy(), co.m[keep].to_numpy())
            cands = {}
            for k, rows in qrows.items():
                s = set(); Ms = []
                for r in rows:
                    M = neutral_mass(pmz[r], add[r])
                    if np.isfinite(M) and M > 0: s.update(midx.window(M, PPM).tolist()); Ms.append(M)
                cands[k] = s
            need = {smi_of(c) for s in cands.values() for c in s} | {smi_of(k) for k in qrows}
            info = characterise(sorted(need), procs=7)
            for k, s in cands.items():
                cls = assign[k]
                if cls == 3: continue                    # Class 3 is unreachable by design
                stage[cls]["queries"] += 1
                truth = scoring.key14(smi_of(k))
                if truth is None: continue
                # 1. is the answer in COCONUT under the scorer's identity?
                if in_coconut(k, truth): stage[cls]["in_db"] += 1
                # 2. did retrieval return it?
                got = [c for c in s if scoring.key14(smi_of(c)) == truth]
                if got: stage[cls]["retrieved"] += 1
                # 3. does at least one copy survive pruning?
                if got and any(info.get(smi_of(c)) is not None and info[smi_of(c)]["full"] for c in got):
                    stage[cls]["scorable"] += 1
            print(f"[{split}] done {time.time()-t0:.0f}s", flush=True)
        out[label] = {str(c): dict(v) for c, v in stage.items()}
        L.append(f"### {label}")
        L.append(f"{'class':6s} {'queries':>8} {'in COCONUT':>12} {'retrieved':>11} {'scorable':>10}")
        for c in (1, 2):
            v = stage[c]; n = max(1, v["queries"])
            L.append(f"C{c:<5d} {v['queries']:>8} {v['in_db']/n:>11.1%} {v['retrieved']/n:>10.1%} "
                     f"{v['scorable']/n:>9.1%}")
        L.append("")
    L += ["reading: 'in COCONUT' is stage 1 (database absence), 'retrieved' folds in stage 2",
          "(the 10 ppm mass window), 'scorable' folds in stage 3 (we can parse and fragment it).",
          "Whatever remains between 'scorable' and the measured Class MRR is stage 4: ranking.",
          f"\nfor reference, the production set's measured MRR: C1 0.8245, C2 0.5250",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E35_ceiling_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E35_ceiling.json", "w"), indent=2)


if __name__ == "__main__":
    main()
