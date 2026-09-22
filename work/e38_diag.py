#!/usr/bin/env python3
"""E38 — the per-query diagnostic ChatGPT asked for: where does each answer actually die?

Every number on the living page is an average of reciprocal ranks, and an average cannot tell
four different failures apart. ChatGPT's request, in his words, was per-query records of whether
the truth was generated, whether it survived into the 25, its final rank, and the reciprocal-rank
change against retrieval alone -- plus an oracle-gated upper bound. That last one is the whole
point: if a PERFECT ranker over the pool we actually build scores 0.42, then every remaining
ranker experiment is competing for the space between 0.37 and 0.42, and the way to a better score
is a better pool, not a better ranker.

This is also the experiment that decides what the pre-gate submission should test. Two candidate
changes are on the table and we can afford to test one before 11 October:

  broader retrieval     widen the mass window, or add databases, so more answers enter the pool
  selective generation  keep manufacturing Class 3 candidates but only where it does not cost
                        Class 2 -- E22 got the net to clear zero, barely, by discounting them

The oracle bound settles it. If generation already puts the Class 3 answer in the pool far more
often than our 0.036 suggests, the loss is ranking and selectivity is the lever. If generation
almost never reaches the answer, then no amount of ranking rescues it and retrieval is the lever.

RECORDED PER QUERY (results/E38_per_query.csv):
  cls, truth_retrieved, truth_generated, truth_in_tail, rank_ret, rank_gen, rr_ret, rr_gen,
  n_mass, n_gen, pool_rank_of_truth (where a perfect ranker would have to place it to score)

PREDICTIONS (before the run, 2026-09-22):
  P1  The oracle bound on the production set is 0.45-0.55 weighted. Our 0.3718 is therefore
      leaving real points on the table, and ranking is still worth working on -- but the ceiling
      is close enough that it, not our ideas, is what limits us.
  P2  Generation reaches the Class 3 answer in under 5% of queries. Class 3's 0.036 is then near
      its own ceiling and selective generation is NOT the lever; retrieval is.
  P3  Adding generation moves a minority of Class 2 queries, and it hurts more of them than it
      helps -- the net is negative and small, which is what E22's discount was papering over.
  P4  Class 1 truths are in the pool over 95% of the time, so Class 1's 0.8475 is a ranking
      result rather than a coverage result.
"""
import csv, json, os, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import build, fit, rank_score, FEATS2, EVAL, TRAIN, W
from casmi_pipeline import TOPN, key14

RESULTS = os.path.expanduser("~/casmi-2026/work/results")


def truth_where(q):
    """Which channel, if any, carries the true structure into this query's candidate pool."""
    ret = gen = False
    for c, (_, s) in q["cand"].items():
        if key14(s) != q["truth"]: continue
        if str(c).startswith("gen:"): gen = True
        else: ret = True
    tail = any(key14(s) == q["truth"] for s in q["tail"])
    return ret, gen, tail


def rank_of(q, w, use_gen):
    """1-based rank of the truth in the full ordered list, or 0 if absent. No top-25 cut."""
    pool = {c: (float(np.dot(w, v)), s) for c, (v, s) in q["cand"].items()
            if use_gen or not str(c).startswith("gen:")}
    order = [pool[c][1] for c in sorted(pool, key=lambda c: (-pool[c][0], c))] + q["tail"]
    seen, i = set(), 0
    for smi in order:
        kk = key14(smi)
        if kk is not None and kk in seen: continue
        if kk is not None: seen.add(kk)
        i += 1
        if kk is not None and kk == q["truth"]: return i
    return 0


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    st = load_structures()
    # pools_fr_ is the sqrt-p ground we actually ship. E22's default prefix is the older
    # cosine-on-intensities pool, and passing it here would diagnose a pipeline we no longer run.
    ev = build(EVAL, st, log=log, pools_prefix="fr_")
    tr_q = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log, pools_prefix="fr_")
    w, npairs = fit(tr_q)
    log(f"trained on {npairs:,} truth/decoy pairs")

    rows, per = [], defaultdict(lambda: defaultdict(list))
    for q in ev:
        cls = q["cls"]
        ret, gen, tail = truth_where(q)
        r_ret, r_gen = rank_of(q, w, False), rank_of(q, w, True)
        rr_ret = rank_score(q, w, use_gen=False)
        rr_gen = rank_score(q, w, use_gen=True)
        # oracle: a perfect ranker can only score what the pool contains, and only within 25 slots
        reachable = ret or gen or tail
        per[cls]["rr_ret"].append(rr_ret); per[cls]["rr_gen"].append(rr_gen)
        per[cls]["oracle"].append(1.0 if reachable else 0.0)
        per[cls]["oracle_ret"].append(1.0 if (ret or tail) else 0.0)
        per[cls]["in_ret"].append(float(ret)); per[cls]["in_gen"].append(float(gen))
        per[cls]["in_tail"].append(float(tail))
        per[cls]["moved"].append(rr_gen - rr_ret)
        rows.append(dict(key=q["k"], cls=cls, truth_retrieved=int(ret), truth_generated=int(gen),
                         truth_in_tail=int(tail), rank_ret=r_ret, rank_gen=r_gen,
                         rr_ret=round(rr_ret, 6), rr_gen=round(rr_gen, 6),
                         n_mass=sum(1 for c in q["cand"] if not str(c).startswith("gen:")),
                         n_gen=sum(1 for c in q["cand"] if str(c).startswith("gen:"))))

    with open(f"{RESULTS}/E38_per_query.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

    def wsum(key): return sum(W[c] * float(np.mean(per[c][key])) for c in (1, 2, 3))

    L = ["E38 — per-query diagnostic: coverage, ranking, and the ceiling above both", "",
         f"{'class':6s} {'n':>5} {'in retrieval':>13} {'generated':>10} {'in tail':>8} "
         f"{'reachable':>10} {'measured':>9} {'oracle':>8} {'headroom':>9}"]
    for c in (1, 2, 3):
        p = per[c]; n = len(p["rr_gen"])
        L.append(f"C{c:<5d} {n:>5} {np.mean(p['in_ret']):>12.1%} {np.mean(p['in_gen']):>9.1%} "
                 f"{np.mean(p['in_tail']):>7.1%} {np.mean(p['oracle']):>9.1%} "
                 f"{np.mean(p['rr_gen']):>9.4f} {np.mean(p['oracle']):>8.4f} "
                 f"{np.mean(p['oracle'])-np.mean(p['rr_gen']):>+9.4f}")
    L += ["", f"weighted measured (+generated)   {wsum('rr_gen'):.4f}",
          f"weighted measured (retrieval only) {wsum('rr_ret'):.4f}",
          f"weighted ORACLE over the pool we build   {wsum('oracle'):.4f}",
          f"weighted oracle, retrieval channel only  {wsum('oracle_ret'):.4f}", ""]

    L.append("what generation does, query by query:")
    L.append(f"{'class':6s} {'helped':>8} {'hurt':>8} {'unchanged':>10} {'net RR':>9}")
    for c in (1, 2, 3):
        m = np.array(per[c]["moved"])
        L.append(f"C{c:<5d} {int((m > 0).sum()):>8} {int((m < 0).sum()):>8} "
                 f"{int((m == 0).sum()):>10} {m.mean():>+9.4f}")
    L += ["", "reading: 'reachable' is the share of queries whose true structure is anywhere in the",
          "pool we hand the ranker -- mass retrieval, generation, or the spectral tail. 'oracle' is",
          "what a perfect ranker would score given that pool, so 'headroom' is everything ranking",
          "could still win. Where headroom is small, better features cannot help and only a better",
          "pool can.", f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E38_diag_2026-09-22.txt", "w").write("\n".join(L) + "\n")
    json.dump({f"C{c}": {k: float(np.mean(v)) for k, v in per[c].items()} for c in (1, 2, 3)},
              open(f"{RESULTS}/E38_diag.json", "w"), indent=2)


if __name__ == "__main__":
    main()
