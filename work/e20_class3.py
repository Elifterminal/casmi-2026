#!/usr/bin/env python3
"""E20 part 2 — Class 3: generate candidates by editing a known relative, and see what it costs.

Class 3 is 39% of the score and sits at exactly 0.000 (L-024): the answer is in no database, so
retrieval can never return it. This builds the missing channel. For each query:

  relatives   the top spectral hits (cosine), capped at MAX_REL
  gap         neutral mass of the query minus the relative's exact mass
  products    every product of every edit whose mass change matches that gap (e20_edits),
              one per site -- the site is the guess
  merge       generated candidates are scored with the SAME learned ranker as retrieved ones and
              compete for the same 25 slots

That last line is the whole risk, and it is the thing nobody in the forum has measured: a
generated candidate that is wrong does not just fail, it displaces a retrieved candidate that
might have been right. So the measurement is not "does Class 3 go up" but "does the weighted
total go up".

Reported separately from the score: GENERATION RECALL -- how often the true Class 3 structure
was built at all. If recall is 0 the ranking question is moot; if recall is high and the score
is not, the problem is ranking, not chemistry.

PREDICTIONS (before the run, 2026-09-20):
  P1  Generation recall on Class 3: 10-25%. E13 put a good relative in the top 5 for 24% of
      queries (46% with better search); placement then divides that.
  P2  Class 3 MRR lands above 0 but at most 0.08.
  P3  Class 1 and Class 2 each lose <= 0.01 to slot competition.
  P4  Net weighted change vs E18's 0.351: between +0.005 and +0.03.
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from twins import db_exclusions
from e20_edits import candidates_for
import casmi_pipeline as cp
from casmi_pipeline import FEATS, NB_TOP, PPM, TOPN, MassIndex, neutral_mass, explain, chance, characterise, key14
import scoring

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
W = {1: 0.16, 2: 0.45, 3: 0.39}
MAX_REL = 10          # relatives per query (top spectral hits)
MAX_SITES = 20        # products per edit per relative
GAP_TOL = 0.003       # Da, matching an observed gap to an edit's mass


def _gen(args):
    """All products of editing one relative to close one mass gap."""
    smi, gap = args
    return [s for s, _ in candidates_for(smi, gap, tol=GAP_TOL, max_sites=MAX_SITES)]


def main():
    t0 = time.time()
    weights = json.load(open(f"{RESULTS}/E18_ranker_coconut.json"))["weights"]
    w = np.array([weights[f] for f in FEATS])
    keys, pmz, add, tr, co, _ = load_structures()
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_db = lambda k: trs.get(k) or cos_.get(k, "")
    exact = {}
    def emass(smi):
        if smi not in exact:
            m = Chem.MolFromSmiles(smi or "")
            exact[smi] = Descriptors.ExactMolWt(m) if m is not None else np.nan
        return exact[smi]

    per = {"retrieval": defaultdict(list), "retrieval+generated": defaultdict(list)}
    gen_stats = defaultdict(int)
    for split in EVAL:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))}
        excl = db_exclusions(sp, tr, co)
        keep = ~co.k.isin(excl)
        midx = MassIndex(co.k[keep].to_numpy(), co.m[keep].to_numpy())

        qs, jobs = [], []
        for k, rows in qrows.items():
            p = pools[k]; cands, Ms = set(), []
            for r in rows:
                M = neutral_mass(pmz[r], add[r])
                if np.isfinite(M) and M > 0: cands.update(midx.window(M, PPM).tolist()); Ms.append(M)
            M = float(np.median(Ms)) if Ms else np.nan
            rel = [h for h, _ in sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_REL]]
            pending = []
            if np.isfinite(M):
                for h in rel:
                    s = smi_db(h); rm = emass(s)
                    if not np.isfinite(rm): continue
                    gap = M - rm
                    if abs(gap) < 1e-6: continue          # same mass: not an edit, skip
                    pending.append((s, gap))
            jobs.append(pending)
            qs.append((k, assign[k], p, cands, M))
        with Pool(7) as pool:
            flat = pool.map(_gen, [j for pend in jobs for j in pend], chunksize=4)
        # regroup products back onto their queries
        gen_per_q, i = [], 0
        for pend in jobs:
            got = set()
            for _ in pend:
                got.update(flat[i]); i += 1
            gen_per_q.append(sorted(got))
        print(f"[{split}] generated {sum(len(g) for g in gen_per_q):,} candidates "
              f"({np.median([len(g) for g in gen_per_q]):.0f} median per query) {time.time()-t0:.0f}s", flush=True)

        need = {smi_db(c) for _, _, _, cands, _ in qs for c in cands}
        need |= {smi_db(h) for _, _, p, _, _ in qs
                 for h, _ in sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]}
        need |= {s for g in gen_per_q for s in g}
        info = characterise(need, procs=7)
        print(f"[{split}] characterised {len(info):,} structures {time.time()-t0:.0f}s", flush=True)

        for (k, cls, p, cands, M), gen in zip(qs, gen_per_q):
            hits = sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
            good = [(h, c) for h, c in hits if info.get(smi_db(h)) is not None]
            hfp = [info[smi_db(h)]["fp"] for h, _ in good]
            hcos = np.array([c for _, c in good])
            truth = p["truth"]
            if cls == 3 and gen:
                gen_stats["c3_queries"] += 1
                if any(key14(s) == truth for s in gen): gen_stats["c3_recall"] += 1
            elif cls == 3:
                gen_stats["c3_queries"] += 1

            def feats(smi, spec_self):
                inf = info.get(smi)
                if inf is None: return None
                e = explain(p["allmz"], p["allit"], inf["full"], p["positive"])
                if hfp:
                    v = hcos * np.array(DataStructs.BulkTanimotoSimilarity(inf["fp"], hfp))
                    nbm, nb10 = float(v.max()), float(v[:10].mean())
                else:
                    nbm = nb10 = 0.0
                f = dict(explain=e, chance_corr=e - chance(inf["full"], M),
                         single=explain(p["allmz"], p["allit"], inf["single"], p["positive"]),
                         nb_max=nbm, nb_mean10=nb10, spec_self=spec_self, heavy=inf["heavy"])
                return float(np.dot(w, [f[x] for x in FEATS]))

            spec = dict(hits)
            retr = {}
            for c in cands:
                s = feats(smi_db(c), p["spec"].get(c, 0.0))
                if s is not None: retr[c] = (s, smi_db(c))
            gensc = {}
            for s in gen:
                sc = feats(s, 0.0)
                if sc is not None: gensc[f"gen:{s}"] = (sc, s)

            def score(pool_):
                ranked = sorted(pool_, key=lambda c: (-pool_[c][0], c))
                tail = sorted(set(spec) - cands)[:TOPN]
                order = [pool_[c][1] for c in ranked] + [smi_db(c) for c in tail]
                seen, out = set(), []
                for smi in order:                      # dedupe by the scorer's identity
                    kk = key14(smi)
                    if kk is not None and kk in seen: continue
                    if kk is not None: seen.add(kk)
                    out.append(kk)
                    if len(out) == TOPN: break
                for i, kk in enumerate(out, 1):
                    if truth is not None and kk == truth: return 1.0 / i
                return 0.0

            per["retrieval"][cls].append(score(retr))
            per["retrieval+generated"][cls].append(score({**retr, **gensc}))
        print(f"[{split}] scored {time.time()-t0:.0f}s", flush=True)

    L = ["E20 — Class 3 by analog editing, merged into the same 25 slots", "",
         f"generation recall on Class 3: {gen_stats['c3_recall']}/{gen_stats['c3_queries']} "
         f"= {gen_stats['c3_recall']/max(1,gen_stats['c3_queries']):.1%} (the true structure was built at all)", "",
         f"{'candidates':22s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}"]
    out = {}
    for name in ("retrieval", "retrieval+generated"):
        c = [float(np.mean(per[name][k])) for k in (1, 2, 3)]
        wt = sum(W[k] * c[k - 1] for k in (1, 2, 3))
        out[name] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=wt)
        L.append(f"{name:22s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt:>9.4f}")
    d = out["retrieval+generated"]["weighted"] - out["retrieval"]["weighted"]
    L.append(f"\nnet change from generating: {d:+.4f} weighted")
    rng = np.random.default_rng(0)
    by = {c: list(zip(per["retrieval"][c], per["retrieval+generated"][c])) for c in (1, 2, 3)}
    boots = [sum(W[c] * np.mean([b - a for a, b in [by[c][i] for i in rng.integers(0, len(by[c]), len(by[c]))]])
                 for c in by) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    L.append(f"paired bootstrap 95% CI: [{lo:+.4f}, {hi:+.4f}]")
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E20_class3_2026-09-20.txt", "w").write("\n".join(L) + "\n")
    json.dump({"results": out, "generation_recall_c3": gen_stats["c3_recall"] / max(1, gen_stats["c3_queries"]),
               "net": d, "ci": [lo, hi]}, open(f"{RESULTS}/E20_class3.json", "w"), indent=2)


if __name__ == "__main__":
    main()
