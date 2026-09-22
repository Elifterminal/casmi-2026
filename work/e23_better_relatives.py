#!/usr/bin/env python3
"""E23 — feed the Class 3 generator better relatives.

E22 made generation pay (+0.0078, CI clear of zero) but it is fed by plain-cosine spectral
hits, and E13 measured exactly how weak that input is: a relative that is both close and one
common edit away sits in the top 5 for 24% of Class 3 queries with cosine, and 46% when
modified-cosine and neutral-loss search are added -- they find DIFFERENT relatives, which is
why the union roughly doubles the reach.

So relatives here are the union of the top MAX_EACH from three searches:
  cosine    the binned cosine already in the frozen pools
  modcos    a peak may also match shifted by the precursor gap (built for the analogue case)
  nlcos     cosine over neutral losses (what the molecule sheds, not what it keeps)

Everything downstream is E22 unchanged: the same 21 edits, the same features plus
`is_generated`, retrained on the training splits, evaluated on splits 10-12.

PREDICTIONS (before the run, 2026-09-20):
  P1  Generation recall on Class 3 roughly doubles: 8.8% -> 15-22%.
  P2  Class 3 MRR rises above 0.05 (from 0.0311).
  P3  Weighted beats E22's 0.3591, and its CI against retrieval-only clears zero.
  P4  Class 2 does NOT fall further than E22's 0.4822: the extra candidates are discounted by
      the same learned offset, and more of them are right.
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e13_analog_search import greedy_cos, top_peaks
from e15_coconut_pools import load_structures
from twins import db_exclusions
from e20_edits import candidates_for
from e22_ranker_v2 import fit, rank_score, FEATS2, MAX_SITES, GAP_TOL, W, EVAL, TRAIN, _gen
from casmi_pipeline import (NB_TOP, PPM, TOPN, MassIndex, neutral_mass, explain, chance,
                            characterise, key14, prep, IDX_PEAKS)
import casmi_pipeline as cp
import scoring

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
MAX_EACH = 10          # relatives kept per search method
SLOW_ROWS, SLOW_REFS = 3, 300   # modcos/nlcos budget per query (E13's, which still beat cosine)


def load_spectra():
    t = pq.read_table(DATA, columns=["inchikey14", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz"])
    keys = np.asarray(t.column("inchikey14")); pmz = t.column("precursor_mz").to_numpy()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    return keys, pmz, peaks


def relatives(split, qrows, pools, keys, pmz, peaks, log):
    """{query key: [structure keys]} -- union of cosine / modcos / nlcos top MAX_EACH."""
    ref_rows = np.load(f"{SPLITS}/ref_rows_{split}.npy")
    idx = cp.SpectralIndex(keys, peaks, ref_rows)
    log(f"[{split}] spectral index for relative search")
    out, tpc = {}, {}
    for k, rows in qrows.items():
        cos_hits = [h for h, _ in sorted(pools[k]["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_EACH]]
        mod, nl = defaultdict(float), defaultdict(float)
        for ri, r in enumerate(rows[:SLOW_ROWS]):
            b, w = prep(*peaks(r)); o = np.argsort(b); b, w = b[o], w[o]
            if len(b) == 0: continue
            gather = [idx.inv[int(bb)] for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]) if int(bb) in idx.inv]
            if not gather: continue
            u, ct = np.unique(np.concatenate(gather), return_counts=True)
            qm, qw = top_peaks(*peaks(r)); qnl = top_peaks(pmz[r] - peaks(r)[0], peaks(r)[1])
            for c in u[np.argsort(-ct)[:SLOW_REFS]]:
                if c not in tpc:
                    tpc[c] = (top_peaks(*peaks(c)), top_peaks(pmz[c] - peaks(c)[0], peaks(c)[1]))
                (rm, rw), rnl = tpc[c]
                kc = keys[c]
                mod[kc] = max(mod[kc], greedy_cos(qm, qw, rm, rw, shift=pmz[r] - pmz[c]))
                nl[kc] = max(nl[kc], greedy_cos(*qnl, *rnl))
        pick = list(cos_hits)
        for d in (mod, nl):
            pick += [h for h, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_EACH]]
        seen, rel = set(), []
        for h in pick:
            if h in seen: continue
            seen.add(h); rel.append(h)
        out[k] = rel
    log(f"[{split}] relatives found (median {np.median([len(v) for v in out.values()]):.0f} per query)")
    return out


def build(splits, st, spectra, excl_keys=frozenset(), log=print):
    keys, pmz, peaks = spectra
    _, qpmz, add, tr, co, _ = st
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_db = lambda k: trs.get(k) or cos_.get(k, "")
    exact = {}
    def emass(smi):
        if smi not in exact:
            m = Chem.MolFromSmiles(smi or "")
            exact[smi] = Descriptors.ExactMolWt(m) if m is not None else np.nan
        return exact[smi]
    out = []
    for split in splits:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))}
        rel_of = relatives(split, qrows, pools, keys, pmz, peaks, log)
        keep = ~co.k.isin(db_exclusions(sp, tr, co))
        midx = MassIndex(co.k[keep].to_numpy(), co.m[keep].to_numpy())
        qs, jobs = [], []
        for k, rows in qrows.items():
            if k in excl_keys: continue
            p = pools[k]; cands, Ms = set(), []
            for r in rows:
                M = neutral_mass(qpmz[r], add[r])
                if np.isfinite(M) and M > 0: cands.update(midx.window(M, PPM).tolist()); Ms.append(M)
            M = float(np.median(Ms)) if Ms else np.nan
            pend = []
            if np.isfinite(M):
                for h in rel_of[k]:
                    s = smi_db(h); rm = emass(s)
                    if np.isfinite(rm) and abs(M - rm) > 1e-6: pend.append((s, M - rm))
            jobs.append(pend); qs.append((k, assign[k], p, cands, M))
        with Pool(7) as pool:
            flat = pool.map(_gen, [j for pend in jobs for j in pend], chunksize=4)
        gen_per_q, i = [], 0
        for pend in jobs:
            got = set()
            for _ in pend: got.update(flat[i]); i += 1
            gen_per_q.append(sorted(got))
        need = {smi_db(c) for _, _, _, cands, _ in qs for c in cands}
        need |= {smi_db(h) for _, _, p, _, _ in qs
                 for h, _ in sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]}
        need |= {s for g in gen_per_q for s in g}
        info = characterise(need, procs=7)
        log(f"[{split}] {len(qs)} queries, {sum(len(g) for g in gen_per_q):,} generated, "
            f"{len(info):,} structures characterised")
        for (k, cls, p, cands, M), gen in zip(qs, gen_per_q):
            hits = sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
            good = [(h, c) for h, c in hits if info.get(smi_db(h)) is not None]
            hfp = [info[smi_db(h)]["fp"] for h, _ in good]
            hcos = np.array([c for _, c in good])
            def vec(smi, spec_self, is_gen):
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
                         nb_max=nbm, nb_mean10=nb10, spec_self=spec_self, heavy=inf["heavy"],
                         is_generated=float(is_gen))
                return np.array([f[x] for x in FEATS2])
            cand = {}
            for c in cands:
                v = vec(smi_db(c), p["spec"].get(c, 0.0), 0)
                if v is not None: cand[c] = (v, smi_db(c))
            for s in gen:
                v = vec(s, 0.0, 1)
                if v is not None: cand[f"gen:{s}"] = (v, s)
            out.append(dict(k=k, cls=cls, truth=p["truth"], cand=cand, gen=gen,
                            tail=[smi_db(c) for c in sorted(set(dict(hits)) - cands)[:TOPN]]))
    return out


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    st = load_structures()
    spectra = load_spectra()
    log("data loaded")
    ev = build(EVAL, st, spectra, log=log)
    tr_q = build(TRAIN, st, spectra, frozenset(q["k"] for q in ev), log=log)
    w, npairs = fit(tr_q)
    log(f"trained on {npairs:,} truth/decoy pairs")
    recall = [q for q in ev if q["cls"] == 3]
    built = sum(1 for q in recall if any(key14(s) == q["truth"] for s in q["gen"]))
    per = defaultdict(lambda: defaultdict(list))
    for q in ev:
        per["retrieval only"][q["cls"]].append(rank_score(q, w, use_gen=False))
        per["retrieval + generated"][q["cls"]].append(rank_score(q, w, use_gen=True))
    L = ["E23 — generation fed by cosine + modified-cosine + neutral-loss relatives", "",
         f"generation recall on Class 3: {built}/{len(recall)} = {built/max(1,len(recall)):.1%}"
         f"   (E22, cosine only: 8.8%)", "",
         "weights: " + ", ".join(f"{f} {v:+.3g}" for f, v in zip(FEATS2, w)), "",
         f"{'variant':24s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}"]
    out = {}
    for name in ("retrieval only", "retrieval + generated"):
        c = [float(np.mean(per[name][k])) for k in (1, 2, 3)]
        wt = sum(W[k] * c[k - 1] for k in (1, 2, 3))
        out[name] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=wt)
        L.append(f"{name:24s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt:>9.4f}")
    rng = np.random.default_rng(0)
    by = {c: list(zip(per["retrieval only"][c], per["retrieval + generated"][c])) for c in (1, 2, 3)}
    d = out["retrieval + generated"]["weighted"] - out["retrieval only"]["weighted"]
    boots = [sum(W[c] * np.mean([b - a for a, b in [by[c][i] for i in rng.integers(0, len(by[c]), len(by[c]))]])
                 for c in by) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    L += ["", f"generated vs retrieval only: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              + ("   <- clears zero" if lo > 0 else ""),
          f"E22 for comparison: 0.3591 weighted, C3 0.0311, recall 8.8%", f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E23_better_relatives_2026-09-20.txt", "w").write("\n".join(L) + "\n")
    json.dump({"weights": dict(zip(FEATS2, map(float, w))), "results": out,
               "generation_recall_c3": built / max(1, len(recall)), "net": [d, lo, hi]},
              open(f"{RESULTS}/E23_better_relatives.json", "w"), indent=2)


if __name__ == "__main__":
    main()
