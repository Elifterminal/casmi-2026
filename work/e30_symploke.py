#!/usr/bin/env python3
"""E30 — Hodos Symploke: does the (m/z x energy) joint carry anything our pooling throws away?

Two experiments (E24, E26) asked whether to ALIGN the collision-energy axis and both came back
null. Reading the family paper, that was the wrong end of the question. The second equation,
Symploke, measures what a relationship creates that is in neither part:

    J   the joint distribution of (A, B) at an instant
    M   the outer product of its marginals -- "the same parts, forbidden from interacting"
    e   = g(J, M), read in the same Fisher-Rao geometry as the first equation

and its stated property is the one we want: THE CONTROL IS A TERM IN THE EQUATION. If the joint
factorises, the surplus vanishes by arithmetic rather than by measurement.

Our pooled spectrum is precisely a marginal. For a molecule measured at several collision
energies, J is the distribution over (m/z bin, energy frame) and pooling keeps only the m/z
marginal. So "is there anything on the energy axis?" is answered by the surplus, before any
alignment machinery is chosen -- and if the surplus is ~0 our two nulls need no further excuse.

Three things measured here:
  1. the surplus e for each molecule, against a permutation null that destroys the m/z-energy
     dependence while keeping the m/z marginal exactly (finite-sample noise is not zero even
     when the arithmetic control is)
  2. whether e varies enough between molecules to be worth acting on
  3. THE LINK: does a molecule's surplus predict whether energy-aligned comparison beat pooled
     comparison for that molecule (E26's per-query outcome)? If the joint carries information
     and our alignment still cannot use it, the opportunity is real and the tool was wrong.

PREDICTIONS (before the run, 2026-09-21):
  P1  e is clearly above its permutation null -- fragmentation genuinely changes with energy,
      this is chemistry, not noise.
  P2  e is nonetheless small on the Fisher-Rao scale (which runs to pi): most of a spectrum's
      mass sits in peaks present at every energy.
  P3  e does NOT predict per-molecule alignment benefit (|r| < 0.15). That combination -- real
      surplus, no exploitable benefit -- would say the axis carries information that DTW over
      saturating frames cannot reach, and would justify one soft-DTW attempt rather than none.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
import casmi_pipeline as cp

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
NBINS = 80000
N_RESCORE = 200
TOPK = 5
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
RNG = np.random.default_rng(20260921)


def frames_of(rows, ce, single, peaks):
    """Single-energy spectra of one molecule, ordered by collision energy: list of (bins, mass)."""
    have = sorted((ce[r], r) for r in rows if single[r] and np.isfinite(ce[r]))
    out = []
    for _, r in have:
        mz, it = peaks(r)
        mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
        if len(it) == 0 or it.max() <= 0: continue
        keep = it >= cp.INT_FLOOR * it.max(); mz, it = mz[keep], it[keep]
        if len(it) > cp.MAX_PEAKS:
            top = np.argpartition(-it, cp.MAX_PEAKS)[:cp.MAX_PEAKS]; mz, it = mz[top], it[top]
        b = np.rint(mz / cp.BIN).astype(np.int64)
        ok = (b >= 0) & (b < NBINS); b, it = b[ok], it[ok]
        if len(b) == 0: continue
        ub, inv = np.unique(b, return_inverse=True)
        agg = np.zeros(len(ub)); np.add.at(agg, inv, it)
        out.append((ub, agg))
    return out


def joint(frames):
    """Dense (bins x frames) joint over the union of bins, normalised to sum 1."""
    bins = np.unique(np.concatenate([b for b, _ in frames]))
    J = np.zeros((len(bins), len(frames)))
    for t, (b, v) in enumerate(frames):
        J[np.searchsorted(bins, b), t] = v
    s = J.sum()
    return (J / s) if s > 0 else None


def surplus(J):
    """Symploke: Fisher-Rao distance between the joint and the product of its marginals."""
    a = J.sum(1); c = J.sum(0)
    M = np.outer(a, c)
    bc = float(np.sum(np.sqrt(J * M)))
    return 2.0 * np.arccos(min(1.0, max(0.0, bc)))


def permuted(J):
    """Null: shuffle each m/z bin's intensities across frames. Keeps the m/z marginal exactly,
    destroys the dependence between which fragment appears and at what energy."""
    K = J.copy()
    for i in range(K.shape[0]):
        RNG.shuffle(K[i])
    return K


def main():
    t0 = time.time()
    t = pq.read_table(DATA, columns=["inchikey14", "ms2_mzs", "ms2_normalized_intensities",
                                     "collision_energy_ev"])
    keys = np.asarray(t.column("inchikey14"))
    ce_raw = t.column("collision_energy_ev").to_pylist()
    ce = np.array([(x[0] if isinstance(x, list) and x else (x if isinstance(x, (int, float)) else np.nan))
                   for x in ce_raw], dtype=float)
    single = np.array([(len(x) if isinstance(x, (list, tuple)) else (0 if x is None else 1)) for x in ce_raw]) == 1
    def flat(c):
        a = t.column(c).combine_chunks(); return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    _, _, _, tr, co, _ = load_structures()
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_of = lambda k: trs.get(k) or cos_.get(k, "")
    fpc = {}
    def fp(k):
        if k not in fpc:
            mm = Chem.MolFromSmiles(smi_of(k) or ""); fpc[k] = FPG.GetFingerprint(mm) if mm is not None else None
        return fpc[k]
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    eps, nulls, benefit, nframes = [], [], [], []
    for split in EVAL:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))}
        ref_rows = np.load(f"{SPLITS}/ref_rows_{split}.npy")
        by_struct = defaultdict(list)
        for r in ref_rows: by_struct[keys[r]].append(int(r))
        cache_s, cache_p = {}, {}
        print(f"[{split}] {len(by_struct):,} reference structures {time.time()-t0:.0f}s", flush=True)
        for n, (k, rows) in enumerate(qrows.items(), 1):
            if assign[k] == 1: continue
            qfr = frames_of(rows, ce, single, peaks)
            if len(qfr) < 2: continue
            J = joint(qfr)
            if J is None: continue
            e = surplus(J)
            eps.append(e); nulls.append(surplus(permuted(J))); nframes.append(len(qfr))
            # per-molecule benefit of aligning vs pooling, on the relative-search yardstick
            tfp = fp(k)
            if tfp is None:
                benefit.append(np.nan); continue
            dq = [np.zeros(NBINS) for _ in qfr]
            for d, (b, v) in zip(dq, qfr):
                s = v.sum()
                if s > 0: d[b] = np.sqrt(v / s)
            allb = np.unique(np.concatenate([b for b, _ in qfr]))
            pv = np.zeros(NBINS)
            for b, v in qfr: pv[b] += v
            tot = pv.sum()
            dq_pool = np.sqrt(pv / tot) if tot > 0 else pv
            cand = [c for c, _ in sorted(pools[k]["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:N_RESCORE]]
            sc_pool, sc_align = [], []
            for c in cand:
                rr = by_struct.get(c, [])
                if not rr: continue
                if c not in cache_s:
                    f = frames_of(rr, ce, single, peaks)
                    cache_s[c] = f
                    if f:
                        pb = np.concatenate([b for b, _ in f]); pv2 = np.concatenate([v for _, v in f])
                        ub, inv = np.unique(pb, return_inverse=True)
                        agg = np.zeros(len(ub)); np.add.at(agg, inv, pv2)
                        s2 = agg.sum()
                        cache_p[c] = (ub, np.sqrt(agg / s2)) if s2 > 0 else None
                    else:
                        cache_p[c] = None
                f, pf = cache_s[c], cache_p[c]
                if not f or len(f) < 2 or pf is None: continue
                sc_pool.append((2 * np.arccos(min(1.0, float(np.dot(dq_pool[pf[0]], pf[1])))), c))
                # path-normalised DTW over Fisher-Rao frames (their A1), as in E24/E26
                nq, nr = len(dq), len(f)
                D = [[float("inf")] * (nr + 1) for _ in range(nq + 1)]
                Lp = [[0] * (nr + 1) for _ in range(nq + 1)]
                D[0][0] = 0.0
                for i in range(1, nq + 1):
                    for j in range(1, nr + 1):
                        b2, v2 = f[j - 1]; s2 = v2.sum()
                        cst = 2 * np.arccos(min(1.0, float(np.dot(dq[i-1][b2], np.sqrt(v2 / s2))))) if s2 > 0 else np.pi
                        best, bl = float("inf"), 0
                        for pi, pj in ((i-1, j-1), (i-1, j), (i, j-1)):
                            if D[pi][pj] + cst < best: best, bl = D[pi][pj] + cst, Lp[pi][pj] + 1
                        D[i][j], Lp[i][j] = best, bl
                sc_align.append((D[nq][nr] / max(1, Lp[nq][nr]), c))
            if not sc_pool or not sc_align:
                benefit.append(np.nan); continue
            def best_tan(lst):
                lst.sort()
                return max((DataStructs.TanimotoSimilarity(tfp, fp(c)) for _, c in lst[:TOPK] if fp(c) is not None),
                           default=0.0)
            benefit.append(best_tan(sc_align) - best_tan(sc_pool))
            if n % 100 == 0: print(f"[{split}]  {n}/{len(qrows)} {time.time()-t0:.0f}s", flush=True)

    eps = np.array(eps); nulls = np.array(nulls); ben = np.array(benefit)
    ok = ~np.isnan(ben)
    r = float(np.corrcoef(eps[ok], ben[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    hi = eps[ok] >= np.median(eps[ok]); lo = ~hi
    L = [f"E30 — Symploke: the (m/z x energy) joint against the product of its marginals", "",
         f"molecules with >=2 single-energy spectra: {len(eps)}   (median {np.median(nframes):.0f} frames)", "",
         f"surplus e (Fisher-Rao, scale 0..pi):",
         f"  real          median {np.median(eps):.4f}   mean {eps.mean():.4f}   90th pct {np.percentile(eps,90):.4f}",
         f"  permuted null median {np.median(nulls):.4f}   mean {nulls.mean():.4f}",
         f"  paired difference (real - null): mean {np.mean(eps-nulls):+.4f}", "",
         f"does the surplus predict where aligning beat pooling?",
         f"  molecules with both measured: {int(ok.sum())}",
         f"  correlation(e, alignment benefit) r = {r:+.4f}",
         f"  alignment benefit, high-surplus half: {ben[ok][hi].mean():+.4f}",
         f"  alignment benefit, low-surplus half:  {ben[ok][lo].mean():+.4f}"]
    rng = np.random.default_rng(0)
    d = eps - nulls
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    loq, hiq = np.percentile(boots, [2.5, 97.5])
    L += ["", f"real vs null: {d.mean():+.4f}  95% CI [{loq:+.4f}, {hiq:+.4f}]"
              + ("   <- the joint carries real dependence" if loq > 0 else "   <- indistinguishable from noise"),
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E30_symploke_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump({"n": len(eps), "e_median": float(np.median(eps)), "null_median": float(np.median(nulls)),
               "paired": [float(d.mean()), float(loq), float(hiq)], "r_with_benefit": r,
               "benefit_high": float(ben[ok][hi].mean()), "benefit_low": float(ben[ok][lo].mean())},
              open(f"{RESULTS}/E30_symploke.json", "w"), indent=2)


if __name__ == "__main__":
    main()
