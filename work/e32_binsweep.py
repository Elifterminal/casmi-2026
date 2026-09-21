#!/usr/bin/env python3
"""E32 — the Symploke measurement at four bin widths, because E30 measured its own sparsity.

E30 ran at our production binning (0.02 Da) and the permutation null landed ON the signal: real
surplus median 1.6146, null 1.6514, both enormous on a 0..pi scale. That is not weak dependence;
it is a measurement that cannot see dependence at all. At 0.02 Da nearly every (mass, energy)
cell is empty, and a sparse joint sits far from the product of its marginals whatever the
dependence is. The same binning is what saturated the Fisher-Rao frames in the alignment arms
(E24, E26). Twice now the representation has been the blocker, not the idea.

Their own standard: a null produced by a crippled representation is not a boundary. So this
repeats the identical measurement at 0.02, 0.1, 0.5 and 2.0 Da, with the permutation null at
each width. The null is the whole point -- it tells us whether the measurement can see anything.

PREDICTIONS (before the run, 2026-09-21):
  P1  At 0.02 Da real and null stay indistinguishable. This reproduces E30 and is the control.
  P2  By 0.5 Da the real surplus sits clearly BELOW its null: shuffling which fragment appears at
      which energy destroys real structure, and once cells are populated that should show.
  P3  Separation grows as bins coarsen, until 2.0 Da starts merging genuinely distinct fragments.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import casmi_pipeline as cp

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
BIN_WIDTHS = [0.02, 0.1, 0.5, 2.0]
RNG = np.random.default_rng(20260921)


def frames_of(rows, ce, single, peaks, bw):
    """Single-energy spectra of one molecule, ordered by collision energy, binned at bw."""
    out = []
    for _, r in sorted((ce[r], r) for r in rows if single[r] and np.isfinite(ce[r])):
        mz, it = peaks(r)
        mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
        if len(it) == 0 or it.max() <= 0: continue
        keep = it >= cp.INT_FLOOR * it.max(); mz, it = mz[keep], it[keep]
        if len(it) > cp.MAX_PEAKS:
            top = np.argpartition(-it, cp.MAX_PEAKS)[:cp.MAX_PEAKS]; mz, it = mz[top], it[top]
        b = np.rint(mz / bw).astype(np.int64)
        ub, inv = np.unique(b, return_inverse=True)
        agg = np.zeros(len(ub)); np.add.at(agg, inv, it)
        out.append((ub, agg))
    return out


def joint(frames):
    bins = np.unique(np.concatenate([b for b, _ in frames]))
    J = np.zeros((len(bins), len(frames)))
    for t, (b, v) in enumerate(frames):
        J[np.searchsorted(bins, b), t] = v
    s = J.sum()
    return (J / s) if s > 0 else None


def surplus(J):
    """Symploke: Fisher-Rao between the joint and the product of its marginals."""
    M = np.outer(J.sum(1), J.sum(0))
    bc = float(np.sum(np.sqrt(J * M)))
    return 2.0 * np.arccos(min(1.0, max(0.0, bc)))


def permuted(J):
    K = J.copy()
    for i in range(K.shape[0]): RNG.shuffle(K[i])
    return K


def main():
    t0 = time.time()
    t = pq.read_table(DATA, columns=["inchikey14", "ms2_mzs", "ms2_normalized_intensities",
                                     "collision_energy_ev"])
    ce_raw = t.column("collision_energy_ev").to_pylist()
    ce = np.array([(x[0] if isinstance(x, list) and x else (x if isinstance(x, (int, float)) else np.nan))
                   for x in ce_raw], dtype=float)
    single = np.array([(len(x) if isinstance(x, (list, tuple)) else (0 if x is None else 1))
                       for x in ce_raw]) == 1
    def flat(c):
        a = t.column(c).combine_chunks(); return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    queries = []
    for split in EVAL:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        for k, rows in sp["query_rows"].items():
            if sp["assign"][k] != 1: queries.append(rows)
    print(f"{len(queries)} Class 2/3 queries", flush=True)

    L = ["E32 — can the Symploke surplus see dependence at all? Four bin widths.", "",
         f"{'bin (Da)':>9} {'molecules':>10} {'cells filled':>13} {'real e':>9} {'null e':>9} {'real-null':>11} {'95% CI':>22}"]
    out = {}
    for bw in BIN_WIDTHS:
        eps, nulls, fill = [], [], []
        for rows in queries:
            fr = frames_of(rows, ce, single, peaks, bw)
            if len(fr) < 2: continue
            J = joint(fr)
            if J is None: continue
            eps.append(surplus(J)); nulls.append(surplus(permuted(J)))
            fill.append(float((J > 0).mean()))
        eps, nulls = np.array(eps), np.array(nulls)
        d = eps - nulls
        rng = np.random.default_rng(0)
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[bw] = dict(n=len(eps), fill=float(np.mean(fill)), real=float(np.median(eps)),
                       null=float(np.median(nulls)), diff=float(d.mean()), ci=[float(lo), float(hi)])
        L.append(f"{bw:>9.2f} {len(eps):>10} {np.mean(fill):>12.1%} {np.median(eps):>9.4f} "
                 f"{np.median(nulls):>9.4f} {d.mean():>+11.4f}   [{lo:+.4f}, {hi:+.4f}]")
        print(L[-1], flush=True)
    L += ["", "reading: 'cells filled' is the share of (mass bin x energy frame) cells with any",
          "intensity. When that is near zero the joint is far from its marginals by sparsity alone",
          "and the surplus cannot distinguish dependence from independence -- which is what E30",
          "measured without noticing.", f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L[-5:]))
    open(f"{RESULTS}/E32_binsweep_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump({str(k): v for k, v in out.items()}, open(f"{RESULTS}/E32_binsweep.json", "w"), indent=2)


if __name__ == "__main__":
    main()
