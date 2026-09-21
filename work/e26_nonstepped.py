#!/usr/bin/env python3
"""E26 — the alignment arm again, on spectra that actually have an energy axis.

E24's alignment arm was null. Two confounds were named at the time and Seda pressed on the
second: ~20% of training spectra and ~25% of the test set are STEPPED acquisitions whose energy
field already holds several values (e.g. [20, 40, 60]) -- the axis is pre-merged, so for a
quarter of the scored data there is nothing to align along. Restricting to single-energy spectra
decides between two very different conclusions:

    signal here  -> the alignment idea is BLOCKED (by my representation choice), still alive
    nothing here -> REFUTED for this task, and we stop spending on bins and smoothing

This also bears on their own programme. The Diastema paper's claim is that "the REPRESENTATION --
not the alignment machinery -- is what earns a measurable advantage", validated on speech and
handwriting. Mass spectra are a third modality and, per Seda, the programme has no independent
replication anywhere. E24 replicated the ground and NOT the curve-over-average claim; E26 is the
clean test of that claim on this modality.

Everything we do treats a molecule's 1-16 spectra as a bag: concatenate the peaks, take the best
cosine per structure. Vanta's Hodos mathematics is built for the other reading -- a signal whose
every instant is a distribution over bins, evolving along an axis, where comparable moments must
be ALIGNED rather than averaged. Collision energy is that axis: at 20 eV a few weak bonds break,
at 60 eV the molecule is confetti. Two related molecules should shed substructures in a similar
ORDER, but a stronger bond delays the event -- which is a warp of the energy axis, not a
different signal.

Prerequisite checked first: 80.1% of training structures have >=3 distinct collision energies
(median 3), so the axis is populated. GNPS has 0%, which caps how many relatives can be aligned.

Three arms, so the GROUND and the ALIGNMENT cannot be confused with each other:

  pooled_cos   one pooled frame per structure, our current binned cosine       (today)
  pooled_fr    one pooled frame per structure, Fisher-Rao on sqrt(p)           (her ground only)
  aligned_fr   the energy series, DTW with Fisher-Rao frames, total cost
               divided by path length -- her A1 master distance                (ground + alignment)

Judged on the job that matters, with E13's yardstick: for Class 2 and Class 3 queries, how
structurally close is the best relative each method puts in its top 5. Relatives feed the
neighbour vote (45% of the score) and analog generation (39%), so this is upstream of both.

A1's normalisation is deliberate: divide the minimised total cost by the path length, NOT
minimise the average. Vanta retired the latter because a path can lower its own average by
padding itself with cheap steps; I verified that padding failure in July (min-mean 0.709 against
0.854 for the same pair).

PREDICTIONS (before the run, 2026-09-21, after E24):
  P1  pooled_fr still beats pooled_cos on this subset (the ground does not depend on the axis).
  P2  aligned_fr still does not beat pooled_fr. I now expect the alignment to be genuinely
      REFUTED for this task rather than blocked -- E24's saturation argument applies just as
      well to single-energy frames, which are exactly as sparse.
  P3  Fewer than half the Class 2/3 queries survive the single-energy filter.

  E24 RESULT (2026-09-21, 756 Class 2/3 queries) -- BOTH MAIN PREDICTIONS WRONG, IN OPPOSITE
  DIRECTIONS. Median best Tanimoto@5: pooled_cos 0.3694, pooled_fr 0.4313, aligned_fr 0.3565.
    P1 FAIL: the GROUND is not second-order. pooled_fr beats our cosine by +0.0285 mean
      [+0.0156, +0.0420] -- the convention we inherited IS costing us.
    P2 FAIL: alignment does nothing (-0.0047, CI crosses zero).
  WHY THE ALIGNMENT ARM PROBABLY FAILED, not yet tested: Fisher-Rao SATURATES on sparse frames.
  At 0.02 Da bins two single spectra barely share bins, so most frame distances sit near pi (the
  maximum) and the alignment has nothing to discriminate with; pooling makes the distributions
  dense enough for the distance to be informative. Also ~20% of training spectra (and 25% of the
  test set) are STEPPED acquisitions already merged across energies, which blurs the axis.
  So: her ground transfers, her alignment is blocked by a representation choice of mine.
  Follow-up if revisited: coarser bins or smoothing for the frame-level ground.
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
from e12_c3_analogs import EDITS as WIDE_EDITS
import casmi_pipeline as cp
from casmi_pipeline import prep, IDX_PEAKS, neutral_mass

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
NBINS = 80000          # 0.02 Da bins up to 1600 Da
MAX_FRAMES = 6         # frames per series, spread over the energy range
N_RESCORE = 200        # candidates rescored per query
TOPK = 5
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
EDIT_MASSES = sorted({abs(v) for v in WIDE_EDITS.values() if v > 0})


def frame(mz, it):
    """One spectrum as sqrt(p) over 0.02 Da bins: a point on the unit sphere (her A3).

    Same peak selection as everything else here (intensity floor, cap), then normalised to sum
    to one -- a probability distribution -- rather than L2-normalised as our cosine does.
    """
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    if len(it) == 0 or it.max() <= 0: return None
    keep = it >= cp.INT_FLOOR * it.max()
    mz, it = mz[keep], it[keep]
    if len(it) > cp.MAX_PEAKS:
        top = np.argpartition(-it, cp.MAX_PEAKS)[:cp.MAX_PEAKS]; mz, it = mz[top], it[top]
    b = np.rint(mz / cp.BIN).astype(np.int64)
    ok = (b >= 0) & (b < NBINS)
    b, it = b[ok], it[ok]
    if len(b) == 0: return None
    ub, inv = np.unique(b, return_inverse=True)
    agg = np.zeros(len(ub)); np.add.at(agg, inv, it)
    s = agg.sum()
    if s <= 0: return None
    return ub, np.sqrt(agg / s)


def fr(dense_q, ref):
    """Fisher-Rao distance between two sqrt(p) frames: 2*arccos(Bhattacharyya)."""
    b, v = ref
    bc = float(np.dot(dense_q[b], v))
    return 2.0 * np.arccos(min(1.0, max(0.0, bc)))


def dtw(series_q_dense, series_r, band=None):
    """A1: minimise TOTAL cost over monotone alignments, then divide by path length."""
    n, m = len(series_q_dense), len(series_r)
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    L = [[0] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = fr(series_q_dense[i - 1], series_r[j - 1])
            best, bl = INF, 0
            for pi, pj in ((i - 1, j - 1), (i - 1, j), (i, j - 1)):
                if D[pi][pj] + c < best:
                    best, bl = D[pi][pj] + c, L[pi][pj] + 1
            D[i][j], L[i][j] = best, bl
    return D[n][m] / max(1, L[n][m])


def pick_frames(rows, ce, peaks, cap=MAX_FRAMES, single=None):
    """Spectra ordered by collision energy, thinned to at most `cap` frames.
    single: boolean mask of non-stepped spectra; rows outside it are dropped (E26)."""
    if single is not None: rows = [r for r in rows if single[r]]
    have = [(ce[r], r) for r in rows if np.isfinite(ce[r])]
    have.sort()
    if not have: have = [(0.0, r) for r in rows]
    if len(have) > cap:
        idx = np.linspace(0, len(have) - 1, cap).round().astype(int)
        have = [have[i] for i in idx]
    out = []
    for _, r in have:
        f = frame(*peaks(r))
        if f is not None: out.append(f)
    return out


def pooled_frame(rows, peaks):
    mz = np.concatenate([peaks(r)[0] for r in rows]); it = np.concatenate([peaks(r)[1] for r in rows])
    return frame(mz, it)


def main():
    t0 = time.time()
    t = pq.read_table(DATA, columns=["inchikey14", "ms2_mzs", "ms2_normalized_intensities",
                                     "collision_energy_ev", "precursor_mz", "adduct"])
    keys = np.asarray(t.column("inchikey14")); pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct"))
    ce_raw = t.column("collision_energy_ev").to_pylist()
    ce = np.array([(x[0] if isinstance(x, list) and x else (x if isinstance(x, (int, float)) else np.nan))
                   for x in ce_raw], dtype=float)
    # E26: a spectrum is "stepped" when its energy field holds more than one value -- the axis is
    # already merged inside that one spectrum, so it cannot be a frame in a series.
    nce = np.array([(len(x) if isinstance(x, (list, tuple)) else (0 if x is None else 1)) for x in ce_raw])
    single = nce == 1
    def flat(c):
        a = t.column(c).combine_chunks(); return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    _, _, _, tr, co, _ = load_structures()
    trs = dict(zip(tr.k, tr.s)); cos_ = dict(zip(co.k, co.s))
    smi_of = lambda k: trs.get(k) or cos_.get(k, "")
    fpc = {}
    def fp(k):
        if k not in fpc:
            m = Chem.MolFromSmiles(smi_of(k) or "")
            fpc[k] = FPG.GetFingerprint(m) if m is not None else None
        return fpc[k]
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    per = defaultdict(list)
    nframes = []
    for split in EVAL:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))}
        ref_rows = np.load(f"{SPLITS}/ref_rows_{split}.npy")
        by_struct = defaultdict(list)
        for r in ref_rows: by_struct[keys[r]].append(int(r))
        print(f"[{split}] {len(by_struct):,} reference structures {time.time()-t0:.0f}s", flush=True)

        cache_series, cache_pool = {}, {}
        for n, (k, rows) in enumerate(qrows.items(), 1):
            cls = assign[k]
            if cls == 1: continue                     # C1's own spectra are present: not the question
            tfp = fp(k)
            if tfp is None: continue
            qs = pick_frames(rows, ce, peaks, single=single)
            if len(qs) < 2: continue          # needs an axis to align along
            qp = pooled_frame(rows, peaks)
            if not qs or qp is None: continue
            dense_q = []
            for b, v in qs:
                d = np.zeros(NBINS); d[b] = v; dense_q.append(d)
            dq_pool = np.zeros(NBINS); dq_pool[qp[0]] = qp[1]

            cand = [c for c, _ in sorted(pools[k]["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:N_RESCORE]]
            scored = {"pooled_cos": [], "pooled_fr": [], "aligned_fr": []}
            for c in cand:
                rows_c = by_struct.get(c, [])
                if not rows_c: continue
                if c not in cache_pool:
                    cache_pool[c] = pooled_frame(rows_c, peaks)
                    cache_series[c] = pick_frames(rows_c, ce, peaks, single=single)
                pf, sf = cache_pool[c], cache_series[c]
                if pf is None or len(sf) < 2: continue
                scored["pooled_cos"].append((-pools[k]["spec"][c], c))          # our current ranking
                scored["pooled_fr"].append((fr(dq_pool, pf), c))
                scored["aligned_fr"].append((dtw(dense_q, sf), c))
                nframes.append(len(sf))
            for meth, lst in scored.items():
                if not lst: continue
                lst.sort()
                best = max((DataStructs.TanimotoSimilarity(tfp, fp(c)) for _, c in lst[:TOPK]
                            if fp(c) is not None), default=0.0)
                per[meth].append(best)
            if n % 100 == 0: print(f"[{split}]  {n}/{len(qrows)} {time.time()-t0:.0f}s", flush=True)

    L = [f"E26 — alignment on NON-STEPPED spectra only, {len(per['pooled_cos'])} Class 2/3 queries", "",
         f"reference series length: median {np.median(nframes):.0f} frames", "",
         f"{'method':12s} {'median best Tanimoto@5':>24} {'mean':>8} {'>=0.6':>8} {'>=0.4':>8}"]
    out = {}
    for meth in ("pooled_cos", "pooled_fr", "aligned_fr"):
        a = np.array(per[meth])
        out[meth] = dict(median=float(np.median(a)), mean=float(a.mean()),
                         share06=float((a >= 0.6).mean()), share04=float((a >= 0.4).mean()), n=len(a))
        L.append(f"{meth:12s} {np.median(a):>24.4f} {a.mean():>8.4f} {(a>=0.6).mean():>8.1%} {(a>=0.4).mean():>8.1%}")
    base = np.array(per["pooled_cos"])
    rng = np.random.default_rng(0)
    L.append("")
    for meth in ("pooled_fr", "aligned_fr"):
        d = np.array(per[meth]) - base
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        L.append(f"{meth:12s} vs pooled_cos: mean Tanimoto {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                 + ("   <- clears zero" if lo > 0 else ""))
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E26_nonstepped_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E26_nonstepped.json", "w"), indent=2)


if __name__ == "__main__":
    main()
