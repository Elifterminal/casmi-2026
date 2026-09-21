#!/usr/bin/env python3
"""E29 — the ground family, now that we have read the paper instead of the summary.

E24/E25 tested exactly one of their grounds (Fisher-Rao on sqrt(p)) against our inherited
cosine, and it won and transferred to the leaderboard (0.277 -> 0.288). The Diastema paper lists
the ground as PLUGGABLE and names the others: the symmetric triangular chi-squared divergence
(their computational default), its metric root sqrt(chi2), Hellinger, and -- for ORDERED bins --
Wasserstein-1. It also supplies the shift quotient: minimise the distance over translations of
the axis, which for a pitch shift is a cyclic roll.

Three things follow for us, and they are why this experiment exists:

  1. Hellinger and Fisher-Rao cannot differ HERE. Both are strictly monotone in the Bhattacharyya
     coefficient, and we only ever use the distance to RANK, so they induce the same order. Not
     worth an arm; worth saying once.
  2. chi2 and sqrt(chi2) are genuinely different orders, and chi2 is what their own engine
     defaults to. Untested by us.
  3. Wasserstein-1 is the interesting one. They exclude it from their shift-isometry theorem for
     a correct reason that does not apply to us: their pitch bins are CYCLIC, and a cyclic roll
     is not a translation on a line. Our bins are ordered by mass and not cyclic -- and a
     structural edit shifts fragment masses by an exact amount, so an edit IS a transport and W1
     measures precisely that. This may be a case their setting excludes rather than one they
     tested and rejected.

Plus their shift quotient, in the form this modality wants: minimise over the EDIT masses, not
over arbitrary shifts. A relative one methylation away has a fragment set displaced by 14.0157 Da
in the substituted part; quotienting by that group should pull true relatives closer.

Judged on E24's yardstick: how structurally close is the best relative in the top 5, for Class 2
and Class 3 queries, over the three twin-safe eval splits.

PREDICTIONS (before the run, 2026-09-21):
  P1  Hellinger ties Fisher-Rao exactly (stated above; the arm is a control on my own reasoning).
  P2  sqrt(chi2) lands within 0.01 of Fisher-Rao -- close cousins near the diagonal, as their
      own chi2 ~ (1/4) d_FR^2 correspondence implies.
  P3  W1 does NOT beat Fisher-Rao outright: mass spectra are spiky, and transport cost is
      dominated by how FAR peaks are rather than whether they match, which is the wrong
      sensitivity for identity.
  P4  The edit-shift quotient beats plain Fisher-Rao: it is the only arm that encodes what makes
      a relative a relative in this domain.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e12_c3_analogs import EDITS as WIDE_EDITS
import casmi_pipeline as cp

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
NBINS = 80000
N_RESCORE = 200
TOPK = 5
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
# the shift quotient's group: the common structural edits, in bins (0.02 Da), both directions
EDIT_BINS = sorted({int(round(abs(v) / cp.BIN)) for v in WIDE_EDITS.values() if abs(v) > 1e-9})


def frame(mz, it):
    """A spectrum as p over 0.02 Da bins (sum 1). Sparse: (bins, probabilities)."""
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    if len(it) == 0 or it.max() <= 0: return None
    keep = it >= cp.INT_FLOOR * it.max(); mz, it = mz[keep], it[keep]
    if len(it) > cp.MAX_PEAKS:
        top = np.argpartition(-it, cp.MAX_PEAKS)[:cp.MAX_PEAKS]; mz, it = mz[top], it[top]
    b = np.rint(mz / cp.BIN).astype(np.int64)
    ok = (b >= 0) & (b < NBINS); b, it = b[ok], it[ok]
    if len(b) == 0: return None
    ub, inv = np.unique(b, return_inverse=True)
    agg = np.zeros(len(ub)); np.add.at(agg, inv, it)
    s = agg.sum()
    return (ub, agg / s) if s > 0 else None


def pooled(rows, peaks):
    mz = np.concatenate([peaks(r)[0] for r in rows]); it = np.concatenate([peaks(r)[1] for r in rows])
    return frame(mz, it)


def bc(dense_sqrt_q, ref):
    b, p = ref
    return float(np.dot(dense_sqrt_q[b], np.sqrt(p)))


def chi2_dist(dense_q, ref, sqrt=False):
    """Their triangular divergence: chi2(p,q) = 1/2 sum (p-q)^2 / (p+q), over the union of bins."""
    b, p = ref
    q = dense_q[b]
    s = p + q
    d = 0.5 * float(np.sum(np.where(s > 0, (p - q) ** 2 / np.maximum(s, 1e-300), 0.0)))
    # bins where the query has mass but the reference has none contribute q/2 each
    d += 0.5 * float(dense_q.sum() - q.sum())
    return np.sqrt(d) if sqrt else d


def w1(qb, qp, rb, rp):
    """Wasserstein-1 on ORDERED bins: integral |CDF_p - CDF_q|, in bin units."""
    bins = np.union1d(qb, rb)
    a = np.zeros(len(bins)); a[np.searchsorted(bins, qb)] = qp
    c = np.zeros(len(bins)); c[np.searchsorted(bins, rb)] = rp
    diff = np.cumsum(a - c)[:-1]
    return float(np.sum(np.abs(diff) * np.diff(bins)))


def main():
    t0 = time.time()
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.expanduser("~/casmi-2026/work/data/train.parquet"),
                      columns=["inchikey14", "ms2_mzs", "ms2_normalized_intensities"])
    keys = np.asarray(t.column("inchikey14"))
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
            m = Chem.MolFromSmiles(smi_of(k) or ""); fpc[k] = FPG.GetFingerprint(m) if m is not None else None
        return fpc[k]
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    ARMS = ["cosine (ours)", "fisher_rao", "hellinger", "sqrt_chi2", "chi2", "wasserstein1", "FR + edit-shift quotient"]
    per = defaultdict(list)
    for split in EVAL:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))}
        ref_rows = np.load(f"{SPLITS}/ref_rows_{split}.npy")
        by_struct = defaultdict(list)
        for r in ref_rows: by_struct[keys[r]].append(int(r))
        cache = {}
        print(f"[{split}] {len(by_struct):,} reference structures {time.time()-t0:.0f}s", flush=True)
        for n, (k, rows) in enumerate(qrows.items(), 1):
            if assign[k] == 1: continue
            tfp = fp(k)
            if tfp is None: continue
            qf = pooled(rows, peaks)
            if qf is None: continue
            qb, qp = qf
            dq = np.zeros(NBINS); dq[qb] = qp
            dq_sqrt = np.sqrt(dq)
            cand = [c for c, _ in sorted(pools[k]["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:N_RESCORE]]
            sc = {a: [] for a in ARMS}
            for c in cand:
                rows_c = by_struct.get(c, [])
                if not rows_c: continue
                if c not in cache: cache[c] = pooled(rows_c, peaks)
                rf = cache[c]
                if rf is None: continue
                rb, rp = rf
                b = bc(dq_sqrt, rf)
                sc["cosine (ours)"].append((-pools[k]["spec"][c], c))
                sc["fisher_rao"].append((2 * np.arccos(min(1.0, max(0.0, b))), c))
                sc["hellinger"].append((np.sqrt(max(0.0, 1.0 - b)), c))
                sc["sqrt_chi2"].append((chi2_dist(dq, rf, sqrt=True), c))
                sc["chi2"].append((chi2_dist(dq, rf), c))
                sc["wasserstein1"].append((w1(qb, qp, rb, rp), c))
                best = b
                for s in EDIT_BINS:                       # quotient by the edit group
                    for sgn in (1, -1):
                        shifted = (rb + sgn * s, rp)
                        ok = (shifted[0] >= 0) & (shifted[0] < NBINS)
                        if ok.any():
                            best = max(best, bc(dq_sqrt, (shifted[0][ok], shifted[1][ok])))
                sc["FR + edit-shift quotient"].append((2 * np.arccos(min(1.0, max(0.0, best))), c))
            for a in ARMS:
                if not sc[a]: continue
                sc[a].sort()
                per[a].append(max((DataStructs.TanimotoSimilarity(tfp, fp(c)) for _, c in sc[a][:TOPK]
                                   if fp(c) is not None), default=0.0))
            if n % 100 == 0: print(f"[{split}]  {n}/{len(qrows)} {time.time()-t0:.0f}s", flush=True)

    L = [f"E29 — the ground family on the relative-search yardstick, {len(per['fisher_rao'])} Class 2/3 queries",
         "", f"edit-shift quotient group: {len(EDIT_BINS)} edit masses, both directions", "",
         f"{'ground':26s} {'median best Tanimoto@5':>24} {'mean':>8} {'>=0.6':>8}"]
    out = {}
    for a in ARMS:
        v = np.array(per[a])
        out[a] = dict(median=float(np.median(v)), mean=float(v.mean()), share06=float((v >= 0.6).mean()))
        L.append(f"{a:26s} {np.median(v):>24.4f} {v.mean():>8.4f} {(v>=0.6).mean():>8.1%}")
    base = np.array(per["fisher_rao"]); rng = np.random.default_rng(0)
    L += ["", "against Fisher-Rao (the ground now in production):"]
    for a in ARMS:
        if a == "fisher_rao": continue
        d = np.array(per[a]) - base
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        L.append(f"  {a:26s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]" + ("   <- beats it" if lo > 0 else ""))
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E29_grounds_2026-09-21.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E29_grounds.json", "w"), indent=2)


if __name__ == "__main__":
    main()
