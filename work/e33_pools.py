#!/usr/bin/env python3
"""E33 part 1 — build three spectral-similarity variants in one pass.

E29 ranked grounds on the relative-search yardstick and two arms were worth taking further:

  chi2     their engine's own default, the symmetric triangular divergence. Best MEDIAN of the
           family (0.4522 against Fisher-Rao's 0.4313) though the paired mean crossed zero.
           Conveniently chi2 lies in [0,1] for probability vectors, so 1 - chi2 is a similarity
           on the same scale as the Bhattacharyya coefficient we ship, and nothing downstream
           needs rescaling.
  eshift   their pitch quotient, adapted: minimise the distance over the group of STRUCTURAL
           EDIT masses rather than over arbitrary shifts. Best mean (0.4791) and best share above
           0.6 (39.0%) in E29, with its interval just touching zero. It is the only thing we have
           tried that encodes what makes a relative a relative in this domain: a molecule one
           methylation away has its substituted fragments displaced by exactly 14.0157 Da.

The gather (inverted index over shared peak bins) is the expensive step and is identical for all
three, so this computes every variant in one pass and writes one pool file each. The quotient is
applied only to the top QUOT_TOP candidates by plain similarity -- 34 shifted comparisons against
every gathered candidate would cost more than it can return.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from candidates import formula_mass, neutral_mass, MassIndex
from e12_c3_analogs import EDITS as WIDE_EDITS
from casmi_pipeline import prep, IDX_PEAKS, N_RESCORE, PPM, TOPN, BIN
import casmi_pipeline as cp


def prep_agg(mz, it):
    """prep(), but peaks landing in the same 0.02 Da bin are SUMMED before normalising.

    Production prep() leaves duplicate bin indices in the vector; the pointer-walk cosine then
    pairs them off, which is self-consistent but double-counts under any dense formulation (it
    produced a Bhattacharyya coefficient of 1.2 here, which is impossible, and that is how this
    was noticed). Aggregating is the correct reading of "a spectrum is a distribution over bins",
    so E33 uses it for every arm and carries fr_agg as its own arm to price the change.
    """
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float32)
    if len(it) == 0 or it.max() <= 0: return np.empty(0, np.int64), np.empty(0, np.float64)
    keep = it >= cp.INT_FLOOR * it.max(); mz, it = mz[keep], it[keep]
    if len(it) > cp.MAX_PEAKS:
        top = np.argpartition(-it, cp.MAX_PEAKS)[:cp.MAX_PEAKS]; mz, it = mz[top], it[top]
    b = np.rint(mz / BIN).astype(np.int64)
    ok = (b >= 0) & (b < NBINS)          # guard: peaks past the axis are dropped, not fatal
    b, it = b[ok], it[ok]
    if len(b) == 0: return np.empty(0, np.int64), np.empty(0, np.float64)
    ub, inv = np.unique(b, return_inverse=True)
    agg = np.zeros(len(ub)); np.add.at(agg, inv, it.astype(np.float64))
    s = agg.sum()
    if s <= 0: return np.empty(0, np.int64), np.empty(0, np.float64)
    return ub, np.sqrt(agg / s)          # sqrt(p): unit L2 norm by construction
from twins import db_exclusions
import scoring

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
NBINS = 600000        # 12,000 Da: a peak at 4,032 Da overflowed the first attempt
QUOT_TOP = 100        # candidates re-scored under the edit-shift quotient
EDIT_SHIFTS = sorted({int(round(abs(v) / BIN)) for v in WIDE_EDITS.values() if abs(v) > 1e-9})


def main(split):
    t0 = time.time()
    sp = json.load(open(f"{SPLITS}/split_{split}.json"))
    assign, qrows = sp["assign"], sp["query_rows"]
    ref_rows = np.load(f"{SPLITS}/ref_rows_{split}.npy")
    t = pq.read_table(DATA, columns=["inchikey14", "normalized_smiles", "molecular_formula",
                                     "precursor_mz", "adduct", "ionization_mode",
                                     "ms2_mzs", "ms2_normalized_intensities"])
    keys = np.asarray(t.column("inchikey14")); smiles = np.asarray(t.column("normalized_smiles"))
    formula = np.asarray(t.column("molecular_formula")); pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct")); mode = np.asarray(t.column("ionization_mode"))
    def flat(c):
        a = t.column(c).combine_chunks(); return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    print(f"[{split}] loaded {time.time()-t0:.0f}s", flush=True)

    # reference frames: sqrt(p) (for BC) and p (for chi2), sparse, plus the inverted index
    sq, pr, inv = {}, {}, defaultdict(list)
    for r in ref_rows:
        b, w = prep_agg(*peaks(r))
        sq[r] = (b, w); pr[r] = (b, w ** 2)          # w = sqrt(p) so w^2 = p
        if len(b):
            for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]): inv[int(bb)].append(r)
    inv = {k: np.asarray(v) for k, v in inv.items()}
    struct = pd.DataFrame({"k": keys, "f": formula, "s": smiles}).drop_duplicates("k")
    massk = {k: formula_mass(f) for k, f in zip(struct["k"], struct["f"])}
    smik = dict(zip(struct["k"], struct["s"]))
    excl = db_exclusions(sp, None)
    dbk = np.array([k for k in struct["k"] if k not in excl and np.isfinite(massk[k])])
    midx = MassIndex(dbk, np.array([massk[k] for k in dbk]))
    print(f"[{split}] indexes {time.time()-t0:.0f}s", flush=True)

    out = {v: [] for v in ("fr_agg", "chi2", "eshift")}
    for n, (k, rows) in enumerate(qrows.items(), 1):
        positive = str(mode[rows[0]]).lower().startswith("pos")
        allmz = np.concatenate([peaks(r)[0] for r in rows])
        allit = np.concatenate([peaks(r)[1] for r in rows])
        spec = {v: defaultdict(float) for v in out}
        for r in rows:
            qb, qw = prep_agg(*peaks(r))
            if len(qb) == 0: continue
            dq = np.zeros(NBINS); dq[qb] = qw           # sqrt(p) dense
            dp = np.zeros(NBINS); dp[qb] = qw ** 2      # p dense
            g = [inv[int(bb)] for bb in np.unique(qb[np.argsort(-qw)[:IDX_PEAKS]]) if int(bb) in inv]
            if not g: continue
            u, ct = np.unique(np.concatenate(g), return_counts=True)
            take = u[np.argsort(-ct)[:N_RESCORE]]
            bcs = []
            for c in take:
                rb, rw = sq[c]
                bc = float(np.dot(dq[rb], rw))
                bcs.append(bc)
                kc = keys[c]
                if bc > spec["fr_agg"][kc]: spec["fr_agg"][kc] = bc
                # chi2 over the union: cells where only one side has mass contribute that mass/2
                rbp, rp = pr[c]
                q_at_r = dp[rbp]
                d = 0.5 * float(np.sum((rp - q_at_r) ** 2 / np.maximum(rp + q_at_r, 1e-300)))
                d += 0.5 * float(1.0 - q_at_r.sum())
                simc = 1.0 - min(1.0, max(0.0, d))
                if simc > spec["chi2"][kc]: spec["chi2"][kc] = simc
            order = np.argsort(-np.asarray(bcs))[:QUOT_TOP]
            for i in order:
                c = take[i]; rb, rw = sq[c]; best = bcs[i]
                for s in EDIT_SHIFTS:
                    for sgn in (1, -1):
                        sb = rb + sgn * s
                        ok = (sb >= 0) & (sb < NBINS)
                        if ok.any():
                            v = float(np.dot(dq[sb[ok]], rw[ok]))
                            if v > best: best = v
                kc = keys[c]
                if best > spec["eshift"][kc]: spec["eshift"][kc] = best
            for kc in list(spec["fr_agg"]):
                spec["eshift"].setdefault(kc, spec["fr_agg"][kc])   # unquotiented keep plain BC
        mcand = set()
        for r in rows:
            M = neutral_mass(pmz[r], add[r])
            if np.isfinite(M) and M > 0: mcand.update(midx.window(M, PPM).tolist())
        truth = scoring.key14(smik.get(k, ""))
        for v in out:
            so = sorted(set(spec[v]) - mcand)[:TOPN]
            out[v].append(dict(k=k, cls=assign[k], truth=truth, positive=positive,
                               allmz=allmz, allit=allit,
                               mass={c: smik.get(c, "") for c in mcand},
                               spec=dict(spec[v]),
                               zsmi={c: smik.get(c, "") for c in so}))
        if n % 100 == 0: print(f"[{split}]  {n}/{len(qrows)} {time.time()-t0:.0f}s", flush=True)

    for v in out:
        p = f"{SPLITS}/pools_{v}_{split}.pkl"
        pickle.dump(out[v], open(p, "wb"))
        print(f"[{split}] wrote {os.path.basename(p)}  runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nptsseed10_n300")
