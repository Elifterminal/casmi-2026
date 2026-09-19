#!/usr/bin/env python3
"""E11 step 1 — freeze the unified candidate pools so rankers can be swapped cheaply.

Every experiment since E08 rebuilt the same thing: spectral index, mass index, and per
query the union of spectral + mass-window candidates. That's ~5 min per split, and the
pool does NOT depend on the ranker. So build it once per split and save it; E11+ then
only re-score candidates.

Saved per query (splits/pools_<split>.pkl, gitignored alongside the splits):
  k, cls, truth (canonical key14), positive, allmz, allit,
  mass   : {cand_key: smiles}           every mass-window candidate
  spec   : {cand_key: best cosine}      every spectral candidate (scores only)
  zsmi   : {cand_key: smiles}           the 25 smallest-key spectral-only candidates

Why zsmi is enough for pure-fragment ranking: spectral-only candidates have frag=0 and
the E10 sort is (-score, key), so the only ones that can reach the top 25 are the 25
with the smallest keys. Anything that needs spectral candidates ranked by spec score
will need a rebuild — noted, not needed for E11.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import formula_mass, neutral_mass, MassIndex
from e08_unified import prep, cosine, IDX_PEAKS, N_RESCORE, PPM, TOPN
import scoring

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")


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
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    def peaks(i): return mzf[mzo[i]:mzo[i+1]], itf[ito[i]:ito[i+1]]
    print(f"[{split}] loaded {time.time()-t0:.0f}s", flush=True)

    prepped = {}
    for r in ref_rows:
        b, w = prep(*peaks(r)); o = np.argsort(b); prepped[r] = (b[o], w[o])
    inv = defaultdict(list)
    for r in ref_rows:
        b, w = prepped[r]
        if len(b) == 0: continue
        for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]): inv[int(bb)].append(r)
    inv = {k: np.asarray(v) for k, v in inv.items()}

    struct = pd.DataFrame({"k": keys, "f": formula, "s": smiles}).drop_duplicates("k")
    massk = {k: formula_mass(f) for k, f in zip(struct["k"], struct["f"])}
    smik = dict(zip(struct["k"], struct["s"]))
    from twins import db_exclusions
    c3 = db_exclusions(sp, None)            # twin-safe splits also drop Class 3 twins
    dbk = np.array([k for k in struct["k"] if k not in c3 and np.isfinite(massk[k])])
    midx = MassIndex(dbk, np.array([massk[k] for k in dbk]))
    print(f"[{split}] indexes {time.time()-t0:.0f}s", flush=True)

    pools = []
    for n, (k, rows) in enumerate(qrows.items(), 1):
        positive = str(mode[rows[0]]).lower().startswith("pos")
        allmz = np.concatenate([peaks(r)[0] for r in rows])
        allit = np.concatenate([peaks(r)[1] for r in rows])
        spec = defaultdict(float)
        for r in rows:          # identical to E08 step 1
            b, w = prep(*peaks(r)); o = np.argsort(b); b, w = b[o], w[o]
            if len(b) == 0: continue
            cand = [inv[int(bb)] for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]) if int(bb) in inv]
            if not cand: continue
            cc = np.concatenate(cand); u, ct = np.unique(cc, return_counts=True)
            for c in u[np.argsort(-ct)[:N_RESCORE]]:
                sim = cosine(b, w, *prepped[c])
                if sim > 0: spec[keys[c]] = max(spec[keys[c]], sim)
        mcand = set()
        for r in rows:          # identical to E08 step 2
            M = neutral_mass(pmz[r], add[r])
            if np.isfinite(M) and M > 0: mcand.update(midx.window(M, PPM).tolist())
        spec_only = sorted(set(spec) - mcand)[:TOPN]
        pools.append(dict(k=k, cls=assign[k], truth=scoring.key14(smik.get(k, "")),
                          positive=positive, allmz=allmz, allit=allit,
                          mass={c: smik.get(c, "") for c in mcand},
                          spec=dict(spec),
                          zsmi={c: smik.get(c, "") for c in spec_only}))
        if n % 100 == 0: print(f"[{split}]  {n}/{len(qrows)} {time.time()-t0:.0f}s", flush=True)

    out = f"{SPLITS}/pools_{split}.pkl"
    pickle.dump(pools, open(out, "wb"))
    sizes = [len(p["mass"]) for p in pools]
    print(f"[{split}] wrote {out}: {len(pools)} queries, mass pool median {np.median(sizes):.0f} "
          f"max {max(sizes)}  runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "seed0_n300")
