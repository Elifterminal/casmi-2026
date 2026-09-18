#!/usr/bin/env python3
"""E15 — the E10 pipeline re-measured on the candidate pools a real submission would face.

Every score since E04 retrieved mass candidates from the TRAINING structures (median C2 pool
11). A submission retrieves from COCONUT and/or the training structures, and E14b showed
those pools are ~2x bigger (19 / 39). E04/E05 showed bigger pools hurt ranking. So re-run
pure-fragment ranking (E10's winner, alpha=0) with the mass-retrieval database swapped:

  train          training structures minus Class 3 (CONTROL: must reproduce E10 per split)
  coconut+truth  COCONUT-2026-09 plus the C1/C2 truths injected (simulates ~99.6% coverage)
  union          COCONUT + training structures (the realistic submission database)

Class 3 truths are excluded from every database (by definition in no database). Spectral
candidates are unchanged (frozen E08 pools). Candidates are ranked by E07's fragment score,
tie-break on key, spectral-only candidates score 0 -- exactly E10.

Class 2 is also broken down by whether the truth is natively in COCONUT: the real test is
natural products, most of our manufactured C2 are not (enveda-180), and COCONUT decoys may be
easier to tell apart from a synthetic truth than from an NP one.

PREDICTIONS (before the run, 2026-09-18):
  P1  'train' reproduces E10 exactly on all three splits (0.3552 / 0.3915 / 0.3828).
  P2  'union' lowers the 3-split weighted mean by 0.03-0.08 (pools ~3.5x the size).
  P3  Within C2, the drop is larger for truths natively in COCONUT (NP decoys look alike).
  RESULT (2026-09-18): P1 PASS (train == E10 on all 3 splits, to 4 dp). P2 PASS: union
      0.3019 (-0.0746 [-0.088,-0.063]); coconut+truth 0.3418 (-0.035 [-0.052,-0.017]).
      P3 PASS, strongly: C2 NP-native truths 0.566 -> 0.389 (union); non-native 0.637 -> 0.565.
      ** The realistic pipeline is BELOW the 0.339 gate. Kill-gate local-CV arm UN-cleared. **
"""
import json, os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from candidates import formula_mass, neutral_mass, MassIndex
from e07_fragment import fragment_masses, explain
from e08_unified import PPM, TOPN
import scoring

DATA = os.path.expanduser("~/casmi-2026/work/data/train.parquet")
COCO = os.path.expanduser("~/casmi-2026/work/data/coconut/coconut_csv_lite-09-2026.csv")
SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
W = {1: 0.16, 2: 0.45, 3: 0.39}
DBS = ("train", "coconut+truth", "union")


def _frag(smi):
    return smi, fragment_masses(Chem.MolFromSmiles(smi or ""))


def load_structures():
    t = pq.read_table(DATA, columns=["inchikey14", "normalized_smiles", "molecular_formula",
                                     "precursor_mz", "adduct"])
    keys = np.asarray(t.column("inchikey14")); pmz = t.column("precursor_mz").to_numpy()
    add = np.asarray(t.column("adduct"))
    tr = pd.DataFrame({"k": keys, "s": np.asarray(t.column("normalized_smiles")),
                       "f": np.asarray(t.column("molecular_formula"))}).drop_duplicates("k")
    tr["m"] = tr.f.map(formula_mass)
    co = pd.read_csv(COCO, usecols=["standard_inchi_key", "canonical_smiles", "molecular_formula"], dtype=str).dropna()
    co["k"] = co.standard_inchi_key.str.split("-").str[0]
    co = co.drop_duplicates("k").rename(columns={"canonical_smiles": "s", "molecular_formula": "f"})
    co["m"] = co.f.map(formula_mass)
    bad = int(co.m.isna().sum())
    return keys, pmz, add, tr[np.isfinite(tr.m)], co[np.isfinite(co.m)], bad


def main():
    t0 = time.time()
    keys, pmz, add, tr, co, bad = load_structures()
    trm, trs = dict(zip(tr.k, tr.m)), dict(zip(tr.k, tr.s))
    com, cos_ = dict(zip(co.k, co.m)), dict(zip(co.k, co.s))
    print(f"loaded {time.time()-t0:.0f}s  training {len(trm):,}  coconut {len(com):,} "
          f"(dropped {bad:,} COCONUT rows our formula parser can't read)", flush=True)

    per = []            # per-query rr under each DB, for the paired bootstrap
    frag = {}
    pool_sizes = defaultdict(list)
    for split in ("seed0_n300", "seed1_n300", "seed2_n300"):
        sp = json.load(open(f"{SPLITS}/split_{split}.json")); assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))}
        c3 = {k for k, c in assign.items() if c == 3}
        truths = {k for k, c in assign.items() if c in (1, 2) and k in trm}
        dbm = {"train": {k: m for k, m in trm.items() if k not in c3},
               "coconut+truth": {**{k: m for k, m in com.items() if k not in c3}, **{k: trm[k] for k in truths}},
               "union": {**{k: m for k, m in com.items() if k not in c3}, **{k: m for k, m in trm.items() if k not in c3}}}
        smi = {**cos_, **trs}          # training SMILES win on a shared key
        idx = {d: MassIndex(np.array(list(v)), np.array(list(v.values()))) for d, v in dbm.items()}

        cands = {}
        for k, rows in qrows.items():
            for d in DBS:
                s = set()
                for r in rows:          # identical to E08 step 2
                    M = neutral_mass(pmz[r], add[r])
                    if np.isfinite(M) and M > 0: s.update(idx[d].window(M, PPM).tolist())
                cands[(k, d)] = s
                pool_sizes[(assign[k], d)].append(len(s))
        need = sorted({smi[c] for s in cands.values() for c in s} - set(frag))
        with Pool(7) as pool:
            for sm, f in pool.imap_unordered(_frag, need, chunksize=16): frag[sm] = f
        print(f"[{split}] pools built, fragmented {len(need):,} new candidates {time.time()-t0:.0f}s", flush=True)

        for k in qrows:
            p = pools[k]; row = {"split": split, "k": k, "cls": assign[k], "native": k in com}
            for d in DBS:
                mc = cands[(k, d)]
                sc = {c: explain(p["allmz"], p["allit"], frag[smi[c]], p["positive"]) for c in mc}
                for c in sorted(set(p["spec"]) - mc)[:TOPN]: sc.setdefault(c, 0.0)
                rr = 0.0
                for i, c in enumerate(sorted(sc, key=lambda c: (-sc[c], c))[:TOPN], 1):
                    if p["truth"] is not None and scoring.key14(smi.get(c, "")) == p["truth"]:
                        rr = 1.0 / i; break
                row[d] = rr
            per.append(row)
        print(f"[{split}] scored {time.time()-t0:.0f}s", flush=True)
    report(per, pool_sizes, t0)


def wmean(rows, d):
    return sum(W[c] * np.mean([r[d] for r in rows if r["cls"] == c]) for c in (1, 2, 3))


def report(per, pool_sizes, t0):
    L = ["E15 — E10 pipeline (pure fragment) on real-database candidate pools", "",
         "median / mean mass-pool size:"]
    for c in (1, 2, 3):
        L.append(f"  C{c}  " + "   ".join(f"{d} {np.median(pool_sizes[(c, d)]):.0f}/{np.mean(pool_sizes[(c, d)]):.0f}" for d in DBS))
    L += ["", f"{'split':12s} {'db':14s} {'C1':>7} {'C2':>7} {'C3':>7} {'weighted':>9}"]
    out = {}
    for s in ("seed0_n300", "seed1_n300", "seed2_n300", "ALL"):
        rows = per if s == "ALL" else [r for r in per if r["split"] == s]
        for d in DBS:
            c = [np.mean([r[d] for r in rows if r["cls"] == k]) for k in (1, 2, 3)]
            w = wmean(rows, d); out[f"{s}/{d}"] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=w)
            L.append(f"{s:12s} {d:14s} {c[0]:>7.4f} {c[1]:>7.4f} {c[2]:>7.4f} {w:>9.4f}")
    L += ["", "3-split mean weighted: " + "   ".join(
        f"{d} {np.mean([out[f'{s}/{d}']['weighted'] for s in ('seed0_n300', 'seed1_n300', 'seed2_n300')]):.4f}" for d in DBS)]
    rng = np.random.default_rng(0)
    by = {c: [r for r in per if r["cls"] == c] for c in (1, 2, 3)}
    for d in DBS[1:]:
        diff = lambda idx: sum(W[c] * np.mean([by[c][i][d] - by[c][i]["train"] for i in idx[c]]) for c in by)
        full = diff({c: np.arange(len(by[c])) for c in by})
        boots = [diff({c: rng.integers(0, len(by[c]), len(by[c])) for c in by}) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        L.append(f"paired Δweighted {d} − train: {full:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    L += ["", "Class 2 by whether the truth is natively in COCONUT:"]
    for nat in (True, False):
        rows = [r for r in by[2] if r["native"] == nat]
        if rows:
            L.append(f"  native={str(nat):5s} n={len(rows):3d}  " + "   ".join(f"{d} {np.mean([r[d] for r in rows]):.4f}" for d in DBS))
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E15_coconut_pools_2026-09-18.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E15_coconut_pools_3seeds.json", "w"), indent=2)
    json.dump(per, open(f"{SPLITS}/E15_perquery.json", "w"))


if __name__ == "__main__":
    main()
