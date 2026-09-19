#!/usr/bin/env python3
"""Tautomer twins: structures the data stores under different InChIKey14s but the scorer
treats as the SAME molecule (it tautomer-canonicalises before taking the InChIKey14).

E17 found the harness leaking through them: it held out by InChIKey14, so a Class 2 answer's
twin kept its spectra in the reference index (19/405 NP C2 queries), and a Class 3 answer's
twin could stay in the candidate database. Everything that holds a structure OUT must hold
out its whole twin group.

Twins share a molecular formula, so only structures in multi-key formula groups need the
(expensive) canonicalisation. The training-set map is computed once and cached.
"""
import json, os
from multiprocessing import Pool
import pandas as pd
import scoring

CACHE = os.path.expanduser("~/casmi-2026/work/splits/train_key14.json")


def _k14(item):
    k, smi = item
    return k, scoring.key14(smi)


def canon_map(keys, smiles, formulas, procs=7):
    """{InChIKey14: scorer key14} for every structure whose formula has >1 structure.
    Singletons map to themselves -- a lone structure can't have a twin in this set."""
    df = pd.DataFrame({"k": keys, "s": smiles, "f": formulas}).drop_duplicates("k")
    multi = df[df.groupby("f").k.transform("size") > 1]
    with Pool(procs) as pool:
        got = dict(pool.map(_k14, list(zip(multi.k, multi.s)), chunksize=256))
    out = {k: k for k in df.k}
    out.update({k: v for k, v in got.items() if v is not None})
    return out


def train_canon(df=None):
    """Cached training-set map. df needs inchikey14 / normalized_smiles / molecular_formula."""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    m = canon_map(df["inchikey14"].to_numpy(), df["normalized_smiles"].to_numpy(),
                  df["molecular_formula"].to_numpy())
    json.dump(m, open(CACHE, "w"))
    return m


def groups(canon):
    """scorer key14 -> set of InChIKey14s sharing it."""
    g = {}
    for k, c in canon.items(): g.setdefault(c, set()).add(k)
    return g


def external_twins(truths, ext_keys, ext_smiles, ext_formulas):
    """Keys in an EXTERNAL database (e.g. COCONUT) that are twins of any truth.
    truths: {InChIKey14: (scorer key14, formula)}. Only same-formula entries are canonicalised."""
    want = {}
    for k, (c, f) in truths.items(): want.setdefault(f, set()).add(c)
    hits = set()
    for k, s, f in zip(ext_keys, ext_smiles, ext_formulas):
        if f in want and k not in truths and scoring.key14(s) in want[f]:
            hits.add(k)
    return hits


def db_exclusions(sp, tr, co=None):
    """Everything a retrieval database must drop for this split's Class 3 answers.
    Legacy splits: the Class 3 InChIKey14s only (reproduces pre-E17 numbers exactly).
    Twin-safe splits: plus their training-side twins (written by the harness as db_exclude)
    and any COCONUT entry that is the same molecule under the scorer key.
    tr / co: frames with columns k, s, f (as returned by e15_coconut_pools.load_structures)."""
    c3 = {k for k, c in sp["assign"].items() if c == 3}
    if not sp.get("twin_safe"):
        return c3
    out = c3 | set(sp.get("db_exclude", []))
    if co is not None:
        canon = train_canon()
        trf = dict(zip(tr.k, tr.f))
        truths = {k: (canon.get(k, k), trf[k]) for k in c3 if k in trf}
        cf = set(f for _, f in truths.values())
        sub = co[co.f.isin(cf)]
        out |= external_twins(truths, sub.k.to_numpy(), sub.s.to_numpy(), sub.f.to_numpy())
    return out
