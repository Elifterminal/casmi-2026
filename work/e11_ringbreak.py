#!/usr/bin/env python3
"""E11 — does breaking RING bonds make the fragment ranker better?

E07's fragmenter only cuts acyclic single bonds. Natural products are mostly fused
rings, so most of their skeleton can never come apart in silico, and peaks from ring
cleavage (retro-Diels-Alder and friends) go unexplained for the TRUE structure as much
as for the decoys. E11 adds ring cleavage and measures it on the frozen E08 pools
(e11_pools.py), ranking by pure fragment score exactly as E10's winner (alpha=0).

Variants (all rank = -score, tie-break on key, spectral-only candidates score 0):
  V0  E07 fragmenter + E07 explain          CONTROL: must reproduce E10 exactly
  V1  new enumerator, acyclic only + E07 explain   CONTROL: must equal V0
  V1x acyclic only, exact-window explain ±1H       isolates E07's nearest-3 shortcut
  V2  + ring pairs (2 single ring bonds, same ring), E07 explain ±1H
  V3  + ring pairs, exact explain, ±ncut H per piece (ring pieces get ±2H)
  V4  V3 + (ring pair + 1 acyclic bond)
  V5  V4 with family weights: acyclic 1.0, ring 0.6, ring+acyclic 0.4
  V6  V3 + hetero-aromatic ring pairs (flavonoid C-ring RDA), exact ±ncut H
  V7  POST-HOC (added after seeing seed0): V3 with family weights
  V8  POST-HOC (added after seeing seed0): all four families, weighted (V5 + ringhet)
      Seed0 was used to pick these; ONLY seeds 1-2 count as evidence for V7/V8.

PREDICTIONS (written before the first run, 2026-09-18):
  P1  V0 reproduces E10 seed0 exactly (C1 0.4861 C2 0.5867 C3 0.0344 w 0.3552).
  P2  V1 == V0 on every query.
  P3  V2 or V3 lifts 3-split mean C2 by >= +0.02 over V0.
  P4  V4 (no weighting) does WORSE than V3: more fragments = more chance matches for
      decoys, so discrimination drops. V5 recovers some of that.
  P5  V6 >= V3 (added after the quercetin unit check showed 0 ring fragments — the
      chromone ring is aromatic in RDKit; prediction still made before any scoring run).
  SEED0 RESULT (2026-09-18): P1 PASS, P2 PASS (0 mismatches), P3 PASS (V3 C2 +0.041),
      P4 PASS (V4 < V3, V5 recovers and beats both), P5 FAIL (V6 0.3805 < V3 0.3842).
      Best V5 0.4027 weighted. Seeds 1-2 = replication.
  REPLICATION (seeds 1-2): FAILED. V5 0.3810/0.3708 vs V0 0.3915/0.3828 — below baseline.
      Paired bootstrap over all 3 splits (e11_bootstrap.py): no variant's gain over V0 clears
      zero at 95% (best V8 +0.0095 [-0.003,+0.022]); V4 is worse (-0.018 [-0.035,+0.001]),
      borderline. (First bootstrap dropped empty-pool queries and called V4 significant — corrected.)
      Seed0's +0.048 was split luck. P3 downgraded to NOT SUPPORTED. E11 = null result.
"""
import os, pickle, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e07_fragment import fragment_masses, explain, MONO, H, PROTON, TOL, TOPPEAKS, MAX_HEAVY
import scoring

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
TOPN = 25
W = (0.16, 0.45, 0.39)
FAMILY_W = {"acyc": 1.0, "ring": 0.6, "ringhet": 0.6, "ringacyc": 0.4}
MAX_ACYC_PAIRS = 40          # same cap as E07


def _components(n, adj, removed):
    seen = [False] * n; out = []
    for s in range(n):
        if seen[s]: continue
        seen[s] = True; stack = [s]; members = [s]
        while stack:
            u = stack.pop()
            for v, b in adj[u]:
                if seen[v] or b in removed: continue
                seen[v] = True; stack.append(v); members.append(v)
        out.append(members)
    return out


def fragment_table(smiles):
    """{family: set of (mass, ncut)} — neutral fragment masses by how they were cut.
    ncut = broken bonds with exactly one end in the piece (sets the H-shift allowance).
    Returns None for unparseable / oversized molecules, like E07."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None or mol.GetNumHeavyAtoms() > MAX_HEAVY:
        return None
    n = mol.GetNumAtoms()
    am = [MONO.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H for a in mol.GetAtoms()]
    ends = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(ends):
        adj[a].append((b, i)); adj[b].append((a, i))
    single = lambda b: b.GetBondType() == Chem.BondType.SINGLE
    acyc = [b.GetIdx() for b in mol.GetBonds() if single(b) and not b.IsInRing()]
    # Benzenoid rings (all-C, all-aromatic) never open. Aromatic bonds in HETERO rings can:
    # RDKit perceives the chromone C-ring of flavonoids as aromatic, and its RDA cleavage is
    # one of the commonest NP fragmentations — kept as its own family so it's measured alone.
    ri = mol.GetRingInfo()
    benzenoid = [set(r) for r, ar in zip(ri.BondRings(), ri.AtomRings())
                 if all(mol.GetAtomWithIdx(a).GetSymbol() == "C" for a in ar)
                 and all(mol.GetBondWithIdx(b).GetIsAromatic() for b in r)]
    def hetaro(i):
        return mol.GetBondWithIdx(i).GetIsAromatic() and not any(i in r for r in benzenoid)
    ring_pairs, het_pairs = set(), set()
    for ring in ri.BondRings():
        rb = [i for i in ring if single(mol.GetBondWithIdx(i)) or hetaro(i)]
        for x in range(len(rb)):
            for y in range(x + 1, len(rb)):
                pr = (min(rb[x], rb[y]), max(rb[x], rb[y]))
                both_single = all(single(mol.GetBondWithIdx(i)) for i in pr)
                (ring_pairs if both_single else het_pairs).add(pr)

    table = {"acyc": {(sum(am), 0)}, "ring": set(), "ringhet": set(), "ringacyc": set()}
    def cut(family, removed):
        comps = _components(n, adj, removed)
        if len(comps) < 2: return
        for c in comps:
            cs = set(c)
            ncut = sum(1 for i in removed if (ends[i][0] in cs) != (ends[i][1] in cs))
            table[family].add((round(sum(am[i] for i in c), 6), ncut))

    combos = [(b,) for b in acyc]
    if len(acyc) <= MAX_ACYC_PAIRS:
        combos += [(a, b) for i, a in enumerate(acyc) for b in acyc[i+1:]]
    for c in combos: cut("acyc", set(c))
    for rp in ring_pairs: cut("ring", set(rp))
    for rp in het_pairs: cut("ringhet", set(rp))
    for rp in ring_pairs:
        for b in acyc: cut("ringacyc", {rp[0], rp[1], b})
    return table


def build_index(table, families, h_by_ncut, weighted):
    """Sorted (mass, weight) arrays with H-shifts expanded, for exact-window lookup."""
    ms, ws = [], []
    for fam in families:
        w = FAMILY_W[fam] if weighted else 1.0
        for m, ncut in table[fam]:
            k = max(1, ncut) if h_by_ncut else 1
            for j in range(-k, k + 1):
                ms.append(m + j * H); ws.append(w)
    if not ms: return None
    o = np.argsort(ms)
    return np.asarray(ms)[o], np.asarray(ws)[o]


def explain_exact(mz, it, idx, positive):
    """Intensity-weighted fraction of the top peaks explained; each peak takes the best
    weight of any fragment within TOL (a true window, not E07's nearest-3 shortcut)."""
    if idx is None: return 0.0
    fm, fw = idx
    order = np.argsort(-it)[:TOPPEAKS]
    mz, it = mz[order].astype(np.float64), it[order]
    neutral = (mz - PROTON) if positive else (mz + PROTON)
    lo = np.searchsorted(fm, neutral - TOL, "left"); hi = np.searchsorted(fm, neutral + TOL, "right")
    got = np.array([fw[a:b].max() if b > a else 0.0 for a, b in zip(lo, hi)])
    s = it.sum()
    return float((got * it).sum() / s) if s > 0 else 0.0


def _masses(table, families):
    return {m for f in families for m, _ in table[f]}


def _work(smi):
    mol = Chem.MolFromSmiles(smi or "")
    return smi, fragment_masses(mol), fragment_table(smi)


def score_variants(p, frag0, tabs):
    """Per-candidate score under every variant for one query."""
    mz, it, pos = p["allmz"], p["allit"], p["positive"]
    out = defaultdict(dict)
    for c, smi in p["mass"].items():
        t = tabs[smi]
        out["V0"][c] = explain(mz, it, frag0[smi], pos)
        if t is None:
            for v in ("V1", "V1x", "V2", "V3", "V4", "V5", "V6", "V7", "V8"): out[v][c] = 0.0
            continue
        out["V1"][c] = explain(mz, it, _masses(t, ["acyc"]), pos)
        out["V1x"][c] = explain_exact(mz, it, build_index(t, ["acyc"], False, False), pos)
        out["V2"][c] = explain(mz, it, _masses(t, ["acyc", "ring"]), pos)
        out["V3"][c] = explain_exact(mz, it, build_index(t, ["acyc", "ring"], True, False), pos)
        out["V4"][c] = explain_exact(mz, it, build_index(t, ["acyc", "ring", "ringacyc"], True, False), pos)
        out["V5"][c] = explain_exact(mz, it, build_index(t, ["acyc", "ring", "ringacyc"], True, True), pos)
        out["V6"][c] = explain_exact(mz, it, build_index(t, ["acyc", "ring", "ringhet"], True, False), pos)
        out["V7"][c] = explain_exact(mz, it, build_index(t, ["acyc", "ring"], True, True), pos)
        out["V8"][c] = explain_exact(mz, it, build_index(t, ["acyc", "ring", "ringhet", "ringacyc"], True, True), pos)
    return out


def rr_of(p, scores):
    cands = {c: scores.get(c, 0.0) for c in p["mass"]}
    for c in p["zsmi"]: cands.setdefault(c, 0.0)
    smi = {**p["zsmi"], **p["mass"]}
    for i, c in enumerate(sorted(cands, key=lambda c: (-cands[c], c))[:TOPN], 1):
        if p["truth"] is not None and scoring.key14(smi[c]) == p["truth"]:
            return 1.0 / i
    return 0.0


def main(split):
    t0 = time.time()
    pools = pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))
    uniq = sorted({s for p in pools for s in p["mass"].values()})
    print(f"[{split}] {len(pools)} queries, {len(uniq)} unique candidates", flush=True)
    frag0, tabs = {}, {}
    with Pool(7) as pool:
        for i, (smi, f0, tb) in enumerate(pool.imap_unordered(_work, uniq, chunksize=8), 1):
            frag0[smi], tabs[smi] = f0, tb
            if i % 1000 == 0: print(f"[{split}]  fragmented {i}/{len(uniq)} {time.time()-t0:.0f}s", flush=True)
    print(f"[{split}] fragmentation done {time.time()-t0:.0f}s", flush=True)

    per = defaultdict(lambda: defaultdict(list)); mismatch_v1 = 0
    perq = []   # per-query rr for every variant -> paired bootstrap (E11b)
    for p in pools:
        sv = score_variants(p, frag0, tabs)
        rrs = {v: rr_of(p, sv[v]) for v in sv}
        if any(abs(sv["V0"][c] - sv["V1"][c]) > 1e-9 for c in p["mass"]): mismatch_v1 += 1
        for v, r in rrs.items(): per[v][p["cls"]].append(r)
        perq.append({"k": p["k"], "cls": p["cls"], **(rrs if p["mass"] else {})})
        if not p["mass"]:
            for v in ("V0", "V1", "V1x", "V2", "V3", "V4", "V5", "V6", "V7", "V8"): per[v][p["cls"]].append(rr_of(p, {}))

    res = {}
    lines = [f"E11 ring-breaking fragmenter — split {split}", "",
             f"{'variant':8s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>10}"]
    for v in ("V0", "V1", "V1x", "V2", "V3", "V4", "V5", "V6", "V7", "V8"):
        c = [float(np.mean(per[v][k])) if per[v][k] else 0.0 for k in (1, 2, 3)]
        w = sum(a * b for a, b in zip(W, c))
        res[v] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=w, n=[len(per[v][k]) for k in (1, 2, 3)])
        lines.append(f"{v:8s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {w:>10.4f}")
    lines += ["", f"queries where V1 per-candidate scores differ from V0: {mismatch_v1}",
              f"runtime {time.time()-t0:.0f}s"]
    print("\n".join(lines))
    import json
    json.dump(perq, open(f"{SPLITS}/E11_perquery_{split}.json", "w"))
    json.dump({"split": split, "variants": res, "v1_v0_mismatch_queries": mismatch_v1},
              open(f"{RESULTS}/E11_ringbreak_{split}.json", "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "seed0_n300")
