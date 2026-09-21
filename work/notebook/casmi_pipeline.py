"""CASMI 2026 inference pipeline — the E18 ranker as one self-contained module.

Ships inside the Kaggle dataset and runs offline. Mirrors, step for step, what E18 measured
(e08 spectral gather, E15/E17 COCONUT mass retrieval, E07 fragments, E17 features, E18 learned
weights), so the local port-equivalence check can hold it to E18's numbers exactly.

Per test molecule (all its spectra fused):
  1. spectral hits  cosine of each spectrum against the training reference index; best per structure
  2. mass pool      COCONUT structures within 10 ppm of the neutral mass (adduct is given)
  3. features       fragment explain / chance-corrected / single-break, neighbour vote over the
                    top-50 spectral hits, the candidate's own spectral match, heavy-atom count
  4. rank           learned linear score, tie-break on key; then spectral-only hits by key; top 25
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import inchi, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# ---- constants: identical to the experiments (e07 / e08 / e17) ---------------------------
BIN, INT_FLOOR, MAX_PEAKS, IDX_PEAKS = 0.02, 0.002, 256, 20
PPM, N_RESCORE, TOPN = 10, 800, 25
PROTON, H = 1.00727646688, 1.007825032
TOL, TOPPEAKS, MAX_HEAVY = 0.01, 30, 60
NB_TOP = 50
FEATS = ["explain", "chance_corr", "single", "nb_max", "nb_mean10", "spec_self", "heavy"]
FEATS2 = FEATS + ["is_generated"]        # E23: retrieved and generated candidates, calibrated together
MAX_EACH = 10                             # relatives kept per search method (cosine / modcos / nlcos)
SLOW_ROWS, SLOW_REFS = 3, 300             # budget for the two expensive searches, per molecule
NPK, MZ_TOL = 50, 0.01                    # peaks per spectrum, and the match window, for modcos/nlcos
GAP_TOL, MAX_SITES = 0.003, 20            # matching a mass gap to an edit, and products per edit

_E = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.9949146196,
      "S": 31.97207100, "P": 30.97376163, "Cl": 34.96885268, "Br": 78.9183371,
      "F": 18.99840322, "I": 126.904473, "Si": 27.9769265325, "Se": 79.9165213,
      "Na": 22.9897692809, "K": 38.96370668, "B": 11.0093054, "As": 74.9215965,
      "Fe": 55.9349375, "D": 2.0141017778}
_ELECTRON = 0.00054857990946
_PROTON_M = _E["H"] - _ELECTRON
_H2O = 2 * _E["H"] + _E["O"]
ADDUCT_SHIFT = {
    "[M+H]+": _PROTON_M, "[M+NH4]+": _E["N"] + 4 * _E["H"] - _ELECTRON,
    "[M-H2O+H]+": _PROTON_M - _H2O, "[M-2H2O+H]+": _PROTON_M - 2 * _H2O,
    "[M+Na]+": _E["Na"] - _ELECTRON, "[M+K]+": _E["K"] - _ELECTRON,
    "[M-H]-": -_PROTON_M, "[M-H2O-H]-": -_PROTON_M - _H2O,
    "[M+CH2O2-H]-": _E["C"] + 2 * _E["H"] + 2 * _E["O"] - _PROTON_M,
    "[M+Cl]-": _E["Cl"] + _ELECTRON,
}
# fragment masses use the e07 table (no As/Fe/D) -- kept separate so scores match exactly
_MONO = {k: _E[k] for k in ("C", "H", "N", "O", "S", "P", "Cl", "Br", "F", "I", "Si", "Se", "B", "Na", "K")}
_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")
_FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_ENUM = rdMolStandardize.TautomerEnumerator()


# ---- masses ---------------------------------------------------------------------------
def formula_mass(f) -> float:
    if not f: return np.nan
    f = str(f).strip()
    if not f or not f[0].isupper(): return np.nan
    m, pos = 0.0, 0
    for sym, cnt in _TOK.findall(f):
        if sym not in _E: return np.nan
        m += _E[sym] * (int(cnt) if cnt else 1); pos += len(sym) + len(cnt)
    return m if pos == len(f) else np.nan


def neutral_mass(mz: float, adduct: str) -> float:
    s = ADDUCT_SHIFT.get(adduct)
    return np.nan if s is None else mz - s


class MassIndex:
    def __init__(self, keys, masses):
        masses = np.asarray(masses, float); ok = np.isfinite(masses)
        keys = np.asarray(keys)[ok]; masses = masses[ok]; o = np.argsort(masses)
        self.keys, self.mass = keys[o], masses[o]

    def window(self, M: float, ppm: float = PPM):
        tol = M * ppm / 1e6
        return self.keys[np.searchsorted(self.mass, M - tol, "left"):np.searchsorted(self.mass, M + tol, "right")]


# ---- spectral index (e08) -------------------------------------------------------------
def prep(mz, it, sqrt_p=False):
    """Bin a spectrum and normalise it for the dot product that follows.

    sqrt_p=True is Vanta's ground (E24): treat the spectrum as a probability distribution over
    bins and take its square root, which puts it on the unit sphere -- the dot product is then
    the Bhattacharyya coefficient and the angle is Fisher-Rao distance. Note sqrt(p) is already
    L2-normalised, since ||sqrt(p)||^2 = sum(p) = 1, so this is literally our own cosine applied
    to square-rooted intensities. E24: finds better relatives, +0.029 mean Tanimoto.
    """
    if len(mz) == 0: return np.empty(0, np.int32), np.empty(0, np.float32)
    it = np.asarray(it, np.float32); mz = np.asarray(mz, np.float64); mx = it.max()
    if mx <= 0: return np.empty(0, np.int32), np.empty(0, np.float32)
    keep = it >= INT_FLOOR * mx; mz, it = mz[keep], it[keep]
    if len(it) > MAX_PEAKS:
        top = np.argpartition(-it, MAX_PEAKS)[:MAX_PEAKS]; mz, it = mz[top], it[top]
    b = np.rint(mz / BIN).astype(np.int32)
    if sqrt_p: it = np.sqrt(it)
    return b, (it / (np.linalg.norm(it) + 1e-12)).astype(np.float32)


def cosine(qb, qw, rb, rw) -> float:
    if len(qb) == 0 or len(rb) == 0: return 0.0
    i = j = 0; s = 0.0
    while i < len(qb) and j < len(rb):
        if qb[i] == rb[j]: s += qw[i] * rw[j]; i += 1; j += 1
        elif qb[i] < rb[j]: i += 1
        else: j += 1
    return float(s)


class SpectralIndex:
    """Reference spectra (training rows) -> per-structure best cosine for a query spectrum."""

    def __init__(self, ref_keys, peaks, rows):
        self.keys = ref_keys
        self.prepped, inv = {}, defaultdict(list)
        for r in rows:
            b, w = prep(*peaks(r)); o = np.argsort(b); self.prepped[r] = (b[o], w[o])
        for r in rows:
            b, w = self.prepped[r]
            if len(b) == 0: continue
            for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]): inv[int(bb)].append(r)
        self.inv = {k: np.asarray(v) for k, v in inv.items()}

    def hits(self, spectra) -> dict:
        spec = defaultdict(float)
        for mz, it in spectra:
            b, w = prep(mz, it); o = np.argsort(b); b, w = b[o], w[o]
            if len(b) == 0: continue
            cand = [self.inv[int(bb)] for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]) if int(bb) in self.inv]
            if not cand: continue
            u, ct = np.unique(np.concatenate(cand), return_counts=True)
            for c in u[np.argsort(-ct)[:N_RESCORE]]:
                sim = cosine(b, w, *self.prepped[c])
                if sim > 0: spec[self.keys[c]] = max(spec[self.keys[c]], sim)
        return dict(spec)


# ---- candidate structure (e07 / e17) --------------------------------------------------
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


def fragment_masses(mol, max_break=2):
    """E07: neutral masses of pieces from breaking 1-2 acyclic single bonds."""
    if mol is None or mol.GetNumHeavyAtoms() > MAX_HEAVY: return None
    acyclic = [b.GetIdx() for b in mol.GetBonds()
               if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
    N = mol.GetNumAtoms()
    am = [_MONO.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H for a in mol.GetAtoms()]
    masses = {sum(am)}
    combos = [(b,) for b in acyclic]
    if max_break >= 2 and len(acyclic) <= 40:
        combos += [(a, b) for i, a in enumerate(acyclic) for b in acyclic[i + 1:]]
    for bonds in combos:
        try:
            frg = Chem.FragmentOnBonds(mol, list(bonds), addDummies=True)
            for idxs in Chem.GetMolFrags(frg, asMols=False):
                masses.add(sum(am[i] for i in idxs if i < N))
        except Exception:
            continue
    return masses


def candidate_info(smi):
    """Query-independent facts about one structure; None if RDKit can't parse it."""
    mol = Chem.MolFromSmiles(smi or "")
    if mol is None: return smi, None
    single = None
    if mol.GetNumHeavyAtoms() <= MAX_HEAVY:
        n = mol.GetNumAtoms()
        am = [_MONO.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H for a in mol.GetAtoms()]
        adj = [[] for _ in range(n)]
        for b in mol.GetBonds():
            adj[b.GetBeginAtomIdx()].append((b.GetEndAtomIdx(), b.GetIdx()))
            adj[b.GetEndAtomIdx()].append((b.GetBeginAtomIdx(), b.GetIdx()))
        single = {sum(am)}
        for b in mol.GetBonds():
            if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing():
                for c in _components(n, adj, {b.GetIdx()}): single.add(sum(am[i] for i in c))
    return smi, dict(full=fragment_masses(mol), single=single, heavy=mol.GetNumHeavyAtoms(),
                     fp=_FPG.GetFingerprint(mol))


def explain(mz, it, frag_masses, positive) -> float:
    """E07: intensity-weighted share of the top peaks matching a fragment (+-1 H)."""
    if not frag_masses: return 0.0
    order = np.argsort(-it)[:TOPPEAKS]; mz, it = mz[order], it[order]
    fm = np.array(sorted(frag_masses)); matched = wsum = 0.0
    for m, i in zip(mz, it):
        wsum += i
        neutral = (m - PROTON) if positive else (m + PROTON)
        j = np.searchsorted(fm, neutral)
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(fm) and min(abs(neutral - fm[k]), abs(neutral - fm[k] - H),
                                        abs(neutral - fm[k] + H)) <= TOL:
                matched += i; break
    return matched / wsum if wsum > 0 else 0.0


@lru_cache(maxsize=1_000_000)
def key14(smiles):
    """The scorer's identity for a structure: tautomer-canonicalise, then InChIKey first block.
    Used only to drop guesses that are the same molecule (a repeat is a wasted slot)."""
    if not smiles: return None
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    try: m = _ENUM.Canonicalize(m)
    except Exception: pass
    try: k = inchi.MolToInchiKey(m)
    except Exception: return None
    return k.split("-")[0] if k else None


def dedupe(cands, smi_of, limit=TOPN):
    """Keep the first of each scorer-identical guess, so 25 slots hold 25 distinct answers."""
    seen, out = set(), []
    for c in cands:
        k = key14(smi_of(c))
        if k is not None and k in seen: continue
        if k is not None: seen.add(k)
        out.append(c)
        if len(out) == limit: break
    return out


def chance(frag, M) -> float:
    if not frag or not np.isfinite(M) or M <= 0: return 0.0
    return min(1.0, 3 * len(frag) * 2 * TOL / M)


# ---- relatives: three searches, because they find different molecules (E13) --------------
def top_peaks(mz, it, n=NPK):
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    if len(it) == 0 or it.max() <= 0: return np.empty(0), np.empty(0)
    o = np.argsort(-it)[:n]; mz, it = mz[o], it[o]
    w = np.sqrt(it); w /= np.linalg.norm(w) + 1e-12
    s = np.argsort(mz)
    return mz[s], w[s]


def greedy_cos(qm, qw, rm, rw, shift=None):
    """Greedy peak-matched cosine; with shift, a query peak may also match ref+shift."""
    if len(qm) == 0 or len(rm) == 0: return 0.0
    pairs = []
    for off in ((0.0,) if shift is None else (0.0, shift)):
        lo = np.searchsorted(rm + off, qm - MZ_TOL, "left"); hi = np.searchsorted(rm + off, qm + MZ_TOL, "right")
        for i, (a, b) in enumerate(zip(lo, hi)):
            for j in range(a, b): pairs.append((qw[i] * rw[j], i, j))
    if not pairs: return 0.0
    pairs.sort(reverse=True); ui, uj = set(), set(); s = 0.0
    for sc, i, j in pairs:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s += sc
    return s


def find_relatives(spectra, spec, sindex, peaks, pmz, cache):
    """Union of the top MAX_EACH structures from cosine, modified-cosine and neutral-loss search."""
    cos_hits = [h for h, _ in sorted(spec.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_EACH]]
    mod, nl = defaultdict(float), defaultdict(float)
    for mz, it, _, qp, _ in spectra[:SLOW_ROWS]:
        b, w = prep(mz, it); o = np.argsort(b); b, w = b[o], w[o]
        if len(b) == 0: continue
        gather = [sindex.inv[int(bb)] for bb in np.unique(b[np.argsort(-w)[:IDX_PEAKS]]) if int(bb) in sindex.inv]
        if not gather: continue
        u, ct = np.unique(np.concatenate(gather), return_counts=True)
        qm, qw = top_peaks(mz, it); qnl = top_peaks(np.asarray(qp) - np.asarray(mz), it)
        for c in u[np.argsort(-ct)[:SLOW_REFS]]:
            if c not in cache:
                cache[c] = (top_peaks(*peaks(c)), top_peaks(pmz[c] - peaks(c)[0], peaks(c)[1]))
            (rm, rw), rnl = cache[c]
            kc = sindex.keys[c]
            mod[kc] = max(mod[kc], greedy_cos(qm, qw, rm, rw, shift=qp - pmz[c]))
            nl[kc] = max(nl[kc], greedy_cos(*qnl, *rnl))
    pick = list(cos_hits)
    for d in (mod, nl):
        pick += [h for h, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_EACH]]
    seen, rel = set(), []
    for h in pick:
        if h in seen: continue
        seen.add(h); rel.append(h)
    return rel


def generate(relatives, M, smi_of, exact_cache, pool=None):
    """Edit each relative to close the mass gap; returns the distinct product SMILES."""
    from e20_edits import candidates_for
    jobs = []
    for h in relatives:
        smi = smi_of(h)
        if not smi: continue
        if smi not in exact_cache:
            m = Chem.MolFromSmiles(smi)
            exact_cache[smi] = Descriptors.ExactMolWt(m) if m is not None else float("nan")
        rm = exact_cache[smi]
        if not np.isfinite(rm) or not np.isfinite(M) or abs(M - rm) < 1e-6: continue
        jobs.append((smi, M - rm))
    out = set()
    for smi, gap in jobs:
        out.update(s for s, _ in candidates_for(smi, gap, tol=GAP_TOL, max_sites=MAX_SITES))
    return sorted(out)


# ---- one molecule ---------------------------------------------------------------------
def features(q, cands, spec, info, smi_of):
    """q: dict(allmz, allit, positive, M). Returns {cand_key: feature dict}."""
    hits = sorted(spec.items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
    good = [(h, c) for h, c in hits if info.get(smi_of(h)) is not None]
    hfp = [info[smi_of(h)]["fp"] for h, _ in good]
    hcos = np.array([c for _, c in good])
    out = {}
    for c in cands:
        inf = info.get(smi_of(c))
        if inf is None: continue
        e = explain(q["allmz"], q["allit"], inf["full"], q["positive"])
        if hfp:
            v = hcos * np.array(DataStructs.BulkTanimotoSimilarity(inf["fp"], hfp))
            nbm, nb10 = float(v.max()), float(v[:10].mean())
        else:
            nbm = nb10 = 0.0
        out[c] = dict(explain=e, chance_corr=e - chance(inf["full"], q["M"]),
                      single=explain(q["allmz"], q["allit"], inf["single"], q["positive"]),
                      nb_max=nbm, nb_mean10=nb10, spec_self=spec.get(c, 0.0), heavy=inf["heavy"])
    return out


def rank(feats, spec, cands, weights, limit=TOPN):
    """Learned score over mass candidates, tie-break on key, then spectral-only hits by key.
    limit > 25 returns spare candidates, so the caller can drop duplicates and still fill 25."""
    w = np.array([weights[f] for f in FEATS])
    sc = {c: float(np.dot(w, [f[x] for x in FEATS])) for c, f in feats.items()}
    ranked = sorted(sc, key=lambda c: (-sc[c], c))
    return (ranked + sorted(set(spec) - set(cands))[:limit])[:limit]


def features2(q, cands, gen, spec, info, smi_of):
    """Feature vectors for both channels: retrieved candidates and generated ones (E23)."""
    hits = sorted(spec.items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]
    good = [(h, c) for h, c in hits if info.get(smi_of(h)) is not None]
    hfp = [info[smi_of(h)]["fp"] for h, _ in good]
    hcos = np.array([c for _, c in good])

    def vec(smi, spec_self, is_gen):
        inf = info.get(smi)
        if inf is None: return None
        e = explain(q["allmz"], q["allit"], inf["full"], q["positive"])
        if hfp:
            v = hcos * np.array(DataStructs.BulkTanimotoSimilarity(inf["fp"], hfp))
            nbm, nb10 = float(v.max()), float(v[:10].mean())
        else:
            nbm = nb10 = 0.0
        f = dict(explain=e, chance_corr=e - chance(inf["full"], q["M"]),
                 single=explain(q["allmz"], q["allit"], inf["single"], q["positive"]),
                 nb_max=nbm, nb_mean10=nb10, spec_self=spec_self, heavy=inf["heavy"],
                 is_generated=float(is_gen))
        return np.array([f[x] for x in FEATS2])

    out = {}
    for c in cands:
        v = vec(smi_of(c), spec.get(c, 0.0), 0)
        if v is not None: out[c] = (v, smi_of(c))
    for smi in gen:
        v = vec(smi, 0.0, 1)
        if v is not None: out[f"gen:{smi}"] = (v, smi)
    return out


def rank2(cand, spec, cands, weights, limit=TOPN):
    """Rank both channels together, then spectral-only hits; duplicates dropped by scorer key."""
    w = np.array([weights[f] for f in FEATS2])
    sc = {c: float(np.dot(w, v)) for c, (v, _) in cand.items()}
    order = [cand[c][1] for c in sorted(sc, key=lambda c: (-sc[c], c))]
    order += [None] * 0
    return order, sorted(set(spec) - set(cands))[:limit]


def characterise(smiles, procs=4, cache=None):
    """candidate_info for many SMILES in parallel (cache dict is extended in place)."""
    cache = {} if cache is None else cache
    need = sorted(set(smiles) - set(cache))
    if procs > 1 and len(need) > 200:
        with Pool(procs) as pool:
            for s, inf in pool.imap_unordered(candidate_info, need, chunksize=16): cache[s] = inf
    else:
        for s in need: cache[s] = candidate_info(s)[1]
    return cache


def predict(molecules, sindex, midx, smi_of, weights, procs=4, log=print, limit=TOPN):
    """molecules: list of dict(id, spectra=[(mz, it, adduct, precursor_mz, mode)]).
    Returns {id: [candidate keys, best first]} -- map keys to SMILES with smi_of."""
    prepared = []
    for m in molecules:
        sp = m["spectra"]
        spec = sindex.hits([(mz, it) for mz, it, *_ in sp])
        cands, Ms = set(), []
        for mz, it, add, pmz, mode in sp:
            M = neutral_mass(pmz, add)
            if np.isfinite(M) and M > 0: cands.update(midx.window(M).tolist()); Ms.append(M)
        q = dict(allmz=np.concatenate([np.asarray(s[0], np.float32) for s in sp]),
                 allit=np.concatenate([np.asarray(s[1], np.float32) for s in sp]),
                 positive=str(sp[0][4]).lower().startswith("pos"),
                 M=float(np.median(Ms)) if Ms else np.nan)
        prepared.append((m["id"], q, cands, spec))
    log(f"  spectral hits + mass pools for {len(prepared)} molecules")
    need = {smi_of(c) for _, _, cands, _ in prepared for c in cands}
    need |= {smi_of(h) for *_, spec in prepared
             for h, _ in sorted(spec.items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]}
    info = characterise(need, procs)
    log(f"  characterised {len(info):,} structures")
    return {mid: rank(features(q, cands, spec, info, smi_of), spec, cands, weights, limit)
            for mid, q, cands, spec in prepared}


def _gen_job(args):
    from e20_edits import candidates_for
    smi, gap = args
    return [s for s, _ in candidates_for(smi, gap, tol=GAP_TOL, max_sites=MAX_SITES)]


def predict2(molecules, sindex, midx, smi_of, weights, peaks, pmz, procs=4, log=print, limit=TOPN,
             generate=True):
    """E23 pipeline: retrieval + analog generation, ranked together.

    molecules: list of dict(id, spectra=[(mz, it, adduct, precursor_mz, mode)]).
    Returns {id: [SMILES, best first]} -- already deduplicated by the scorer's key.
    """
    prepared, jobs, exact = [], [], {}
    tpc = {}
    for m in molecules:
        sp = m["spectra"]
        spec = sindex.hits([(mz, it) for mz, it, *_ in sp])
        cands, Ms = set(), []
        for mz, it, add, p_, mode in sp:
            M = neutral_mass(p_, add)
            if np.isfinite(M) and M > 0: cands.update(midx.window(M).tolist()); Ms.append(M)
        M = float(np.median(Ms)) if Ms else np.nan
        q = dict(allmz=np.concatenate([np.asarray(s[0], np.float32) for s in sp]),
                 allit=np.concatenate([np.asarray(s[1], np.float32) for s in sp]),
                 positive=str(sp[0][4]).lower().startswith("pos"), M=M)
        # generate=False is the A/B arm: identical ranking, generation channel switched off.
        # Local: 0.3663 with generation, 0.3519 without (E23).
        rel = find_relatives(sp, spec, sindex, peaks, pmz, tpc) if (generate and np.isfinite(M)) else []
        pend = []
        for h in rel:
            smi = smi_of(h)
            if not smi: continue
            if smi not in exact:
                mm = Chem.MolFromSmiles(smi)
                exact[smi] = Descriptors.ExactMolWt(mm) if mm is not None else float("nan")
            rm = exact[smi]
            if np.isfinite(rm) and abs(M - rm) > 1e-6: pend.append((smi, M - rm))
        jobs.append(pend)
        prepared.append((m["id"], q, cands, spec))
    log(f"  relatives + pools for {len(prepared)} molecules")

    flat = []
    allj = [j for pend in jobs for j in pend]
    if allj:
        if procs > 1:
            with Pool(procs) as pool: flat = pool.map(_gen_job, allj, chunksize=4)
        else:
            flat = [_gen_job(j) for j in allj]
    gen_per, i = [], 0
    for pend in jobs:
        got = set()
        for _ in pend: got.update(flat[i]); i += 1
        gen_per.append(sorted(got))
    log(f"  generated {sum(len(g) for g in gen_per):,} candidates")

    need = {smi_of(c) for _, _, cands, _ in prepared for c in cands}
    need |= {smi_of(h) for *_, spec in prepared
             for h, _ in sorted(spec.items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]}
    need |= {s for g in gen_per for s in g}
    info = characterise(need, procs)
    log(f"  characterised {len(info):,} structures")

    w = np.array([weights[f] for f in FEATS2])
    out = {}
    for (mid, q, cands, spec), gen in zip(prepared, gen_per):
        cand = features2(q, cands, gen, spec, info, smi_of)
        sc = {c: float(np.dot(w, v)) for c, (v, _) in cand.items()}
        order = [cand[c][1] for c in sorted(sc, key=lambda c: (-sc[c], c))]
        # tail = spectral-only hits, drawn from the SAME top-NB_TOP window the features use.
        # Drawing from every spectral hit instead differs from E23 by 7e-5 on Class 1 (caught by
        # validate_port2.py) -- small, but the port must match the experiment exactly.
        top_hits = dict(sorted(spec.items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP])
        order += [smi_of(c) for c in sorted(set(top_hits) - set(cands))[:limit]]
        seen, keep = set(), []
        for smi in order:
            k = key14(smi)
            if k is not None and k in seen: continue
            if k is not None: seen.add(k)
            keep.append(smi)
            if len(keep) == limit: break
        out[mid] = keep
    return out
