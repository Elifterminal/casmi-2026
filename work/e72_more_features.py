#!/usr/bin/env python3
"""E72 — E71 said the model class is not the lever. So test the other half: more features.

E71 changed only the model class, seven features held fixed, and gradient-boosted trees LOST to the
linear pairwise fit: -0.0058 as-protocol, -0.0065 scaffold-disjoint, both CIs straddling zero. With
seven features and ~520 training queries there is no interaction structure for trees to exploit that
the linear model is not already capturing. That leaves the features as the remaining explanation for
why the teams above us run thirty-one of them.

WHAT THIS ADDS: chemistry of the candidate, plus how well its mass actually fits the precursor.

  ppm_err, ppm_abs   signed and absolute mass error against the query's neutral mass. Candidates
                     arrive from a 10 ppm window, so every one of them "fits" -- but WHERE in that
                     window is information the ranker has never been given.
  mw                 exact monoisotopic mass
  rings, arom_rings  ring count and aromatic ring count
  rot_bonds          rotatable bonds
  tpsa, logp         polar surface area and lipophilicity
  frac_csp3          saturation
  n_N, n_O, n_S, n_hal   heteroatom counts
  hbd, hba           donors and acceptors

Fourteen new, twenty-one total. Every one is a property of the candidate structure or of its mass
against the precursor, so all are computable for a test molecule with no spectra of its own.

WHAT THIS DELIBERATELY DOES NOT ADD, and why. The obvious extra features are "how much data exists
about this molecule" -- how many training spectra it has, how many libraries it appears in. Those
are exactly the family that produced the E48 artifact. Our splits REMOVE a Class 2 or 3 truth's
spectra from the reference index, so a spectra-count feature would read zero for truths and non-zero
for many decoys, and the model would learn our split construction rather than any chemistry. It
might even transfer, since test truths are genuinely novel -- but I cannot tell the two apart from
local numbers, so it stays out of this round and gets its own experiment with a proper control if
the chemical features prove worth building on.

Both model classes are refitted on the enlarged feature set, because "more features" and "trees"
could interact: trees may need the extra columns before they can beat a linear fit. Every arm keeps
E71's scaffold-disjoint control.

PREDICTIONS (before the run, 2026-09-27):
  P1  The enlarged feature set beats seven features by >= +0.010 weighted, for at least one model
      class, in the scaffold-disjoint arm.
  P2  ppm_abs is among the top five features by gain. Its absence from a mass-spectrometry ranker
      looks like our clearest oversight, and if it is NOT important that is worth knowing.
  P3  With twenty-one features, trees now beat linear -- the reverse of E71 -- because interactions
      need columns to interact between. If trees still lose at twenty-one features, the model class
      is settled and I stop proposing it.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from casmi_pipeline import FEATS, TOPN, key14, neutral_mass
import e71_gbm_reranker as E71

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS, RESULTS = os.path.join(WORK, "splits"), os.path.join(WORK, "results")
EVAL = E71.EVAL
TRAIN = E71.TRAIN
W = E71.W
BOOT = 4000
DESC_CACHE = os.path.join(SPLITS, "e72_descriptors.pkl")
EXTRA = ["ppm_err", "ppm_abs", "mw", "rings", "arom_rings", "rot_bonds", "tpsa", "logp",
         "frac_csp3", "n_N", "n_O", "n_S", "n_hal", "hbd", "hba"]
HAL = {"F", "Cl", "Br", "I"}


def descriptors(smi):
    """Query-independent chemistry of one candidate. None if RDKit cannot parse it."""
    m = Chem.MolFromSmiles(smi or "")
    if m is None: return None
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    return dict(mw=float(Descriptors.ExactMolWt(m)),
                rings=float(rdMolDescriptors.CalcNumRings(m)),
                arom_rings=float(rdMolDescriptors.CalcNumAromaticRings(m)),
                rot_bonds=float(rdMolDescriptors.CalcNumRotatableBonds(m)),
                tpsa=float(rdMolDescriptors.CalcTPSA(m)),
                logp=float(Crippen.MolLogP(m)),
                frac_csp3=float(rdMolDescriptors.CalcFractionCSP3(m)),
                n_N=float(sum(s == "N" for s in sym)), n_O=float(sum(s == "O" for s in sym)),
                n_S=float(sum(s == "S" for s in sym)), n_hal=float(sum(s in HAL for s in sym)),
                hbd=float(rdMolDescriptors.CalcNumHBD(m)), hba=float(rdMolDescriptors.CalcNumHBA(m)))


def query_masses(splits):
    """{query key: neutral mass} rebuilt the way e22's build computes it, from the split's rows."""
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["precursor_mz", "adduct"])
    pmz = t.column("precursor_mz").to_numpy(); add = np.asarray(t.column("adduct")); del t
    out = {}
    for sp_name in splits:
        sp = json.load(open(f"{SPLITS}/split_{sp_name}.json"))
        for k, rows in sp["query_rows"].items():
            Ms = [neutral_mass(pmz[r], add[r]) for r in rows]
            Ms = [m for m in Ms if np.isfinite(m) and m > 0]
            if Ms: out[k] = float(np.median(Ms))
    return out


def build_extended(qs, M_of, desc, log, tag):
    """Append the new columns to each candidate's existing 7-vector, in EXTRA order."""
    miss = 0
    for q in qs:
        M = M_of.get(q["k"])
        newc = {}
        for c, (v, smi) in q["cand"].items():
            d = desc.get(smi)
            if d is None or M is None or M <= 0:
                miss += 1
                continue
            ppm = (d["mw"] - M) / M * 1e6
            extra = [ppm, abs(ppm)] + [d[f] for f in EXTRA[2:]]
            newc[c] = (np.concatenate([v, np.array(extra, dtype=float)]), smi)
        q["cand_ext"] = newc
    log(f"  {tag}: extended; {miss} candidates dropped for an unparseable structure or mass")


def as_ext(q):
    """A view of the query whose 'cand' is the extended feature set, for reuse of E71's helpers."""
    return dict(k=q["k"], cls=q["cls"], truth=q["truth"], cand=q["cand_ext"], tail=q["tail"],
                _tk=q.get("_tk"))


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    cache = os.path.join(SPLITS, "e71_built_fr.pkl")
    ev, tr_all = pickle.load(open(cache, "rb"))
    log(f"loaded cached build: {len(ev)} eval, {len(tr_all)} train queries")

    smis = {smi for q in ev + tr_all for (_, smi) in q["cand"].values() if smi}
    log(f"{len(smis):,} distinct candidate structures")
    if os.path.exists(DESC_CACHE):
        desc = pickle.load(open(DESC_CACHE, "rb"))
        need = [s for s in smis if s not in desc]
    else:
        desc, need = {}, sorted(smis)
    if need:
        from multiprocessing import Pool
        with Pool(7) as p:
            for s, d in zip(need, p.map(descriptors, need, chunksize=256)):
                desc[s] = d
        pickle.dump(desc, open(DESC_CACHE, "wb"))
    log(f"descriptors ready ({sum(1 for v in desc.values() if v is None)} unparseable)")

    M_of = query_masses(EVAL + TRAIN)
    log(f"neutral masses for {len(M_of):,} queries")
    build_extended(ev, M_of, desc, log, "eval")
    build_extended(tr_all, M_of, desc, log, "train")

    # scaffold-disjoint training set, exactly as E71 defined it
    ev_scaf, ev_keys = set(), set()
    for q in ev:
        tk = E71.truth_key(q)
        if tk:
            s = E71.scaffold(q["cand"][tk][1])
            if s: ev_scaf.add(s)
        if q["truth"]: ev_keys.add(q["truth"])
    tr_disj = []
    for q in tr_all:
        tk = E71.truth_key(q)
        if tk is None or q["truth"] in ev_keys: continue
        s = E71.scaffold(q["cand"][tk][1])
        if s and s in ev_scaf: continue
        tr_disj.append(q)
    log(f"train: {len(tr_all)} as-protocol, {len(tr_disj)} scaffold-disjoint")

    rng = np.random.default_rng(72)
    arms, gbms = {}, {}
    for nf, view in ((len(FEATS), lambda q: q), (len(FEATS) + len(EXTRA), as_ext)):
        label = f"{nf} feats"
        for tag, trq in (("as-protocol", tr_all), ("scaffold-disjoint", tr_disj)):
            TR = [view(q) for q in trq]
            EV = [view(q) for q in ev]
            w, _ = E71.fit_linear(TR)
            if w is not None:
                arms[f"linear · {label} · {tag}"] = E71.per_class(
                    EV, lambda q, w=w: E71.linear_scores(q, w))
            g = E71.fit_gbm(TR, rng, log)
            if g is not None:
                arms[f"trees  · {label} · {tag}"] = E71.per_class(
                    EV, lambda q, g=g: E71.gbm_scores(q, g))
                gbms[f"{label} · {tag}"] = g
            log(f"  fitted {label}, {tag}")

    def wmrr(per):
        return sum(W[k] * float(np.mean(per[k])) for k in (1, 2, 3) if per[k])

    def paired(a, b):
        d = []
        for _ in range(BOOT):
            tot = 0.0
            for k in (1, 2, 3):
                xa, xb = np.asarray(a[k]), np.asarray(b[k])
                if len(xa) == 0: continue
                i = rng.integers(0, len(xa), len(xa))
                tot += W[k] * (xa[i].mean() - xb[i].mean())
            d.append(tot)
        d = np.array(d)
        return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)

    L = [f"E72 — {len(FEATS)} features against {len(FEATS)+len(EXTRA)}, both model classes", "",
         f"added: {', '.join(EXTRA)}", "",
         f"{'arm':40s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}"]
    for name, per in arms.items():
        c = [float(np.mean(per[k])) if per[k] else float("nan") for k in (1, 2, 3)]
        L.append(f"{name:40s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wmrr(per):>9.4f}")

    L.append("")
    n7, n21 = f"{len(FEATS)} feats", f"{len(FEATS)+len(EXTRA)} feats"
    for tag in ("as-protocol", "scaffold-disjoint"):
        for model in ("linear", "trees "):
            a, b = f"{model} · {n21} · {tag}", f"{model} · {n7} · {tag}"
            if a in arms and b in arms:
                m, lo, hi = paired(arms[a], arms[b])
                L.append(f"{model} {n21} minus {n7}, {tag:18s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                         f"  {'CLEARS ZERO' if lo > 0 else 'does not clear zero'}")
        a, b = f"trees  · {n21} · {tag}", f"linear · {n21} · {tag}"
        if a in arms and b in arms:
            m, lo, hi = paired(arms[a], arms[b])
            L.append(f"trees minus linear at {n21}, {tag:15s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                     f"  {'CLEARS ZERO' if lo > 0 else 'does not clear zero'}")

    key = f"{n21} · scaffold-disjoint"
    if key in gbms:
        imp = gbms[key].feature_importances_
        names = list(FEATS) + EXTRA
        order = np.argsort(-imp)
        L += ["", "feature gain, trees at 21 features (scaffold-disjoint):"]
        for i in order[:12]:
            L.append(f"  {names[i]:14s} {imp[i]:>8.0f}")
    L += ["", "P1 wanted the larger feature set to win by >= +0.010 in the scaffold-disjoint arm.",
          "P2 wanted ppm_abs in the top five by gain.",
          "P3 wanted trees to beat linear once there are 21 features.", "",
          "DELIBERATELY EXCLUDED: spectra counts and library-membership counts. Our splits strip a",
          "Class 2/3 truth's spectra from the reference index, so such a feature reads zero for the",
          "answer and non-zero for many decoys -- the E48 artifact family. It needs its own control.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E72_more_features_2026-09-27.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
