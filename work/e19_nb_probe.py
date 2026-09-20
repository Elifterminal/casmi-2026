#!/usr/bin/env python3
"""E19 — how much of the neighbour vote rides on near-duplicate relatives?

Seda's review (2026-09-20) asked for a leak probe on the neighbour vote, since it was the
signal the tautomer-twin leak inflated most. Her version -- hold out every same-formula
structure -- measures a harsher world than the competition: a Class 2 molecule genuinely has
published relatives, and using them is the intended game. The sharper question is how much of
the +0.097 depends on relatives so close they are nearly the answer:

  none    every spectral hit kept (= E18)
  <0.95   hits with Tanimoto >= 0.95 to the TRUE structure dropped (near-duplicates)
  <0.80   hits with Tanimoto >= 0.80 dropped (close analogues too)

Dropping by similarity-to-truth uses the label, so this is a DIAGNOSTIC, not a ranker: it
measures dependence, it is not a pipeline we could ship. Everything else matches E18 exactly,
including the E18 weights (no refitting -- refitting would confound the answer).

PREDICTIONS (before the run, 2026-09-20):
  P1  At <0.95 the learned ranker loses < 0.02 weighted: near-duplicates are rare (E17's leak
      check found 11.6% of C2 queries with a Tanimoto >= 0.95 hit).
  P2  At <0.80 it loses more, 0.03-0.08: this is the analogue signal the feature is built on,
      so removing it should hurt -- that is the feature working, not a leak.
  P3  Class 1 falls hardest at <0.95 (its own other spectra are the near-duplicates).
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e15_coconut_pools import load_structures
from twins import db_exclusions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import casmi_pipeline as cp          # the port-equivalent module (validate_port.py holds it to E18)
from casmi_pipeline import FEATS, NB_TOP, PPM, TOPN, MassIndex, neutral_mass, explain, chance, characterise
import scoring

SPLITS = os.path.expanduser("~/casmi-2026/work/splits")
RESULTS = os.path.expanduser("~/casmi-2026/work/results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
W = {1: 0.16, 2: 0.45, 3: 0.39}
CUTS = [("none", 2.0), ("<0.95", 0.95), ("<0.80", 0.80)]


def main():
    t0 = time.time()
    weights = json.load(open(f"{RESULTS}/E18_ranker_coconut.json"))["weights"]
    w = np.array([weights[f] for f in FEATS])
    keys, pmz, add, tr, co, _ = load_structures()
    trs, cos_ = dict(zip(tr.k, tr.s)), dict(zip(co.k, co.s))
    smi_of = lambda k: trs.get(k) or cos_.get(k, "")
    per = {name: defaultdict(list) for name, _ in CUTS}
    dropped = defaultdict(int)

    for split in EVAL:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        assign, qrows = sp["assign"], sp["query_rows"]
        pools = {p["k"]: p for p in pickle.load(open(f"{SPLITS}/pools_{split}.pkl", "rb"))}
        excl = db_exclusions(sp, tr, co)
        keep = ~co.k.isin(excl)
        midx = MassIndex(co.k[keep].to_numpy(), co.m[keep].to_numpy())

        qs = []
        for k, rows in qrows.items():
            p = pools[k]; cands, Ms = set(), []
            for r in rows:
                M = neutral_mass(pmz[r], add[r])
                if np.isfinite(M) and M > 0: cands.update(midx.window(M, PPM).tolist()); Ms.append(M)
            qs.append((k, assign[k], p, cands, float(np.median(Ms)) if Ms else np.nan))
        need = {smi_of(c) for _, _, _, cands, _ in qs for c in cands}
        need |= {smi_of(h) for _, _, p, _, _ in qs
                 for h, _ in sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))[:NB_TOP]}
        need |= {smi_of(k) for k, *_ in qs}
        info = characterise(need, procs=7)
        print(f"[{split}] {len(qs)} queries, {len(info):,} structures characterised {time.time()-t0:.0f}s", flush=True)

        for k, cls, p, cands, M in qs:
            tfp = (info.get(smi_of(k)) or {}).get("fp")
            hits_all = sorted(p["spec"].items(), key=lambda kv: (-kv[1], kv[0]))
            for name, cut in CUTS:
                if cut >= 2.0 or tfp is None:
                    hits = hits_all[:NB_TOP]
                else:
                    hits = []
                    for h, c in hits_all:
                        hfp = (info.get(smi_of(h)) or {}).get("fp")
                        if hfp is not None and DataStructs.TanimotoSimilarity(tfp, hfp) >= cut:
                            dropped[name] += 1; continue
                        hits.append((h, c))
                        if len(hits) == NB_TOP: break
                good = [(h, c) for h, c in hits if info.get(smi_of(h)) is not None]
                hfp = [info[smi_of(h)]["fp"] for h, _ in good]
                hcos = np.array([c for _, c in good])
                spec = dict(hits_all)
                sc = {}
                for c in cands:
                    inf = info.get(smi_of(c))
                    if inf is None: continue
                    e = explain(p["allmz"], p["allit"], inf["full"], p["positive"])
                    if hfp:
                        v = hcos * np.array(DataStructs.BulkTanimotoSimilarity(inf["fp"], hfp))
                        nbm, nb10 = float(v.max()), float(v[:10].mean())
                    else:
                        nbm = nb10 = 0.0
                    f = dict(explain=e, chance_corr=e - chance(inf["full"], M),
                             single=explain(p["allmz"], p["allit"], inf["single"], p["positive"]),
                             nb_max=nbm, nb_mean10=nb10, spec_self=spec.get(c, 0.0), heavy=inf["heavy"])
                    sc[c] = float(np.dot(w, [f[x] for x in FEATS]))
                ranked = sorted(sc, key=lambda c: (-sc[c], c)) + sorted(set(spec) - cands)[:TOPN]
                rr = 0.0
                for i, c in enumerate(ranked[:TOPN], 1):
                    if p["truth"] is not None and scoring.key14(smi_of(c)) == p["truth"]: rr = 1.0 / i; break
                per[name][cls].append(rr)
        print(f"[{split}] scored {time.time()-t0:.0f}s", flush=True)

    L = ["E19 — neighbour-vote dependence on near-duplicate relatives (learned ranker, E18 weights)", "",
         f"{'hits kept':10s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9} {'vs none':>9}  hits dropped"]
    base = None
    out = {}
    for name, _ in CUTS:
        c = [float(np.mean(per[name][k])) for k in (1, 2, 3)]
        wt = sum(W[k] * c[k - 1] for k in (1, 2, 3))
        base = wt if base is None else base
        out[name] = dict(C1=c[0], C2=c[1], C3=c[2], weighted=wt)
        L.append(f"{name:10s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt:>9.4f} {wt-base:>+9.4f}  {dropped[name]:,}")
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E19_nb_probe_2026-09-20.txt", "w").write("\n".join(L) + "\n")
    json.dump(out, open(f"{RESULTS}/E19_nb_probe.json", "w"), indent=2)


if __name__ == "__main__":
    main()
