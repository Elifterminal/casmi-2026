#!/usr/bin/env python3
"""E54 — Lee's ordering argument, tested with an instrument that can actually see.

E52 tried this with our own fragmenter and failed for a measured reason: a true structure
generates ~29 fragments and a median of FOUR match any observed peak, so there was nothing to
order. ICEBERG matches real peaks and, crucially, predicts a DIFFERENT spectrum at each collision
energy. E53 re-ran it keeping the frames separate -- the structure E47 destroyed by merging.

THE SHARPER FORM OF THE ARGUMENT. Rather than asking when a fragment first appears, build a small
matrix per candidate:

        M[i][j] = agreement( predicted at energy e_i , observed at energy e_j )

If the candidate is the right structure, its predicted energy dependence should track the
observed one, so the DIAGONAL (matched energies) should beat the OFF-DIAGONAL (mismatched). A
decoy that happens to share masses has no reason for its energy behaviour to line up -- it may
agree with the observed spectrum overall while agreeing equally at every energy, which is the
signature of a coincidence rather than a mechanism.

    diag_advantage = mean(M[i][i]) - mean(M[i][j], i != j)

That is Lee's "order of decay" in a form that does not depend on our fragmenter being any good,
and it is a quantity no public notebook computes: they score per-spectrum, none cross-compares
energies.

TWO USES, and the second is the one Lee actually proposed:
  RANKING     diag_advantage as a feature, on top of the plain per-frame agreement
  ELIMINATION a candidate whose off-diagonal BEATS its diagonal is behaving impossibly -- its
              predicted chemistry moves the wrong way as the energy rises. Deleting those shortens
              the list without scoring anything.

PREDICTIONS (before the run, 2026-09-24):
  P1  Truths show a larger diagonal advantage than the median decoy in 55-70% of queries. Real but
      not decisive, like every single feature on this project.
  P2  As an eliminator it beats E52's useless 1.03x. I will call anything above 1.5x decoys-removed
      per truth-lost a result worth building on.
  P3  The plain per-frame agreement (the diagonal alone) is a better RANKER than the merged cosine
      from E50, because that is what the field measured at +0.019 and we have never had it.
  RISK: most of our queries carry only two distinct energies, which gives a 2x2 matrix and exactly
  two off-diagonal cells. The estimate will be noisy where the ladder is short.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e47c_validate import sparse_cos
from casmi_pipeline import key14
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
import os as _os
if _os.environ.get("E54_CONTROL"):          # the contaminated control: molecules ICEBERG knows
    SPLITS_ALL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
    PRED_FILES = ("iceberg_frames_known.npz",)
    WORKLISTS = ("iceberg_known_control.json",)
    TAG = "control_known"
else:
    SPLITS_ALL = ("npxmtsseed60_n300", "npxmtsseed61_n300", "npxmtsseed62_n300")
    PRED_FILES = ("iceberg_frames_seed60.npz", "iceberg_frames_seed6162.npz")
    WORKLISTS = ("iceberg_top25_d60_n300_c12.json",
                 "iceberg_top25_d61_n300_d62_n300_c12.json")
    TAG = "all3"
TOP_OBS = 40


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    pred = defaultdict(lambda: defaultdict(dict))     # query -> cand -> energy -> (mz,it)
    for pf in PRED_FILES:
        d = np.load(f"{EXPORTS}/{pf}")
        for k in d.keys():
            q, rest, kind = k.split("|")
            cand, _, e = rest.rpartition("@")
            pred[q][cand].setdefault(float(e), {})[kind] = d[k]
    log(f"{len(pred)} queries with per-energy predictions")

    wl = {}
    for wf in WORKLISTS:
        for q in json.load(open(f"{EXPORTS}/{wf}"))["queries"]: wl[q["key"]] = q
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"])
    ce_raw = t.column("collision_energy_ev").to_pylist()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    sp = {"query_rows": {}, "assign": {}}
    pools = {}
    for sname in SPLITS_ALL:
        one = json.load(open(f"{SPLITS}/split_{sname}.json"))
        sp["query_rows"].update(one["query_rows"]); sp["assign"].update(one["assign"])
        for pp in pickle.load(open(f"{SPLITS}/pools_fr_{sname}.pkl", "rb")): pools[pp["k"]] = pp
    log("observed spectra loaded")

    rows = []
    for qk, cands in pred.items():
        p = pools.get(qk)
        if p is None or not p["truth"]: continue
        obs = {}
        for r in sp["query_rows"][qk]:
            ce = ce_raw[r]
            vals = ce if isinstance(ce, list) else ([ce] if isinstance(ce, (int, float)) else [])
            vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
            if len(vals) != 1: continue           # blended frames carry no ordering
            mz, it = mzf[mzo[r]:mzo[r+1]], itf[ito[r]:ito[r+1]]
            if len(mz) == 0: continue
            o = np.argsort(-it)[:TOP_OBS]
            obs.setdefault(vals[0], (mz[o], it[o]))
        if len(obs) < 2: continue                 # need a ladder for an off-diagonal to exist
        oe = sorted(obs)
        per = {}
        for c, byE in cands.items():
            pe = [e for e in oe if e in byE and "mz" in byE[e] and len(byE[e]["mz"])]
            if len(pe) < 2: continue
            M = np.zeros((len(pe), len(oe)))
            for i, ei in enumerate(pe):
                pm, pi = byE[ei]["mz"], byE[ei]["it"]
                for j, ej in enumerate(oe):
                    om, oi = obs[ej]
                    M[i, j] = sparse_cos(pm, pi, om, oi)
            diag = [M[i, oe.index(ei)] for i, ei in enumerate(pe)]
            off = [M[i, j] for i, ei in enumerate(pe) for j, ej in enumerate(oe) if ej != ei]
            if not off: continue
            per[c] = dict(diag=float(np.mean(diag)), off=float(np.mean(off)),
                          adv=float(np.mean(diag) - np.mean(off)))
        if len(per) < 2: continue
        smi_of = {c: wl[qk]["candidates"][i]["smiles"]
                  for i, cc in enumerate(wl.get(qk, {}).get("candidates", []))
                  for c in [cc["key"]]} if qk in wl else {}
        tk = [c for c in per if smi_of.get(c) and key14(smi_of[c]) == p["truth"]]
        if not tk: continue
        tc = tk[0]
        others = [v for c, v in per.items() if c not in tk]
        if not others: continue
        rows.append(dict(query=qk, cls=sp["assign"][qk], n_energies=len(oe), n_cands=len(per),
                         truth_adv=per[tc]["adv"], truth_diag=per[tc]["diag"],
                         med_adv=float(np.median([o["adv"] for o in others])),
                         med_diag=float(np.median([o["diag"] for o in others])),
                         truth_impossible=bool(per[tc]["adv"] < 0),
                         decoy_impossible=float(np.mean([o["adv"] < 0 for o in others])),
                         rank_by_diag=1 + sum(1 for o in others if o["diag"] > per[tc]["diag"]),
                         rank_by_adv=1 + sum(1 for o in others if o["adv"] > per[tc]["adv"])))
    log(f"{len(rows)} queries scored")

    L = ["E54 — does a candidate's predicted energy dependence track the observed one?", "",
         f"{len(rows)} queries with a ladder of >=2 energies and the truth among the candidates", ""]
    if rows:
        ta = np.array([r["truth_adv"] for r in rows]); ma = np.array([r["med_adv"] for r in rows])
        td = np.array([r["truth_diag"] for r in rows]); md = np.array([r["med_diag"] for r in rows])
        ti = np.mean([r["truth_impossible"] for r in rows])
        di = np.mean([r["decoy_impossible"] for r in rows])
        L += ["DIAGONAL ADVANTAGE (matched energies minus mismatched):",
              f"  truth        mean {ta.mean():+.4f}  median {np.median(ta):+.4f}",
              f"  median decoy mean {ma.mean():+.4f}  median {np.median(ma):+.4f}",
              f"  truth beats median decoy: {np.mean(ta > ma):.1%}", "",
              "PLAIN PER-FRAME AGREEMENT (the diagonal alone -- the field's per-spectrum channel):",
              f"  truth {td.mean():.4f} vs median decoy {md.mean():.4f}; "
              f"truth ahead {np.mean(td > md):.1%}",
              f"  truth ranked 1st by per-frame agreement: "
              f"{np.mean([r['rank_by_diag'] == 1 for r in rows]):.1%}", "",
              "AS AN ELIMINATOR (off-diagonal beating the diagonal = impossible energy behaviour):",
              f"  truths flagged impossible {ti:.1%}",
              f"  decoys flagged impossible {di:.1%}",
              f"  -> decoys removed per truth lost: {di/max(ti,1e-9):.2f}x   "
              f"(E52 managed 1.03x)"]
        for ne in sorted({r["n_energies"] for r in rows}):
            s = [r for r in rows if r["n_energies"] == ne]
            L.append(f"  {ne} energies: n={len(s):>3}  truth beats median decoy on advantage "
                     f"{np.mean([r['truth_adv'] > r['med_adv'] for r in s]):.1%}")
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E54_energy_matrix_{TAG}_2026-09-24.txt", "w").write("\n".join(L) + "\n")
    json.dump(rows, open(f"{RESULTS}/E54_energy_matrix_{TAG}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
