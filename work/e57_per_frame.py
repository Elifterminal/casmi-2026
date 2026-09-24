#!/usr/bin/env python3
"""E57 — score each energy frame separately instead of merging. The field's +0.019, in our terms.

The published 0.336 notebook reports that averaging a per-spectrum model with a merged-spectrum
model is worth +0.019 on the leaderboard -- the largest single effect its author measured. They
add the line that matters most to us: "Both local holdouts said the opposite." Their local
validation preferred merged-alone (0.5173 against 0.4904) while the board preferred the pair
(0.335 against 0.311).

We have never had a per-frame channel. Every feature we compute reads `allmz`/`allit`, which is
every collision-energy frame concatenated into one spectrum before scoring. That is a choice we
made early and never tested, and it throws away the structure Lee identified: which pieces appear
gently and which need force.

This needs no borrowed model. Our own `explain` feature -- the intensity-weighted share of the top
peaks a candidate's fragments can account for -- can be computed against each frame separately:

  explain_merged   what we ship today
  explain_mean     mean of the per-frame values
  explain_max      best single frame (a candidate that explains ONE energy well)
  explain_min      worst frame (a candidate that explains EVERY energy, which is the stronger claim)
  explain_spread   max minus min, the energy-dependence of the match itself

AND A WARNING THE FIELD HANDED US FOR FREE. If their local validation inverted on exactly this
change, ours may too. A null here does not clear the change -- it reproduces their local result,
and the only way to settle it would be a submission. I am recording that BEFORE the run so the
interpretation is not chosen afterwards.

PREDICTIONS (before the run, 2026-09-24):
  P1  The per-frame features beat merged-only locally, but modestly: +0.005 to +0.020 weighted.
  P2  explain_min carries more than explain_max. Explaining every energy is a stronger claim about
      a structure than explaining one, and a coincidental mass match should fail somewhere.
  P3  Following the field's warning, I give this a real chance of going the other way: maybe 1 in 3
      that merged-only wins locally, as it did for them. If that happens the honest reading is that
      our local yardstick has the same blind spot theirs did, NOT that the change is bad.
"""
import json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from rdkit import RDLogger
from sklearn.linear_model import LogisticRegression
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import build, FEATS2, W
from casmi_pipeline import explain, key14, TOPN, characterise, candidate_info
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
CLEAN = ("npxmtsseed60_n300", "npxmtsseed61_n300", "npxmtsseed62_n300")
NEW = ["explain_mean", "explain_max", "explain_min", "explain_spread"]


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"])
    ce_raw = t.column("collision_energy_ev").to_pylist()
    def flat(c):
        a = t.column(c).combine_chunks()
        return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()
    mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
    log("spectra loaded")

    st = load_structures()
    ev = build(CLEAN, st, log=log, pools_prefix="fr_", generate=False)
    log(f"{len(ev)} queries built")

    # per-frame spectra for every query
    frames = {}
    for split in CLEAN:
        sp = json.load(open(f"{SPLITS}/split_{split}.json"))
        for k, rows in sp["query_rows"].items():
            fr = []
            for r in rows:
                mz, it = mzf[mzo[r]:mzo[r+1]], itf[ito[r]:ito[r+1]]
                if len(mz): fr.append((mz, it))
            if fr: frames[k] = fr

    pools = {}
    for split in CLEAN:
        for p in pickle.load(open(f"{SPLITS}/pools_fr_{split}.pkl", "rb")): pools[p["k"]] = p

    need = sorted({s for q in ev for _, s in q["cand"].values()})
    info = characterise(need, procs=7)
    log(f"{len(info):,} structures characterised")

    n_multi = 0
    for q in ev:
        fr = frames.get(q["k"], [])
        p = pools.get(q["k"])
        pos = bool(p["positive"]) if p else True
        if len(fr) > 1: n_multi += 1
        q["fv"] = {}
        for c, (vec, smi) in q["cand"].items():
            f = {n: float(v) for n, v in zip(FEATS2, vec)}
            inf = info.get(smi)
            vals = []
            if inf and inf.get("full"):
                for mz, it in fr:
                    vals.append(explain(mz, it, inf["full"], pos))
            if vals:
                f["explain_mean"] = float(np.mean(vals))
                f["explain_max"] = float(np.max(vals))
                f["explain_min"] = float(np.min(vals))
                f["explain_spread"] = float(np.max(vals) - np.min(vals))
            else:
                for n in NEW: f[n] = 0.0
            q["fv"][c] = f
    log(f"per-frame features built; {n_multi}/{len(ev)} queries have >1 frame")

    ARMS = {"merged only (shipped)": FEATS2,
            "+ per-frame": FEATS2 + NEW,
            "per-frame, no merged explain": [f for f in FEATS2 if f != "explain"] + NEW}

    def fit_w(qs, feats):
        X, y = [], []
        for q in qs:
            tk = [c for c, (_, s) in q["cand"].items() if key14(s) == q["truth"]]
            if not tk: continue
            t_ = np.array([q["fv"][tk[0]][f] for f in feats])
            for c in q["cand"]:
                if c in tk: continue
                d = t_ - np.array([q["fv"][c][f] for f in feats]); X += [d, -d]; y += [1, 0]
        if not X: return np.zeros(len(feats))
        X = np.array(X); sd = X.std(0) + 1e-9
        return LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X / sd, y).coef_[0] / sd

    def rr(q, w, feats):
        sc = {c: float(np.dot(w, [q["fv"][c][f] for f in feats])) for c in q["cand"]}
        order = [q["cand"][c][1] for c in sorted(sc, key=lambda c: (-sc[c], str(c)))] + q["tail"]
        seen, out = set(), []
        for smi in order:
            kk = key14(smi)
            if kk is not None and kk in seen: continue
            if kk is not None: seen.add(kk)
            out.append(kk)
            if len(out) == TOPN: break
        for i, kk in enumerate(out, 1):
            if q["truth"] is not None and kk == q["truth"]: return 1.0 / i
        return 0.0

    ids = sorted({q["truth"] or q["k"] for q in ev})
    rng = np.random.default_rng(0); rng.shuffle(ids)
    fold = {i: n % 2 for n, i in enumerate(ids)}
    per = {a: defaultdict(list) for a in ARMS}
    byid = defaultdict(lambda: defaultdict(list))
    for name, feats in ARMS.items():
        for f in (0, 1):
            trq = [q for q in ev if fold[q["truth"] or q["k"]] != f]
            teq = [q for q in ev if fold[q["truth"] or q["k"]] == f]
            w = fit_w(trq, feats)
            for q in teq:
                s = rr(q, w, feats)
                per[name][q["cls"]].append(s)
                byid[q["truth"] or q["k"]][name].append((q["cls"], s))
        log(f"  {name}: done")

    def wt(n): return sum(W[c] * float(np.mean(per[n][c])) for c in (1, 2, 3) if per[n][c])
    L = ["E57 — per-frame scoring against merged (the field's +0.019, in our features)", "",
         f"{len(ev)} queries, {n_multi} with more than one energy frame", "",
         f"{'arm':32s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9} {'vs merged':>11}"]
    for n in ARMS:
        c = [float(np.mean(per[n][k])) if per[n][k] else float('nan') for k in (1, 2, 3)]
        L.append(f"{n:32s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt(n):>9.4f} "
                 f"{wt(n)-wt('merged only (shipped)'):>+11.4f}")
    idl = list(byid); rng = np.random.default_rng(1)
    L.append("")
    for n in ARMS:
        if n == "merged only (shipped)": continue
        boots = []
        for _ in range(3000):
            pick = rng.integers(0, len(idl), len(idl))
            a, b = defaultdict(list), defaultdict(list)
            for j in pick:
                for cls, s in byid[idl[j]]["merged only (shipped)"]: a[cls].append(s)
                for cls, s in byid[idl[j]][n]: b[cls].append(s)
            boots.append(sum(W[c] * (np.mean(b[c]) - np.mean(a[c])) for c in (1, 2, 3) if a[c] and b[c]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        L.append(f"{n:32s} {wt(n)-wt('merged only (shipped)'):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                 + ("   <- clears zero" if lo > 0 else ""))
    L += ["", "the field measured this change at +0.019 on the BOARD while their own local holdouts",
          "preferred merged-only. If ours does the same, that reproduces their local result and the",
          "question can only be settled by a submission -- recorded before the run, not after.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E57_per_frame_2026-09-24.txt", "w").write("\n".join(L) + "\n")
    json.dump({a: dict(weighted=wt(a)) for a in ARMS}, open(f"{RESULTS}/E57_per_frame.json", "w"), indent=2)


if __name__ == "__main__":
    main()
