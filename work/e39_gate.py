#!/usr/bin/env python3
"""E39 — a gate that decides when to manufacture candidates.

E38 priced this. Generation helps 32 Class 3 queries and hurts 36 Class 2 queries, so switching
it on everywhere is worth +0.0051 while switching it on only where it helps would be worth
+0.0190 -- five times the E37 tail fix, and the best-priced thing left on the board.

WHAT THE GATE PREDICTS, and why not the obvious thing. The obvious label is "did generation help
this query", but E38 says that is 68 usable labels out of 900. The causal quantity underneath it
is denser and better posed: IS THE TRUE STRUCTURE ABSENT FROM THE RETRIEVAL POOL? When it is,
generated candidates are the only way to reach the answer and cost nothing, because there is no
right answer for them to outrank. When it is not, every generated candidate is a decoy. That
label covers every query and is 39% positive in our splits.

THE HARD PART, stated up front. Class 2 and Class 3 are spectrally IDENTICAL by construction:
neither has its own spectrum in the reference index, so spectral self-similarity separates
Class 1 from the other two and cannot separate 2 from 3. The only signal that can is
candidate-set confidence -- does the pool contain something that actually fits the spectrum?
Every feature here is a different way of asking that:

  best_score / score_margin     how good is the top candidate, and how far clear of the second
  best_explain / best_chance    does any candidate's fragment set explain the peaks, above chance
  best_nbmax                    does any candidate resemble the molecules whose spectra matched
  spec_max / spec_gap           Class 1 detector: the query's own spectrum sitting in the index
  n_mass                        pool size, as a crowding control

THE THREAT TO VALIDITY, which I do not think I can fully remove. Our Class 3 queries are
MANUFACTURED by deleting the true structure from the database. A real Class 3 molecule is novel
chemistry, not a deleted row. A gate can learn "the answer was removed from this pool" in ways
that do not transfer to "this molecule is genuinely new". The features above are chosen to be
the ones that would behave the same either way -- they measure fit, not bookkeeping -- but that
is an argument, not evidence, and the leaderboard is the only place it can be tested.

PROTOCOL. Ranker weights fit on the training splits as in E22. Gate fit on the SAME training
splits, so it never sees an evaluation query. Threshold chosen on training only; the evaluation
sweep is reported for information but the headline number is the pre-committed threshold.
Caveat: gate features on the training splits come from a ranker that was fit on those queries,
so training confidence is optimistic and the chosen threshold may sit slightly wrong.

PREDICTIONS (before the run, 2026-09-22):
  P1  Gate AUC for "truth not in the retrieval pool" lands in 0.70-0.85. The fit signal is real
      but blunt, because a crowded 10 ppm pool always contains something that fits reasonably.
  P2  The learned gate beats always-on (+0.0051) but captures well under half the perfect-gate
      prize: final weighted between 0.377 and 0.384.
  P3  Against retrieval-only the interval clears zero. This is the one I am least sure of --
      always-on gains only 0.0051 and that interval almost certainly crosses.
  P4  Most of the gain comes from NOT generating on Class 2 queries rather than from generating
      on more Class 3 queries. The gate's value is refusal, not permission.
"""
import hashlib, json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from e15_coconut_pools import load_structures
from e22_ranker_v2 import build, fit, rank_score, FEATS2, EVAL, TRAIN, W
from casmi_pipeline import key14

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
CACHE = os.path.join(SPLITS, "built_gen_fr.pkl")
GFEATS = ["best_score", "score_margin", "best_explain", "best_chance", "best_nbmax",
          "spec_max", "spec_gap", "n_mass"]


def code_hash():
    """Fingerprint of the code that shapes the cached build, so a stale cache cannot be used.

    Two silent-stale-input bugs have already cost this project a day each (a Kaggle dataset that
    published without the file it was meant to carry, and pools built on the pre-sqrt-p ground).
    A cache keyed on nothing would be the third.
    """
    h = hashlib.sha256()
    for p in ("e22_ranker_v2.py", "e20_edits.py", os.path.join("notebook", "casmi_pipeline.py")):
        h.update(open(os.path.join(WORK, p), "rb").read())
    return h.hexdigest()[:16]


def built(st, log):
    """Retrieval + generated candidates for eval and train splits, cached across experiments."""
    want = code_hash()
    if os.path.exists(CACHE):
        got = pickle.load(open(CACHE, "rb"))
        if got.get("hash") == want:
            log(f"cache hit ({CACHE}, hash {want})")
            return got["ev"], got["tr"]
        log(f"cache stale (built from {got.get('hash')}, code is {want}) -- rebuilding")
    ev = build(EVAL, st, log=log, pools_prefix="fr_")
    tr = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log, pools_prefix="fr_")
    pickle.dump(dict(hash=want, ev=ev, tr=tr), open(CACHE, "wb"))
    log(f"cached build -> {CACHE}")
    return ev, tr


def gate_features(q, w):
    """Per-query confidence summary. Uses no information a real submission would not have."""
    ret = [(float(np.dot(w, v)), v) for c, (v, s) in q["cand"].items()
           if not str(c).startswith("gen:")]
    if not ret:
        return np.zeros(len(GFEATS))
    ret.sort(key=lambda t: -t[0])
    i = {f: j for j, f in enumerate(FEATS2)}
    top = ret[0][1]
    spec = [v[i["spec_self"]] for _, v in ret]
    smax = max(spec) if spec else 0.0
    s10 = float(np.mean(sorted(spec, reverse=True)[:10])) if spec else 0.0
    f = dict(best_score=ret[0][0],
             score_margin=ret[0][0] - (ret[1][0] if len(ret) > 1 else ret[0][0]),
             best_explain=max(v[i["explain"]] for _, v in ret),
             best_chance=max(v[i["chance_corr"]] for _, v in ret),
             best_nbmax=max(v[i["nb_max"]] for _, v in ret),
             spec_max=smax, spec_gap=smax - s10,
             n_mass=float(np.log1p(len(ret))))
    return np.array([f[x] for x in GFEATS])


def truth_absent(q):
    """Label: the true structure is NOT among the retrieved candidates (generation's only chance)."""
    for c, (_, s) in q["cand"].items():
        if not str(c).startswith("gen:") and key14(s) == q["truth"]: return 0
    for s in q["tail"]:
        if key14(s) == q["truth"]: return 0
    return 1


def scored(qs, w, fire):
    """Weighted MRR when generation is switched on exactly for the queries where fire[k] is true."""
    per = defaultdict(list)
    for q in qs:
        per[q["cls"]].append(rank_score(q, w, use_gen=bool(fire.get(q["k"], False))))
    return sum(W[c] * float(np.mean(per[c])) for c in (1, 2, 3)), per


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    st = load_structures()
    ev, tr = built(st, log)
    w, npairs = fit(tr)
    log(f"ranker trained on {npairs:,} pairs")

    Xtr = np.array([gate_features(q, w) for q in tr]); ytr = np.array([truth_absent(q) for q in tr])
    Xev = np.array([gate_features(q, w) for q in ev]); yev = np.array([truth_absent(q) for q in ev])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    gate = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    ptr, pev = gate.predict_proba((Xtr - mu) / sd)[:, 1], gate.predict_proba((Xev - mu) / sd)[:, 1]
    auc_tr, auc_ev = roc_auc_score(ytr, ptr), roc_auc_score(yev, pev)
    log(f"gate AUC  train {auc_tr:.4f}  eval {auc_ev:.4f}  (positives: train {ytr.mean():.1%}, "
        f"eval {yev.mean():.1%})")

    # threshold chosen on TRAIN only
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tr_curve = [(float(t), scored(tr, w, {q["k"]: p >= t for q, p in zip(tr, ptr)})[0]) for t in grid]
    t_star = max(tr_curve, key=lambda kv: kv[1])[0]
    log(f"threshold chosen on training: {t_star:.2f}")

    fire = {q["k"]: p >= t_star for q, p in zip(ev, pev)}
    arms = {
        "retrieval only": {q["k"]: False for q in ev},
        "generation always on": {q["k"]: True for q in ev},
        f"learned gate (p>={t_star:.2f})": fire,
        "perfect gate (oracle)": None,
    }
    out, per_arm = {}, {}
    for name, f in arms.items():
        if f is None:
            per = defaultdict(list)
            for q in ev:
                per[q["cls"]].append(max(rank_score(q, w, use_gen=False), rank_score(q, w, use_gen=True)))
            wt = sum(W[c] * float(np.mean(per[c])) for c in (1, 2, 3))
        else:
            wt, per = scored(ev, w, f)
        out[name] = dict(weighted=wt, **{f"C{c}": float(np.mean(per[c])) for c in (1, 2, 3)})
        per_arm[name] = per

    L = ["E39 — a gate that decides when to manufacture candidates", "",
         f"gate AUC for 'truth absent from retrieval pool':  train {auc_tr:.4f}   eval {auc_ev:.4f}",
         "gate weights: " + ", ".join(f"{f} {v:+.3g}" for f, v in zip(GFEATS, gate.coef_[0])), "",
         f"{'arm':30s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9} {'vs ret-only':>12}"]
    base = out["retrieval only"]["weighted"]
    for name in arms:
        o = out[name]
        L.append(f"{name:30s} {o['C1']:>8.4f} {o['C2']:>8.4f} {o['C3']:>8.4f} {o['weighted']:>9.4f} "
                 f"{o['weighted']-base:>+12.4f}")

    rng = np.random.default_rng(0)
    L.append("")
    for name in arms:
        if name == "retrieval only": continue
        a, b = per_arm["retrieval only"], per_arm[name]
        boots = [sum(W[c] * np.mean([b[c][i] - a[c][i] for i in rng.integers(0, len(a[c]), len(a[c]))])
                     for c in (1, 2, 3)) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        d = out[name]["weighted"] - base
        L.append(f"{name:30s} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]" +
                 ("   <- clears zero" if lo > 0 else ""))

    # what the gate actually did, per class
    L += ["", "gate decisions on the evaluation set:",
          f"{'class':6s} {'queries':>8} {'gate fired':>11} {'truth absent':>13} {'correct':>9}"]
    for c in (1, 2, 3):
        idx = [j for j, q in enumerate(ev) if q["cls"] == c]
        fired = np.array([fire[ev[j]["k"]] for j in idx]); lab = yev[idx]
        L.append(f"C{c:<5d} {len(idx):>8} {fired.mean():>10.1%} {lab.mean():>12.1%} "
                 f"{(fired == lab.astype(bool)).mean():>8.1%}")

    L += ["", "threshold sweep on the EVALUATION set (for information only -- the headline number",
          "above uses the threshold picked on training, before any of this was visible):",
          f"{'threshold':>10} {'weighted':>10}"]
    for t in grid:
        wt, _ = scored(ev, w, {q["k"]: p >= t for q, p in zip(ev, pev)})
        L.append(f"{t:>10.2f} {wt:>10.4f}")
    L.append(f"\nruntime {time.time()-t0:.0f}s")
    print("\n".join(L))
    open(f"{RESULTS}/E39_gate_2026-09-22.txt", "w").write("\n".join(L) + "\n")
    json.dump(dict(auc_train=auc_tr, auc_eval=auc_ev, threshold=t_star, arms=out,
                   weights=dict(zip(GFEATS, map(float, gate.coef_[0])))),
              open(f"{RESULTS}/E39_gate.json", "w"), indent=2)


if __name__ == "__main__":
    main()
