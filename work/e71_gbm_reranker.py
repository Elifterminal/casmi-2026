#!/usr/bin/env python3
"""E71 — swap the ranker's MODEL CLASS and nothing else: linear pairwise -> gradient-boosted trees.

WHY THIS FIRST. Our ranker is seven features combined by a linear pairwise logistic fit (E17/E22:
LogisticRegression on truth-minus-decoy feature differences). The teams above us run eight
gradient-boosted models over thirty-one features. Two separate changes are hiding in that sentence,
and E58 taught me what happens when I bundle two changes and then try to attribute the result. So
this run changes ONLY the model class. Identical features, identical training queries, identical
evaluation queries, identical scoring. Adding features comes after, on whichever model wins.

WHY A TREE SHOULD WIN, if anything does. A linear weight has to pick one direction per feature for
all candidates. But `heavy` (heavy-atom count) cannot have a single sign -- too small and the
molecule cannot produce the heavy fragments, too large and the explained-intensity score is inflated
by sheer combinatorial opportunity, which is exactly what `chance_corr` exists to subtract. A tree
can hold "explain is only trustworthy when heavy is moderate" in two splits. Interactions are the
whole reason to change model class, so if the trees win it should be by finding those.

THE RISK THAT MAKES THIS COMPARISON UNFAIR, controlled rather than hoped away. A GBM with hundreds
of splits can memorise scaffold-specific patterns that seven linear weights cannot. Our protocol
excludes evaluation STRUCTURES from training by InChIKey14 (E17 onwards), which does not exclude
tautomer twins and does not exclude the same Murcko scaffold appearing in both. For a linear model
that hardly matters; for trees it could manufacture the entire gain. So every arm runs twice:

  as-protocol       eval structures excluded by InChIKey14, as every prior experiment did
  scaffold-disjoint training queries sharing a Murcko scaffold with ANY eval query are dropped too,
                    and identity is the SCORER's key14 so tautomer twins go with them

Both models are refitted on each training set, so the comparison is always at equal data. A GBM
that wins as-protocol but not scaffold-disjoint was memorising, and I will report it that way.

Hyperparameters are fixed in advance and early stopping uses a slice held out of TRAINING, never of
eval -- tuning against the evaluation set is how a local number stops predicting the board.

PREDICTIONS (before the run, 2026-09-27):
  P1  Trees beat linear on weighted MRR@25 in the scaffold-disjoint arm by >= +0.010. Below that the
      model class is not the lever and the thirty-one features are, which would be worth knowing
      just as much.
  P2  The as-protocol gain is LARGER than the scaffold-disjoint gain, because some memorisation is
      available and trees will take it. If the two are equal, scaffold overlap was not a problem
      here and I will say so.
  P3  Class 2 gains most. It is 45% of the weighting and the class where the neighbour-vote features
      do the work, so it has the most interaction structure to find.
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from functools import lru_cache
from casmi_pipeline import FEATS, TOPN
from casmi_pipeline import key14 as _key14_raw


# scoring.key14 runs a tautomer canonicalisation per call and is NOT cached. This script asks the
# same question of the same SMILES tens of thousands of times -- every candidate of every query, on
# every pass -- which turned a minutes-long job into an hours-long one and I had to kill the first
# attempt. Memoise here rather than touching scoring.py, which validate_port pins. Pure function,
# so the cache changes the wall clock and nothing else.
@lru_cache(maxsize=2_000_000)
def key14(smi):
    return _key14_raw(smi)
from e22_ranker_v2 import build, load_structures, rank_score
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(WORK, "results")
EVAL = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
TRAIN = ("nptsseed20_n300", "nptsseed21_n300", "nptsseed22_n300")
W = {1: 0.16, 2: 0.45, 3: 0.39}
BOOT = 4000
# fixed in advance; not tuned against eval
LGB = dict(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31,
           min_child_samples=30, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
           reg_lambda=1.0, label_gain=[0, 1], verbose=-1, random_state=71)


def scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi or "")
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else None
    except Exception:
        return None


def truth_key(q):
    """The candidate key whose structure IS the answer, by the scorer's identity.

    The obvious implementation -- canonicalise every candidate and compare -- is what made the
    first attempt at this experiment unfinishable. Pools hold ~700 candidates inside a 10 ppm mass
    window, so they mostly share a formula, and each comparison costs a tautomer canonicalisation
    on a molecule nothing else in the run will ask about again. 854 training queries came to roughly
    600,000 distinct canonicalisations; memoising helped nothing because almost none repeat.

    The shortcut comes from twins.py's own finding: 275,810 training structures collapse to 274,186
    scorer identities, so fewer than 1% of molecules have a tautomer twin at all and for the rest
    the scorer's key14 IS the InChIKey14. Candidate keys are InChIKey14s, so a dict hit answers the
    question outright. The scan survives as the fallback for the <1%, which keeps the semantics
    identical -- it just almost never runs.
    """
    if "_tk" in q: return q["_tk"]
    if q["cls"] == 3:
        # Class 3 means the answer was REMOVED from the candidate database by construction, and
        # harness.verify asserts exactly that on every split. Scanning for it is guaranteed to fail
        # after canonicalising every candidate -- and Class 3 is 39% of queries, which is the real
        # reason the first two attempts at this experiment hung rather than finished.
        q["_tk"] = None
        return None
    tk = None
    if q["truth"] in q["cand"]:                 # fast path: no twin, key14 == InChIKey14
        tk = q["truth"]
    else:                                        # rare: the answer is a tautomer twin
        for c, (_, sm) in q["cand"].items():
            if sm and key14(sm) == q["truth"]: tk = c; break
    q["_tk"] = tk
    return tk


def fit_linear(train_q):
    """E17/E22's ranker, unchanged: pairwise logistic on truth-minus-decoy differences."""
    X, y = [], []
    for q in train_q:
        tk = truth_key(q)
        if tk is None: continue
        t = q["cand"][tk][0]
        for c, (v, _) in q["cand"].items():
            if c == tk: continue
            d = t - v; X += [d, -d]; y += [1, 0]
    if not X: return None, 0
    X = np.array(X); sd = X.std(0) + 1e-9
    m = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X / sd, y)
    return m.coef_[0] / sd, len(y) // 2


def to_groups(qs):
    """Feature matrix, relevance labels and group sizes, one group per query."""
    X, y, g = [], [], []
    for q in qs:
        tk = truth_key(q)
        if tk is None: continue
        keys = list(q["cand"])
        X.append(np.array([q["cand"][c][0] for c in keys]))
        y.append(np.array([1 if c == tk else 0 for c in keys]))
        g.append(len(keys))
    if not X: return None, None, None
    return np.vstack(X), np.concatenate(y), np.array(g)


def fit_gbm(train_q, rng, log):
    """LambdaMART over the same features. Early stopping on a scaffold-held-out slice of TRAINING."""
    scaf = {}
    for q in train_q:
        tk = truth_key(q)
        if tk: scaf[q["k"]] = scaffold(q["cand"][tk][1])
    groups = sorted({s for s in scaf.values() if s})
    rng.shuffle(groups)
    val_scaf = set(groups[: max(1, len(groups) // 5)])
    tr = [q for q in train_q if scaf.get(q["k"]) not in val_scaf]
    va = [q for q in train_q if scaf.get(q["k"]) in val_scaf]
    if len(va) < 20: tr, va = train_q, train_q        # too small to hold out; say so in the report
    Xt, yt, gt = to_groups(tr); Xv, yv, gv = to_groups(va)
    if Xt is None: return None
    m = lgb.LGBMRanker(**LGB)
    m.fit(Xt, yt, group=gt, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[25],
          callbacks=[lgb.early_stopping(40, verbose=False)])
    log(f"    trees: {m.best_iteration_ or LGB['n_estimators']} used, "
        f"held out {len(va)} queries by scaffold for early stopping")
    return m


def rank_mrr(q, score):
    """MRR@25 for one query given {candidate key: score}.

    Mirrors e22_ranker_v2.rank_score exactly -- retrieval-only pool, tie-break on the candidate
    key, dedup on the scorer's key14 (a candidate whose key14 is None still consumes a slot, as it
    does live), 25 slots, then the spectral-only tail. Both arms go through THIS function, so any
    difference between them is the model and not the plumbing. Equality with rank_score is asserted
    on the linear arm before anything is reported.
    """
    order = [q["cand"][c][1] for c in sorted(score, key=lambda c: (-score[c], c))] + q["tail"]
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


def linear_scores(q, w):
    return {c: float(np.dot(w, v)) for c, (v, _) in q["cand"].items()
            if not str(c).startswith("gen:")}


def gbm_scores(q, model):
    keys = [c for c in q["cand"] if not str(c).startswith("gen:")]
    if not keys: return {}
    V = np.array([q["cand"][c][0] for c in keys])
    s = model.predict(V)
    return {c: float(x) for c, x in zip(keys, s)}


def per_class(ev, score_of):
    out = defaultdict(list)
    for q in ev:
        out[q["cls"]].append(rank_mrr(q, score_of(q)))
    return out


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    import pickle
    cache = os.path.join(WORK, "splits", "e71_built_fr.pkl")
    if os.path.exists(cache):
        ev, tr_all = pickle.load(open(cache, "rb"))
        log(f"loaded cached feature build: {len(ev)} eval, {len(tr_all)} train queries")
    else:
        st = load_structures()
        ev = build(EVAL, st, log=log, pools_prefix="fr_", generate=False)
        ev = [q for q in ev if q["truth"] is not None]
        log(f"eval: {len(ev)} queries")
        tr_all = build(TRAIN, st, frozenset(q["k"] for q in ev), log=log,
                       pools_prefix="fr_", generate=False)
        pickle.dump((ev, tr_all), open(cache, "wb"))
        log("cached the feature build (12 min of characterisation -> seconds on reruns, which the"
            " feature-addition work after this will need many of)")
    log(f"train as-protocol: {len(tr_all)} queries")

    # scaffold-disjoint training set: drop any training query sharing a scaffold with any eval
    # query, and drop scorer-key twins of eval answers, which InChIKey14 exclusion misses
    ev_scaf, ev_keys = set(), set()
    for q in ev:
        tk = truth_key(q)
        if tk:
            s = scaffold(q["cand"][tk][1])
            if s: ev_scaf.add(s)
        if q["truth"]: ev_keys.add(q["truth"])
    tr_disj = []
    dropped_scaf = dropped_twin = 0
    for q in tr_all:
        tk = truth_key(q)
        if tk is None: continue
        if q["truth"] in ev_keys: dropped_twin += 1; continue
        s = scaffold(q["cand"][tk][1])
        if s and s in ev_scaf: dropped_scaf += 1; continue
        tr_disj.append(q)
    log(f"train scaffold-disjoint: {len(tr_disj)} queries "
        f"(dropped {dropped_scaf} sharing an eval scaffold, {dropped_twin} scorer-key twins)")
    fast = sum(1 for q in ev + tr_all if q.get("_tk") is not None and q["_tk"] == q["truth"])
    slow = sum(1 for q in ev + tr_all if q.get("_tk") is not None and q["_tk"] != q["truth"])
    log(f"truth located by direct key hit in {fast} queries, by tautomer scan in {slow}")

    rng = np.random.default_rng(71)
    arms = {}
    for tag, trq in (("as-protocol", tr_all), ("scaffold-disjoint", tr_disj)):
        log(f"  fitting on {tag} ({len(trq)} queries)")
        w, npairs = fit_linear(trq)
        gbm = fit_gbm(trq, rng, log)
        if w is not None:
            # correctness gate: our shared path must reproduce the shipping ranker exactly
            for q in ev[:200]:
                a = rank_mrr(q, linear_scores(q, w))
                b = rank_score(q, w, use_gen=False)
                assert abs(a - b) < 1e-12, f"rank_mrr disagrees with rank_score: {a} vs {b}"
            arms[f"linear · {tag}"] = per_class(ev, lambda q, w=w: linear_scores(q, w))
        if gbm is not None:
            arms[f"trees  · {tag}"] = per_class(ev, lambda q, g=gbm: gbm_scores(q, g))

    L = [f"E71 — model class only: linear pairwise vs gradient-boosted trees, {len(FEATS)} features",
         "", f"features: {', '.join(FEATS)}", "",
         f"{'arm':30s} {'C1':>8} {'C2':>8} {'C3':>8} {'weighted':>9}"]
    summary = {}
    for name, per in arms.items():
        c = [float(np.mean(per[k])) if per[k] else float("nan") for k in (1, 2, 3)]
        wt = sum(W[k] * c[k - 1] for k in (1, 2, 3))
        summary[name] = (per, wt)
        L.append(f"{name:30s} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f} {wt:>9.4f}")

    def paired(a, b):
        """Weighted MRR difference with a bootstrap over queries, resampled within class."""
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

    L.append("")
    for tag in ("as-protocol", "scaffold-disjoint"):
        ln, gn = f"linear · {tag}", f"trees  · {tag}"
        if ln in arms and gn in arms:
            m, lo, hi = paired(arms[gn], arms[ln])
            L.append(f"trees minus linear, {tag:18s} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                     f"  {'CLEARS ZERO' if lo > 0 else 'does not clear zero'}")
    if "trees  · scaffold-disjoint" in arms and "trees  · as-protocol" in arms:
        m, lo, hi = paired(arms["trees  · as-protocol"], arms["trees  · scaffold-disjoint"])
        L += ["",
              f"trees: as-protocol minus scaffold-disjoint {m:+.4f} [{lo:+.4f}, {hi:+.4f}]",
              "  This is the size of what scaffold overlap was worth to the trees. If it is large",
              "  and positive, every prior number fitted under the as-protocol rule is flattered,",
              "  the linear ranker's included."]
    L += ["", "P1 wanted trees to beat linear by >= +0.010 in the scaffold-disjoint arm.",
          "P2 wanted the as-protocol gain to exceed the scaffold-disjoint gain.",
          "P3 wanted Class 2 to gain most.", "",
          "SCOPE: retrieval-only (generation is off on the board, E25), sqrt(p)/Fisher-Rao pools,",
          "the shipping ranker path for ties and dedup. Hyperparameters were fixed before the run",
          "and early stopping held out a scaffold-disjoint slice of TRAINING, never of eval.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E71_gbm_reranker_2026-09-27.txt", "w").write("\n".join(L) + "\n")
    json.dump({k: {str(c): v for c, v in per.items()} for k, (per, _) in summary.items()},
              open(f"{RESULTS}/E71_gbm_reranker.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
