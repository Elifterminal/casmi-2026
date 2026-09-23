#!/usr/bin/env python3
"""E44 — the overlap audit, run BEFORE any gain number. ChatGPT's condition, and he is right.

    "Extend the identity holdout to the predictor's pretraining data. Holding a molecule out of
     your own libraries is insufficient if the imported model trained on it. An overlap-contaminated
     gain cannot justify extending the project."

The open ICEBERG weights are trained on MassSpecGym. Our evaluation molecules come from the
competition's training libraries. If those overlap, a local gain measures the predictor's memory
rather than its chemistry, and would send us spending weeks we do not have.

THERE IS A SHARPER VERSION OF THIS RISK THAN THE ONE I WAS WARNED ABOUT, and it is specific to how
our harness manufactures its classes:

  our Class 2 and Class 3 queries are MANUFACTURED by deleting a molecule from our database and,
  for Class 3, from the spectral index too. But the molecule still HAS a public spectrum -- that
  is precisely why it is in our libraries in the first place. So it can sit in MassSpecGym.

  A REAL Class 2 or Class 3 molecule is defined by having no public spectrum anywhere. It
  therefore CANNOT be in MassSpecGym.

If that asymmetry holds, then any Class 2/3 gain we measure locally from a MassSpecGym-trained
predictor could be memorisation of a spectrum the real test molecules will never have had -- a
gain that is real in our harness and structurally unavailable on the board. That is worse than a
null: it is a null wearing a positive's clothes, on exactly the classes worth 84% of the score.

WHAT THIS MEASURES (identity at the SCORER's key, not raw InChIKey, so tautomers count as the
same molecule -- the same rule our twin-safe harness uses):
  1. share of our evaluation queries, per class, present in MassSpecGym
  2. the same for the molecules our ranker trains on
  3. for the overlapping ones, whether MassSpecGym holds the same adduct too, since a predictor
     that saw the exact (structure, adduct) pair is the strongest form of the leak

PREDICTIONS (before the run, 2026-09-23):
  P1  Evaluation overlap is HIGH, above 60%, and roughly equal across our three classes -- because
      our class labels are a property of the harness, not of the molecule.
  P2  That equality is the finding. It would mean our manufactured Class 2/3 cannot be used to
      estimate a real Class 2/3 gain from this predictor, whatever the number says.
  P3  Adduct-level overlap is substantially lower than structure-level overlap.
"""
import csv, json, os, pickle, sys, time
from collections import defaultdict
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
EVAL_SPLITS = ("nptsseed10_n300", "nptsseed11_n300", "nptsseed12_n300")
TRAIN_SPLITS = ("nptsseed20_n300", "nptsseed21_n300", "nptsseed22_n300")
MSG = os.path.expanduser(
    "/tmp/claude-1000/-home-lee/c046c9a5-0d6b-4917-ae00-430a3bba9009/scratchpad/msg15.tsv")
csv.field_size_limit(10 ** 9)


def _parse_msg(log):
    msg_keys, msg_adducts, n = set(), defaultdict(set), 0
    with open(MSG, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            n += 1
            k = scoring.key14(row.get("smiles", ""))
            if k:
                msg_keys.add(k)
                a = (row.get("adduct") or "").strip()
                if a: msg_adducts[k].add(a)
            if n % 100000 == 0: log(f"  MassSpecGym rows {n:,}, {len(msg_keys):,} structures")
    return msg_keys, msg_adducts, n


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

    # MassSpecGym: structure identity at the scorer's key, plus the adducts seen per structure.
    # Cached -- canonicalising 231k structures costs 13 minutes and the file never changes.
    MCACHE = os.path.join(SPLITS, "massspecgym_keys.pkl")
    if os.path.exists(MCACHE):
        d = pickle.load(open(MCACHE, "rb"))
        msg_keys, msg_adducts, n = d["keys"], d["adducts"], d["rows"]
        log(f"MassSpecGym from cache: {n:,} rows, {len(msg_keys):,} structures")
    else:
        msg_keys, msg_adducts, n = _parse_msg(log)
        pickle.dump(dict(keys=msg_keys, adducts=dict(msg_adducts), rows=n), open(MCACHE, "wb"))
    log(f"MassSpecGym: {n:,} rows -> {len(msg_keys):,} distinct structures at the scorer's key")

    # our queries, with the adduct each was observed under
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["inchikey14", "normalized_smiles", "adduct"])
    keys = np.asarray(t.column("inchikey14")); smis = np.asarray(t.column("normalized_smiles"))
    adds = np.asarray(t.column("adduct"))
    smi_of_key, add_of_key = {}, defaultdict(set)
    for k, s, a in zip(keys, smis, adds):
        smi_of_key.setdefault(k, s); add_of_key[k].add(str(a))
    del t
    log("train.parquet indexed")

    def audit(queries, label):
        per = defaultdict(lambda: defaultdict(int))
        rows = []
        for q in queries:
            cls = q.get("cls", 0)
            sk = scoring.key14(smi_of_key.get(q["k"], ""))
            if sk is None: continue
            hit = sk in msg_keys
            same_add = bool(hit and (add_of_key.get(q["k"], set()) & msg_adducts.get(sk, set())))
            per[cls]["n"] += 1; per[cls]["hit"] += hit; per[cls]["adduct"] += same_add
            rows.append(dict(set=label, query=q["k"], cls=cls, in_massspecgym=bool(hit),
                             same_adduct=bool(same_add)))
        return per, rows

    def queries_of(splits):
        out = []
        for sp_name in splits:
            sp = json.load(open(f"{SPLITS}/split_{sp_name}.json"))
            for k in sp["query_rows"]:
                out.append(dict(k=k, cls=sp["assign"][k]))
        return out
    ev_per, ev_rows = audit(queries_of(EVAL_SPLITS), "eval")
    tr_per, tr_rows = audit(queries_of(TRAIN_SPLITS), "train")

    L = ["E44 — do the open ICEBERG weights already know our evaluation molecules?", "",
         f"MassSpecGym 1.5: {n:,} spectra, {len(msg_keys):,} distinct structures (scorer key)", "",
         f"{'set':7s} {'class':6s} {'queries':>8} {'in MassSpecGym':>15} {'same adduct too':>16}"]
    for label, per in (("eval", ev_per), ("train", tr_per)):
        for c in (1, 2, 3):
            v = per[c]
            if not v["n"]: continue
            L.append(f"{label:7s} C{c:<5d} {v['n']:>8} {v['hit']/v['n']:>14.1%} "
                     f"{v['adduct']/v['n']:>15.1%}")
        tot = {k: sum(per[c][k] for c in (1, 2, 3)) for k in ("n", "hit", "adduct")}
        if tot["n"]:
            L.append(f"{label:7s} {'all':6s} {tot['n']:>8} {tot['hit']/tot['n']:>14.1%} "
                     f"{tot['adduct']/tot['n']:>15.1%}")
    L += ["",
          "THE READING THAT MATTERS. Our Class 2 and Class 3 queries are manufactured by deleting a",
          "molecule from our database -- but the molecule still HAS a public spectrum, which is why",
          "it is in our libraries at all, and so it can sit in MassSpecGym. A REAL Class 2 or 3",
          "molecule has no public spectrum anywhere and therefore CANNOT be in MassSpecGym. If the",
          "overlap above is high and roughly equal across our classes, then our harness cannot",
          "estimate a real Class 2/3 gain from a MassSpecGym-trained predictor at all: the local",
          "number would measure memory the board will never supply.",
          f"\nruntime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E44_overlap_2026-09-23.txt", "w").write("\n".join(L) + "\n")
    json.dump(dict(massspecgym_rows=n, massspecgym_structures=len(msg_keys),
                   eval={f"C{c}": dict(ev_per[c]) for c in (1, 2, 3)},
                   train={f"C{c}": dict(tr_per[c]) for c in (1, 2, 3)}),
              open(f"{RESULTS}/E44_overlap.json", "w"), indent=2, default=int)
    with open(f"{RESULTS}/E44_overlap.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(ev_rows[0])); wr.writeheader()
        wr.writerows(ev_rows + tr_rows)


if __name__ == "__main__":
    main()
