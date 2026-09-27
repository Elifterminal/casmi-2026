#!/usr/bin/env python3
"""E61 — a validation query set for the fork, controlling what we can and measuring what we cannot.

WHY THIS EXISTS. Our local yardstick has read 0.03-0.08 high all project, which made it useless for
comparing against the board or against other teams' published numbers. senanuretin's validation set
tracks his board score within 0.01; his recipe is obscure structures, held out at scaffold level,
stratified so the set sees decoy costs as well as coverage benefits.

WHAT WE CANNOT DO, stated first because it bounds everything below. The fork's rankers train on
precomputed rows shipped as feature matrices: rank_train.npz carries X (142,762 x 31), Y, M and G,
where G is a query-group index. There is NO InChIKey, SMILES or structure field anywhere in it. So
we cannot determine which structures the fork's ranker trained on, and therefore cannot hold our
validation structures out of it by scaffold or by any other key. Regenerating those rows ourselves
would mean reimplementing the part of their system that makes it good.

So this controls three things and leaves the fourth measured-but-uncontrolled:

  1. FAMOUSNESS (controlled). Structures absent from MassSpecGym. E46 measured fame at 0.064 of
     our score -- the largest known contributor to the level gap, and the one senanuretin
     independently found ("a validation set of famous natural products barely sees decoy costs").
  2. COCONUT BALANCE (controlled). Half the queries are COCONUT members, half are known only
     through the training libraries. A set where every answer is in COCONUT can only ever see the
     COST of adding a database, never the benefit -- the flaw that made his V-in reject PubChemLite
     for the wrong reason, and the flaw in every set we have used.
  3. SCAFFOLD RARITY (partially controlled). We prefer structures whose Murcko scaffold is rare in
     the training libraries. This lowers the chance the fork's ranker met that scaffold in its 819
     training groups. It does not prove it did not.
  4. RANKER-TRAINING LEAKAGE (UNCONTROLLED, direction known). Residual risk that a validation
     structure's scaffold sat in those 819 groups. Its effect would be to make our local number read
     HIGH -- the same direction our instrument has always erred -- so local stays an upper bound
     rather than a measurement, and that is how it should be quoted.

Twin-safety at the scorer's key is inherited from the harness, so tautomers of a query can never
appear as a separate query or survive in the reference index.
"""
import json, os, pickle, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))
import scoring

WORK = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(WORK, "splits")
RESULTS = os.path.join(WORK, "results")
COCO = os.path.expanduser("~/casmi-2026/work/data/coconut/coconut_csv_lite-09-2026.csv")
N_PER_ARM = 300          # 300 COCONUT + 300 non-COCONUT
SCAF_MAX = 3             # a scaffold appearing more than this often in the libraries is "common"
MASS_LO, MASS_HI = 100.0, 1300.0


def _scaf(s):
    try:
        m = Chem.MolFromSmiles(s or "")
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else None
    except Exception:
        return None


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
    import pandas as pd

    t = pq.read_table(os.path.join(WORK, "data", "train.parquet"),
                      columns=["inchikey14", "normalized_smiles", "precursor_mz", "ingest_lib"])
    keys = np.asarray(t.column("inchikey14")); smis = np.asarray(t.column("normalized_smiles"))
    pmz = t.column("precursor_mz").to_numpy(); lib = np.asarray(t.column("ingest_lib"))
    del t
    per = defaultdict(lambda: [0, None, []])
    for k, s, m, l in zip(keys, smis, pmz, lib):
        e = per[k]; e[0] += 1
        if e[1] is None: e[1] = s
        e[2].append(m)
    log(f"{len(per):,} structures in the training libraries")

    msg = pickle.load(open(f"{SPLITS}/massspecgym_keys.pkl", "rb"))["keys"]
    co = set(pd.read_csv(COCO, usecols=["standard_inchi_key"], dtype=str)
             .dropna().standard_inchi_key.str.split("-").str[0])
    log(f"MassSpecGym {len(msg):,} structures | COCONUT {len(co):,}")

    # eligibility: enough spectra to hold one out, sane mass, and NOT famous
    elig = {}
    for k, (n, s, ms) in per.items():
        if n < 2 or not s: continue
        med = float(np.median(ms))
        if not (MASS_LO <= med <= MASS_HI): continue
        sk = scoring.key14(s)
        if sk is None or sk in msg: continue          # control 1: famousness
        elig[k] = (s, sk)
    log(f"eligible after mass + spectra-count + MassSpecGym exclusion: {len(elig):,}")

    with Pool(7) as p:
        scafs = dict(zip(elig, p.map(_scaf, [v[0] for v in elig.values()], chunksize=64)))
    freq = Counter(v for v in scafs.values() if v)
    log(f"{len(freq):,} distinct Murcko scaffolds among eligible structures")

    rare = {k for k, v in scafs.items() if v and freq[v] <= SCAF_MAX}
    acyclic = {k for k, v in scafs.items() if v == ""}
    log(f"scaffold-rare (<= {SCAF_MAX} occurrences): {len(rare):,}; acyclic (no scaffold): "
        f"{len(acyclic):,} -- acyclic are EXCLUDED, since rarity is undefined for them")

    # control 2: COCONUT balance, twin-safe within each arm
    rng = np.random.default_rng(2026)
    arms = {"coconut": [], "non_coconut": []}
    seen = set()
    for k in rng.permutation(sorted(rare)):
        s, sk = elig[k]
        if sk in seen: continue                        # twin-safe at the scorer's key
        arm = "coconut" if k in co else "non_coconut"
        if len(arms[arm]) >= N_PER_ARM: continue
        seen.add(sk); arms[arm].append(k)
        if all(len(v) >= N_PER_ARM for v in arms.values()): break

    out = {a: sorted(v) for a, v in arms.items()}
    path = f"{SPLITS}/valset_fork_queries.json"
    json.dump(dict(n_per_arm=N_PER_ARM, scaffold_max_freq=SCAF_MAX,
                   controls=dict(famousness="MassSpecGym excluded",
                                 coconut_balance="half in / half out",
                                 scaffold_rarity=f"Murcko scaffold seen <= {SCAF_MAX} times",
                                 ranker_training_leakage="UNCONTROLLED -- the fork's training rows "
                                                         "carry no structure identity"),
                   queries=out), open(path, "w"))

    L = ["E61 — validation query set for the fork", "",
         f"{'arm':16s} {'selected':>9}",
         f"{'COCONUT':16s} {len(out['coconut']):>9}",
         f"{'non-COCONUT':16s} {len(out['non_coconut']):>9}",
         f"{'total':16s} {sum(len(v) for v in out.values()):>9}", "",
         "controls applied:",
         "  1. famousness      every structure is ABSENT from MassSpecGym (E46: fame = 0.064)",
         "  2. COCONUT balance half in, half out, so the set sees coverage BENEFIT as well as",
         "                     decoy COST -- every previous set of ours could only see cost",
         f"  3. scaffold rarity Murcko scaffold appears <= {SCAF_MAX} times in the libraries",
         "  4. twin-safety     one query per scorer identity; tautomers cannot recur",
         "",
         "NOT CONTROLLED, and quoted with the set: the fork's rankers train on precomputed rows",
         "(rank_train.npz: X, Y, M, G) with no structure identity, so we cannot hold these queries",
         "out of their training. The residual bias would make local read HIGH, which is the",
         "direction our instrument has always erred -- so local remains an UPPER BOUND on the board,",
         "not an estimate of it.",
         f"\nwrote {path}",
         f"runtime {time.time()-t0:.0f}s"]
    print("\n".join(L))
    open(f"{RESULTS}/E61_valset_2026-09-26.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
