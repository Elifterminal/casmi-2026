# CASMI 2026 — submission notebook (offline).
#
# Everything the model knows ships in the attached dataset: the pipeline module, COCONUT
# (trimmed to key / SMILES / mass), the learned E18 weights, and pinned RDKit wheels.
# The same file runs locally for the dry run; paths are detected, not hard-coded.
import glob, json, os, subprocess, sys, time

T0 = time.time()
ON_KAGGLE = os.path.exists("/kaggle/input")
if ON_KAGGLE:
    COMP = glob.glob("/kaggle/input/*casmi26*molecule*")[0]
    ASSETS = [d for d in glob.glob("/kaggle/input/*") if os.path.exists(f"{d}/casmi_pipeline.py")][0]
    OUT = "/kaggle/working/submission.csv"
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
    COMP = os.path.join(HERE, "..", "data")
    ASSETS = os.path.join(HERE, "dataset")
    OUT = os.path.join(HERE, "submission.csv")


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ---- RDKit, pinned to the version the scorer uses ------------------------------------
try:
    import rdkit
    have = rdkit.__version__
except ImportError:
    have = None
if have != "2026.03.3":
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    whl = glob.glob(f"{ASSETS}/wheels/rdkit-2026.3.3-{tag}-*.whl")
    if not whl:
        raise RuntimeError(f"no bundled RDKit wheel for {tag}; have {have}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", whl[0]], check=True)
    log(f"installed {os.path.basename(whl[0])} (was {have})")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, ASSETS)
import casmi_pipeline as cp
from rdkit import rdBase
log(f"rdkit {rdBase.rdkitVersion}, python {sys.version.split()[0]}, kaggle={ON_KAGGLE}")

# ---- reference library: every training spectrum --------------------------------------
t = pq.read_table(f"{COMP}/train.parquet",
                  columns=["inchikey14", "normalized_smiles", "molecular_formula", "ms2_mzs", "ms2_normalized_intensities"])
keys = np.asarray(t.column("inchikey14"))
# Training SMILES only for structures whose formula parses -- exactly the set E17/E18 used
# (validate_port.py holds this module to E18's numbers with that mapping).
_st = pd.DataFrame({"k": keys, "s": np.asarray(t.column("normalized_smiles")),
                    "f": np.asarray(t.column("molecular_formula"))}).drop_duplicates("k")
_st = _st[np.isfinite(_st.f.map(cp.formula_mass))]
train_smi = dict(zip(_st.k, _st.s)); del _st


def flat(c):
    a = t.column(c).combine_chunks()
    return a.values.to_numpy(zero_copy_only=False).astype(np.float32), a.offsets.to_numpy()


mzf, mzo = flat("ms2_mzs"); itf, ito = flat("ms2_normalized_intensities"); del t
def peaks(i): return mzf[mzo[i]:mzo[i + 1]], itf[ito[i]:ito[i + 1]]
log(f"train loaded: {len(keys):,} spectra, {len(train_smi):,} structures")
sindex = cp.SpectralIndex(keys, peaks, np.arange(len(keys)))
log("spectral index built")

# ---- candidate database: COCONUT -----------------------------------------------------
co = pd.read_parquet(f"{ASSETS}/coconut_min.parquet")
midx = cp.MassIndex(co.inchikey14.to_numpy(), co.mass.to_numpy())
coco_smi = dict(zip(co.inchikey14, co.smiles)); del co
smi_of = lambda k: train_smi.get(k) or coco_smi.get(k, "")
weights = json.load(open(f"{ASSETS}/weights.json"))["weights"]
log(f"COCONUT: {len(coco_smi):,} structures; weights {list(weights)}")

# ---- test molecules --------------------------------------------------------------------
test = pd.read_parquet(f"{COMP}/test.parquet")
mols = [dict(id=mid, spectra=[(np.asarray(r.ms2_mzs, np.float32), np.asarray(r.ms2_normalized_intensities, np.float32),
                               r.adduct, float(r.precursor_mz), r.ionization_mode) for r in g.itertuples()])
        for mid, g in test.groupby("molecule_id", sort=False)]
log(f"test: {len(test):,} spectra, {len(mols)} molecules")

pred = cp.predict(mols, sindex, midx, smi_of, weights, procs=os.cpu_count() or 4, log=log)

# ---- submission ------------------------------------------------------------------------
FALLBACK = "CCO"          # never emit an empty row: a missing/null prediction rejects the file
rows, empty = [], 0
for m in mols:
    smis = [s for s in (smi_of(k) for k in pred.get(m["id"], [])) if s and ";" not in s][:cp.TOPN]
    if not smis: smis, empty = [FALLBACK], empty + 1
    rows.append((m["id"], ";".join(smis)))
sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])

# the competition's rejection rules, checked before writing
assert sub.molecule_id.is_unique, "duplicate molecule_id"
assert set(sub.molecule_id) == set(test.molecule_id), "molecule_id set differs from test"
assert sub.smiles.notna().all() and (sub.smiles.str.len() > 0).all(), "null/empty prediction"
assert sub.smiles.str.split(";").map(len).max() <= cp.TOPN, "more than 25 guesses"
sub.to_csv(OUT, index=False)
log(f"wrote {OUT}: {len(sub)} molecules, median {int(sub.smiles.str.split(';').map(len).median())} guesses, "
    f"{empty} fallback rows")
