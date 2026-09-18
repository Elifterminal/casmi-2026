#!/usr/bin/env python3
"""MRR@25 exactly as the competition scores it.

Matching rule from the competition page: both the prediction and the answer are
passed through RDKit's tautomer canonicalization (pinned 2026.03.3) and reduced to
the first block of their InChIKey (InChIKey14), then compared. So stereochemistry
and tautomer form are free.

Nothing here is trusted until it reproduces the competition's own worked example.
"""
from functools import lru_cache
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_ENUM = rdMolStandardize.TautomerEnumerator()
MAX_GUESSES = 25


@lru_cache(maxsize=1_000_000)
def key14(smiles: str):
    """Canonical InChIKey14 for a SMILES, or None if unparseable."""
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        m = _ENUM.Canonicalize(m)
    except Exception:
        pass                      # fall back to the raw molecule
    try:
        k = inchi.MolToInchiKey(m)
    except Exception:
        return None
    return k.split("-")[0] if k else None


def rr(truth_smiles, guesses, precomputed_truth_key=None):
    """Reciprocal rank of the first correct guess, 0.0 if none in 25."""
    t = precomputed_truth_key or key14(truth_smiles)
    if t is None:
        return 0.0
    for i, g in enumerate(guesses[:MAX_GUESSES], start=1):
        if key14(g) == t:
            return 1.0 / i
    return 0.0


def mrr(records):
    """records: iterable of (truth_smiles, [guess_smiles, ...])"""
    rs = [rr(t, g) for t, g in records]
    return (sum(rs) / len(rs)) if rs else 0.0


def weighted_mrr(per_class, weights=(0.16, 0.45, 0.39)):
    """Combine per-class MRR into one number comparable to the leaderboard.

    NOTE: the 16/45/39 weights are UNVERIFIED — reverse-engineered by a competitor
    and posted publicly; the real split is hidden. Always report the three class
    scores alongside this, never this number alone.
    """
    w1, w2, w3 = weights
    return (w1 * per_class.get(1, 0.0)
            + w2 * per_class.get(2, 0.0)
            + w3 * per_class.get(3, 0.0))


if __name__ == "__main__":
    import sys
    ok = True

    # --- positive control: the competition's OWN worked example -------------
    # Both of these are stated on the competition page to score identically
    # against an answer of glucose, reducing to InChIKey14 WQZGKKKJIJFFOK.
    stereo = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
    flat   = "OCC1OC(O)C(O)C(O)C1O"
    print("competition's published example (both must be WQZGKKKJIJFFOK):")
    for s in (stereo, flat):
        k = key14(s)
        good = k == "WQZGKKKJIJFFOK"
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {k}   {s}")

    print("\nstereochemistry must be free:")
    got = rr(stereo, [flat])
    print(f"  {'ok  ' if got == 1.0 else 'FAIL'} flat guess vs stereo truth -> rr={got}")
    ok &= got == 1.0

    # --- negative controls: the scorer must be able to say NO --------------
    print("\nnegative controls (a scorer that never fails is worthless):")
    cases = [
        ("wrong molecule scores 0", rr("CCO", ["c1ccccc1"]), 0.0),
        ("unparseable guess scores 0", rr("CCO", ["not_a_smiles"]), 0.0),
        ("unparseable truth scores 0", rr("!!!", ["CCO"]), 0.0),
        ("empty guess list scores 0", rr("CCO", []), 0.0),
        ("rank 2 scores 0.5", rr("CCO", ["c1ccccc1", "CCO"]), 0.5),
        ("rank 25 scores 0.04", rr("CCO", ["c1ccccc1"] * 24 + ["CCO"]), 1 / 25),
        ("past 25 scores 0", rr("CCO", ["c1ccccc1"] * 25 + ["CCO"]), 0.0),
    ]
    for name, got, want in cases:
        good = abs(got - want) < 1e-12
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:34s} got {got:.4f} want {want:.4f}")

    print("\nRESULT:", "scorer matches the competition's stated rules" if ok else "SCORER IS WRONG")
    sys.exit(0 if ok else 1)
