#!/usr/bin/env python3
"""E20 part 1 — the chemistry: turn a known relative into plausible unknowns.

Class 3 molecules are in no database, so retrieval cannot reach them and they score exactly
0.000 (L-024). The route E12/E13 measured is analog editing: take a known relative, read the
precursor-mass difference, and apply the structural edit that explains it. E13 says a relative
that is both close (Tanimoto >= 0.6) and one common edit away sits in the top 5 for 24% of
Class 3 queries with plain cosine search, 46% with better search. That is the ceiling; getting
the edit onto the right ATOM is the part nobody has measured.

This module is only the chemistry. Each edit takes a molecule and returns every plausible
product, one per site. The site list is the point: a methylation on a natural product with
four hydroxyls is four different molecules, and only one of them is the answer.

Every edit declares the neutral mass it adds, and `selftest` checks each product actually has
that mass (a wrong delta silently generates candidates with the wrong formula, which can never
match and just burn submission slots).
"""
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

MAX_SITES = 40          # per edit per relative: natural products have a lot of hydroxyls
# Glucosyl written so the FIRST atom is the anomeric carbon: that is the atom that bonds to the
# parent's oxygen. Writing it from the 6-OH instead built an O-O peroxide and added 178 Da, not 162.
HEXOSE = "C1OC(CO)C(O)C(O)C1O"


def _mol(smi):
    m = Chem.MolFromSmiles(smi)
    return m


def _clean(m):
    """Sanitise and canonicalise a built molecule; None if RDKit rejects it."""
    try:
        s = Chem.MolToSmiles(m)
        mm = Chem.MolFromSmiles(s)
        return s if mm is not None else None
    except Exception:
        return None


def _has_h(a):
    return a.GetTotalNumHs() > 0


def _attach(mol, idx, frag_smiles):
    """Replace one H on atom idx with frag_smiles (attached by its first atom)."""
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None: return None
    combo = Chem.RWMol(Chem.CombineMols(mol, frag))
    off = mol.GetNumAtoms()
    a = combo.GetAtomWithIdx(idx)
    if a.GetNumExplicitHs() > 0: a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
    combo.AddBond(idx, off, Chem.BondType.SINGLE)
    return _clean(combo)


def _delete(mol, idxs):
    """Remove atoms (indices) and cap the neighbours with implicit H."""
    rw = Chem.RWMol(mol)
    for i in sorted(idxs, reverse=True):
        rw.RemoveAtom(i)
    return _clean(rw)


def methylate(mol):
    return [_attach(mol, a.GetIdx(), "C") for a in mol.GetAtoms()
            if a.GetSymbol() in ("O", "N") and _has_h(a)]


def demethylate(mol):
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C" or a.GetDegree() != 1 or a.GetTotalNumHs() != 3: continue
        nb = a.GetNeighbors()[0]
        if nb.GetSymbol() in ("O", "N"): out.append(_delete(mol, [a.GetIdx()]))
    return out


def hydroxylate(mol):
    return [_attach(mol, a.GetIdx(), "O") for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and _has_h(a)]


def dehydroxylate(mol):
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == "O" and a.GetDegree() == 1 and a.GetTotalNumHs() == 1:
            if a.GetNeighbors()[0].GetSymbol() == "C": out.append(_delete(mol, [a.GetIdx()]))
    return out


def acetylate(mol):
    return [_attach(mol, a.GetIdx(), "C(C)=O") for a in mol.GetAtoms()
            if a.GetSymbol() in ("O", "N") and _has_h(a)]


def deacetylate(mol):
    """Strip an O- or N-acetyl: the C(=O)CH3 unit, leaving the OH / NH."""
    patt = Chem.MolFromSmarts("[O,N;!H0,H0]-[CX3](=O)[CH3]")
    out = []
    for match in mol.GetSubstructMatches(patt):
        _, c, o, me = match
        out.append(_delete(mol, [c, o, me]))
    return out


def glycosylate(mol):
    return [_attach(mol, a.GetIdx(), HEXOSE) for a in mol.GetAtoms()
            if a.GetSymbol() == "O" and _has_h(a)]


def deglycosylate(mol):
    """Remove a hexose ring attached through an oxygen, leaving that oxygen as OH."""
    patt = Chem.MolFromSmarts("[OX2]([#6])[CH1]1[OX2][CH1]([CH2][OX2H])[CH1]([OX2H])[CH1]([OX2H])[CH1]1[OX2H]")
    out = []
    for match in mol.GetSubstructMatches(patt):
        out.append(_delete(mol, list(match[2:])))     # keep the linking O, drop the sugar
    return out


def prenylate(mol):
    sites = [a.GetIdx() for a in mol.GetAtoms()
             if (a.GetSymbol() == "O" and _has_h(a)) or (a.GetSymbol() == "C" and a.GetIsAromatic() and _has_h(a))]
    return [_attach(mol, i, "CC=C(C)C") for i in sites]


# ---- edit library v2 (E21): the rest of the list catalogued in E12 -----------------------
# E20 used nine edits and reached a matching mass gap for 19.4% of Class 3 queries; the
# seventeen-edit list from E12 would have reached 36.2%. These are the missing ones, limited to
# edits with an unambiguous structural operation (NH, O2 and CH2O are left out: no single
# obvious site chemistry, so they would generate noise rather than candidates).
PENTOSE = "C1OCC(O)C(O)C1O"              # xylopyranosyl, anomeric carbon first
DEOXYHEXOSE = "C1OC(C)C(O)C(O)C1O"       # rhamnosyl
GLUCURONYL = "C1OC(C(=O)O)C(O)C(O)C1O"   # glucuronic acid, anomeric carbon first


def dehydrate(mol):
    """-H2O: drop a hydroxyl and an H from the neighbouring carbon, leaving a double bond."""
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "O" or a.GetDegree() != 1 or a.GetTotalNumHs() != 1: continue
        c = a.GetNeighbors()[0]
        if c.GetSymbol() != "C": continue
        for nb in c.GetNeighbors():
            if nb.GetIdx() == a.GetIdx() or nb.GetSymbol() != "C" or not _has_h(nb): continue
            if c.GetIsAromatic() or nb.GetIsAromatic(): continue
            rw = Chem.RWMol(mol)
            rw.GetBondBetweenAtoms(c.GetIdx(), nb.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
            rw.RemoveAtom(a.GetIdx())
            out.append(_clean(rw))
    return out


def hydrate(mol):
    """+H2O: add H and OH across a C=C."""
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE: continue
        x, y = b.GetBeginAtom(), b.GetEndAtom()
        if x.GetIsAromatic() or y.GetIsAromatic(): continue
        if x.GetSymbol() != "C" or y.GetSymbol() != "C": continue
        for target in (x.GetIdx(), y.GetIdx()):
            rw = Chem.RWMol(mol)
            rw.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()).SetBondType(Chem.BondType.SINGLE)
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(target, o, Chem.BondType.SINGLE)
            out.append(_clean(rw))
    return out


def decarboxylate(mol):
    """-CO2: remove a carboxylic acid, leaving H on the carbon it hung from."""
    patt = Chem.MolFromSmarts("[#6]-[CX3](=O)[OX2H1]")
    return [_delete(mol, list(m[1:])) for m in mol.GetSubstructMatches(patt)]


def carboxylate(mol):
    return [_attach(mol, a.GetIdx(), "C(=O)O") for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and _has_h(a)]


def sulfate(mol):
    return [_attach(mol, a.GetIdx(), "S(=O)(=O)O") for a in mol.GetAtoms()
            if a.GetSymbol() == "O" and _has_h(a)]


def desulfate(mol):
    patt = Chem.MolFromSmarts("[OX2]-[SX4](=O)(=O)[OX2H1]")
    return [_delete(mol, list(m[1:])) for m in mol.GetSubstructMatches(patt)]


def pentosylate(mol):
    return [_attach(mol, a.GetIdx(), PENTOSE) for a in mol.GetAtoms()
            if a.GetSymbol() == "O" and _has_h(a)]


def deoxyhexosylate(mol):
    return [_attach(mol, a.GetIdx(), DEOXYHEXOSE) for a in mol.GetAtoms()
            if a.GetSymbol() == "O" and _has_h(a)]


def glucuronidate(mol):
    return [_attach(mol, a.GetIdx(), GLUCURONYL) for a in mol.GetAtoms()
            if a.GetSymbol() == "O" and _has_h(a)]


def ethylate(mol):
    return [_attach(mol, a.GetIdx(), "CC") for a in mol.GetAtoms()
            if a.GetSymbol() in ("O", "N") and _has_h(a)]


def reduce_bond(mol):
    """+H2: saturate a C=C, or turn a ketone/aldehyde into an alcohol."""
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE: continue
        x, y = b.GetBeginAtom(), b.GetEndAtom()
        if x.GetIsAromatic() or y.GetIsAromatic(): continue
        pair = {x.GetSymbol(), y.GetSymbol()}
        if pair == {"C"} or pair == {"C", "O"}:
            rw = Chem.RWMol(mol)
            rw.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()).SetBondType(Chem.BondType.SINGLE)
            out.append(_clean(rw))
    return out


def oxidise(mol):
    """-H2: turn a secondary or primary alcohol into a carbonyl."""
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "O" or a.GetDegree() != 1 or a.GetTotalNumHs() != 1: continue
        c = a.GetNeighbors()[0]
        if c.GetSymbol() != "C" or c.GetIsAromatic() or not _has_h(c): continue
        rw = Chem.RWMol(mol)
        rw.GetBondBetweenAtoms(c.GetIdx(), a.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
        out.append(_clean(rw))
    return out


# name -> (neutral mass added, function). Negative edits remove that mass.
EDITS = {
    "+CH2":      (14.015650, methylate),
    "-CH2":     (-14.015650, demethylate),
    "+O":        (15.994915, hydroxylate),
    "-O":       (-15.994915, dehydroxylate),
    "+C2H2O":    (42.010565, acetylate),
    "-C2H2O":   (-42.010565, deacetylate),
    "+C6H10O5": (162.052824, glycosylate),
    "-C6H10O5": (-162.052824, deglycosylate),
    "+C5H8":     (68.062600, prenylate),
    "-H2O":     (-18.010565, dehydrate),
    "+H2O":      (18.010565, hydrate),
    "-CO2":     (-43.989829, decarboxylate),
    "+CO2":      (43.989829, carboxylate),
    "+SO3":      (79.956815, sulfate),
    "-SO3":     (-79.956815, desulfate),
    "+C5H8O4":  (132.042259, pentosylate),
    "+C6H10O4": (146.057909, deoxyhexosylate),
    "+C6H8O6":  (176.032088, glucuronidate),
    "+C2H4":     (28.031300, ethylate),
    "+H2":        (2.015650, reduce_bond),
    "-H2":       (-2.015650, oxidise),
}


def apply_edit(smiles, name, max_sites=MAX_SITES):
    """Every distinct product of applying one edit to one relative."""
    mol = _mol(smiles)
    if mol is None or name not in EDITS: return []
    try:
        prods = EDITS[name][1](mol)
    except Exception:
        return []
    seen, out = set(), []
    for s in prods:
        if not s or s in seen: continue
        seen.add(s); out.append(s)
        if len(out) >= max_sites: break
    return out


def candidates_for(smiles, delta, tol=0.003, max_sites=MAX_SITES):
    """Products of every edit whose mass change matches the observed gap."""
    out = []
    for name, (dm, _) in EDITS.items():
        if abs(dm - delta) <= tol:
            out += [(s, name) for s in apply_edit(smiles, name, max_sites)]
    return out


def selftest():
    """Every edit must change the mass by exactly what it claims, and produce valid molecules."""
    cases = {
        "+CH2": "Oc1ccc(O)cc1", "-CH2": "COc1ccc(O)cc1", "+O": "c1ccccc1C", "-O": "Oc1ccc(O)cc1",
        "+C2H2O": "Oc1ccc(O)cc1", "-C2H2O": "CC(=O)Oc1ccccc1", "+C6H10O5": "Oc1ccc(O)cc1",
        "-C6H10O5": "OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O", "+C5H8": "Oc1ccc(O)cc1",
        "-H2O": "CC(O)CC", "+H2O": "CC=CC", "-CO2": "OC(=O)Cc1ccccc1", "+CO2": "Cc1ccccc1",
        "+SO3": "Oc1ccc(O)cc1", "-SO3": "OS(=O)(=O)Oc1ccccc1", "+C5H8O4": "Oc1ccc(O)cc1",
        "+C6H10O4": "Oc1ccc(O)cc1", "+C6H8O6": "Oc1ccc(O)cc1", "+C2H4": "Oc1ccc(O)cc1",
        "+H2": "CC=CC", "-H2": "CC(O)CC",
    }
    ok = True
    for name, (dm, _) in EDITS.items():
        parent = cases[name]
        pm = Descriptors.ExactMolWt(_mol(parent))
        prods = apply_edit(parent, name)
        if not prods:
            print(f"  FAIL {name:10s} produced nothing from {parent}"); ok = False; continue
        bad = []
        for s in prods:
            m = _mol(s)
            if m is None: bad.append((s, "unparseable")); continue
            d = Descriptors.ExactMolWt(m) - pm
            if abs(d - dm) > 1e-4: bad.append((s, f"delta {d:+.5f} wanted {dm:+.5f}"))
        good = "ok  " if not bad else "FAIL"
        ok &= not bad
        print(f"  {good} {name:10s} {len(prods):2d} product(s) from {parent}"
              + (f"   first: {prods[0]}" if prods else "")
              + (f"\n       {bad[0]}" if bad else ""))
    # negative control: an edit that does not match the observed gap must yield nothing
    n = candidates_for("Oc1ccc(O)cc1", 99.9)
    print(f"  {'ok  ' if not n else 'FAIL'} unmatched gap (99.9 Da) -> {len(n)} candidates (expect 0)")
    ok &= not n
    # site enumeration: a molecule with two OH must give two distinct methylation products
    # site enumeration, on an ASYMMETRIC molecule: hydroquinone's two hydroxyls are equivalent,
    # so it correctly gives ONE product -- the earlier version of this check was simply wrong.
    sym = apply_edit("Oc1ccc(O)cc1", "+CH2")
    asym = apply_edit("OCc1ccccc1O", "+CH2")        # benzylic OH vs phenol OH
    print(f"  {'ok  ' if len(sym) == 1 else 'FAIL'} symmetric hydroquinone -> {len(sym)} product (expect 1)")
    print(f"  {'ok  ' if len(asym) == 2 else 'FAIL'} asymmetric diol -> {len(asym)} products (expect 2)")
    ok &= len(sym) == 1 and len(asym) == 2
    # round trip: glycosylate then deglycosylate must return the parent
    par = "Oc1ccc(O)cc1"
    gly = apply_edit(par, "+C6H10O5")
    back = apply_edit(gly[0], "-C6H10O5") if gly else []
    hit = bool(back) and Chem.CanonSmiles(back[0]) == Chem.CanonSmiles(par)
    print(f"  {'ok  ' if hit else 'FAIL'} glycosylate -> deglycosylate round trip: {back[0] if back else 'nothing'}")
    ok &= hit
    print("\nRESULT:", "edit chemistry verified" if ok else "EDIT CHEMISTRY WRONG")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
