#!/usr/bin/env python3
"""Bond dissociation energies, so "how much had to break" stops meaning "how many".

E66 found its one real signal by counting the bonds crossing a fragment boundary: the correct
molecule explains gentle-energy peaks with FEWER broken bonds than its same-mass rivals (0.574
against 0.500 chance, label-controlled). Counting treats an amide C-N, an aromatic ring bond and a
C-S thioether as one unit each, which is precisely the simplification Lee objected to when he said
T bonds to S harder than to N. Summing real energies instead is strictly more physical, and it is
the measure his argument actually describes.

VALUES are mean homolytic bond dissociation energies in kJ/mol, the standard tabulated averages
(Luo, Comprehensive Handbook of Chemical Bond Energies; CRC). They are AVERAGES over molecular
contexts, which matters and is the honest limit of this approach: the real BDE of a given bond
depends on what surrounds it -- resonance, ring strain, neighbouring electronegativity -- and a
lookup keyed on element pair and bond order captures only the first-order part of that. It is a
clear improvement on counting and not a substitute for a quantum calculation.

AROMATIC bonds get a value between single and double, reflecting delocalisation: breaking one
aromatic C:C does not release a fragment, and opening a ring requires breaking two, so aromatic
systems are the most expensive way for a candidate to reach a given fragment mass. That asymmetry
is invisible to a count and is the main thing this table buys.
"""

# (element A, element B, bond order as RDKit reports it) -> kJ/mol.  Keys are stored with the two
# symbols sorted so lookup never depends on bond direction.
_BDE = {
    ("C", "H", 1): 413.0, ("C", "C", 1): 348.0, ("C", "C", 2): 614.0, ("C", "C", 3): 839.0,
    ("C", "N", 1): 293.0, ("C", "N", 2): 615.0, ("C", "N", 3): 891.0,
    ("C", "O", 1): 358.0, ("C", "O", 2): 799.0,
    ("C", "S", 1): 259.0, ("C", "S", 2): 573.0,
    ("C", "F", 1): 485.0, ("C", "Cl", 1): 328.0, ("C", "Br", 1): 276.0, ("C", "I", 1): 240.0,
    ("C", "P", 1): 264.0, ("C", "Si", 1): 301.0, ("B", "C", 1): 372.0,
    ("H", "N", 1): 391.0, ("N", "N", 1): 163.0, ("N", "N", 2): 418.0, ("N", "N", 3): 941.0,
    ("N", "O", 1): 201.0, ("N", "O", 2): 607.0,
    ("H", "O", 1): 463.0, ("O", "O", 1): 146.0, ("O", "O", 2): 495.0,
    ("O", "P", 1): 335.0, ("O", "P", 2): 544.0, ("O", "S", 1): 265.0, ("O", "S", 2): 522.0,
    ("H", "S", 1): 339.0, ("S", "S", 1): 266.0,
    ("F", "H", 1): 565.0, ("Cl", "H", 1): 431.0, ("Br", "H", 1): 366.0, ("H", "I", 1): 299.0,
    ("H", "P", 1): 322.0, ("H", "Si", 1): 318.0,
}

# aromatic bonds, keyed on the element pair alone
_AROM = {("C", "C"): 518.0, ("C", "N"): 480.0, ("C", "O"): 440.0, ("C", "S"): 400.0,
         ("N", "N"): 420.0, ("N", "O"): 380.0}

# fallback by bond order when a pair is not tabulated: the median of the tabulated values at that
# order, so an exotic pair costs something typical rather than zero or infinity
_BY_ORDER = {1: 330.0, 2: 580.0, 3: 890.0}
_AROM_DEFAULT = 480.0


def bond_energy(bond):
    """Tabulated BDE in kJ/mol for one RDKit bond. Never returns 0, so a broken bond always costs."""
    a, b = bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
    pair = tuple(sorted((a, b)))
    if bond.GetIsAromatic():
        return _AROM.get(pair, _AROM_DEFAULT)
    order = int(round(bond.GetBondTypeAsDouble()))
    if order not in (1, 2, 3):          # dative, unspecified and other oddities
        order = 1
    v = _BDE.get(pair + (order,))
    return v if v is not None else _BY_ORDER[order]


def bond_table(mol):
    """Per-bond (atom i, atom j, BDE, is_ring) for a molecule, in RDKit bond order."""
    out = []
    for b in mol.GetBonds():
        out.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_energy(b), bool(b.IsInRing())))
    return out


def broken(table, mask):
    """What it cost this fragment to exist: (count, summed kJ/mol, ring bonds broken).

    A bond is broken when the mask puts its two atoms on opposite sides of the boundary. The mask
    covers heavy atoms only, so hydrogens are never counted -- consistent with ICEBERG, which
    enumerates fragments over the heavy-atom skeleton.
    """
    n = len(mask)
    cnt = ring = 0
    energy = 0.0
    for i, j, e, in_ring in table:
        if i < n and j < n and mask[i] != mask[j]:
            cnt += 1; energy += e; ring += in_ring
    return cnt, energy, ring
