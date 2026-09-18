#!/usr/bin/env python3
"""E04 step 1 — candidate retrieval by precursor mass. Is the answer even reachable?

Before spending GPU hours learning to RANK candidates, establish whether the true
structure is in the candidate pool at all. If recall is poor, no re-ranker helps and
we have a much earlier kill signal than November.

Neutral monoisotopic mass is recovered from the measured precursor m/z by removing
the adduct, then matched against a structure database within a ppm window.
"""
import re
import numpy as np

# monoisotopic element masses
E = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.9949146196,
     "S": 31.97207100, "P": 30.97376163, "Cl": 34.96885268, "Br": 78.9183371,
     "F": 18.99840322, "I": 126.904473, "Si": 27.9769265325, "Se": 79.9165213,
     "Na": 22.9897692809, "K": 38.96370668, "B": 11.0093054, "As": 74.9215965,
     "Fe": 55.9349375, "D": 2.0141017778}
ELECTRON = 0.00054857990946
PROTON = E["H"] - ELECTRON
H2O = 2 * E["H"] + E["O"]
FORMIC = E["C"] + 2 * E["H"] + 2 * E["O"]

# m/z = M * z_sign_adjusted + SHIFT, for singly charged ions.
# Each entry: measured m/z -> neutral M is  M = mz - shift
ADDUCT_SHIFT = {
    "[M+H]+":        PROTON,
    "[M+NH4]+":      E["N"] + 4 * E["H"] - ELECTRON,
    "[M-H2O+H]+":    PROTON - H2O,
    "[M-2H2O+H]+":   PROTON - 2 * H2O,
    "[M+Na]+":       E["Na"] - ELECTRON,
    "[M+K]+":        E["K"] - ELECTRON,
    "[M-H]-":       -PROTON,
    "[M-H2O-H]-":   -PROTON - H2O,
    "[M+CH2O2-H]-":  FORMIC - PROTON,
    "[M+Cl]-":       E["Cl"] + ELECTRON,
}

_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(f):
    """Monoisotopic mass of a molecular formula string, or nan if unparseable."""
    if not f:
        return np.nan
    f = str(f).strip()
    if not f or not f[0].isupper():
        return np.nan
    m, pos = 0.0, 0
    for sym, cnt in _TOK.findall(f):
        if sym not in E:
            return np.nan
        m += E[sym] * (int(cnt) if cnt else 1)
        pos += len(sym) + len(cnt)
    if pos != len(f):          # trailing junk (charges, brackets, isotopes)
        return np.nan
    return m


def neutral_mass(mz, adduct):
    s = ADDUCT_SHIFT.get(adduct)
    return np.nan if s is None else mz - s


class MassIndex:
    """Structures sorted by neutral mass, queried with a ppm window."""

    def __init__(self, keys, masses):
        ok = np.isfinite(masses)
        self.keys = np.asarray(keys)[ok]
        m = np.asarray(masses)[ok]
        o = np.argsort(m)
        self.keys, self.mass = self.keys[o], m[o]

    def window(self, M, ppm):
        tol = M * ppm / 1e6
        lo = np.searchsorted(self.mass, M - tol, "left")
        hi = np.searchsorted(self.mass, M + tol, "right")
        return self.keys[lo:hi]


if __name__ == "__main__":
    ok = True
    # positive control: glucose C6H12O6 = 180.0633881
    got = formula_mass("C6H12O6")
    good = abs(got - 180.0633881) < 1e-5
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} C6H12O6 -> {got:.7f} (expect 180.0633881)")
    # caffeine C8H10N4O2 = 194.0803756
    got = formula_mass("C8H10N4O2")
    good = abs(got - 194.0803756) < 1e-5
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} C8H10N4O2 -> {got:.7f} (expect 194.0803756)")

    print("\n  adduct round-trips (neutral -> m/z -> neutral):")
    for ad, shift in ADDUCT_SHIFT.items():
        M = 180.0633881
        back = neutral_mass(M + shift, ad)
        good = abs(back - M) < 1e-9
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {ad:15s} m/z {M+shift:10.5f} -> {back:.7f}")

    print("\n  negative controls:")
    for bad in ("", "Xx99", "C6H12O6junk", None):
        v = formula_mass(bad)
        good = np.isnan(v)
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {str(bad)!r:16s} -> {v}")
    v = neutral_mass(200.0, "[M+Unknown]+")
    ok &= np.isnan(v)
    print(f"  {'ok  ' if np.isnan(v) else 'FAIL'} unknown adduct -> {v}")

    print("\nRESULT:", "mass maths verified" if ok else "MASS MATHS WRONG")
