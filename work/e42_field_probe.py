#!/usr/bin/env python3
"""E42 — shape one real candidate pool into a panaesthesis field and see if it reads.

Vanta's proposal: connect the query AND its candidate pool as the PARTS of one field, so each
candidate's relation to the query and to the rest of the pool is read at once instead of as 65
independent pairwise scores. Her own discipline applies to me here -- "it is easy to form a view
about a tool before running it, and that view is not a measurement" -- so this runs it rather
than reasoning about whether it would work.

THE CONNECTOR'S CONTRACT (read from adapters.py, not assumed):
    data:<path> -> .npz with 'values' or 'inputs' shaped (observations, parts, profile)
    plus optional 'parts' names. Nothing about any domain is assumed; every column is a tap.

THE SHAPING, and the one honest compromise in it:
    parts        the query, then every candidate in its 10 ppm pool
    profile      a shared mass axis -- the union of observed peaks and candidate fragment masses,
                 quantised, so query and candidates become commensurate vectors
    observations the query's collision-energy frames

    The compromise: a candidate's profile does NOT vary across collision energy, because a
    candidate is a structure and all we can compute is the SET of masses it could produce -- not
    how much of each, and not how that changes with energy. So candidate parts are constant down
    the observation axis while the query part moves. Diastema reads fine against that (it
    compares parts by their profiles). Symploke should read candidate-candidate coupling as
    degenerate, and if it does that is the shaping being honest, not the instrument failing.

    That asymmetry is exactly the thing our scalar scoring throws away, and it is the reason to
    look here at all.

WHAT WOULD COUNT AS A RESULT (declared before the run, 2026-09-23):
  The pool is Class 2 with the truth at rank 4 -- our exact failure mode: the answer is present
  and we do not put it first.
  P1  The field reads non-degenerate: the Diastema distances between parts are not all equal and
      not collapsed to zero.
  P2  WEAK TEST, and the one that matters: the truth's relation to the field differs from the
      median decoy's by more than the rank-4 ordering already implies. If the truth looks exactly
      like a decoy in field terms, this door is closed and I will say so.
  P3  Symploke on candidate-candidate pairs is degenerate, for the reason above.
"""
import json, os, subprocess, sys
import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))
EXPORTS = os.path.expanduser("~/casmi-2026/exports")
VENV = os.path.expanduser("~/.panaesthesis-venv/bin/python")
QUERY = "AFTBPUXZTDLRSP"        # class 2, truth at rank 4, 66 candidates
QUANT = 0.05                     # Da per profile bin
MAX_BINS = 2000
MAX_PARTS = 80


def main():
    meta = json.load(open(f"{EXPORTS}/casmi_pools.json"))
    q = next(x for x in meta["queries"] if x["query"] == QUERY)
    d = np.load(f"{EXPORTS}/casmi_pools.npz")
    print(f"query {QUERY}  class {q['cls']}  {q['n_candidates']} candidates  "
          f"truth rank {q['truth_rank']}  {q['n_frames']} frames")

    frames = [(d[f"{QUERY}/spec{j}_mz"], d[f"{QUERY}/spec{j}_intensity"])
              for j in range(q["n_frames"])]
    frag = d[f"{QUERY}/fragments"]; off = d[f"{QUERY}/fragment_offsets"]
    cands = q["candidates"][:MAX_PARTS - 1]

    # shared mass axis: bins any part actually occupies, most-populated first
    allm = [mz for mz, _ in frames] + [frag[off[i]:off[i + 1]] for i in range(len(cands))]
    bins = np.unique(np.rint(np.concatenate(allm) / QUANT).astype(np.int64))
    if len(bins) > MAX_BINS:
        counts = np.zeros(len(bins))
        for m in allm:
            b = np.rint(np.asarray(m) / QUANT).astype(np.int64)
            idx = np.searchsorted(bins, b)
            idx = idx[(idx >= 0) & (idx < len(bins))]
            np.add.at(counts, idx, 1.0)
        bins = np.sort(bins[np.argsort(-counts)[:MAX_BINS]])
    W = len(bins)
    print(f"profile axis: {W} bins at {QUANT} Da")

    def place(mz, wt):
        v = np.zeros(W, np.float32)
        b = np.rint(np.asarray(mz, np.float64) / QUANT).astype(np.int64)
        idx = np.searchsorted(bins, b)
        ok = (idx >= 0) & (idx < len(bins))
        ok &= bins[np.clip(idx, 0, len(bins) - 1)] == b
        np.add.at(v, idx[ok], np.asarray(wt, np.float64)[ok] if np.ndim(wt) else wt)
        return v

    P = 1 + len(cands)
    values = np.zeros((len(frames), P, W), np.float32)
    for n, (mz, it) in enumerate(frames):
        values[n, 0] = place(mz, it)                       # part 0 = the query, per frame
        for i, c in enumerate(cands, 1):
            f = frag[off[i - 1]:off[i]]
            values[n, i] = place(f, 1.0)                   # candidates: presence, constant
    # normalise each part-profile to a distribution, which is what the equations read
    s = values.sum(axis=2, keepdims=True)
    values = np.divide(values, np.maximum(s, 1e-12))

    names = ["QUERY"] + [f"c{c['rank']:03d}{'_TRUTH' if c['is_truth'] else ''}" for c in cands]
    out = f"{EXPORTS}/field_{QUERY}.npz"
    # 'inputs' rather than 'values': the model adapter accepts either, but the STIMULUS loader
    # requires 'inputs', and one file has to serve as both (the field CLI otherwise falls back to
    # the tool's built-in 24x24 demo glyphs and dies on the shape mismatch).
    np.savez_compressed(out, inputs=values, parts=np.array(names))
    print(f"wrote {out}  values{values.shape}  "
          f"(observations, parts, profile)  nonzero {np.count_nonzero(values):,}")
    truth = [i for i, n in enumerate(names) if n.endswith("_TRUTH")]
    print(f"truth part index: {truth}")

    outdir = f"{EXPORTS}/field_run_{QUERY}"
    cmd = [VENV, "-m", "hodos_monitor", "field", "--model", f"data:{out}",
           "--stimulus", f"array:{out}", "--out", outdir]
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print("STDERR:\n" + r.stderr[-3000:])
    print(f"exit {r.returncode}")
    if os.path.isdir(outdir):
        print("produced:", sorted(os.listdir(outdir)))


if __name__ == "__main__":
    main()
