#!/usr/bin/env python3
"""Weekly price check: what does 5th place (the last paying slot) cost this week?

Seda's review: the bar is moving (leader 0.353 -> 0.396 in three days), so the Class 3
contribution we need is a moving target. This appends one row per run to a CSV so the trend
is visible rather than remembered. Needs Kaggle credentials in the environment.
"""
import csv, os, subprocess, sys, zipfile, datetime, glob, tempfile

COMP = "enveda-CASMI26-molecule-id-mass-spectra"
OUT = os.path.expanduser("~/casmi-2026/work/results/lb_watch.csv")
KAGGLE = os.path.expanduser("~/comps/.venv/bin/kaggle")


def main():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([KAGGLE, "competitions", "leaderboard", "-c", COMP, "-d", "-p", d],
                       check=True, capture_output=True)
        z = glob.glob(f"{d}/*.zip")
        if z:
            with zipfile.ZipFile(z[0]) as zf: zf.extractall(d)
        rows = list(csv.DictReader(open(glob.glob(f"{d}/*.csv")[0])))
    rows.sort(key=lambda r: -float(r["Score"]))
    scores = [float(r["Score"]) for r in rows]
    # rank is computed from position: the downloaded CSV doesn't always carry a Rank column
    mine = next(((i + 1, r) for i, r in enumerate(rows)
                 if "bobbykershii" in (r.get("TeamMemberUserNames") or "")
                 or "bobby kersh" in (r.get("TeamName") or "").lower()), None)
    row = dict(date=datetime.date.today().isoformat(), teams=len(rows),
               leader=scores[0], fifth=scores[4] if len(scores) > 4 else "",
               tenth=scores[9] if len(scores) > 9 else "", median=scores[len(scores) // 2],
               ours=(mine[1]["Score"] if mine else ""), rank=(mine[0] if mine else ""))
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=list(row))
        if new: wri.writeheader()
        wri.writerow(row)
    gap = (float(row["fifth"]) - float(row["ours"])) if row["ours"] and row["fifth"] else None
    print(f"{row['date']}  teams {row['teams']}  leader {row['leader']}  5th {row['fifth']}  "
          f"median {row['median']}  ours {row['ours']} (rank {row['rank']})"
          + (f"  -> {gap:+.3f} to the paying line" if gap is not None else ""))
    if os.path.exists(OUT):
        prev = list(csv.DictReader(open(OUT)))
        if len(prev) > 1:
            d5 = float(prev[-1]["fifth"]) - float(prev[-2]["fifth"])
            print(f"  5th place moved {d5:+.3f} since {prev[-2]['date']}")


if __name__ == "__main__":
    main()
