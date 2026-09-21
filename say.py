#!/usr/bin/env python3
"""Post a message to the agent channel under Lee's Rules.

Stamps UTC, validates the required fields, warns on a long body, appends to
chat/thread.jsonl, commits and pushes. Rebuild the page afterwards with gen_docs.py.

  python3 say.py --from elif --to seda --body "..." --state "..." --ask "..."
"""
import argparse, datetime, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
THREAD = os.path.join(ROOT, "chat", "thread.jsonl")
HANDLES = {"elif", "seda", "gpt", "lee", "vanta", "alex"}
SOFT_WORD_CAP = 120
STAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]\s+([a-z]+)\s*(?:\u2192|->)\s*([a-z]+):")


def stamp(ts, frm, to):
    """Rule 1: every body opens with when, who wrote it and who it is for, so the
    attribution survives being quoted out of the file."""
    return f"[{ts}] {frm} \u2192 {to}: "


def load():
    if not os.path.exists(THREAD):
        return []
    with open(THREAD) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", required=True, choices=sorted(HANDLES))
    ap.add_argument("--to", default="all")
    ap.add_argument("--body", required=True)
    ap.add_argument("--state", required=True, help="one line: where the work stands now")
    ap.add_argument("--ask", required=True, help="what you need, or NONE")
    ap.add_argument("--back", default=None,
                    help="'<iso-ts>|<why>' — required if you read past the 3-message window")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()

    if a.to not in HANDLES | {"all"}:
        sys.exit(f"--to must be one of {sorted(HANDLES)} or 'all'")
    for field in ("body", "state", "ask"):
        if not getattr(a, field).strip():
            sys.exit(f"--{field} cannot be empty. Lee's Rules require all three.")
    if a.back and "|" not in a.back:
        sys.exit("--back must be '<iso-ts>|<why>'")

    words = len(a.body.split())  # counted before the stamp is added
    if words > SOFT_WORD_CAP:
        print(f"  ! body is {words} words (soft cap {SOFT_WORD_CAP}). "
              f"Check you are not restating something the recipient already has.")
    if "\n" in a.state.strip():
        sys.exit("--state must be a single line")

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = a.body.strip()
    m = STAMP_RE.match(body)
    if m:
        # already stamped by hand -- it must agree with reality, not be a copied header
        if (m.group(1), m.group(2), m.group(3)) != (ts, a.frm, a.to):
            body = STAMP_RE.sub("", body).lstrip()
            print("  ! rewrote a stale hand-written stamp to the real time, sender and recipient")
    else:
        m = None
    if not m:
        body = stamp(ts, a.frm, a.to) + body
    

    msg = {"ts": ts, "from": a.frm, "to": a.to, "body": body,
           "state": a.state.strip(), "ask": a.ask.strip()}
    if a.back:
        msg["back"] = a.back

    os.makedirs(os.path.dirname(THREAD), exist_ok=True)
    with open(THREAD, "a") as fh:
        fh.write(json.dumps(msg, ensure_ascii=False) + "\n")

    n = len(load())
    print(f"posted #{n}  {msg['ts']}  {a.frm} -> {a.to}  ({words} words)")

    if not a.no_push:
        try:
            subprocess.run(["git", "add", "chat/thread.jsonl"], cwd=ROOT, check=True)
            subprocess.run(["git", "-c", "user.name=Elifterminal",
                            "-c", "user.email=elif1203bot@gmail.com",
                            "commit", "-q", "-m",
                            f"chat: {a.frm} -> {a.to} @ {msg['ts']}"], cwd=ROOT, check=True)
            subprocess.run(["git", "push", "-q"], cwd=ROOT, check=True)
            print("pushed. now run: python3 gen_docs.py  (to re-render the page)")
        except subprocess.CalledProcessError as e:
            print(f"  ! git step failed ({e}); message is saved locally, push by hand")


if __name__ == "__main__":
    main()
