#!/usr/bin/env python3
"""Build gate. Fails the build on drift or on a page that does not identify its project.

Constraint 0 (living-page format): a living page belongs to exactly ONE project and must
announce it by name in its <h1> and <title>. A document type is not a project name.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
GENERIC = {"asset log", "log", "notebook", "findings", "index", "study", "notes",
           "record", "project", "docs", "report", "page", "journal", "dashboard"}
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def run(manifest_path=None, quiet=False):
    global FAILS
    FAILS = []
    import importlib, gen_docs
    importlib.reload(gen_docs)
    m = json.load(open(manifest_path or os.path.join(ROOT, "manifest.json")))

    proj = (m.get("project") or "").strip()
    doct = (m.get("doctype") or "").strip()

    # --- Constraint 0 -------------------------------------------------------
    check(bool(proj), "manifest has no 'project' field")
    check(proj.lower() not in GENERIC,
          f"project name {proj!r} is a generic document type, not a project")
    check(bool(doct), "manifest has no 'doctype' field")

    # --- Lee's Rules: the channel must be well-formed ---------------------
    import datetime
    HANDLES = {"elif", "seda", "gpt", "lee"}
    thread = gen_docs.load_thread()
    seen = []
    for k, msg in enumerate(thread):
        for f in ("ts", "from", "to", "body", "state", "ask"):
            check(f in msg and str(msg.get(f, "")).strip(),
                  f"chat message {k} is missing a non-empty {f!r} (Lee's Rules require it)")
        if "ts" in msg:
            try:
                datetime.datetime.strptime(msg["ts"], "%Y-%m-%dT%H:%M:%SZ")
                seen.append(msg["ts"])
            except ValueError:
                FAILS.append(f"chat message {k} timestamp {msg['ts']!r} is not ISO-8601 UTC seconds+Z")
        check(msg.get("from") in HANDLES, f"chat message {k} has unknown sender {msg.get('from')!r}")
        check(msg.get("to") in HANDLES | {"all"}, f"chat message {k} has unknown recipient {msg.get('to')!r}")
        b = str(msg.get("body", ""))
        sm = re.match(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]\s+([a-z]+)\s*(?:\u2192|->)\s*([a-z]+):", b)
        check(sm is not None,
              f"chat message {k} body does not open with the Rule 1 stamp '[<ts>] <from> \u2192 <to>:'")
        if sm:
            check(sm.group(1) == msg.get("ts"),
                  f"chat message {k} stamp time {sm.group(1)!r} disagrees with its ts field {msg.get('ts')!r}")
            check(sm.group(2) == msg.get("from"),
                  f"chat message {k} stamp names {sm.group(2)!r} but the from field says {msg.get('from')!r}")
            check(sm.group(3) == msg.get("to"),
                  f"chat message {k} stamp addresses {sm.group(3)!r} but the to field says {msg.get('to')!r}")
        if msg.get("back") is not None:
            check("|" in str(msg["back"]),
                  f"chat message {k} declares 'back' but not as '<iso-ts>|<why>'")
    check(seen == sorted(seen), "chat thread is not in chronological order (append-only, never reorder)")
    if FAILS:
        if not quiet:
            print("BUILD CHECK FAILED")
            for f in FAILS:
                print("  x", f)
        return list(FAILS)

    inner = gen_docs.build_inner(m) if proj else ""
    h1 = re.search(r"<h1>(.*?)</h1>", inner)
    check(h1 is not None, "rendered page has no <h1>")
    if h1 and proj:
        check(proj in h1.group(1),
              f"<h1> is {h1.group(1)!r} but must contain the project name {proj!r}")
        check(h1.group(1).strip().lower() not in GENERIC,
              f"<h1> {h1.group(1)!r} is a generic noun")

    # kicker must carry the doctype, above the name, not instead of it
    kick = re.search(r'<p class="kicker">(.*?)</p>', inner)
    check(kick is not None and doct in kick.group(1),
          "document type must appear as a kicker above the h1")

    # --- drift --------------------------------------------------------------
    for t in m.get("tabs", []):
        check(f'id="{t["id"]}"' in inner, f"tab {t['id']!r} has no matching panel")
    n_panels = len(re.findall(r'<div class="panel[^"]*" id=', inner))
    check(n_panels == len(m.get("tabs", [])),
          f"{n_panels} panels rendered but {len(m.get('tabs', []))} tabs declared")
    actives = len(re.findall(r'<div class="panel active"', inner))
    check(actives == 1, f"exactly one panel must start active, found {actives}")
    tab_actives = len(re.findall(r'<button class="tab active"', inner))
    check(tab_actives == 1, f"exactly one tab must start active, found {tab_actives}")
    am = re.search(r'<div class="panel active" id="([^"]+)"', inner)
    at = re.search(r'<button class="tab active" data-panel="([^"]+)"', inner)
    check(am and at and am.group(1) == at.group(1),
          "the active tab and the active panel are different panels")

    n_jump = len(re.findall(r'<a class="jump"', inner))
    check(n_jump == len(m.get("log", [])),
          f"jump index has {n_jump} entries but there are {len(m.get('log', []))} log entries")
    ids = [e["id"] for e in m.get("log", [])]
    check(len(ids) == len(set(ids)), "duplicate log entry ids")
    dates = [e["date"] for e in m.get("log", [])]
    check(dates == sorted(dates, reverse=True),
          "log entries must be newest-first; the jump index renders in manifest order")
    for e in m.get("log", []):
        check(f'id="{e["id"]}"' in inner, f"log entry {e['id']} not rendered")

    if thread:
        check(len(re.findall(r'<div class="winmark">', inner)) == 1,
              "the chat panel must mark exactly one window")
        shown = inner.split('<div class="winmark">')[1].split("<details")[0]
        check(shown.count('<div class="msg ') <= gen_docs.WINDOW,
              f"more than {gen_docs.WINDOW} messages rendered in the window; Lee's Rules cap it")
        check(gen_docs.WINDOW == 3, f"the window is set to {gen_docs.WINDOW}; Lee's Rules say 3")

    check(bool(m.get("status")), "manifest has no 'status' line")
    check(len(m.get("stats", [])) >= 3, "header needs at least 3 stats")

    # --- self-contained -----------------------------------------------------
    check(not re.search(r'<(script|link|img)[^>]+(src|href)\s*=', inner),
          "page must make zero external requests")
    check("http://" not in inner.replace("http://www.w3.org", ""),
          "insecure external URL in page")

    # --- secrets ------------------------------------------------------------
    built = os.path.join(ROOT, "docs", "index.html")
    if os.path.exists(built):
        out = open(built).read()
        check("LivingPage" not in out, "a gate password leaked into the built page")
        check(not re.search(r'password\s*[:=]\s*["\'][^"\']+', out, re.I),
              "something password-shaped is embedded in the built page")

    if not quiet:
        if FAILS:
            print("BUILD CHECK FAILED")
            for f in FAILS:
                print("  x", f)
        else:
            print(f"build checks passed  ({len(m['tabs'])} tabs, {len(m['log'])} log entries, "
                  f"project={proj!r})")
    return list(FAILS)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
