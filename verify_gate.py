#!/usr/bin/env python3
"""Proves the shipped page unlocks with each real credential, rejects wrong ones,
and carries no plaintext. Reads passwords from ~/.elif_accounts.json, never prints them."""
import base64, hashlib, json, os, sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs/index.html")).read()
i = html.index("var VAULT=") + len("var VAULT=")
depth = 0
for j in range(i, len(html)):
    if html[j] == "{": depth += 1
    elif html[j] == "}":
        depth -= 1
        if depth == 0:
            V = json.loads(html[i:j+1]); break
d = base64.b64decode

def unlock(user, pw):
    rec = next((r for r in V["users"] if r["u"] == user), None)
    if not rec: return None
    kek = hashlib.pbkdf2_hmac("sha256", pw.encode(), d(rec["s"]) + user.encode(), V["it"], 32)
    try:
        ck = AESGCM(kek).decrypt(d(rec["i"]), d(rec["w"]), user.encode())
        return AESGCM(ck).decrypt(d(V["iv"]), d(V["ct"]), None).decode()
    except Exception:
        return None

blob = json.load(open(os.path.expanduser("~/.elif_accounts.json")))
creds = [(a["username"], a["password"]) for a in blob[0]["accounts"]
         if a.get("site") == "casmi-2026 living page"]
ok = True
print(f"iterations: {V['it']:,} | users in vault: {len(V['users'])} | ciphertext: {len(V['ct']):,} b64 chars")
print("real credentials (all must UNLOCK):")
for u, pw in sorted(creds):
    p = unlock(u, pw)
    good = p is not None and "<h1>CASMI 2026</h1>" in p
    print(f"  {u:8s} {'UNLOCKS' if good else '!!! FAILED'}  {len(p) if p else 0:,} bytes")
    ok &= good
print("negative controls (all must be REJECTED):")
first = sorted(creds)[0][0]
for u, pw, why in [(first, "LivingPage%9", "right user, wrong password"),
                   (first, sorted(creds)[0][1].lower(), "password wrong case"),
                   ("user9", sorted(creds)[0][1], "unknown username"),
                   (first, "", "empty password")]:
    r = unlock(u, pw)
    print(f"  {why:28s} {'REJECTED' if r is None else '!!! ACCEPTED'}")
    ok &= (r is None)
leaks = [s for s in ("Class 2", "mass spectrum", "kill gate", "COCONUT") if s in html]
print("plaintext absent from shipped file:", "PASS" if not leaks else f"!!! FAIL {leaks}")
ok &= not leaks
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
