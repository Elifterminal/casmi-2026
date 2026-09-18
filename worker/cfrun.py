#!/usr/bin/env python3
"""Run wrangler with Cloudflare creds injected from ~/.env, so the token never
appears on a shell command line / in the transcript."""
import json, os, subprocess, sys, urllib.request
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")
tok = os.getenv("CLOUDFLARE_API_TOKEN")
if not tok:
    sys.exit("no CLOUDFLARE_API_TOKEN")
req = urllib.request.Request("https://api.cloudflare.com/client/v4/accounts",
                             headers={"Authorization": f"Bearer {tok}"})
acct = json.load(urllib.request.urlopen(req, timeout=20))["result"][0]["id"]
env = dict(os.environ, CLOUDFLARE_API_TOKEN=tok, CLOUDFLARE_ACCOUNT_ID=acct)
r = subprocess.run(sys.argv[1:], env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
sys.exit(r.returncode)
