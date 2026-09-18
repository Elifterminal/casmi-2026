# CASMI 2026

Working notebook for our entry into [Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
on Kaggle — predicting a molecule's 2D structure from its tandem mass spectrum.

**The page:** https://elifterminal.github.io/casmi-2026/

It's behind a login. Ask Lee for a credential.

## How this repo works

`manifest.json` is the source of truth for the page. It is **not** in this repo, and it
shouldn't be — publishing it would make the login decorative. It lives locally.

```
manifest.json      the record (local only, gitignored)
figures.py         hand-emitted inline SVG, no chart library
gen_docs.py        manifest -> docs/index.html, encrypts the payload
check_manifest.py  build gate; fails on drift or a page that doesn't name its project
verify_gate.py     proves the built page unlocks with each real credential and rejects wrong ones
docs/index.html    what gets published — ciphertext plus a login form
```

Rebuild:

```bash
python3 check_manifest.py && python3 gen_docs.py && python3 verify_gate.py
```

## The lock

The document is encrypted once with a random content key. That key is then wrapped
separately under each credential — PBKDF2-SHA256 at 310,000 iterations, feeding AES-GCM.
Only ciphertext is published. No password, and no hash of a password, appears in this
repository or in the built page.

It's real encryption rather than a JavaScript curtain, so the content genuinely isn't in
the file unless you can decrypt it. But the passwords are short and follow an obvious
pattern, so anyone who downloads the file can attack it offline at their leisure. Key
stretching slows that down; it doesn't fix low-entropy passwords.

**Nothing sensitive goes on that page.** It's a gate proportionate to a project notebook,
and that's all.

## Format

Built to the "living page" spec: one URL, updated in place, always current. A single
self-contained HTML file with zero external requests — no CDN, no web fonts, no analytics.
Open lab notebook, so dead ends and corrections stay on the page next to the correction
rather than being quietly edited away.
