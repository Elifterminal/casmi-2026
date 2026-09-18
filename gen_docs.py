#!/usr/bin/env python3
"""manifest.json -> docs/index.html

Single self-contained file, zero external requests. The page body is encrypted:
a random content key encrypts the document once, then that key is wrapped once per
credential (PBKDF2-SHA256 -> AES-GCM). Only ciphertext ships. No password, and no
hash of a password, appears anywhere in the output or in this script.

Credentials are read from ~/.elif_accounts.json (chmod 600, never committed).
"""
import base64, json, os, secrets, sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import hashlib
import figures

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "docs", "index.html")
CRED_SITE = "casmi-2026 living page"
PBKDF2_ITERS = 310_000

b64 = lambda x: base64.b64encode(x).decode()

CSS = r"""
:root{--bg:#fbfbfc;--panel:#fff;--fg:#16181d;--mut:#636a76;--line:#e3e5ea;
      --accent:#2563eb;--warn:#dc2626;--ok:#15803d;--code:#f3f4f6;}
@media (prefers-color-scheme:dark){
 :root{--bg:#0d0f13;--panel:#14171d;--fg:#e9eaee;--mut:#98a0ad;--line:#262b34;
       --accent:#6ea0ff;--warn:#ff7a70;--ok:#68d391;--code:#1b1f26;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.62 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:1120px;margin:0 auto;padding:48px 28px 96px}
header{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:26px}
.kicker{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);
 font-weight:600;margin:0 0 6px}
h1{font-size:34px;letter-spacing:-.02em;margin:0 0 10px;line-height:1.15}
h2{font-size:20px;letter-spacing:-.02em;margin:26px 0 10px}
h3{font-size:15.5px;letter-spacing:-.01em;margin:20px 0 6px}
.sub{color:var(--mut);margin:0;font-size:14px}
.status{display:inline-block;margin-top:14px;padding:7px 12px;border-radius:7px;
 background:var(--code);border-left:4px solid var(--warn);font-size:13px;color:var(--fg)}
nav.tabs{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid var(--line);margin-bottom:30px}
button.tab{background:none;border:0;border-bottom:2px solid transparent;padding:10px 14px;
 font:inherit;font-size:14px;color:var(--mut);cursor:pointer;border-radius:6px 6px 0 0}
button.tab:hover{background:var(--code);color:var(--fg)}
button.tab.active{color:var(--fg);border-bottom-color:var(--accent);font-weight:600}
.panel{display:none}.panel.active{display:block}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:18px 0}
.read{border-left:4px solid var(--accent);background:var(--code);padding:13px 16px;
 border-radius:0 8px 8px 0;margin:14px 0;font-size:14.2px}
.read.warn{border-left-color:var(--warn)}
.read.ok{border-left-color:var(--ok)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px}
.stat .n{font-size:23px;font-weight:700;letter-spacing:-.02em;display:block;line-height:1.2}
.stat .k{font-size:11.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.tag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600;
 text-transform:uppercase;letter-spacing:.04em}
.tag.ok{background:#dcfce7;color:#15803d}.tag.warn{background:#fef3c7;color:#92400e}
.tag.mut{background:var(--code);color:var(--mut)}.tag.bad{background:#fee2e2;color:#991b1b}
@media (prefers-color-scheme:dark){
 .tag.ok{background:#14331f;color:#68d391}.tag.warn{background:#3b2f10;color:#fbbf24}
 .tag.bad{background:#4c1d1d;color:#fca5a5}}
.fig{overflow-x:auto;margin:20px 0;padding-bottom:6px}
.fig svg{min-width:640px}
.q{border-left:4px solid var(--warn);background:var(--code);padding:12px 15px;
 border-radius:0 8px 8px 0;margin:14px 0;font-size:14px}
.jump-card{border-left:4px solid var(--mut)}
a.jump{display:block;padding:6px 9px;border-radius:5px;text-decoration:none;color:var(--fg);
 font-size:13px;line-height:1.35}
a.jump:hover{background:var(--code)}
a.jump b{display:inline-block;min-width:52px;color:var(--mut);font-weight:600}
@media(min-width:700px){a.jump{display:inline-block;width:calc(50% - 6px);vertical-align:top}}
.entry{border-top:1px solid var(--line);padding-top:20px;margin-top:24px}
.entry .meta{font-size:12px;color:var(--mut);margin-bottom:6px}
.entry h3{margin-top:2px}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13.5px}
th{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);color:var(--mut);
 font-weight:600;text-transform:uppercase;font-size:11.5px;letter-spacing:.05em}
td{padding:9px;border-bottom:1px solid var(--line);vertical-align:top}
td:first-child{white-space:nowrap;width:1%}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12.8px}
ol,ul{padding-left:22px}li{margin:6px 0}
.phase{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);
 border-radius:0 10px 10px 0;padding:16px 19px;margin:16px 0}
.phase .pn{font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);font-weight:700}
footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);color:var(--mut);font-size:12.5px}
"""

LOGIN_CSS = r"""
#gate{max-width:380px;margin:12vh auto;padding:0 24px}
#gate h1{font-size:23px;margin:0 0 6px}
#gate p.k{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);
 font-weight:600;margin:0 0 6px}
#gate p.d{color:var(--mut);font-size:13.5px;margin:0 0 22px}
#gate label{display:block;font-size:12px;color:var(--mut);margin:12px 0 4px;font-weight:600}
#gate input{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:8px;
 background:var(--panel);color:var(--fg);font:inherit;font-size:14px}
#gate input:focus{outline:2px solid var(--accent);outline-offset:-1px;border-color:transparent}
#gate button{width:100%;margin-top:18px;padding:11px;border:0;border-radius:8px;
 background:var(--accent);color:#fff;font:inherit;font-weight:600;font-size:14.5px;cursor:pointer}
#gate button:disabled{opacity:.6;cursor:default}
#err{color:var(--warn);font-size:13px;min-height:19px;margin-top:11px}
"""

JS_TABS = r"""
document.querySelectorAll('.tab').forEach(function(t){
  t.addEventListener('click',function(){
    document.querySelectorAll('.tab').forEach(function(x){x.classList.remove('active');});
    document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active');});
    t.classList.add('active');
    document.getElementById(t.dataset.panel).classList.add('active');
    window.scrollTo({top:0,behavior:'smooth'});
  });
});
document.querySelectorAll('a.jump').forEach(function(a){
  a.addEventListener('click',function(e){
    e.preventDefault();
    var el=document.getElementById(a.getAttribute('href').slice(1));
    if(el) el.scrollIntoView({behavior:'smooth',block:'start'});
  });
});
"""

JS_GATE = r"""
var B=function(s){var r=atob(s),a=new Uint8Array(r.length);
  for(var i=0;i<r.length;i++)a[i]=r.charCodeAt(i);return a;};
var form=document.getElementById('f'),err=document.getElementById('err'),
    btn=document.getElementById('go');
form.addEventListener('submit',function(e){
  e.preventDefault();
  var u=document.getElementById('u').value.trim().toLowerCase(),
      p=document.getElementById('p').value;
  err.textContent='';btn.disabled=true;btn.textContent='Checking…';
  var rec=null;
  for(var i=0;i<VAULT.users.length;i++){if(VAULT.users[i].u===u){rec=VAULT.users[i];break;}}
  var fail=function(){btn.disabled=false;btn.textContent='Open';
    err.textContent='Wrong username or password.';};
  if(!rec){setTimeout(fail,450);return;}
  var enc=new TextEncoder();
  crypto.subtle.importKey('raw',enc.encode(p),'PBKDF2',false,['deriveKey']).then(function(base){
    var salt=new Uint8Array(B(rec.s).length+enc.encode(u).length);
    salt.set(B(rec.s),0);salt.set(enc.encode(u),B(rec.s).length);
    return crypto.subtle.deriveKey(
      {name:'PBKDF2',salt:salt,iterations:VAULT.it,hash:'SHA-256'},
      base,{name:'AES-GCM',length:256},false,['decrypt']);
  }).then(function(kek){
    return crypto.subtle.decrypt({name:'AES-GCM',iv:B(rec.i),additionalData:enc.encode(u)},
                                 kek,B(rec.w));
  }).then(function(ckRaw){
    return crypto.subtle.importKey('raw',ckRaw,'AES-GCM',false,['decrypt']);
  }).then(function(ck){
    return crypto.subtle.decrypt({name:'AES-GCM',iv:B(VAULT.iv)},ck,B(VAULT.ct));
  }).then(function(plain){
    document.open();document.write(new TextDecoder().decode(plain));document.close();
  }).catch(fail);
});
"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_inner(m):
    F = figures.ALL
    active_id = next((t['id'] for t in m['tabs'] if t.get('active')), None)
    def pnl(pid):
        return f'<div class="panel active" id="{pid}">' if pid == active_id \
               else f'<div class="panel" id="{pid}">'
    p = []
    p.append('<div class="wrap"><header>')
    p.append(f'<p class="kicker">{esc(m["doctype"])}</p>')
    p.append(f'<h1>{esc(m["project"])}</h1>')
    p.append(f'<p class="sub">{m["sub"]}</p>')
    p.append(f'<div class="status"><b>Status:</b> {m["status"]}</div>')
    p.append('</header>')

    p.append('<div class="grid">')
    for s in m["stats"]:
        p.append(f'<div class="stat"><span class="n">{esc(s["n"])}</span>'
                 f'<span class="k">{esc(s["k"])}</span></div>')
    p.append('</div>')

    p.append('<nav class="tabs">')
    for t in m["tabs"]:
        cls = "tab active" if t.get("active") else "tab"
        p.append(f'<button class="{cls}" data-panel="{t["id"]}">{esc(t["label"])}</button>')
    p.append('</nav>')

    # ---------------- brief ----------------
    p.append(pnl('brief'))
    p.append('<h2>You get a mass spectrum. Name the molecule.</h2>')
    p.append('<p>A mass spectrometer takes a molecule, gives it an electric charge, then smashes it '
             'with gas until it breaks apart, and records the mass of every fragment. What comes out '
             'is a histogram of masses. From that histogram you have to reconstruct the molecule and '
             'write it down as a <code>SMILES</code> string, which is just a text encoding of a '
             'chemical structure — paracetamol is <code>CC(=O)Nc1ccc(O)cc1</code>.</p>')
    p.append('<div class="read"><b>It is the analytical-chemistry version of reconstructing a vase '
             'from a photograph of its shards.</b> You know the total weight and you know every piece, '
             'but the arrangement is what you are missing, and many different vases produce a similar '
             'pile of pieces.</div>')
    p.append('<h3>What is actually in the test set</h3>')
    p.append('<p>About 400 molecules, measured across roughly 1,500 spectra, so most molecules were '
             'recorded two or three times — at different collision energies, or carrying a different '
             'charge. You submit one ranked list per <i>molecule</i>, not per spectrum, so part of the '
             'job is fusing several noisy views of the same thing into a single answer.</p>')
    p.append('<p>These are natural products: plant, microbe and mammal chemistry. That matters, '
             'because it is a much stranger region of chemical space than the drug-like compounds '
             'that most published reference data covers.</p>')
    p.append('<h3>How it is scored, and why that shapes everything</h3>')
    p.append('<p>The metric is MRR@25. You submit up to 25 candidate structures per molecule, ranked. '
             'First guess right scores 1.0, second 0.5, tenth 0.1, twenty-fifth 0.04. None right '
             'scores zero. Only your first correct guess counts.</p>')
    p.append(F["mrr"]())
    p.append('<div class="read ok"><b>Two consequences drive the entire strategy.</b><br>'
             'A wrong guess costs nothing except the slot it occupies — there is no precision penalty, '
             'so you always submit all 25.<br>And because only position matters, this is a ranking and '
             'calibration problem rather than a prediction problem. You do not have to be sure. Your '
             '<i>ordering</i> has to be right.</div>')
    p.append('<p>One more gift: matching ignores stereochemistry and tautomers. Both sides get '
             'canonicalised and reduced to the flat skeleton before comparison, so a whole category of '
             'chemistry you would normally have to get right is free here.</p>')
    p.append('<h3>The rules that matter to us</h3>')
    p.append('<table><thead><tr><th>Constraint</th><th>What it means</th></tr></thead><tbody>'
             '<tr><td>Internet disabled</td><td>Notebooks run with no network. Any external database '
             'has to be pre-staged as an attached dataset. This is also why the competition satisfies '
             'our no-external-AI-API rule by its own design rather than by us being careful.</td></tr>'
             '<tr><td>9 hour runtime</td><td>CPU or GPU. The same ceiling for everyone, which flattens '
             'the hardware advantage at submission time.</td></tr>'
             '<tr><td>3GB training data</td><td>~2.5m spectra over ~275k structures. Fits in RAM on the '
             'desktop. This is not a GPU arms race.</td></tr>'
             '<tr><td>5 paid places</td><td>$16k / $12k / $9k / $7k / $6k. Fifth still pays.</td></tr>'
             '<tr><td>Solo is fine</td><td>No video, no writeup, no team needed.</td></tr>'
             '</tbody></table>')
    p.append('</div>')

    # ---------------- thesis ----------------
    p.append(pnl('thesis'))
    p.append('<h2>Every test molecule is in one of three hidden classes</h2>')
    p.append('<p>This is the single fact that decides the competition, and it is the reason the '
             'leaderboard looks the way it does.</p>')
    p.append('<table><thead><tr><th>Class</th><th>What it means</th><th>How you would find it</th>'
             '</tr></thead><tbody>'
             '<tr><td><span class="tag ok">Class 1</span></td><td>The molecule already has published '
             'reference spectra.</td><td>Match it against the training data. Basically solved — '
             'spectral matching reaches about 0.93 on this group.</td></tr>'
             '<tr><td><span class="tag warn">Class 2</span></td><td>Nobody has ever published a '
             'spectrum for it, but the structure exists in PubChem or in COCONUT, the natural-products '
             'database.</td><td>You cannot match it. You retrieve candidates by mass and then rank '
             'them.</td></tr>'
             '<tr><td><span class="tag bad">Class 3</span></td><td>The structure is not in any database. '
             'Nobody has recorded it, ever.</td><td>You have to construct it.</td></tr>'
             '</tbody></table>')
    p.append('<p>The split is roughly 16% / 45% / 39%. Those numbers are not published — the '
             'competitor currently in 7th reverse-engineered them and posted his estimate, and it is '
             'consistent with what the leaderboard is doing.</p>')
    p.append(F["classes"]())
    p.append('<h2>Now do the arithmetic</h2>')
    p.append('<p>Class 1 is nearly solved, but it is only 16% of the score, so it can contribute at '
             'most about 0.15. The leader is at 0.353. Which means essentially the entire leaderboard '
             'is Class 1 plus a slice of Class 2, and Class 3 is scoring roughly nothing for '
             'everybody.</p>')
    p.append(F["headroom"]())
    p.append('<div class="read ok"><b>Solve Classes 1 and 2 perfectly, give up entirely on Class 3, '
             'and you score 0.61.</b> The leader is at 0.353. There is an enormous amount of unclaimed '
             'ground in the middle of this problem, and it sits in the 45% chunk that is pure retrieval '
             'and ranking. That is engineering. It is not chemistry.</div>')
    p.append('<h2>And the field cannot currently tell whether it is improving</h2>')
    p.append('<p>There is a public thread titled <i>"using the 250 example structures as CV does not '
             'seem very helpful."</i> People are scoring 0.33 to 0.37 in local testing and 0.15 on the '
             'leaderboard. That gap is the most interesting thing on the board.</p>')
    p.append(F["cv"]())
    p.append('<p>The organisers shipped 250 natural products as a domain-matched example set, and '
             'everyone holds those out to validate against. But the same 250 structures also appear in '
             'five other training libraries. Holding out by <i>source label</i> leaves the structures '
             'sitting in the reference index under a different name.</p>')
    p.append('<div class="read warn"><b>So the holdout is contaminated, and what it secretly measures '
             'is Class 1 — the 16% that barely moves the score.</b> One team reported their leaderboard '
             'score falling from 0.152 to 0.145 after an "improvement": they made library matching more '
             'aggressive, got better at Class 1, and made Class 2 ranking worse. That is a field '
             'optimising itself sideways, in public, with three months left.</div>')
    p.append('<div class="q"><b>The thesis in one line.</b> The money is not in being a better mass '
             'spectrometrist. It is in Class 2, and in being the only team with a number it can '
             'trust.</div>')
    p.append('</div>')

    # ---------------- plan ----------------
    p.append(pnl('plan'))
    p.append('<h2>Four phases</h2>')
    p.append('<p>Nothing here has been run yet. This is the plan as written on 17 September, before '
             'entry.</p>')

    p.append('<div class="phase"><span class="pn">Phase 1</span>'
             '<h3>Build an honest scoreboard before writing a single model</h3>'
             '<p>Hold structures out by chemical identity — InChIKey14, across every library at once — '
             'rather than by source label. Then manufacture the three classes deliberately:</p>'
             '<ul><li><b>Class 2 query:</b> strip every spectrum of that structure from the reference '
             'index, but leave the structure in the candidate database.</li>'
             '<li><b>Class 3 query:</b> remove it from the candidate database too.</li></ul>'
             '<p>Weight the result 16/45/39 to match the real test composition.</p>'
             '<div class="read"><b>This is unglamorous and it is the whole game.</b> It is also exactly '
             'the discipline the Skill Lift campaign produced — paired testing, validity gating, '
             'predeclared stopping rules. That project\'s real output was not the skills, it was '
             'learning how not to fool ourselves. Here the entire field is fooling itself in a '
             'documented, specific way. If we are the only team with a trustworthy number, every '
             'decision we make afterwards is better than theirs.</div></div>')

    p.append('<div class="phase"><span class="pn">Phase 2</span>'
             '<h3>Go at Class 2 — the 45%</h3>'
             '<p><b>Candidate generation.</b> Given the precursor mass and the adduct, pull every '
             'plausible structure out of a pre-staged COCONUT and PubChem subset. The tension here is '
             'real: a bigger pool is more likely to contain the right answer <i>and</i> harder to rank. '
             'The public baseline literally ships a flag to disable extra databases because of what its '
             'author calls decoy dilution. I think the answer is weighting candidates by '
             'natural-product plausibility, not just making the pool bigger.</p>'
             '<p><b>Ranking</b>, which is where the points are. Predict a molecular fingerprint from the '
             'spectrum, score candidates against it. The baseline already does this. Two places it '
             'looks like it is leaving value:</p>'
             '<ol><li><b>Evidence fusion.</b> Different collision energies fragment a molecule '
             'differently, so the 1&ndash;16 spectra per molecule are complementary evidence, not '
             'repeated measurements of the same thing.</li>'
             '<li><b>The domain split everybody pools away.</b> 46% of the training data is '
             'instrument-matched to the test set but chemically wrong — synthetic drug-like screening '
             'compounds. The actual natural-product chemistry comes from different instruments. Everyone '
             'is throwing both into one bucket. Modelling that mismatch explicitly should be worth '
             'something.</li></ol></div>')

    p.append('<div class="phase"><span class="pn">Phase 3</span>'
             '<h3>Class 3, and the economics of a slot</h3>'
             '<p>Novel structures score exactly zero under retrieval, by definition. The realistic '
             'approach is analog editing: find the closest known relative, read the mass difference, '
             'and apply the chemical modification it implies. Natural products vary in patterned ways — '
             'a sugar added here, a methyl or a hydroxyl there — so the edits are not arbitrary.</p>'
             '<p>But there is a catch somebody has already spotted and nobody has answered: generated '
             'candidates compete for the same 25 slots as retrieved ones, so <b>generation can lose you '
             'points before it gains you any.</b></p>'
             '<div class="read">That is precisely the question an honest validation harness can answer '
             'and a contaminated one cannot. Which loops straight back to Phase 1.</div></div>')

    p.append('<div class="phase"><span class="pn">Phase 4</span>'
             '<h3>Treat the 25 slots as a portfolio</h3>'
             '<p>Three channels produce candidates and they all have to be merged into one ordered '
             'list. Most teams will concatenate them. The right move is to calibrate each channel\'s '
             'confidence onto a common scale and interleave.</p>'
             '<p>Free MRR, sitting there, purely from ordering.</p></div>')

    p.append('<h2>Why us, honestly</h2>')
    p.append('<p>Not chemistry. I would be learning that as I went. The fit is that this is a ranking, '
             'calibration and measurement-hygiene problem wearing a chemistry costume, and the field '
             'has publicly documented that its measurement hygiene is broken.</p>')
    p.append('<p>The structural advantages are real but they are not an edge on their own: entering on '
             'day four of ninety-one, internet-disabled notebooks, data that fits in RAM, five places '
             'that pay.</p>')
    p.append('</div>')

    # ---------------- log ----------------
    p.append(pnl('log'))
    p.append('<h2>Working log</h2>')
    p.append('<p>Newest first. Dead ends and corrections stay on the page in their original wording, '
             'next to the correction — that is the point of keeping a notebook in the open.</p>')
    p.append('<div class="card jump-card"><h3 style="margin-top:0">Jump to</h3>')
    for e in m["log"]:
        p.append(f'<a class="jump" href="#{e["id"]}"><b>{esc(e["date"][5:])}</b> {esc(e["title"])}</a>')
    p.append('</div>')
    for e in m["log"]:
        p.append(f'<div class="entry" id="{e["id"]}">')
        p.append(f'<div class="meta"><code>{e["id"]}</code> &middot; {esc(e["date"])} &middot; '
                 f'<span class="tag {e["tag"]}">{esc(e["tagtext"])}</span></div>')
        p.append(f'<h3>{esc(e["title"])}</h3>')
        p.append(f'<p>{e["body"]}</p>')
        if e.get("read"):
            cls = ("read " + e["read"]["cls"]).strip()
            p.append(f'<div class="{cls}">{e["read"]["html"]}</div>')
        if e["id"] == "L-002":
            p.append(F["lb"]())
        p.append('</div>')
    p.append('</div>')

    # ---------------- risks ----------------
    p.append(pnl('risks'))
    p.append('<h2>What worries me</h2>')
    p.append('<div class="read warn"><b>The GPU is a real constraint, and I oversold this earlier.</b> '
             'I told Lee the 1080 Ti did not matter here. That is true for data size — 3GB fits in RAM — '
             'but not for training the neural ranker, which is where a good chunk of the headroom lives. '
             'Kaggle\'s free 30 GPU-hours a week would be our actual training rig. Anyone with a 4090 '
             'iterates faster than we do.</div>')
    p.append('<p><b>The field is sharp and shares fast.</b> The current leader gave away a strong '
             'solution for nothing, twenty hours after building it. Expect that to happen again, and '
             'expect any edge with a short description to have a short life.</p>')
    p.append('<p><b>The private leaderboard is 67% of the test set</b> and the class mix is hidden, so '
             'there will be a shakeup at the end. That cuts both ways. It punishes everyone overfitting '
             'a contaminated holdout, which is most of the field — but it also means our own number is '
             'never fully trustworthy.</p>')
    p.append('<p><b>CASMI draws specialists.</b> It is a long-running academic challenge, and the '
             'people behind SIRIUS, CSI:FingerID and MIST live in this problem.</p>')
    p.append('<h2>The kill gate</h2>')
    p.append('<div class="q"><b>Predeclared, before any work starts.</b> By the end of the first real '
             'work block we have to beat the published 0.339 baseline on <i>both</i> our own honest '
             'validation and the public leaderboard, by a margin bigger than seed noise. If three weeks '
             'cannot clear a free public notebook, the headroom is not reachable from where we are '
             'standing and we stop.<br><br>Same rule that retired the ROGII entry rather than grinding '
             'it. No three-month grind on a flat signal.</div>')
    p.append('<h2>About the lock on this page</h2>')
    p.append('<p>The gate is real encryption, not a JavaScript curtain. The document is encrypted with '
             'a random key; that key is then wrapped separately under each credential using PBKDF2 at '
             f'{PBKDF2_ITERS:,} iterations feeding AES-GCM. Only ciphertext is published. No password, '
             'and no hash of a password, appears in the file or in the repository that builds it.</p>')
    p.append('<div class="read warn"><b>What that does not buy.</b> The passwords are short and follow '
             'an obvious pattern, so anyone who downloads the file can attack it offline at their own '
             'pace. Key stretching slows that down; it does not fix low-entropy passwords. This is a '
             'gate proportionate to a project notebook, and nothing on this page should ever be '
             'something that actually needs protecting.</div>')
    p.append('</div>')

    p.append(f'<footer>{esc(m["project"])} &mdash; {esc(m["doctype"])}. One URL, updated in place, '
             'always current. Generated from a manifest, so the counts in the header cannot drift.'
             '</footer>')
    p.append('</div>')
    p.append(f'<script>{JS_TABS}</script>')
    return "".join(p)


def load_creds():
    path = os.path.expanduser("~/.elif_accounts.json")
    with open(path) as fh:
        blob = json.load(fh)
    out = []
    for a in blob[0].get("accounts", []):
        if a.get("site") == CRED_SITE:
            out.append((a["username"].strip().lower(), a["password"]))
    return sorted(out)


def main():
    with open(os.path.join(ROOT, "manifest.json")) as fh:
        m = json.load(fh)
    creds = load_creds()
    if not creds:
        sys.exit(f"no credentials found for site '{CRED_SITE}' in ~/.elif_accounts.json")

    inner = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
             "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
             f"<title>{esc(m['project'])} &mdash; {esc(m['doctype'])}</title>"
             f"<style>{CSS}</style></head><body>{build_inner(m)}</body></html>")

    ck = AESGCM.generate_key(bit_length=256)
    iv = secrets.token_bytes(12)
    ct = AESGCM(ck).encrypt(iv, inner.encode(), None)

    users = []
    for username, password in creds:
        salt = secrets.token_bytes(16)
        kek = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                  salt + username.encode(), PBKDF2_ITERS, 32)
        wiv = secrets.token_bytes(12)
        wrapped = AESGCM(kek).encrypt(wiv, ck, username.encode())
        users.append({"u": username, "s": b64(salt), "i": b64(wiv), "w": b64(wrapped)})

    vault = {"it": PBKDF2_ITERS, "iv": b64(iv), "ct": b64(ct), "users": users}

    shell = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{esc(m["project"])} &mdash; {esc(m["doctype"])}</title>'
        f'<style>{CSS}{LOGIN_CSS}</style></head><body><div id="gate">'
        f'<p class="k">{esc(m["doctype"])}</p>'
        f'<h1>{esc(m["project"])}</h1>'
        '<p class="d">This notebook is encrypted. Sign in to decrypt it in your browser.</p>'
        '<form id="f" autocomplete="off">'
        '<label for="u">Username</label><input id="u" name="u" autocapitalize="off" '
        'autocorrect="off" spellcheck="false" required>'
        '<label for="p">Password</label><input id="p" name="p" type="password" required>'
        '<button id="go" type="submit">Open</button><div id="err" role="alert"></div>'
        '</form></div>'
        f'<script>var VAULT={json.dumps(vault, separators=(",", ":"))};{JS_GATE}</script>'
        '</body></html>')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(shell)
    print(f"wrote {OUT}  ({len(shell):,} bytes, payload {len(inner):,} bytes plaintext, "
          f"{len(creds)} credentials)")


if __name__ == "__main__":
    main()
