"""Make ICEBERG/GLACIER load, using the installer's OWN idempotency marker.

WHAT THE SOURCE SAYS (ice_runner.py, downloaded and read rather than guessed at):

    def install_site(pkg, site, log):
        src  = sorted(glob(pkg/wheels/*.whl.ice) + glob(pkg/wheels/*.whl))
        mark = site/'.ice_site_ok'
        want = '\\n'.join(basename(w) for w in src)
        if exists(mark) and open(mark).read() == want: return      # <-- the bypass
        ...
        pip install --no-index --no-deps --target site <all wheels>
        if returncode: raise RuntimeError('pip install failed: ...')

It installs EVERY wheel, including rdkit-2025.3.6-cp312, which pip refuses on this Python 3.13
image -- so the batch fails and both engines score nothing. `--any-rdkit` does NOT help: it only
relaxes a later version check (line 170), not the install.

My first attempt passed a filtered wheels directory. That failed because run_ice has no `wheels`
parameter at all -- the log said `run_ice takes wheels=False` -- so ICE used the dataset's own
directory again, and the shared site dir was left without a marker, so GL failed the same way.

THE FIX, using the author's own mechanism: pre-install the installable wheels into the site
directory ourselves, then write `.ice_site_ok` with exactly the basename list install_site computes.
Both engines then return early from install_site and use what we installed. Verified locally against
the real function: with the marker present it returns without invoking pip.

`--any-rdkit` goes in via `extra_args` (a documented pass-through on run_ice) because the site will
now carry the image's RDKit 2026.03 rather than the pinned 2025.03, and without the flag the runner
aborts on that version check. Results are no longer "panel-exact" by the author's definition; that
is the price of the image having moved, and it is stated rather than hidden.

GLACIER's `wheels=` argument is restored to the original ICE_PKG/wheels so that the basename list it
computes matches the marker we wrote -- one marker satisfying both engines.
"""
import json, os, shutil

p = os.path.expanduser("~/casmi-2026/fork-v44/casmi26-fork-v44.ipynb")
bak = "/tmp/claude-1000/-home-lee/c046c9a5-0d6b-4917-ae00-430a3bba9009/scratchpad/v44.ipynb.bak"
shutil.copyfile(bak, p)          # start from the pristine fork, not from the failed patch
nb = json.load(open(p))

PREP = '''
# ELIF FIX: install_site() in the dataset's ice_runner.py pip-installs EVERY wheel in pkg/wheels,
# including rdkit-2025.3.6-cp312. This image is Python 3.13, pip refuses that wheel, the whole batch
# fails, and both engines score nothing -- our first run logged ICE and GL at "n_mols_scored": 0 with
# fusion stats {'ice': {'molecules': 0}, 'gl': {'molecules': 0}}. --any-rdkit does not fix it; that
# only relaxes a later version check.
# install_site returns early when site/.ice_site_ok holds exactly the wheel-basename list it would
# compute, so: install the installable wheels here, write that marker, and let both engines skip pip.
# Verified against the real function locally before spending a run on it.
import glob as _g, shutil as _sh, subprocess as _sp
_SITE = '/kaggle/working/ice_site'
_src = sorted(_g.glob(os.path.join(ICE_PKG, 'wheels', '*.whl.ice')) +
              _g.glob(os.path.join(ICE_PKG, 'wheels', '*.whl')))
_want = '\\n'.join(os.path.basename(_w) for _w in _src)
_good = [_w for _w in _src if 'rdkit' not in os.path.basename(_w).lower()]
os.makedirs(_SITE, exist_ok=True)
_wdir = os.path.join(_SITE, '.elif_wheels'); os.makedirs(_wdir, exist_ok=True)
_whl = []
for _w in _good:
    _b = os.path.basename(_w)
    _d = os.path.join(_wdir, _b[:-4] if _b.endswith('.ice') else _b)
    if not os.path.exists(_d): _sh.copyfile(_w, _d)
    _whl.append(_d)
_r = _sp.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
              '--disable-pip-version-check', '--upgrade', '--target', _SITE] + _whl,
             stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
print('ELIF pre-install rc=%d (%d wheels, rdkit skipped)' % (_r.returncode, len(_whl)), flush=True)
if _r.returncode == 0:
    with open(os.path.join(_SITE, '.ice_site_ok'), 'w') as _f: _f.write(_want)
    print('ELIF marker written -> install_site will skip pip for both engines', flush=True)
else:
    print('ELIF pre-install FAILED, engines will fail as before:', _r.stdout[-500:], flush=True)
_XA = ('--any-rdkit',)   # the site now carries the image RDKit, not the pinned 2025.03
'''

OLD_ICE = """    ICE_SCORES = ice_fuse.run_ice(ICE_PKG, items, workdir='/kaggle/working/ice_work', device='cuda', budget_s=ICE_BUDGET,
                                  site='/kaggle/working/ice_site')"""
NEW_ICE = """    ICE_SCORES = ice_fuse.run_ice(ICE_PKG, items, workdir='/kaggle/working/ice_work', device='cuda', budget_s=ICE_BUDGET,
                                  site='/kaggle/working/ice_site', extra_args=_XA)"""
OLD_GL = """                               site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'))"""
NEW_GL = """                               site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'),
                               **({'extra_args': _XA} if 'extra_args' in _insp.signature(gl_fuse.run_gl).parameters else {}))"""

n_ice = n_gl = 0
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    s = "".join(c["source"])
    if OLD_ICE in s:
        anchor = "    import fuse as ice_fuse\n"
        assert anchor in s, "ice_fuse anchor missing"
        s = s.replace(anchor, anchor + "".join("    " + ln + "\n" for ln in PREP.strip().split("\n")), 1)
        s = s.replace(OLD_ICE, NEW_ICE, 1); n_ice += 1
    if OLD_GL in s:
        s = s.replace(OLD_GL, NEW_GL, 1)
        if "import inspect as _insp" not in s:
            s = "import inspect as _insp\n" + s
        n_gl += 1
    c["source"] = s.splitlines(keepends=True)

assert n_ice == 1 and n_gl == 1, f"ICE {n_ice}, GL {n_gl}"
bad = 0
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code": continue
    s = "".join(c["source"]); first = s.split("\n", 1)[0]
    body = s.split("\n", 1)[1] if first.startswith("%%writefile") else s
    if body.lstrip().startswith(("%", "!")): continue
    try:
        compile(body, f"cell{i}", "exec")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR cell {i} line {e.lineno}: {e.msg}"); bad += 1
assert not bad
json.dump(nb, open(p, "w"), indent=1)
full = "".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
for probe in (".ice_site_ok", "extra_args=_XA", "--any-rdkit", "_insp.signature(gl_fuse.run_gl)"):
    assert probe in full, f"missing {probe}"
print("patched from pristine backup: marker bypass + --any-rdkit, all cells compile")
