"""Hand-emitted inline SVG. No chart library, no image files, no external requests.
Colours are CSS vars so the figures follow the page theme."""

def _wrap(w, h, body, cls="fig"):
    return (f'<div class="{cls}"><svg viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-width:{w}px;height:auto" role="img">{body}</svg></div>')

def _txt(x, y, s, size=12, fill="var(--fg)", anchor="start", weight="400"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="ui-sans-serif,system-ui,sans-serif">{s}</text>')

def fig_classes():
    """Composition of the test set, and what each class can contribute."""
    W, H = 760, 300
    x0, bw = 150, 540
    rows = [
        ("Class 1", 16, "has published spectra", "var(--ok)", 0.93),
        ("Class 2", 45, "in a database, no spectrum", "var(--accent)", 0.0),
        ("Class 3", 39, "not in any database", "var(--warn)", 0.0),
    ]
    b = [_txt(0, 20, "What the ~400 test molecules are made of", 13, "var(--fg)", weight="600")]
    # composition bar
    y = 44
    cx = x0
    for name, pct, _, col, _ in rows:
        w = bw * pct / 100
        b.append(f'<rect x="{cx:.1f}" y="{y}" width="{w:.1f}" height="34" fill="{col}" opacity=".85"/>')
        if pct > 10:
            b.append(_txt(cx + w/2, y+22, f"{pct}%", 13, "#fff", "middle", "700"))
        cx += w
    b.append(_txt(x0-10, y+22, "composition", 12, "var(--mut)", "end"))
    # per-class detail rows
    y = 110
    for name, pct, desc, col, solved in rows:
        b.append(f'<rect x="{x0-10}" y="{y-14}" width="4" height="20" fill="{col}"/>')
        b.append(_txt(0, y, name, 13, "var(--fg)", weight="600"))
        b.append(_txt(x0, y, desc, 12.5, "var(--mut)"))
        # max contribution bar: pct/100 of total score
        contrib = pct / 100
        b.append(f'<rect x="{x0+250}" y="{y-11}" width="{contrib*220:.1f}" height="14" fill="{col}" opacity=".3"/>')
        b.append(_txt(x0+250+contrib*220+8, y, f"max {contrib:.2f} of the score", 11.5, "var(--mut)"))
        y += 34
    # the punchline
    b.append(f'<line x1="0" y1="218" x2="{W}" y2="218" stroke="var(--line)"/>')
    b.append(_txt(0, 242, "Class 1 is nearly solved — spectral matching reaches ~0.93 MRR on it.", 12.5, "var(--mut)"))
    b.append(_txt(0, 262, "But it can only ever be worth 0.15 of the total. The leader sits at 0.353.", 12.5, "var(--mut)"))
    b.append(_txt(0, 286, "So essentially the whole leaderboard is Class 1 plus a slice of Class 2.", 13, "var(--fg)", weight="600"))
    return _wrap(W, H, "".join(b))

def fig_headroom():
    """Where the ceiling is vs where the field is."""
    W, H = 760, 250
    x0, bw = 200, 480
    rows = [
        ("Current leader", 0.353, "var(--warn)", "everyone else is at 0.339"),
        ("Class 1 + 2 perfect", 0.610, "var(--accent)", "giving up on Class 3 entirely"),
        ("Every molecule, rank 1", 1.000, "var(--mut)", "not happening"),
    ]
    b = [_txt(0, 20, "How much of this problem is unclaimed", 13, "var(--fg)", weight="600")]
    y = 52
    for name, v, col, note in rows:
        b.append(f'<rect x="{x0}" y="{y}" width="{bw}" height="26" fill="var(--code)"/>')
        b.append(f'<rect x="{x0}" y="{y}" width="{bw*v:.1f}" height="26" fill="{col}" opacity=".8"/>')
        b.append(_txt(x0-10, y+18, name, 12.5, "var(--fg)", "end"))
        b.append(_txt(x0+bw*v+8 if v < .8 else x0+bw*v-8, y+18, f"{v:.3f}",
                      12.5, "var(--fg)" if v < .8 else "#fff", "start" if v < .8 else "end", "700"))
        b.append(_txt(x0, y+44, note, 11.5, "var(--mut)"))
        y += 62
    b.append(f'<line x1="{x0}" y1="40" x2="{x0}" y2="{y-24}" stroke="var(--line)"/>')
    b.append(_txt(0, H-12, "The gap between 0.353 and 0.61 is the part that is pure retrieval and ranking — engineering, not chemistry.",
                  12.5, "var(--fg)", weight="600"))
    return _wrap(W, H, "".join(b))

def fig_mrr():
    """What a rank is worth under MRR@25."""
    W, H = 760, 260
    x0, y0, pw, ph = 60, 40, 640, 160
    b = [_txt(0, 20, "What each slot is worth: score = 1 / rank of your first correct guess", 13, "var(--fg)", weight="600")]
    b.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="var(--line)"/>')
    b.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="var(--line)"/>')
    pts = []
    for r in range(1, 26):
        x = x0 + (r-0.5) * (pw/25)
        v = 1.0/r
        h = v * ph
        pts.append(f"{x:.1f},{y0+ph-h:.1f}")
        b.append(f'<rect x="{x-9:.1f}" y="{y0+ph-h:.1f}" width="18" height="{h:.1f}" '
                 f'fill="var(--accent)" opacity="{0.85 if r<=3 else 0.35}"/>')
    b.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="var(--warn)" stroke-width="1.5" opacity=".7"/>')
    for r, lab in [(1, "1.00"), (2, "0.50"), (5, "0.20"), (10, "0.10"), (25, "0.04")]:
        x = x0 + (r-0.5) * (pw/25)
        b.append(_txt(x, y0+ph+16, str(r), 11, "var(--mut)", "middle"))
        b.append(_txt(x, y0+ph-(1.0/r)*ph-6, lab, 10.5, "var(--mut)", "middle"))
    b.append(_txt(x0+pw/2, y0+ph+34, "rank of the first correct structure", 11.5, "var(--mut)", "middle"))
    b.append(_txt(0, H-24, "A wrong guess costs nothing but the slot it sits in, so you always submit all 25.", 12.5, "var(--fg)", weight="600"))
    b.append(_txt(0, H-6, "Only the ordering matters. That makes this a ranking and calibration problem, not a prediction problem.", 12.5, "var(--mut)"))
    return _wrap(W, H, "".join(b))

def fig_lb():
    """The overnight jump when the leader open-sourced."""
    W, H = 760, 230
    x0, y0, pw, ph = 70, 40, 620, 130
    b = [_txt(0, 20, "The leaderboard, before and after one competitor published his solution", 13, "var(--fg)", weight="600")]
    b.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="var(--line)"/>')
    def yv(v): return y0 + ph - (v/0.40)*ph
    for v in (0.10, 0.20, 0.30, 0.40):
        b.append(f'<line x1="{x0}" y1="{yv(v):.1f}" x2="{x0+pw}" y2="{yv(v):.1f}" stroke="var(--line)" opacity=".5"/>')
        b.append(_txt(x0-8, yv(v)+4, f"{v:.2f}", 11, "var(--mut)", "end"))
    bx = x0 + pw*0.62
    b.append(f'<line x1="{bx}" y1="{y0-6}" x2="{bx}" y2="{y0+ph}" stroke="var(--warn)" stroke-width="1.5" stroke-dasharray="4 3"/>')
    b.append(_txt(bx+8, y0+6, "solution published", 11.5, "var(--warn)", weight="600"))
    b.append(f'<polyline points="{x0+20},{yv(0.148):.1f} {x0+120},{yv(0.152):.1f} {bx-30},{yv(0.155):.1f} {bx},{yv(0.158):.1f}" fill="none" stroke="var(--mut)" stroke-width="2"/>')
    b.append(f'<polyline points="{bx},{yv(0.158):.1f} {bx+30},{yv(0.339):.1f} {x0+pw-20},{yv(0.353):.1f}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>')
    b.append(f'<circle cx="{x0+pw-20}" cy="{yv(0.353):.1f}" r="4" fill="var(--accent)"/>')
    b.append(_txt(x0+pw-28, yv(0.353)-10, "0.353", 12, "var(--accent)", "end", "700"))
    b.append(_txt(x0+30, yv(0.148)+20, "~0.15 for weeks", 11.5, "var(--mut)"))
    b.append(_txt(0, H-24, "Thirty teams now sit at exactly 0.339 — that is his notebook, run unmodified.", 12.5, "var(--fg)", weight="600"))
    b.append(_txt(0, H-6, "This field is sharp and gives strong work away fast. Expect it to happen again.", 12.5, "var(--mut)"))
    return _wrap(W, H, "".join(b))

def fig_cv():
    """Why everyone's validation is contaminated."""
    W, H = 780, 300
    b = [_txt(0, 20, "Why the whole field's local testing disagrees with the leaderboard", 13, "var(--fg)", weight="600")]
    libs = ["riken", "gnps", "mona", "massbank", "pluskal"]
    def panel(px, title, col, held_out_all, caption, score):
        o = [f'<rect x="{px}" y="44" width="350" height="180" rx="10" fill="var(--panel)" stroke="var(--line)"/>']
        o.append(_txt(px+16, 68, title, 12.5, col, weight="700"))
        o.append(f'<rect x="{px+16}" y="80" width="150" height="26" rx="4" fill="{col}" opacity=".18" stroke="{col}"/>')
        o.append(_txt(px+91, 97, "enveda-np-examples", 10.5, "var(--fg)", "middle", "600"))
        o.append(_txt(px+176, 97, "held out", 10.5, col, weight="600"))
        o.append(_txt(px+16, 126, "same 250 structures also live in:", 11, "var(--mut)"))
        lx = px+16
        for L in libs:
            w = 8 + len(L)*6
            fill = col if held_out_all else "var(--warn)"
            op = ".18" if held_out_all else ".18"
            o.append(f'<rect x="{lx}" y="134" width="{w}" height="20" rx="4" fill="{fill}" opacity="{op}" stroke="{fill}"/>')
            o.append(_txt(lx+w/2, 148, L, 9.5, "var(--fg)", "middle"))
            if held_out_all:
                o.append(f'<line x1="{lx+2}" y1="152" x2="{lx+w-2}" y2="136" stroke="{col}" stroke-width="1.5"/>')
            lx += w + 6
        o.append(_txt(px+16, 176, caption, 11, "var(--mut)"))
        o.append(_txt(px+16, 206, score, 12, "var(--fg)", weight="700"))
        return "".join(o)
    b.append(panel(0, "HOLD OUT BY SOURCE LABEL  (what everyone does)", "var(--warn)", False,
                   "still in the reference index, under another name", "local 0.33–0.37   →   leaderboard 0.15"))
    b.append(panel(400, "HOLD OUT BY STRUCTURE KEY  (what we'd do)", "var(--ok)", True,
                   "removed everywhere at once, by InChIKey14", "local score you can actually trust"))
    b.append(f'<line x1="0" y1="244" x2="{W}" y2="244" stroke="var(--line)"/>')
    b.append(_txt(0, 266, "A contaminated holdout secretly measures Class 1 — the 16% that barely moves the score.", 12.5, "var(--fg)", weight="600"))
    b.append(_txt(0, 286, "So the field is tuning hard against the wrong number. One team's score fell 0.152 → 0.145 doing exactly that.", 12.5, "var(--mut)"))
    return _wrap(W, H, "".join(b))

ALL = {"classes": fig_classes, "headroom": fig_headroom, "mrr": fig_mrr, "lb": fig_lb, "cv": fig_cv}
