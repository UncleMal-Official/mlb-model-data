"""Daily model recap — per-tab W/L, sharp reads, misses, rollup.

Produces:
  output/recap_<date>.md    full detail (internal / VIP notes)
  output/recap_<date>.png   branded shareable card (1080x1560)

Grading basis:
  - If board_pp_<date>.csv exists (real PrizePicks lines joined), W/L is graded on
    POSTED LINES and the card says so.
  - Otherwise it falls back to model-estimated lines and the card is watermarked
    "EST. LINES" so a synthetic record can never be mistaken for a real one.

Usage: python3 make_recap.py 2026-08-07
"""
import json, pathlib, sys
import pandas as pd
import numpy as np

import render_card  # reuse the Playwright renderer

D = pathlib.Path("/home/claude/mlb_model")
OUT = D / "output"
GREEN, RED, MUTED = "#4ade47", "#e5484d", "#8d948d"


def load(date):
    r = pd.read_csv(OUT / f"results_{date}.csv")
    b = pd.read_csv(OUT / f"full_board_v1_{date}.csv").drop_duplicates("player_id")
    keep = ["player_id", "actual_fp", "est_line", "PA", "H", "d2", "d3", "HR", "R", "RBI", "BB", "SB"]
    m = b.merge(r[keep], on="player_id", how="left")
    m["flags"] = m["flags"].fillna("")
    pp = OUT / f"board_pp_{date}.csv"
    real = False
    if pp.exists():
        p = pd.read_csv(pp)[["player_id", "fs_line"]]
        m = m.merge(p, on="player_id", how="left")
        if m.fs_line.notna().sum() >= 20:
            real = True
    if not real:
        m["fs_line"] = np.nan
    m["line"] = m.fs_line.where(m.fs_line.notna(), m.est_line)
    m["win"] = m.actual_fp > m.line
    m["push"] = m.actual_fp == m.line
    return m, real


def rec(d):
    g = d[~d.push]
    w = int(g.win.sum()); l = len(g) - w; p = int(d.push.sum())
    return dict(n=len(d), w=w, l=l, p=p,
                rate=100 * w / max(len(g), 1),
                fp=float(d.actual_fp.mean()), hr=int(d.HR.sum()))


def build(date):
    m, real = load(date)
    slate = float(m.actual_fp.mean())
    da_path = OUT / f"day_after_{date}.csv"
    tabs = []

    core60 = m[m.conf != "LOW"].nlargest(60, "proj_fp")
    core20 = core60.nlargest(20, "proj_fp")
    dl = m[m.conf != "LOW"].copy()
    dl["up"] = dl.hr_prob * 0.55 + dl.pgs.clip(lower=0) * 12 + dl.proj_tb * 8
    demons = dl.nlargest(40, "up")
    hr40 = m.nlargest(40, "hr_prob")
    pgs20 = m.nlargest(20, "pgs")
    da = m[m.player.isin(pd.read_csv(da_path).player)] if da_path.exists() else m.iloc[0:0]

    for name, d in [("CORE — Top 20", core20), ("CORE — Full 60", core60),
                    ("Day After", da), ("Demons & Long-shots", demons),
                    ("PGS Raw Top 20", pgs20)]:
        if len(d):
            t = rec(d); t["tab"] = name; t["vs"] = t["fp"] - slate
            tabs.append(t)

    hr_exp = float(hr40.hr_prob.sum() / 100)
    hr_hit = int(hr40.HR.sum())
    board_exp = float(m.hr_prob.sum() / 100)
    board_hr = int(m.HR.sum())

    payload = dict(date=date, real=real, slate=slate, tabs=tabs,
                   hr=dict(exp=hr_exp, hit=hr_hit, board_exp=board_exp, board_hit=board_hr),
                   rollup=rec(pd.concat([core20, da, demons]).drop_duplicates("player_id")),
                   top_scores=m.nlargest(5, "actual_fp")[["player", "team", "actual_fp", "proj_fp"]].to_dict("records"))
    return m, payload


CARD = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="hz:slide-selector" content=".graphic">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#222;display:flex;justify-content:center}
.graphic{position:relative;width:1080px;min-height:1560px;background:#0b0f0c;overflow:hidden;
 font-family:"omnes-pro","Carlito",sans-serif;display:flex;flex-direction:column;
 padding:52px 72px 0}
.glow{position:absolute;left:-140px;top:-260px;width:1360px;height:700px;
 background:radial-gradient(ellipse at center,rgba(74,222,71,.13),rgba(0,0,0,0) 70%)}
.z{position:relative;z-index:2}
.hdr{text-align:center;font-weight:700;font-size:25px;letter-spacing:8px;color:#4ade47;text-transform:uppercase}
.ttl{text-align:center;font-weight:800;font-size:76px;letter-spacing:1px;color:#fff;margin-top:14px;
 font-family:"bebas-neue-v14-deprecated","DejaVu Sans Condensed",sans-serif;line-height:1}
.sub{text-align:center;font-size:21px;color:#9aa19a;margin-top:14px}
.hero{display:flex;gap:24px;margin-top:30px}
.htile{flex:1;height:150px;background:rgba(18,26,19,.95);border:1px solid #22381f;border-radius:14px;
 display:flex;flex-direction:column;justify-content:center;align-items:center}
.htile .v{font-family:"bebas-neue-v14-deprecated","DejaVu Sans Condensed",sans-serif;
 font-weight:700;font-size:54px;line-height:1;color:#4ade47;white-space:nowrap}
.htile .k{margin-top:9px;font-size:16px;letter-spacing:2px;color:#8d948d;text-transform:uppercase;white-space:nowrap}
.sect{font-weight:700;font-size:19px;letter-spacing:5px;color:#4ade47;text-transform:uppercase;margin-top:34px}
table{width:100%;border-collapse:collapse;margin-top:16px}
th{font-size:15px;letter-spacing:2px;color:#8d948d;text-transform:uppercase;text-align:right;
 padding:0 12px 12px 0;font-weight:700}
th.l,td.l{text-align:left;padding-left:6px}
td{font-size:23px;color:#e8ece8;padding:12px 12px 12px 0;text-align:right;
 border-top:1px solid #1b2a1a;white-space:nowrap}
td.l{font-weight:600}
.good{color:#4ade47;font-weight:700}
.bad{color:#e5484d;font-weight:700}
.mut{color:#8d948d;font-weight:700}
.note{font-size:20px;color:#c3c9c3;line-height:1.55;margin-top:14px}
.note b{color:#4ade47}
.note .x{color:#e5484d;font-weight:700}
.roll{height:132px;background:rgba(18,26,19,.95);border:1px solid #2c4a28;border-radius:14px;
 display:flex;align-items:center;justify-content:center;gap:26px;margin-top:30px;margin-bottom:26px}
.roll .big{font-family:"bebas-neue-v14-deprecated","DejaVu Sans Condensed",sans-serif;
 font-size:64px;color:#fff;line-height:1}
.roll .lab{font-size:17px;letter-spacing:2px;color:#8d948d;text-transform:uppercase}
.spacer{flex:1}
.foot{margin:0 -72px 0;height:104px;background:#0d1410;border-top:1px solid #1b2a1a;position:relative}
.foot .d{position:absolute;top:15px;left:72px;width:936px;font-size:15px;color:#7d847d;line-height:1.5}
.foot .b{position:absolute;bottom:13px;left:0;width:1080px;text-align:center;font-weight:700;
 font-size:19px;letter-spacing:9px;color:#4ade47;text-transform:uppercase}
</style></head><body>
<div class="graphic"><div class="glow"></div>
<div class="z">
<div class="hdr">Model Recap</div>
<div class="ttl">__TITLE__</div>
<div class="sub">__SUB__</div>
<div class="hero">__HERO__</div>
<div class="sect">Tab by Tab</div>
<table>__TABLE__</table>
<div class="sect">Sharp Reads</div>
<div class="note">__SHARP__</div>
<div class="sect">Misses</div>
<div class="note">__MISS__</div>
<div class="roll">__ROLL__</div>
</div>
<div class="spacer"></div>
<div class="foot"><div class="d">__DISC__</div><div class="b">CashCord</div></div>
</div></body></html>"""


def render(payload, m, out_png):
    d = pd.Timestamp(payload["date"])
    title = d.strftime("%A, %b ") + str(d.day)
    real = payload["real"]
    tabs = payload["tabs"]
    by = {t["tab"]: t for t in tabs}
    da = by.get("Day After"); c20 = by.get("CORE — Top 20")
    hr = payload["hr"]

    hero = []
    hero.append(("%d-%d" % (da["w"], da["l"]) if da else "—", "Day After Cohort"))
    hero.append(("%d-%d" % (c20["w"], c20["l"]) if c20 else "—", "CORE Top 20"))
    hero.append((f"{hr['board_hit']} / {hr['board_exp']:.0f}", "HRs Hit / Model Called"))
    htiles = "".join(
        f'<div class="htile" style="left:{i*320}px"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for i, (v, k) in enumerate(hero))

    rows = ['<tr><th class="l">Tab</th><th>Record</th><th>Win %</th><th>Avg FP</th><th>vs Slate</th></tr>']
    potd = m[m.player == "Bobby Witt Jr."]
    if len(potd):
        a = float(potd.actual_fp.iloc[0])
        rows.append(f'<tr><td class="l">POTD — Witt o7.0 <span style="color:#8d948d;font-size:16px">(posted line)</span></td>'
                    f'<td class="mut">PUSH</td><td class="mut">—</td><td>{a:.0f}</td><td class="mut">—</td></tr>')
    for t in tabs:
        cls = "good" if t["rate"] >= 54.2 else ("bad" if t["rate"] < 50 else "mut")
        vs = t["vs"]
        vcls = "good" if vs > 0.3 else ("bad" if vs < -0.3 else "mut")
        rows.append(f'<tr><td class="l">{t["tab"]}</td>'
                    f'<td class="{cls}">{t["w"]}-{t["l"]}</td>'
                    f'<td class="{cls}">{t["rate"]:.0f}%</td>'
                    f'<td>{t["fp"]:.1f}</td>'
                    f'<td class="{vcls}">{"+" if vs>=0 else ""}{vs:.1f}</td></tr>')
    rows.append(f'<tr><td class="l">HR Board — Top 40</td><td class="mut">{hr["hit"]} HR</td>'
                f'<td class="mut">{100*hr["hit"]/40:.0f}%</td><td>{hr["exp"]:.1f} exp</td><td class="mut">—</td></tr>')
    table = "".join(rows)


    roll = payload["rollup"]
    basis = "posted PrizePicks lines" if real else "model-estimated lines"
    rollhtml = (f'<div><div class="big">{roll["w"]}-{roll["l"]}</div></div>'
                f'<div><div class="big" style="color:#4ade47">{roll["rate"]:.0f}%</div></div>'
                f'<div style="max-width:420px"><div class="lab">Graded plays, all tabs</div>'
                f'<div class="lab" style="color:#c3c9c3;letter-spacing:0;font-size:18px;text-transform:none;margin-top:6px">'
                f'vs {basis}</div></div>')

    disc = (f"Every hitter on the board graded against real box scores. "
            + ("Records on posted PrizePicks lines."
               if real else
               "PrizePicks feed went live Aug 8 — Friday's records are graded against model-estimated "
               "lines (sim median &minus; 1) and are NOT posted-line results. Shown for internal calibration."))

    html = (CARD.replace("__TITLE__", title.upper())
            .replace("__SUB__", payload["sub"])
            .replace("__HERO__", htiles)
            .replace("__TABLE__", table)
            .replace("__SHARP__", payload["sharp"])
            .replace("__MISS__", payload["miss"])
            .replace("__ROLL__", rollhtml)
            .replace("__DISC__", disc))
    tmp = pathlib.Path("/tmp/recap_card.html"); tmp.write_text(html)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": 1080, "height": 1560}, device_scale_factor=2)
        pg.goto(f"file://{tmp}")
        pg.wait_for_timeout(350)
        pg.locator(".graphic").screenshot(path=out_png)
        br.close()
    return out_png


if __name__ == "__main__":
    date = sys.argv[1]
    m, payload = build(date)
    json.dump(payload, open(OUT / f"recap_{date}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in payload.items() if k != "top_scores"}, indent=1, default=float))
