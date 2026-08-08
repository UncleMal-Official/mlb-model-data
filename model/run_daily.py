"""Autopilot orchestrator — runs on GitHub Actions after every data fetch.

Steps (each non-fatal where sensible):
  1. Symlink compatibility paths so the existing model scripts run unmodified.
  2. Refresh pitcher platoon splits for today's probables (statsapi, direct).
  3. Ensure park/BPP configs exist for today (neutral PENDING fallback).
  4. Build the board (recency-vs-mix filter included) with snapshot + What Changed diff.
  5. Grade YESTERDAY's board vs real box scores -> results CSV + tracking.csv append.
  6. Day After + Yesterday's Lessons.
  7. Join PrizePicks lines if a JSON exists at data/pp/pp_<today>*.json (Mike uploads).
  8. Workbook + auto-POTD card (real player art from data/headshots/).
  9. Copy everything to outputs/<today>/ for commit.
"""
import json, os, pathlib, re, shutil, subprocess, sys
from datetime import date, timedelta

import pandas as pd
import requests

TODAY = os.environ.get("MODEL_DATE") or date.today().isoformat()
YDAY = (date.fromisoformat(TODAY) - timedelta(days=1)).isoformat()
WS = pathlib.Path(os.environ.get("GITHUB_WORKSPACE", pathlib.Path(__file__).resolve().parent.parent))
MODEL = WS / "model"
OUT = MODEL / "output"
OUT.mkdir(exist_ok=True)
(MODEL / "data").mkdir(exist_ok=True)

# ---- 1. compatibility paths (scripts reference /home/claude/...)
for link, target in (("/home/claude/mlb_model", MODEL), ("/home/claude/mlb-model-data", WS)):
    p = pathlib.Path(link)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(target)
# scripts expect config/ + data/ under mlb_model
for sub in ("config", "assets"):
    pass
if not (MODEL / "data" / "pitcher_splits_2026.json").exists():
    shutil.copy(MODEL / "config" / "pitcher_splits_2026.json", MODEL / "data" / "pitcher_splits_2026.json")
if not (OUT / "logo_b64.txt").exists():
    shutil.copy(MODEL / "assets" / "logo_b64.txt", OUT / "logo_b64.txt")
sys.path.insert(0, str(MODEL))

API = "https://statsapi.mlb.com/api/v1"
sch = json.load(open(WS / "data" / "schedule.json"))


def log(m):
    print(f"[run_daily] {m}", flush=True)


# ---- 2. refresh splits for today's probables
def refresh_splits():
    ps = json.load(open(MODEL / "data" / "pitcher_splits_2026.json"))
    ids = []
    for g in sch["games"]:
        for s in ("away", "home"):
            pid = g[s].get("probable_id")
            if pid:
                ids.append((int(pid), g[s].get("probable_name"), g[s].get("probable_throws") or "R", g[s]["team"]))
    added = 0
    for pid, name, throws, team in ids:
        try:
            j = requests.get(f"{API}/people/{pid}/stats",
                             params=dict(stats="statSplits", group="pitching", sitCodes="vl,vr"),
                             timeout=20).json()
            splits = {}
            for sp in (j.get("stats") or []):
                for s in sp.get("splits", []):
                    code = (s.get("split") or {}).get("code")
                    st = s.get("stat", {})
                    if code in ("vl", "vr"):
                        splits["vsL" if code == "vl" else "vsR"] = dict(
                            pa=st.get("battersFaced", 0) or st.get("plateAppearances", 0), ab=st.get("atBats", 0),
                            h=st.get("hits", 0), d2=st.get("doubles", 0), d3=st.get("triples", 0),
                            hr=st.get("homeRuns", 0), bb=st.get("baseOnBalls", 0), hbp=st.get("hitByPitch", 0),
                            so=st.get("strikeOuts", 0), avg=float(st.get("avg", 0) or 0),
                            obp=float(st.get("obp", 0) or 0), slg=float(st.get("slg", 0) or 0))
            if splits:
                ps["pitchers"][str(pid)] = dict(name=name, team=team, throws=throws,
                                                vsL=splits.get("vsL", {}), vsR=splits.get("vsR", {}))
                added += 1
        except Exception as e:
            log(f"splits {name}: {e}")
    json.dump(ps, open(MODEL / "data" / "pitcher_splits_2026.json", "w"))
    log(f"splits refreshed for {added} probables")


# ---- 3. park/bpp fallbacks
def ensure_configs():
    pk = MODEL / "config" / f"park_factors_{TODAY}.json"
    if not pk.exists():
        TEAM_ABBR = json.loads((MODEL / "config" / "team_abbr.json").read_text()) \
            if (MODEL / "config" / "team_abbr.json").exists() else None
        import re
        src = (MODEL / "build_board_v2.py").read_text()
        TEAM_ABBR = eval(re.search(r"TEAM_ABBR\s*=\s*\{[^}]+\}", src, re.S).group(0).split("=", 1)[1])
        games = {}
        for g in sch["games"]:
            key = f"{TEAM_ABBR[g['away']['team']]}@{TEAM_ABBR[g['home']['team']]}"
            games[key] = dict(park=g.get("venue"), time_et="", hr=1.0, xbh=1.0, single=1.0, runs=1.0,
                              rain_flag=False, wind="", notes="BPP pending - neutral")
        for k in list(games):
            for a, b in (("WSH", "WAS"), ("CHW", "CWS"), ("ATH", "OAK")):
                if a in k:
                    games[k.replace(a, b)] = games[k]
        json.dump(dict(source=f"NEUTRAL placeholder {TODAY} - park layer PENDING", games=games), open(pk, "w"), indent=1)
        log("park factors: neutral placeholder written")
    bp = MODEL / "config" / f"bpp_features_{TODAY}.json"
    if not bp.exists():
        json.dump(dict(source=f"none {TODAY}", notable_matchups=[], hr_solid_matchups=[],
                       most_likely_hr_bp_odds=[], most_hrs_allowed_starters=[], sim_most_likely=[],
                       sim_pitcher_props=[], bvp_history=[], outlier_odds_report=[]), open(bp, "w"), indent=1)



# ---- 3b. regenerate fallback tables from repo parquets (always fresh: call-ups included)
def ensure_local_tables():
    import re as _re
    src = (MODEL / "build_board_v2.py").read_text()
    TEAM_ABBR = eval(_re.search(r"TEAM_ABBR\s*=\s*\{[^}]+\}", src, _re.S).group(0).split("=", 1)[1])
    hb = pd.read_parquet(WS / "data" / "hitting_basic.parquet")
    l21 = hb[hb.window == "last21"].copy()
    l21["team"] = l21["team"].astype(str).map(TEAM_ABBR).fillna(l21["team"].astype(str))
    l21 = l21.rename(columns={"doubles": "d2", "triples": "d3"})
    cols = ["player_id", "name", "team", "pa", "ab", "h", "d2", "d3", "hr", "r", "rbi", "bb", "so", "sb", "avg", "slg"]
    l21[[c for c in cols if c in l21.columns]].to_csv(MODEL / "data" / "batters_21d.csv", index=False)

    ba = pd.read_parquet(WS / "data" / "batter_agg.parquet")
    a = ba[(ba.window == "season") & (ba.pitch_grp == "ALL")]
    if len(a):
        g = a.groupby(["batter", "stand"], as_index=False).pa.sum()
        top = g.sort_values("pa", ascending=False).drop_duplicates("batter")
        top = top.rename(columns={"batter": "player_id", "stand": "bat_side"})
        top[["player_id", "bat_side"]].to_csv(MODEL / "data" / "bat_sides.csv", index=False)
    log(f"fallback tables rebuilt: {len(l21)} batters_21d rows")


# ---- 4. board (dated exec of the canonical builder)
def build_board():
    src = (MODEL / "build_board_v2.py").read_text().replace("2026-08-07", TODAY)
    prior = OUT / f"full_board_v1_{TODAY}.csv"
    snap = None
    if prior.exists():
        snap = f"/tmp/board_snap_{TODAY}.csv"
        shutil.copy(prior, snap)
    cwd = os.getcwd(); os.chdir(MODEL)
    try:
        exec(compile(src, "build_board_dated.py", "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(cwd)
    if snap:
        from diff_boards import diff
        d = diff(snap, str(prior))
        d.to_csv(OUT / f"board_changes_{TODAY}.csv", index=False)
        log(f"what changed: {len(d)} rows")


# ---- 5. grade yesterday
def grade_yesterday():
    yb = OUT / f"full_board_v1_{YDAY}.csv"
    if not yb.exists():
        yb = WS / "outputs" / YDAY / f"full_board_v1_{YDAY}.csv"
    if not yb.exists():
        log("no yesterday board to grade"); return
    b = pd.read_csv(yb).drop_duplicates("player_id")
    ids = b.player_id.astype(int).tolist()
    rows = {}
    for i in range(0, len(ids), 100):
        try:
            j = requests.get(f"{API}/people", params=dict(
                personIds=",".join(map(str, ids[i:i + 100])),
                hydrate=f"stats(group=[hitting],type=[byDateRange],startDate={YDAY},endDate={YDAY})"),
                timeout=30).json()
            for p in j.get("people", []):
                for st in (p.get("stats") or []):
                    for s in st.get("splits", []):
                        t = s.get("stat", {})
                        rows[p["id"]] = t
        except Exception as e:
            log(f"grade chunk: {e}")
    def fp(t):
        s1 = t.get("hits", 0) - t.get("doubles", 0) - t.get("triples", 0) - t.get("homeRuns", 0)
        return (3 * s1 + 5 * t.get("doubles", 0) + 8 * t.get("triples", 0) + 10 * t.get("homeRuns", 0)
                + 2 * (t.get("runs", 0) + t.get("rbi", 0) + t.get("baseOnBalls", 0) + t.get("hitByPitch", 0))
                + 5 * t.get("stolenBases", 0))
    b["actual_fp"] = b.player_id.map(lambda i: fp(rows[i]) if i in rows else None)
    b["actual_pa"] = b.player_id.map(lambda i: rows.get(i, {}).get("plateAppearances"))
    b.to_csv(OUT / f"results_{YDAY}.csv", index=False)
    played = b[b.actual_pa.fillna(0) > 0]
    core = played[(played.conf.isin(["HIGH", "MED"])) & (played.proj_fp >= 7.0)].nlargest(20, "proj_fp")
    tr = MODEL / "tracking.csv"
    t = pd.read_csv(tr) if tr.exists() else pd.DataFrame(columns=["date", "cohort", "n", "hits", "notes"])
    est = (core.proj_fp * 0.72 - 1.6).round()  # crude est-line proxy for unattended runs
    hits = int((core.actual_fp > est).sum())
    t = pd.concat([t, pd.DataFrame([dict(date=YDAY, cohort="CORE20_auto", n=len(core), hits=hits,
                                         notes=f"mean proj {core.proj_fp.mean():.1f} vs actual {core.actual_fp.mean():.1f} (autopilot grade)")])])
    t.to_csv(tr, index=False)
    log(f"graded {len(played)} players from {YDAY}; CORE20 {hits}/{len(core)}")


# ---- 6-8: day after, lessons, lines, workbook, card
def rest():
    try:
        subprocess.run([sys.executable, "-c", f"""
import sys; sys.path.insert(0, "{MODEL}")
import pandas as pd, json
from yesterdays_lessons import build_lessons
R = "{WS}/data"
rb = pd.read_parquet(f"{{R}}/recent_bbe.parquet"); ba = pd.read_parquet(f"{{R}}/batter_agg.parquet")
hb = pd.read_parquet(f"{{R}}/hitting_basic.parquet"); pa = pd.read_parquet(f"{{R}}/pitcher_agg.parquet")
L = build_lessons(rb, ba, hb, pa, "{YDAY}")
L.to_csv("{OUT}/lessons_{YDAY}.csv")
print("lessons:", len(L))
"""], check=False, cwd=MODEL)
    except Exception as e:
        log(f"lessons: {e}")
    # PrizePicks JSON: accept data/pp/ OR repo root, any pp_<date>.json / projections json
    cands = []
    for d in (WS / "data" / "pp", WS, MODEL / "data"):
        if d.exists():
            cands += [p for p in d.glob("*.json")
                      if re.search(r"^(pp[_-]|prizepicks|projections)", p.name, re.I)
                      and (TODAY in p.name or re.search(r"projections", p.name, re.I))]
    pp = sorted(set(cands))
    cal_k = None
    if pp:
        r = subprocess.run([sys.executable, str(MODEL / "join_pp_lines.py"), *[str(x) for x in pp],
                            "--date", TODAY], capture_output=True, text=True, cwd=MODEL)
        print(r.stdout[-3000:], flush=True)
        m = re.search(r"calibration: k = ([0-9.]+)", r.stdout or "")
        if m:
            cal_k = float(m.group(1))
        log(f"PP lines joined from {[p.name for p in pp]} | cal k={cal_k}")
    else:
        log("no PrizePicks JSON found (drop pp_<date>.json in data/pp/ or repo root) - board ships without lines")

    subprocess.run([sys.executable, str(MODEL / "make_xlsx_v2.py"), TODAY, "v1"], check=False, cwd=MODEL)

    board_arg = ["--board", str(OUT / f"full_board_v1_{TODAY}.csv"), "--date", TODAY]
    subprocess.run([sys.executable, str(MODEL / "make_potd_card.py"), *board_arg], check=False, cwd=MODEL)

    # If real lines exist, re-render the card with the pick's ACTUAL line + market calibration
    pp_csv = OUT / f"board_pp_{TODAY}.csv"
    meta_p = OUT / f"potd_auto_{TODAY}.json"
    if cal_k and pp_csv.exists() and meta_p.exists():
        try:
            meta = json.load(open(meta_p))
            dfp = pd.read_csv(pp_csv)
            hit = dfp[dfp.player_id == meta["player_id"]]
            line = float(hit.fs_line.iloc[0]) if len(hit) and pd.notna(hit.fs_line.iloc[0]) else None
            if line:
                subprocess.run([sys.executable, str(MODEL / "make_potd_card.py"), *board_arg,
                                "--player", meta["player"], "--line", str(line),
                                "--cal-k", str(cal_k)], check=False, cwd=MODEL)
                log(f"card re-rendered with REAL line {line} (cal k={cal_k})")
            else:
                log(f"{meta['player']} has no PP fantasy line - card keeps model estimate")
        except Exception as e:
            log(f"card real-line pass: {e}")


# ---- 9. stage outputs for commit
def stage():
    dest = WS / "outputs" / TODAY
    dest.mkdir(parents=True, exist_ok=True)
    for pat in (f"full_board_v1_{TODAY}.csv", f"MLB_Model_Board_v1_{TODAY}.xlsx",
                f"potd_auto_{TODAY}.png", f"potd_auto_{TODAY}.json",
                f"board_changes_{TODAY}.csv", f"lessons_{YDAY}.csv",
                f"results_{YDAY}.csv", f"day_after_{TODAY}.csv", f"board_pp_{TODAY}.csv"):
        p = OUT / pat
        if p.exists():
            shutil.copy(p, dest / pat)
    shutil.copy(MODEL / "tracking.csv", dest / "tracking.csv")
    log(f"outputs staged -> outputs/{TODAY}/")


if __name__ == "__main__":
    refresh_splits()
    ensure_configs()
    ensure_local_tables()
    build_board()
    grade_yesterday()
    try:  # Day After tool (dated exec, same trick as the board)
        da_src = (MODEL / "run_day_after.py").read_text().replace("2026-08-07", TODAY).replace("2026-08-06", YDAY)
        cwd = os.getcwd(); os.chdir(MODEL)
        exec(compile(da_src, "run_day_after_dated.py", "exec"), {"__name__": "__main__"})
        os.chdir(cwd)
    except Exception as e:
        log(f"day after: {e}")
    rest()
    stage()
    log("done")
