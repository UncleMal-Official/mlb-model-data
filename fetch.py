"""
Daily MLB data fetcher.
Pulls pitch-level Statcast data (incremental, cached), plus MLB Stats API
schedule/probables/lineups, and writes compact aggregate tables the model consumes:

  data/batter_agg.parquet   - batter x platoon x pitch-group x window (barrels, EV, xwOBAcon, K/BB, swing profile)
  data/pitcher_agg.parquet  - pitcher x batter-side x pitch-type + group x window (usage, damage allowed, K/BB)
  data/hitting_basic.parquet- per-batter basic hitting rates (season + last 21d) incl. R/RBI/SB from Stats API
  data/schedule.json        - today's games, probables (with throwing hand), posted lineups (with batting side)
  data/meta.json            - freshness info

Raw pitch data is kept in raw_cache/ (GitHub Actions cache), NOT committed.
"""
import json, os, sys, time, warnings
from datetime import date, datetime, timedelta

import pandas as pd
import requests

warnings.filterwarnings("ignore")

SEASON_START = "2026-03-18"
TODAY = date.today()
RAW_DIR = "raw_cache"
OUT_DIR = "data"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

FASTBALLS = {"FF", "SI", "FC"}
BREAKING = {"SL", "ST", "SV", "CU", "KC", "CS"}
OFFSPEED = {"CH", "FS", "FO", "SC", "EP", "KN"}

def pitch_group(pt):
    if pt in FASTBALLS: return "FB"
    if pt in BREAKING: return "BRK"
    if pt in OFFSPEED: return "OFF"
    return "OTH"

# ---------------------------------------------------------------- raw statcast
def month_key(d): return d.strftime("%Y-%m")

def fetch_statcast_range(start, end):
    """Pull one date range via pybaseball (chunks internally)."""
    from pybaseball import statcast
    for attempt in range(3):
        try:
            df = statcast(start_dt=start, end_dt=end, verbose=False)
            return df
        except Exception as e:
            print(f"  retry {attempt+1} for {start}..{end}: {e}")
            time.sleep(20 * (attempt + 1))
    print(f"  FAILED range {start}..{end}, continuing")
    return pd.DataFrame()

KEEP_COLS = [
    "game_date", "game_pk", "batter", "pitcher", "player_name", "stand", "p_throws",
    "pitch_type", "events", "description", "zone", "balls", "strikes",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "at_bat_number", "pitch_number", "inning", "home_team", "away_team",
]

def update_raw():
    """Incrementally maintain monthly parquet shards in raw_cache/."""
    shards = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".parquet"))
    last_date = None
    if shards:
        newest = pd.read_parquet(os.path.join(RAW_DIR, shards[-1]), columns=["game_date"])
        if len(newest):
            last_date = pd.to_datetime(newest["game_date"]).max().date()
    start = (last_date - timedelta(days=2)) if last_date else datetime.strptime(SEASON_START, "%Y-%m-%d").date()
    end = TODAY - timedelta(days=1)
    if start > end:
        print("raw cache already current")
        return
    print(f"pulling statcast {start} .. {end}")
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=5), end)
        df = fetch_statcast_range(cur.isoformat(), chunk_end.isoformat())
        if len(df):
            df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
            df["game_date"] = pd.to_datetime(df["game_date"])
            for mk, sub in df.groupby(df["game_date"].dt.strftime("%Y-%m")):
                path = os.path.join(RAW_DIR, f"statcast_{mk}.parquet")
                if os.path.exists(path):
                    old = pd.read_parquet(path)
                    sub = pd.concat([old, sub], ignore_index=True)
                    sub = sub.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"], keep="last")
                sub.to_parquet(path, index=False)
            print(f"  {cur}..{chunk_end}: {len(df)} pitches")
        cur = chunk_end + timedelta(days=1)

def load_raw():
    parts = []
    for f in sorted(os.listdir(RAW_DIR)):
        if f.endswith(".parquet"):
            parts.append(pd.read_parquet(os.path.join(RAW_DIR, f)))
    if not parts:
        return pd.DataFrame(columns=KEEP_COLS)
    df = pd.concat(parts, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitch_grp"] = df["pitch_type"].map(pitch_group)
    df["is_swing"] = df["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
        "hit_into_play", "hit_into_play_score", "hit_into_play_no_out", "missed_bunt", "foul_bunt",
    ])
    df["is_whiff"] = df["description"].isin(["swinging_strike", "swinging_strike_blocked", "missed_bunt"])
    df["is_bbe"] = df["description"].str.startswith("hit_into_play") & df["launch_speed"].notna()
    df["is_barrel"] = (df["launch_speed_angle"] == 6)
    df["is_hardhit"] = df["launch_speed"] >= 95
    df["out_zone"] = df["zone"] >= 11
    df["is_chase"] = df["out_zone"] & df["is_swing"]
    ev = df["events"].fillna("")
    df["pa_end"] = ev != ""
    df["e_k"] = ev.isin(["strikeout", "strikeout_double_play"])
    df["e_bb"] = ev == "walk"
    df["e_hbp"] = ev == "hit_by_pitch"
    df["e_1b"] = ev == "single"
    df["e_2b"] = ev == "double"
    df["e_3b"] = ev == "triple"
    df["e_hr"] = ev == "home_run"
    return df

AGG_SPEC = dict(
    pitches=("pitch_type", "size"), swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"),
    chases=("is_chase", "sum"), out_zone_p=("out_zone", "sum"), bbe=("is_bbe", "sum"),
    barrels=("is_barrel", "sum"), hardhit=("is_hardhit", "sum"),
    ev_sum=("launch_speed", "sum"), la_sum=("launch_angle", "sum"),
    xwobacon_sum=("estimated_woba_using_speedangle", "sum"),
    woba_sum=("woba_value", "sum"), woba_den=("woba_denom", "sum"),
    pa=("pa_end", "sum"), k=("e_k", "sum"), bb=("e_bb", "sum"), hbp=("e_hbp", "sum"),
    b1=("e_1b", "sum"), b2=("e_2b", "sum"), b3=("e_3b", "sum"), hr=("e_hr", "sum"),
)

def window_frames(df):
    w21 = df[df["game_date"] >= pd.Timestamp(TODAY - timedelta(days=21))]
    w45 = df[df["game_date"] >= pd.Timestamp(TODAY - timedelta(days=45))]
    return {"w21": w21, "w45": w45, "season": df}

def build_batter_agg(df):
    rows = []
    for wname, wdf in window_frames(df).items():
        for keys, g in [(["batter", "stand", "p_throws", "pitch_grp"], None), (["batter", "stand", "p_throws"], "ALL")]:
            grp = wdf.groupby(keys).agg(**AGG_SPEC).reset_index()
            if g == "ALL":
                grp["pitch_grp"] = "ALL"
            grp["window"] = wname
            rows.append(grp)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(f"{OUT_DIR}/batter_agg.parquet", index=False)
    print(f"batter_agg: {len(out)} rows")

def build_pitcher_agg(df):
    rows = []
    for wname, wdf in window_frames(df).items():
        for keys, g in [(["pitcher", "p_throws", "stand", "pitch_type"], None),
                        (["pitcher", "p_throws", "stand", "pitch_grp"], "GRP"),
                        (["pitcher", "p_throws", "stand"], "ALL")]:
            grp = wdf.groupby(keys).agg(**AGG_SPEC).reset_index()
            if g == "ALL":
                grp["pitch_type"] = "ALL"
            elif g == "GRP":
                grp = grp.rename(columns={"pitch_grp": "pitch_type"})
                grp["pitch_type"] = "GRP_" + grp["pitch_type"]
            grp["window"] = wname
            rows.append(grp)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(f"{OUT_DIR}/pitcher_agg.parquet", index=False)
    print(f"pitcher_agg: {len(out)} rows")

# ------------------------------------------------------------- stats api side
S = requests.Session()
S.headers["User-Agent"] = "mlb-model-fetcher/1.0"
API = "https://statsapi.mlb.com/api/v1"

def get(url, **params):
    r = S.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_hitting_basic():
    frames = []
    for label, extra in [("season", {"stats": "season"}),
                         ("last21", {"stats": "byDateRange",
                                     "startDate": (TODAY - timedelta(days=21)).isoformat(),
                                     "endDate": TODAY.isoformat()})]:
        offset, rows = 0, []
        while True:
            j = get(f"{API}/stats", group="hitting", sportId=1, season=2026,
                    limit=200, offset=offset, playerPool="ALL", **extra)
            splits = (j.get("stats") or [{}])[0].get("splits", [])
            if not splits: break
            for s in splits:
                st, p = s.get("stat", {}), s.get("player", {})
                rows.append(dict(window=label, player_id=p.get("id"), name=p.get("fullName"),
                                 team=(s.get("team") or {}).get("abbreviation") or (s.get("team") or {}).get("name"),
                                 pa=st.get("plateAppearances"), ab=st.get("atBats"), h=st.get("hits"),
                                 doubles=st.get("doubles"), triples=st.get("triples"), hr=st.get("homeRuns"),
                                 r=st.get("runs"), rbi=st.get("rbi"), bb=st.get("baseOnBalls"),
                                 hbp=st.get("hitByPitch"), so=st.get("strikeOuts"), sb=st.get("stolenBases"),
                                 avg=st.get("avg"), obp=st.get("obp"), slg=st.get("slg"), ops=st.get("ops")))
            offset += 200
            if offset > 3000: break
        frames.append(pd.DataFrame(rows))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(f"{OUT_DIR}/hitting_basic.parquet", index=False)
    print(f"hitting_basic: {len(out)} rows")

def fetch_schedule():
    j = get(f"{API}/schedule", sportId=1, date=TODAY.isoformat(),
            hydrate="probablePitcher,lineups,weather,venue")
    games, person_ids = [], set()
    for d in j.get("dates", []):
        for g in d.get("games", []):
            def side(s):
                t = g["teams"][s]
                pp = t.get("probablePitcher") or {}
                return dict(team=t["team"]["name"], team_id=t["team"]["id"],
                            probable_id=pp.get("id"), probable_name=pp.get("fullName"))
            ln = g.get("lineups") or {}
            def lineup(key):
                return [dict(id=p.get("id"), name=p.get("fullName")) for p in (ln.get(key) or [])]
            gg = dict(game_pk=g["gamePk"], game_date_utc=g["gameDate"],
                      status=(g.get("status") or {}).get("detailedState"),
                      venue=(g.get("venue") or {}).get("name"),
                      away=side("away"), home=side("home"),
                      away_lineup=lineup("awayPlayers"), home_lineup=lineup("homePlayers"),
                      doubleheader=g.get("doubleHeader"), game_number=g.get("gameNumber"))
            games.append(gg)
            for s in ("away", "home"):
                if gg[s]["probable_id"]: person_ids.add(gg[s]["probable_id"])
            for key in ("away_lineup", "home_lineup"):
                for p in gg[key]: person_ids.add(p["id"])
    hands = {}
    ids = list(person_ids)
    for i in range(0, len(ids), 100):
        j2 = get(f"{API}/people", personIds=",".join(map(str, ids[i:i+100])))
        for p in j2.get("people", []):
            hands[p["id"]] = dict(bat=(p.get("batSide") or {}).get("code"),
                                  throw=(p.get("pitchHand") or {}).get("code"),
                                  pos=((p.get("primaryPosition") or {}).get("abbreviation")))
    for g in games:
        for s in ("away", "home"):
            pid = g[s]["probable_id"]
            g[s]["probable_throws"] = (hands.get(pid) or {}).get("throw")
        for key in ("away_lineup", "home_lineup"):
            for idx, p in enumerate(g[key]):
                info = hands.get(p["id"]) or {}
                p["bat_side"], p["position"], p["order"] = info.get("bat"), info.get("pos"), idx + 1
    with open(f"{OUT_DIR}/schedule.json", "w") as f:
        json.dump(dict(fetched_utc=datetime.utcnow().isoformat(), date=TODAY.isoformat(), games=games), f, indent=1)
    print(f"schedule: {len(games)} games")

def fetch_prizepicks():
    """Best-effort PrizePicks board pull (may be bot-blocked from CI; failure is non-fatal)."""
    try:
        rows, players = [], {}
        url = "https://api.prizepicks.com/projections"
        params = {"league_id": 2, "per_page": 500, "single_stat": "true"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
                   "Accept": "application/json"}
        for page in range(1, 8):
            params["page"] = page
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code != 200:
                print(f"prizepicks: HTTP {r.status_code}, stopping")
                break
            j = r.json()
            for inc in j.get("included", []):
                if inc.get("type") == "new_player":
                    a = inc.get("attributes", {})
                    players[inc["id"]] = dict(name=a.get("display_name"), team=a.get("team"),
                                              position=a.get("position"))
            data = j.get("data", [])
            for d in data:
                a = d.get("attributes", {})
                pid = (((d.get("relationships") or {}).get("new_player") or {}).get("data") or {}).get("id")
                rows.append(dict(projection_id=d.get("id"), player_id=pid,
                                 stat_type=a.get("stat_type"), line=a.get("line_score"),
                                 odds_type=a.get("odds_type"), status=a.get("status"),
                                 game_id=a.get("game_id"), board_time=a.get("board_time"),
                                 start_time=a.get("start_time"), description=a.get("description")))
            nxt = (j.get("links") or {}).get("next")
            if not data or not nxt:
                break
        if rows:
            df = pd.DataFrame(rows)
            df["player_name"] = df["player_id"].map(lambda i: (players.get(i) or {}).get("name"))
            df["team"] = df["player_id"].map(lambda i: (players.get(i) or {}).get("team"))
            df["position"] = df["player_id"].map(lambda i: (players.get(i) or {}).get("position"))
            df.to_parquet(f"{OUT_DIR}/prizepicks.parquet", index=False)
            print(f"prizepicks: {len(df)} projections, {len(players)} players")
        else:
            print("prizepicks: no rows")
    except Exception as e:
        print(f"prizepicks fetch failed (non-fatal): {e}")

def main():
    t0 = time.time()
    fetch_schedule()
    fetch_prizepicks()
    fetch_hitting_basic()
    update_raw()
    df = load_raw()
    print(f"raw pitches loaded: {len(df)}")
    if len(df):
        build_batter_agg(df)
        build_pitcher_agg(df)
        max_date = str(df["game_date"].max().date())
    else:
        max_date = None
    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(dict(updated_utc=datetime.utcnow().isoformat(), raw_through=max_date,
                       raw_rows=int(len(df)), runtime_sec=round(time.time() - t0)), f, indent=1)
    print(f"done in {round(time.time()-t0)}s")

if __name__ == "__main__":
    main()
