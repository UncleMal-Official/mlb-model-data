"""Day After tool — unlucky hard contact yesterday + similar look today = candidate.

Activates when the fetcher's data/recent_bbe.parquet exists (fetch.py v4+).
Logic per Mike's spec (2026-08-07):
  1. Yesterday: strong HR process, weak HR result — barrels/near-barrels, high EV,
     HR-band launch angles, big distances... but 0 HR (or badly under xwOBA).
  2. Today: platoon lines up and today's SP mix overlaps the pitch groups the
     player was crushing yesterday ("major HR pitches overlap and platoons").
  3. Output a ranked Day After list + a DA tag that surfaces on the main board.
Psychology being exploited: the public plays a guy, he "fails," they're mad and
off him — while the contact quality says he's a STRONGER play today.
"""
import pandas as pd
import numpy as np

HR_LA_LO, HR_LA_HI = 18, 38


def yesterday_unlucky(recent_bbe: pd.DataFrame, yesterday) -> pd.DataFrame:
    """Score each batter's HR process vs result for a given date."""
    d = recent_bbe[recent_bbe.game_date == pd.Timestamp(yesterday)].copy()
    d = d.drop_duplicates()          # cache-append can duplicate rows
    if d.empty:
        return pd.DataFrame()
    d["batter"] = d["batter"].astype("int64")
    d["pitch_grp"] = d["pitch_grp"].astype(str)
    d["is_barrel"] = d.launch_speed_angle == 6
    d["near_barrel"] = (d.launch_speed >= 98) & d.launch_angle.between(HR_LA_LO, HR_LA_HI)
    d["deep_out"] = (d.hit_distance_sc >= 370) & (d.events != "home_run")
    d["is_hr"] = d.events == "home_run"
    g = d.groupby(["batter", "player_name"]).agg(
        bbe=("events", "size"), barrels=("is_barrel", "sum"), near=("near_barrel", "sum"),
        deep_outs=("deep_out", "sum"), max_ev=("launch_speed", "max"),
        max_dist=("hit_distance_sc", "max"), hr=("is_hr", "sum"),
        xwobacon=("estimated_woba_using_speedangle", "mean"),
    ).reset_index()
    # groups the batter was punishing yesterday (for mix-overlap vs today's SP)
    g["batter"] = g["batter"].astype("int64")
    hot = {b: s.value_counts(normalize=True).to_dict()
           for b, s in d[d.is_barrel | d.near_barrel].groupby("batter").pitch_grp}
    g["hot_groups"] = g["batter"].map(hot)
    g["process"] = (g.barrels * 2.0 + g.near * 1.0 + g.deep_outs * 1.5
                    + (g.max_ev.fillna(0) >= 105) * 1.0 + (g.max_dist.fillna(0) >= 390) * 1.0)
    g["unlucky"] = (g.hr == 0) & (g.process >= 2.0)
    return g[g.unlucky].drop_duplicates("batter").sort_values("process", ascending=False)


def day_after_candidates(unlucky: pd.DataFrame, today_board: pd.DataFrame,
                         mix_of_today_sp) -> pd.DataFrame:
    """Join yesterday's unlucky list to today's slate; require platoon + mix overlap.

    mix_of_today_sp: callable(batter_row) -> dict of {grp: usage} for today's opposing SP
    vs that batter's side (reuse pitcher_mix from the board builder).
    """
    rows = []
    tb = today_board.set_index("player_id")
    for _, u in unlucky.iterrows():
        if u.batter not in tb.index:
            continue
        t = tb.loc[u.batter]
        mix = mix_of_today_sp(t) or {}
        hot = u.hot_groups if isinstance(u.hot_groups, dict) else {}
        overlap = sum(min(hot.get(k, 0.0), mix.get(k, 0.0)) for k in set(hot) | set(mix))
        # "similar look": today's SP throws >=35% overlap of the groups he was barreling
        if overlap >= 0.35 or (not hot and mix):
            rows.append(dict(
                player=t.player, team=t.team, game=t.game, opp_sp=t.opp_sp,
                yesterday=(f"{int(u.barrels)} barrel(s), {int(u.near)} near, "
                           f"{int(u.deep_outs)} deep out(s), max {u.max_ev:.0f} mph"
                           + (f" / {u.max_dist:.0f} ft" if pd.notna(u.max_dist) else "")),
                process_score=round(float(u.process), 1), mix_overlap=round(float(overlap), 2),
                hr_prob_today=t.hr_prob, proj_fp=t.proj_fp, pgs=t.pgs,
                da_score=round(float(u.process) * (0.5 + overlap) * (t.hr_prob / 15.0), 2)))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.drop_duplicates("player").sort_values("da_score", ascending=False)
    return out
