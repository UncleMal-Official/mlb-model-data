"""Yesterday's Lessons — auto-written, data-backed takeaways from the last slate.

Categories (all computed, no vibes):
  UNLUCKY       - contact quality >> results (xwOBAcon vs actual on batted balls)
  LUCKY         - results >> contact quality (regression warning before the public chases)
  FELL OFF      - 21-day damage collapsed vs season baseline, with the pitch-group why
  STILL HOT     - hot bat, but yesterday was a brutal matchup - don't drop him
  PITCHER       - starters whose process and results diverged (escaped vs deserved better)
Max 10 lessons, ranked by signal size.
"""
import pandas as pd
import numpy as np

WOBA_EVENT = {"single": 0.9, "double": 1.25, "triple": 1.6, "home_run": 2.0}


def build_lessons(recent_bbe, batter_agg, hitting_basic, pitcher_agg, yesterday, pitcher_names=None):
    rb = recent_bbe[recent_bbe.game_date == pd.Timestamp(yesterday)].drop_duplicates().copy()
    if rb.empty:
        return pd.DataFrame()
    rb["batter"] = rb.batter.astype("int64")
    names = hitting_basic[hitting_basic.window == "season"].set_index("player_id")["name"].to_dict()
    teams = hitting_basic[hitting_basic.window == "season"].set_index("player_id")["team"].to_dict()
    rb["actual_woba"] = rb.events.map(WOBA_EVENT).fillna(0.0)
    rb["is_barrel"] = rb.launch_speed_angle == 6
    rb["deep_out"] = (rb.hit_distance_sc >= 370) & (rb.events != "home_run")

    lessons = []

    def add(cat, score, headline, data_line):
        lessons.append(dict(category=cat, score=round(float(score), 2),
                            headline=headline, data=data_line))

    # ---------- batter luck (yesterday) ----------
    g = rb.groupby("batter").agg(
        bbe=("events", "size"), xw=("estimated_woba_using_speedangle", "mean"),
        aw=("actual_woba", "mean"), barrels=("is_barrel", "sum"),
        deep=("deep_out", "sum"), max_ev=("launch_speed", "max"),
        max_dist=("hit_distance_sc", "max"), hr=("events", lambda s: (s == "home_run").sum()))
    g = g[g.bbe >= 3]
    g["gap"] = g.xw - g.aw
    ug = g[(g.gap >= 0.22) & ((g.barrels >= 1) | (g.deep >= 1) | (g.xw >= 0.5))]
    for pid, r in ug.sort_values("gap", ascending=False).head(4).iterrows():
        nm = names.get(pid, str(pid))
        add("UNLUCKY", r.gap * 10 + r.barrels,
            f"{nm} ({teams.get(pid,'?')}) got robbed - the process was elite",
            f"{int(r.bbe)} batted balls at {r.xw:.3f} expected wOBA vs {r.aw:.3f} actual; "
            f"{int(r.barrels)} barrel(s), max {r.max_ev:.0f} mph"
            + (f", {r.max_dist:.0f} ft" if pd.notna(r.max_dist) else "")
            + ". Contact like this pays off within days - Day After watch.")
    for pid, r in g[g.gap <= -0.22].sort_values("gap").head(3).iterrows():
        nm = names.get(pid, str(pid))
        add("LUCKY", abs(r.gap) * 9,
            f"{nm} ({teams.get(pid,'?')}) cashed weak contact - regression warning",
            f"Actual {r.aw:.3f} wOBA on {int(r.bbe)} batted balls vs just {r.xw:.3f} expected. "
            f"The box score flatters him - beware the public chasing this today.")

    # ---------- fell off (w21 vs season damage) ----------
    def agg_side(df, keys):
        return df.groupby(keys).agg(bbe=("bbe", "sum"), xsum=("xwobacon_sum", "sum"),
                                    ch=("chases", "sum"), oz=("out_zone_p", "sum"),
                                    wh=("whiffs", "sum"), sw=("swings", "sum")).reset_index()
    yb = set(rb.batter.unique())
    w21 = agg_side(batter_agg[(batter_agg.window == "w21") & (batter_agg.pitch_grp == "ALL")], ["batter"]).set_index("batter")
    sea = agg_side(batter_agg[(batter_agg.window == "season") & (batter_agg.pitch_grp == "ALL")], ["batter"]).set_index("batter")
    both = w21.join(sea, lsuffix="_21", rsuffix="_s")
    both = both[(both.bbe_21 >= 25) & (both.bbe_s >= 120)]
    both["x21"] = both.xsum_21 / both.bbe_21
    both["x_sea"] = both.xsum_s / both.bbe_s
    both["drop"] = both["x_sea"] - both["x21"]
    both = both[both.index.isin(yb)]
    for pid, r in both[both["drop"] >= 0.075].sort_values("drop", ascending=False).head(3).iterrows():
        nm = names.get(pid, str(pid))
        chase21 = r.ch_21 / max(r.oz_21, 1); chases = r.ch_s / max(r.oz_s, 1)
        whiff21 = r.wh_21 / max(r.sw_21, 1); whiffs = r.wh_s / max(r.sw_s, 1)
        why = []
        if chase21 - chases >= 0.03: why.append(f"chase rate up {100*(chase21-chases):.0f} pts")
        if whiff21 - whiffs >= 0.03: why.append(f"whiff rate up {100*(whiff21-whiffs):.0f} pts")
        why_s = "; ".join(why) if why else "contact quality itself has slipped"
        add("FELL OFF", r["drop"] * 8,
            f"{nm} ({teams.get(pid,'?')}) has genuinely fallen off - not just variance",
            f"xwOBA on contact: {r.x_sea:.3f} season vs {r.x21:.3f} last 21 days "
            f"({int(r.bbe_21)} recent batted balls). Why: {why_s}.")

    # ---------- still hot, bad matchup yesterday ----------
    h21 = hitting_basic[hitting_basic.window == "last21"].set_index("player_id")
    pit_sea = pitcher_agg[(pitcher_agg.window == "season") & (pitcher_agg.pitch_type == "ALL")]
    pq = pit_sea.groupby("pitcher").apply(lambda d: d.xwobacon_sum.sum() / max(d.bbe.sum(), 1))
    tough = set(pq[pq <= pq.quantile(0.25)].index)   # top-quartile contact suppressors
    faced = rb.groupby("batter").pitcher.agg(lambda s: s.mode().iat[0])
    quiet = g[(g.aw <= 0.25) & (g.hr == 0)].index
    cand = []
    for pid in quiet:
        if pid not in h21.index: continue
        row = h21.loc[pid]
        try: slg = float(str(row.slg).replace(".", "0.", 1)) if str(row.slg).startswith(".") else float(row.slg)
        except Exception: continue
        if float(row.pa or 0) >= 45 and slg >= 0.500 and faced.get(pid) in tough:
            cand.append((pid, slg))
    for pid, slg in sorted(cand, key=lambda x: -x[1])[:2]:
        nm = names.get(pid, str(pid))
        add("STILL HOT", slg * 4,
            f"{nm} ({teams.get(pid,'?')}) - quiet night, but blame the pitcher, not the bat",
            f"Slugging {slg:.3f} over the last 21 days and ran into a top-25% contact-suppressing "
            f"starter yesterday. One bad matchup is not a cold streak - hold.")

    # ---------- pitcher process vs results ----------
    pg = rb.groupby("pitcher").agg(bbe=("events", "size"),
                                   xw=("estimated_woba_using_speedangle", "mean"),
                                   aw=("actual_woba", "mean"),
                                   barrels=("is_barrel", "sum"))
    pn = pitcher_names or {}
    pg = pg[pg.bbe >= 10]
    pg["gap"] = pg.aw - pg.xw
    pnames = {}
    for pid, r in pg[pg.gap <= -0.13].sort_values("gap").head(2).iterrows():
        add("PITCHER", abs(r.gap) * 7,
            f"Pitcher {pn.get(pid) or names.get(pid, pid)} ESCAPED - the contact against him was loud",
            f"Allowed {r.xw:.3f} expected wOBA on {int(r.bbe)} batted balls but only {r.aw:.3f} actual, "
            f"{int(r.barrels)} barrels against. Box score says fine; the bats say target his next start.")
    for pid, r in pg[pg.gap >= 0.13].sort_values("gap", ascending=False).head(1).iterrows():
        add("PITCHER", r.gap * 6,
            f"Pitcher {pn.get(pid) or names.get(pid, pid)} deserved better than his line",
            f"Only {r.xw:.3f} expected wOBA allowed on {int(r.bbe)} batted balls vs {r.aw:.3f} actual. "
            f"Don't overreact to the runs column next time out.")

    out = pd.DataFrame(lessons)
    if len(out):
        out = out.sort_values("score", ascending=False).head(10).reset_index(drop=True)
        out.index = out.index + 1
    return out


if __name__ == "__main__":
    R = "/home/claude/mlb-model-data/data"
    rb = pd.read_parquet(f"{R}/recent_bbe.parquet")
    ba = pd.read_parquet(f"{R}/batter_agg.parquet")
    hb = pd.read_parquet(f"{R}/hitting_basic.parquet")
    pa = pd.read_parquet(f"{R}/pitcher_agg.parquet")
    L = build_lessons(rb, ba, hb, pa, "2026-08-06", pitcher_names={656492: "Foster Griffin", 669160: "Dustin May"})
    L.to_csv("/home/claude/mlb_model/output/lessons_2026-08-06.csv")
    print(f"lessons: {len(L)}")
    for i, r in L.iterrows():
        print(f"\n{i}. [{r.category}] {r.headline}\n   {r.data}")
