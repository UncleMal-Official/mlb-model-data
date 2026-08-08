"""Day After demo: Aug 6 unlucky hard contact -> tonight's slate."""
import json
import pandas as pd
from day_after import yesterday_unlucky, day_after_candidates

R = "/home/claude/mlb-model-data/data"
D = "/home/claude/mlb_model"

rb = pd.read_parquet(f"{R}/recent_bbe.parquet")
hb = pd.read_parquet(f"{R}/hitting_basic.parquet")
names = hb[hb.window == "season"].set_index("player_id")["name"].to_dict()
rb["player_name"] = rb.batter.map(names).fillna(rb.batter.astype(str))  # statcast player_name is the pitcher; replace with batter name

board = pd.read_csv(f"{D}/output/full_board_v1_2026-08-07.csv")
board["player_id"] = board["player_id"].astype("int64")
board = board.drop_duplicates("player_id")

# map (game, team) -> opposing SP id, from repo schedule
sch = json.load(open(f"{R}/schedule.json"))
TEAM_ABBR = {"New York Mets":"NYM","Pittsburgh Pirates":"PIT","Toronto Blue Jays":"TOR","Philadelphia Phillies":"PHI",
"Cincinnati Reds":"CIN","Washington Nationals":"WSH","Atlanta Braves":"ATL","New York Yankees":"NYY",
"Athletics":"ATH","Boston Red Sox":"BOS","Los Angeles Angels":"LAA","Miami Marlins":"MIA",
"Cleveland Guardians":"CLE","Chicago White Sox":"CHW","Minnesota Twins":"MIN","Milwaukee Brewers":"MIL",
"Chicago Cubs":"CHC","Kansas City Royals":"KC","Colorado Rockies":"COL","St. Louis Cardinals":"STL",
"Baltimore Orioles":"BAL","Texas Rangers":"TEX","Los Angeles Dodgers":"LAD","Arizona Diamondbacks":"ARI",
"Houston Astros":"HOU","San Diego Padres":"SD","Tampa Bay Rays":"TB","Seattle Mariners":"SEA",
"Detroit Tigers":"DET","San Francisco Giants":"SF"}
opp_sp = {}
for g in sch["games"]:
    a, h = TEAM_ABBR[g["away"]["team"]], TEAM_ABBR[g["home"]["team"]]
    opp_sp[a] = g["home"]["probable_id"]
    opp_sp[h] = g["away"]["probable_id"]

pa_ = pd.read_parquet(f"{R}/pitcher_agg.parquet")
def pitcher_mix_fn(row):
    pid = opp_sp.get(row.team)
    if not pid: return None
    bs = row.bats
    eff = ("L" if row.sp_throws == "R" else "R") if bs == "S" else bs
    g = pa_[(pa_.pitcher == pid) & (pa_.stand == eff) & (pa_.window == "season") & (pa_.pitch_type.str.startswith("GRP_"))]
    tot = g.pitches.sum()
    if tot < 30: return None
    return {r.pitch_type[4:]: r.pitches / tot for _, r in g.iterrows() if r.pitch_type[4:] != "OTH"}

unlucky = yesterday_unlucky(rb, "2026-08-06")
print(f"unlucky yesterday (Aug 6): {len(unlucky)}")
print(unlucky.head(15)[["player_name","bbe","barrels","near","deep_outs","max_ev","max_dist","process"]].to_string())

cands = day_after_candidates(unlucky, board, pitcher_mix_fn)
print(f"\nDAY AFTER candidates on tonight's slate: {len(cands)}")
if len(cands):
    print(cands.head(12).to_string())
    cands.to_csv(f"{D}/output/day_after_2026-08-07.csv", index=False)
