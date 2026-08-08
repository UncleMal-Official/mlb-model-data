"""Auto-POTD: pick the single best CORE value on the board, build the receipts,
render the branded card. Runs on the FIRST board build of the day; Mike reviews
the PNG (and can override player/market/line) before posting.

Selection = CORE discipline, scored:
  z(PGS) 0.45 + z(proj FP vs slate) 0.35 + confidence bonus + corroboration
  (BPP tags, Day After tag) - penalties (rain, projected-only lineup, small SP sample)

Line handling: until the PrizePicks feed lands, the line printed on the card is
MODEL-ESTIMATED (sim median - 1.0, rounded to PP-style 0.5). The run summary
flags it loudly; --line overrides.
"""
import argparse, json, pathlib, re
import numpy as np
import pandas as pd

import render_card

WORK = pathlib.Path("/home/claude/mlb_model")
REPO = pathlib.Path("/home/claude/mlb-model-data/data")
OUT = WORK / "output"

NICK2 = {"Red Sox", "White Sox", "Blue Jays"}
PARKS = {"NYY": "Yankee Stadium", "BOS": "Fenway", "TB": "The Trop", "TOR": "Rogers Centre",
         "BAL": "Camden Yards", "CLE": "Progressive", "DET": "Comerica", "KC": "Kauffman",
         "MIN": "Target Field", "CWS": "Rate Field", "HOU": "Daikin Park", "LAA": "Angel Stadium",
         "ATH": "Sutter Health", "SEA": "T-Mobile", "TEX": "Globe Life", "ATL": "Truist",
         "MIA": "LoanDepot", "NYM": "Citi Field", "PHI": "Citizens Bank", "WSH": "Nationals Park",
         "CHC": "Wrigley", "CIN": "GABP", "MIL": "AmFam Field", "PIT": "PNC Park",
         "STL": "Busch", "ARI": "Chase Field", "COL": "Coors", "LAD": "Dodger Stadium",
         "SD": "Petco", "SF": "Oracle Park",
         "WAS": "Nationals Park", "CWS": "Rate Field", "OAK": "Sutter Health"}
# PrizePicks batter scoring
PTS = dict(single=3, double=5, triple=8, hr=10, r=2, rbi=2, bb=2, sb=5)


def nickname(full):
    parts = str(full).split()
    if len(parts) >= 2 and " ".join(parts[-2:]) in NICK2:
        return " ".join(parts[-2:])
    return parts[-1] if parts else full


def zscore(s):
    s = s.astype(float)
    return (s - s.mean()) / max(s.std(), 1e-9)


# ---------------------------------------------------------------- selection
def select_potd(board: pd.DataFrame, force_player: str | None = None):
    b = board.drop_duplicates("player_id").copy()
    b["flags"] = b["flags"].fillna("")
    b["bpp_tags"] = b["bpp_tags"].fillna("")
    elig = b[(b.conf.isin(["HIGH", "MED"])) & (b.exp_pa >= 4.0) & (b.proj_fp >= 7.0)].copy()
    if force_player:
        pick = b[b.player.str.lower() == force_player.lower()]
        if pick.empty:
            raise SystemExit(f"--player '{force_player}' not on board")
        return pick.iloc[0], elig
    if elig.empty:
        elig = b.nlargest(10, "proj_fp").copy()
    score = zscore(elig.pgs) * 0.45 + zscore(elig.proj_fp) * 0.35
    score += np.where(elig.conf == "HIGH", 0.35, 0.0)
    score += np.where(elig.bpp_tags.str.len() > 0, 0.25, 0.0)
    score += np.where(elig["flags"].str.contains("DAY-AFTER"), 0.15, 0.0)
    score += np.where(elig["flags"].str.contains("RECENCY-HOT"), 0.20, 0.0)
    score -= np.where(elig["flags"].str.contains("RECENCY-COLD"), 0.25, 0.0)
    score -= np.where(elig["flags"].str.contains("RAIN"), 0.30, 0.0)
    score -= np.where(elig["flags"].str.contains("LINEUP-PROJ"), 0.20, 0.0)
    score -= np.where(elig["flags"].str.contains("SP-SMALL-SAMPLE"), 0.15, 0.0)
    elig["potd_score"] = score
    elig = elig.sort_values("potd_score", ascending=False)
    return elig.iloc[0], elig


# ---------------------------------------------------------------- simulator
def simulate(row, n=20000, seed=7):
    """Monte Carlo fantasy-score dist from the board row's component projections."""
    rng = np.random.default_rng(seed)
    epa = float(row.exp_pa)
    hr_mean = -np.log(max(1e-9, 1 - float(row.hr_prob) / 100.0))
    hits = float(row.proj_hits); tb = float(row.proj_tb)
    dbl = max(0.0, tb - hits - 3 * hr_mean)          # TB - H = 2B + 3*HR (triples folded in)
    dbl = min(dbl, hits - hr_mean)
    sng = max(0.0, hits - dbl - hr_mean)
    bb = float(row.proj_bb); sb = float(row.proj_sb)
    r_x = max(0.0, float(row.proj_r) - hr_mean); rbi_x = max(0.0, float(row.proj_rbi) - hr_mean)
    lo = int(np.floor(epa)); frac = epa - lo
    pa_n = lo + (rng.random(n) < frac)
    p = np.array([sng, dbl, hr_mean, bb]) / epa      # per-PA event probs
    p = np.clip(p, 0, None)
    out_p = max(0.0, 1 - p.sum())
    probs = np.append(p, out_p)
    ev = np.zeros(n)
    for k in np.unique(pa_n):
        m = pa_n == k
        draws = rng.multinomial(int(k), probs, size=m.sum())
        ev[m] = (draws[:, 0] * PTS["single"] + draws[:, 1] * PTS["double"]
                 + draws[:, 2] * PTS["hr"] + draws[:, 3] * PTS["bb"])
        hr_draw = draws[:, 2]
        ev[m] += hr_draw * (PTS["r"] + PTS["rbi"])   # HR scores a run + at least 1 RBI
    ev += rng.poisson(r_x, n) * PTS["r"] + rng.poisson(rbi_x, n) * PTS["rbi"]
    ev += rng.poisson(sb, n) * PTS["sb"]
    return ev


# ---------------------------------------------------------------- receipts
def load_ctx():
    ctx = {}
    ctx["hb"] = pd.read_parquet(REPO / "hitting_basic.parquet")
    ps = json.load(open(WORK / "data/pitcher_splits_2026.json"))["pitchers"]
    ctx["sp_by_name"] = {v["name"]: v for v in ps.values()}
    # League split baselines (PA-weighted over all starters on file), so quoted numbers
    # carry "vs league" context - a .386 SLG allowed to LHB is BELOW the RHP-league norm.
    lg = {}
    for tp in ("R", "L"):
        for tb_side in ("L", "R"):
            tb = ab = hr = bb = pa = 0
            for v in ps.values():
                if v.get("throws") != tp:
                    continue
                s = v.get("vsL" if tb_side == "L" else "vsR") or {}
                if not s.get("pa"):
                    continue
                sg = s["h"] - s["d2"] - s["d3"] - s["hr"]
                tb += sg + 2 * s["d2"] + 3 * s["d3"] + 4 * s["hr"]
                ab += s["ab"]; hr += s["hr"]; bb += s["bb"]; pa += s["pa"]
            if ab:
                lg[(tp, tb_side)] = dict(slg=tb / ab, hr_pa=hr / pa, bb_pa=bb / pa)
    ctx["lg_split"] = lg
    for f in sorted(WORK.glob("config/park_factors_*.json")):
        ctx["parks"] = json.load(open(f)).get("games", {})
    for f in sorted(WORK.glob("config/bpp_features_*.json")):
        ctx["bpp"] = json.load(open(f))
    return ctx


def compose(row, ctx, date_str, line_override=None, cal_k=None):
    if cal_k:
        row = row.copy()
        for c in ("proj_tb", "proj_hits", "proj_bb", "proj_r", "proj_rbi", "proj_sb", "proj_fp"):
            row[c] = float(row[c]) * cal_k
        row["hr_prob"] = 100.0 * (1.0 - (1.0 - float(row.hr_prob) / 100.0) ** cal_k)
    hb = ctx["hb"]
    sea = hb[(hb.window == "season") & (hb.player_id == int(row.player_id))]
    l21 = hb[(hb.window == "last21") & (hb.player_id == int(row.player_id))]
    sea = sea.iloc[0] if len(sea) else None
    l21 = l21.iloc[0] if len(l21) else None

    sim = simulate(row)
    med = float(np.median(sim))
    if line_override is not None:
        line = float(line_override); line_src = "manual"
    else:
        line = max(5.5, round((med - 1.0) * 2) / 2); line_src = "MODEL-ESTIMATED - confirm vs PrizePicks"
    win = float((sim > line).mean() * 100)

    # matchup line
    team_nick = nickname(sea.team) if sea is not None else row.team
    home = row.game.split("@")[1]
    park = PARKS.get(home, home)
    pk = ""
    parks = ctx.get("parks", {})
    pf = parks.get(row.game) or parks.get(row.game.replace("WSH", "WAS")) or {}
    hrm = pf.get("hr")
    if hrm and abs(hrm - 1) >= 0.06:
        pk = f" &middot; {park} {'+' if hrm > 1 else '&minus;'}{abs(hrm - 1) * 100:.0f}% HR"
    elif pf.get("runs") and abs(pf["runs"] - 1) >= 0.06:
        pk = f" &middot; {park} {'+' if pf['runs'] > 1 else '&minus;'}{abs(pf['runs'] - 1) * 100:.0f}% Runs"
    pos = "" if pd.isna(row.pos) or str(row.pos).strip() in ("", "nan") else f" &middot; {row.pos}"
    matchup = f"{team_nick}{pos} &middot; vs {row.opp_sp} ({row.sp_throws}){pk}"

    season_line = "&middot;"
    if sea is not None:
        season_line = (f"Season: <b>{sea.avg} AVG</b> &middot; <b>{int(sea.hr)} HR</b> "
                       f"&middot; <b>{int(sea.sb)} SB</b> &middot; <b>{sea.ops} OPS</b>")

    # tiles
    tiles = [dict(val=f"{row.proj_fp:.1f}", lab="Model Projection"),
             dict(val=f"{win:.0f}%", lab="Sim Win &middot; 20K Runs"),
             dict(val=f"{med:.0f}", lab="Median Sim Score")]
    pool = []
    if "RECENCY-HOT" in str(row.get("flags", "")) and float(row.get("recency_ratio", 1)) >= 1.10:
        pool.append(dict(val=f"+{(float(row.recency_ratio) - 1) * 100:.0f}%",
                         lab="L21 Damage vs This Mix"))
    spl = ctx["sp_by_name"].get(row.opp_sp)
    side = "L" if row.bats == "L" else "R"
    vs = (spl or {}).get("vsL" if side == "L" else "vsR")
    bvp = None
    for h in ctx.get("bpp", {}).get("bvp_history", []):
        if h["batter"] == row.player and h["pitcher"] == row.opp_sp:
            bvp = h
    if bvp and (bvp["hits"] >= 3 or bvp["hr"] >= 1):
        tiles_val = f"{bvp['hits']}-{bvp['ab']}, {bvp['hr']} HR"
        pool.append(dict(val=tiles_val, lab=f"Career vs {row.opp_sp.split()[-1]}"))
    if pd.notna(row.mix_barrel) and row.mix_bbe >= 25:
        pool.append(dict(val=f"{row.mix_barrel:.0f}%", lab="Barrels vs Tonight's Mix"))
    if vs and vs.get("pa", 0) >= 60:
        # pick the receipt that is a real leak RELATIVE TO LEAGUE for this platoon
        lgs = ctx.get("lg_split", {}).get((str(row.sp_throws), side)) or {}
        sp_last = row.opp_sp.split()[-1]
        slg_d = vs["slg"] - lgs.get("slg", vs["slg"])
        hr_d = vs["hr"] / vs["pa"] - lgs.get("hr_pa", vs["hr"] / vs["pa"])
        bb_d = vs["bb"] / vs["pa"] - lgs.get("bb_pa", vs["bb"] / vs["pa"])
        if slg_d >= 0.015:
            pool.append(dict(val=f".{int(round(vs['slg'] * 1000)):03d} SLG",
                             lab=f"{sp_last} vs {side}HB &middot; lg .{int(round(lgs['slg'] * 1000)):03d}"))
        elif hr_d >= 0.004:
            pool.append(dict(val=f"{vs['hr']} HR",
                             lab=f"{sp_last} allowed vs {side}HB"))
        elif bb_d >= 0.010:
            pool.append(dict(val=f"{100 * vs['bb'] / vs['pa']:.0f}% BB",
                             lab=f"{sp_last} vs {side}HB &middot; walk floor"))
        else:
            pool.append(dict(val=f".{int(round(vs['slg'] * 1000)):03d} SLG",
                             lab=f"{sp_last} vs {side}HB &middot; {vs['hr']} HR"))
    if hrm and hrm >= 1.10:
        pool.append(dict(val=f"+{(hrm - 1) * 100:.0f}% HR", lab=f"{park} Tonight"))
    if l21 is not None and float(l21.pa) >= 40:
        pool.append(dict(val=f"{l21.slg} SLG", lab="Last 21 Days"))
    if pd.notna(row.hr_prob) and row.hr_prob >= 18:
        pool.append(dict(val=f"{row.hr_prob:.0f}%", lab="Model HR Probability"))
    tiles += pool[:3]
    while len(tiles) < 6:
        tiles.append(dict(val=f"{row.exp_pa:.1f}", lab="Projected Plate Apps"))

    # support line
    floor_pts = (row.proj_bb * PTS["bb"] + row.proj_r * PTS["r"]
                 + row.proj_rbi * PTS["rbi"] + row.proj_sb * PTS["sb"])
    floor_share = 100 * floor_pts / max(row.proj_fp, 1e-9)
    if vs:
        lgs = ctx.get("lg_split", {}).get((str(row.sp_throws), side)) or {}
        lg_note = f" (lg .{int(round(lgs['slg'] * 1000)):03d})" if lgs else ""
        support = (f"{row.opp_sp} vs {side}HB: <b>.{int(round(vs['slg'] * 1000)):03d} SLG{lg_note} &middot; "
                   f"{vs['hr']} HR &middot; {100 * vs['bb'] / max(vs['pa'], 1):.0f}% BB</b> &nbsp;&mdash;&nbsp; "
                   f"<b>{floor_share:.0f}%</b> of the projected score is BB / R / RBI / SB floor")
    else:
        support = (f"<b>{row.exp_pa:.1f}</b> projected plate appearances &nbsp;&mdash;&nbsp; "
                   f"<b>{floor_share:.0f}%</b> of the projected score is BB / R / RBI / SB floor")

    dt = pd.Timestamp(date_str)
    spec = dict(
        kicker_date=dt.strftime("%a %b ") + str(dt.day),
        name=row.player, matchup=matchup, season_line=season_line,
        market="Fantasy Score", hero_line=f"OVER {line:.1f}",
        tiles=tiles, support=support,
        initials="".join(w[0] for w in row.player.split()[:2]).upper(),
    )
    meta = dict(player=row.player, player_id=int(row.player_id), team=row.team, game=row.game,
                market="Fantasy Score", line=line, line_source=line_src,
                proj_fp=float(row.proj_fp), sim_win_pct=round(win, 1), sim_median=med,
                pgs=float(row.pgs), conf=row.conf, flags=row["flags"],
                alt_markets=dict(total_bases=float(row.proj_tb), hits=float(row.proj_hits),
                                 hr_prob=float(row.hr_prob), hit_prob=float(row.hit_prob)))
    return spec, meta


# ---------------------------------------------------------------- photo
def resolve_photo(spec, player_id, photo=None, photo_b64=None):
    if photo_b64:  # base64 text file (e.g. tonight's Witt upload)
        spec["photo_mode"] = "action"
        spec["photo_src"] = "data:image/jpeg;base64," + pathlib.Path(photo_b64).read_text().strip()
        return "action (manual)"
    if photo:
        spec["photo_mode"] = "action"
        spec["photo_src"] = render_card.img_to_data_uri(photo)
        return "action (manual)"
    hero = REPO / "headshots" / f"{player_id}_hero.jpg"
    silo = REPO / "headshots" / f"{player_id}_silo.png"
    if hero.exists():
        spec["photo_mode"] = "action"
        spec["photo_src"] = render_card.img_to_data_uri(str(hero))
        return "action (auto MLB hero)"
    if silo.exists():
        spec["photo_mode"] = "silo"
        spec["photo_src"] = render_card.img_to_data_uri(str(silo))
        return "silo (auto)"
    spec["photo_mode"] = "placeholder"
    return "placeholder (no photo on file yet)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--player", help="force a specific player (manual override)")
    ap.add_argument("--line", type=float, help="actual PrizePicks line (overrides estimate)")
    ap.add_argument("--photo", help="path to a manual action photo")
    ap.add_argument("--photo-b64", help="path to a base64 txt of a manual action photo")
    ap.add_argument("--cal-k", type=float, help="market calibration scale from join_pp_lines (e.g. 0.842)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()

    board = pd.read_csv(a.board)
    ctx = load_ctx()
    row, ranked = select_potd(board, a.player)
    spec, meta = compose(row, ctx, a.date, a.line, a.cal_k)
    meta["photo_mode"] = resolve_photo(spec, meta["player_id"], a.photo, a.photo_b64)

    png = OUT / f"potd_auto_{a.date}{a.suffix}.png"
    info = render_card.render(spec, str(png), out_html=str(OUT / f"potd_auto_{a.date}{a.suffix}.html"))
    meta["render"] = info
    top5 = ranked.head(5)[["player", "team", "proj_fp", "pgs", "conf"]].to_dict("records") \
        if "potd_score" in ranked.columns else []
    meta["runner_ups"] = top5
    json_path = OUT / f"potd_auto_{a.date}{a.suffix}.json"
    json_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"\ncard: {png}")


if __name__ == "__main__":
    main()
