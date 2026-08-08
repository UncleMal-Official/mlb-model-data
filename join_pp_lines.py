"""Ingest a PrizePicks projections JSON (saved from Mike's browser) and join
Line / Edge columns onto the day's board.

Usage: python3 join_pp_lines.py <pp_json_file> [<pp_json_file2> ...] --date 2026-08-08

Mike's 20-second daily step: open these in any browser tab (already logged-in Chrome sails
through DataDome), Ctrl+S each as .json, drop into chat or the repo:
  https://api.prizepicks.com/projections?league_id=2&per_page=500&page=1
  https://api.prizepicks.com/projections?league_id=2&per_page=500&page=2   (only if page 1 says next)

Outputs:
  data/pp_lines_<date>.csv            all MLB projections (player, team, stat, line, odds_type)
  output/board_pp_<date>.csv          board + Line + model win% + Edge for the FS market
  printed: top-10 CORE edges + demon/goblin notes, POTD real-line rerender command
"""
import argparse, json, pathlib, sys, unicodedata
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/mlb_model")
from make_potd_card import simulate

D = pathlib.Path("/home/claude/mlb_model")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "")
    for suf in (" jr", " sr", " ii", " iii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.split())


def parse_pp(files):
    rows = []
    for f in files:
        j = json.load(open(f))
        players = {}
        for inc in j.get("included", []):
            if inc.get("type") == "new_player":
                a = inc.get("attributes", {})
                players[inc["id"]] = dict(name=a.get("display_name") or a.get("name"),
                                          team=a.get("team"), pos=a.get("position"))
        for d in j.get("data", []):
            a = d.get("attributes", {})
            rel = ((d.get("relationships", {}).get("new_player", {}) or {}).get("data", {}) or {})
            p = players.get(rel.get("id"), {})
            rows.append(dict(pp_player=p.get("name"), pp_team=p.get("team"), pp_pos=p.get("pos"),
                             stat=a.get("stat_type"), line=a.get("line_score"),
                             odds_type=a.get("odds_type"), status=a.get("status"),
                             start_time=a.get("start_time")))
    df = pd.DataFrame(rows).dropna(subset=["pp_player", "stat", "line"])
    df["line"] = pd.to_numeric(df.line, errors="coerce")
    return df.drop_duplicates(["pp_player", "stat", "line", "odds_type"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--date", required=True)
    a = ap.parse_args()

    pp = parse_pp(a.files)
    pp_path = D / f"data/pp_lines_{a.date}.csv"
    pp.to_csv(pp_path, index=False)
    print(f"PP projections parsed: {len(pp)} rows -> {pp_path}")
    print(pp.stat.value_counts().head(12).to_string())

    board = pd.read_csv(D / f"output/full_board_v1_{a.date}.csv").drop_duplicates("player_id")
    board["nkey"] = board.player.map(norm)
    fs = pp[pp.stat.str.contains("Fantasy", case=False, na=False)].copy()
    fs["nkey"] = fs.pp_player.map(norm)
    std = fs[fs.odds_type.fillna("standard").str.lower().isin(["standard", ""])]
    m = board.merge(std[["nkey", "line", "odds_type"]], on="nkey", how="left")

    # ---- market calibration: find global scale k so mean P(over) across all
    # standard FS lines ~ 0.50 (PP sets lines to balance action; our absolute
    # FP runs hot - measured +0.7 on Friday). Rankings stay ours; level is market-anchored.
    joined_rows = m[m.line.notna()]

    def scaled(r, k):
        r = r.copy()
        for c in ("proj_tb", "proj_hits", "proj_bb", "proj_r", "proj_rbi", "proj_sb"):
            r[c] = float(r[c]) * k
        p = float(r.hr_prob) / 100.0
        r["hr_prob"] = 100.0 * (1.0 - (1.0 - p) ** k)
        return r

    def mean_over(k, n=4000):
        ws = [float((simulate(scaled(r, k), n=n) > r.line).mean())
              for _, r in joined_rows.iterrows()]
        return float(np.mean(ws))

    lo, hi = 0.70, 1.10
    for _ in range(10):
        mid = (lo + hi) / 2
        if mean_over(mid) > 0.50:
            hi = mid
        else:
            lo = mid
    k = round((lo + hi) / 2, 3)
    print(f"\nmarket calibration: k = {k} (mean P(over) across {len(joined_rows)} lines -> ~50%)")

    wins, wins_raw, edges = {}, {}, {}
    for _, r in joined_rows.iterrows():
        w = float((simulate(scaled(r, k), n=12000) > r.line).mean())
        wr = float((simulate(r, n=12000) > r.line).mean())
        wins[r.player_id] = round(w * 100, 1)
        wins_raw[r.player_id] = round(wr * 100, 1)
        edges[r.player_id] = round((w - 0.542) * 100, 1)   # vs flex breakeven
    m["fs_line"] = m.line
    m["win_pct"] = m.player_id.map(wins)          # calibrated
    m["win_pct_raw"] = m.player_id.map(wins_raw)  # uncalibrated (hot) - for tracking
    m["edge_vs_flex"] = m.player_id.map(edges)
    out = D / f"output/board_pp_{a.date}.csv"
    m.drop(columns=["nkey", "line", "odds_type"]).to_csv(out, index=False)

    joined = m[m.fs_line.notna()]
    print(f"\nFS lines joined: {len(joined)} of {len(board)} board hitters "
          f"(unmatched PP names: {len(std) - len(joined)})")
    top = joined[joined.conf.isin(['HIGH', 'MED'])].nlargest(10, "edge_vs_flex")
    print("\nTOP CORE EDGES (win% vs 54.2% flex breakeven):")
    print(top[["player", "team", "opp_sp", "fs_line", "proj_fp", "win_pct", "edge_vs_flex", "conf"]].to_string(index=False))
    demons = pp[pp.odds_type.fillna("").str.lower() == "demon"]
    goblins = pp[pp.odds_type.fillna("").str.lower() == "goblin"]
    print(f"\ndemons on board: {len(demons)}, goblins: {len(goblins)} (graded in Demons tab on next build)")
    pot = top.iloc[0] if len(top) else None
    if pot is not None:
        print(f"\nPOTD re-render with the real line:\n  python3 make_potd_card.py "
              f"--board output/full_board_v1_{a.date}.csv --date {a.date} "
              f"--player \"{pot.player}\" --line {pot.fs_line}")


if __name__ == "__main__":
    main()
