"""Board version diff — the What Changed (good/bad/ugly) engine.

Compares two board CSVs and categorizes every meaningful delta so users can see
exactly what moved when lineups posted, slots changed, or data refreshed.
Now part of the permanent pipeline: build_board saves a snapshot each run.
"""
import pandas as pd
import numpy as np

FP_NOTABLE = 0.4
FP_BIG = 1.0


def diff(old_csv: str, new_csv: str) -> pd.DataFrame:
    o = pd.read_csv(old_csv).drop_duplicates("player_id").set_index("player_id")
    n = pd.read_csv(new_csv).drop_duplicates("player_id").set_index("player_id")
    o["flags"] = o["flags"].fillna(""); n["flags"] = n["flags"].fillna("")
    rows = []

    def add(cat, sev, player, team, game, change, was, now, why):
        rows.append(dict(category=cat, severity=sev, player=player, team=team, game=game,
                         change=change, was=was, now=now, why=why))

    # ---- players who left the board
    for pid in o.index.difference(n.index):
        r = o.loc[pid]
        if "LINEUP-PROJ" in r["flags"]:
            add("OUT", "BAD", r.player, r.team, r.game, "Projected starter NOT in posted lineup",
                f"proj {r.proj_fp} FP (slot {r.slot})", "off board",
                "Lineup posted without him - projection miss, not a scratch")
        else:
            add("OUT", "UGLY", r.player, r.team, r.game, "SCRATCHED from posted lineup",
                f"{r.proj_fp} FP (slot {r.slot})", "off board",
                "Was in a posted lineup earlier - verify before playing any leftover entries")

    # ---- players who joined the board
    for pid in n.index.difference(o.index):
        r = n.loc[pid]
        if r.proj_fp >= 7.0:
            add("IN", "GOOD", r.player, r.team, r.game, "NEW to board (lineup posted)",
                "not listed", f"{r.proj_fp} FP, slot {r.slot}, {r.hr_prob}% HR",
                "Entered a posted lineup")

    # ---- movers among common players
    common = o.index.intersection(n.index)
    for pid in common:
        a, b = o.loc[pid], n.loc[pid]
        why_bits = []
        if "LINEUP-PROJ" in a["flags"] and "LINEUP-PROJ" not in b["flags"]:
            why_bits.append("lineup confirmed")
        if int(a.slot) != int(b.slot):
            why_bits.append(f"slot {int(a.slot)} to {int(b.slot)}")
        if str(a.opp_sp) != str(b.opp_sp):
            why_bits.append(f"SP change: {a.opp_sp} to {b.opp_sp}")
        if str(a.conf) != str(b.conf):
            why_bits.append(f"confidence {a.conf} to {b.conf}")
        d = round(float(b.proj_fp) - float(a.proj_fp), 2)
        why = ", ".join(why_bits) if why_bits else "data refresh"
        if abs(d) >= FP_NOTABLE:
            sev = "GOOD" if d > 0 else ("UGLY" if d <= -FP_BIG else "BAD")
            add("MOVE", sev, b.player, b.team, b.game,
                f"Projection {'UP' if d > 0 else 'DOWN'} {abs(d):.1f} FP",
                f"{a.proj_fp} FP", f"{b.proj_fp} FP", why)
        elif why_bits and (int(a.slot) != int(b.slot) or str(a.opp_sp) != str(b.opp_sp)):
            add("INFO", "GOOD" if int(b.slot) < int(a.slot) else "BAD",
                b.player, b.team, b.game,
                "Slot/matchup update", f"slot {int(a.slot)}", f"slot {int(b.slot)}", why)

    out = pd.DataFrame(rows)
    if len(out):
        sev_rank = {"UGLY": 0, "BAD": 1, "GOOD": 2}
        out["_r"] = out.severity.map(sev_rank)
        out = out.sort_values(["_r", "category"]).drop(columns="_r")
    return out


if __name__ == "__main__":
    d = diff("/tmp/board_before.csv", "/home/claude/mlb_model/output/full_board_v1_2026-08-07.csv")
    d.to_csv("/home/claude/mlb_model/output/board_changes_v1.2_to_v1.3.csv", index=False)
    print(f"changes: {len(d)}")
    print(d.groupby(["severity", "category"]).size().to_string())
    print()
    print(d.head(30).to_string())
