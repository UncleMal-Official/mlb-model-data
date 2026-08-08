"""Recency-vs-mix filter — the audit Mike ran by hand on Abrams/Swanson/Salvy,
now applied to EVERY board row automatically.

For each hitter: mix-weight their last-21-day damage (xwOBAcon by pitch group,
vs tonight's SP hand) using the SP's actual arsenal shares, compare to their
season mix-weighted damage, shrink by sample (n/(n+35)), and produce:

  recency_ratio   shrunken w21/season damage ratio (1.00 = no signal)
  w21_mix_xw      raw last-21 mix-weighted xwOBAcon
  mix_bbe_21      batted-ball sample behind it
  flags           RECENCY-HOT (>=1.10, n>=25) / RECENCY-COLD (<=0.92, n>=25) /
                  RECENCY-THIN (n21 < 15 - can't verify the hot/cold story)

Projections are adjusted through the ratio: damage-driven FP points scale by it,
HR probability scales exponentially (power is most recency-sensitive). The naive
public projection is deliberately NOT adjusted - recency-vs-specific-mix is
exactly the non-public granularity PGS exists to capture.
"""
import numpy as np
import pandas as pd

SHRINK_N = 35.0
HOT_T, COLD_T = 1.10, 0.92
MIN_N_FLAG, MIN_N_THIN = 25, 15


def _mix_use(pa, sp_id, side):
    b = pa[(pa.pitcher == sp_id) & (pa.stand == side) & (pa.window == "season")
           & (pa.pitch_type.str.startswith("GRP_"))]
    tot = b.pitches.sum()
    if not tot:
        return {}
    return {r.pitch_type[4:]: r.pitches / tot for _, r in b.iterrows()}


def _mix_xw(ba, bat, side, throws, window, use):
    a = ba[(ba.batter == bat) & (ba.stand == side) & (ba.p_throws == throws)
           & (ba.window == window) & (ba.pitch_grp.isin(["FB", "BRK", "OFF"]))]
    xw = cov = nb = 0.0
    for _, r in a.iterrows():
        u = use.get(r.pitch_grp, 0)
        if r.bbe >= 5 and u > 0:
            xw += u * (r.xwobacon_sum / r.bbe)
            cov += u
            nb += int(r.bbe)
    if cov <= 0.4:
        return None, 0
    return xw / cov, int(nb)


def apply_recency(df, batter_agg, pitcher_agg, sp_name_to_id, scoring):
    """Mutates/returns board df with recency columns, adjusted projections, flags."""
    ratios, w21s, n21s = [], [], []
    for _, r in df.iterrows():
        sp_id = sp_name_to_id.get(r.opp_sp)
        side = r.bats if r.bats in ("L", "R") else "R"
        ratio, w21_xw, n21 = 1.0, np.nan, 0
        if sp_id:
            use = _mix_use(pitcher_agg, sp_id, side)
            if use:
                sea, _ = _mix_xw(batter_agg, int(r.player_id), side, str(r.sp_throws), "season", use)
                w21, n21 = _mix_xw(batter_agg, int(r.player_id), side, str(r.sp_throws), "w21", use)
                if sea and w21 and sea > 0:
                    wgt = n21 / (n21 + SHRINK_N)
                    ratio = (wgt * w21 + (1 - wgt) * sea) / sea
                    w21_xw = w21
        ratios.append(round(ratio, 3)); w21s.append(w21_xw); n21s.append(n21)
    df["recency_ratio"] = ratios
    df["w21_mix_xw"] = [round(x, 3) if pd.notna(x) else "" for x in w21s]
    df["mix_bbe_21"] = n21s

    # ---- adjust projections through the ratio
    rr = df.recency_ratio.astype(float)
    for c in ("proj_tb", "proj_hits", "proj_r", "proj_rbi"):
        df[c] = (df[c].astype(float) * rr).round(2)
    p = df.hr_prob.astype(float) / 100.0
    df["hr_prob"] = (100 * (1 - (1 - p) ** (rr ** 1.5))).round(1)
    floor_pts = (scoring["walk"] * df.proj_bb.astype(float)
                 + scoring["stolen_base"] * df.proj_sb.astype(float))
    dmg_pts = df.proj_fp.astype(float) - floor_pts
    df["proj_fp"] = (floor_pts + dmg_pts * rr).round(2)
    df["pgs"] = (df.proj_fp.astype(float) - df.naive_fp.astype(float)).round(2)
    df["pgs"] = (df["pgs"] - df["pgs"].mean()).round(2)

    # ---- flags
    def add_flag(fl, new):
        return f"{fl},{new}" if fl else new
    out_flags = []
    for _, r in df.iterrows():
        fl = r["flags"] or ""
        if r.mix_bbe_21 >= MIN_N_FLAG and r.recency_ratio >= HOT_T:
            fl = add_flag(fl, "RECENCY-HOT")
        elif r.mix_bbe_21 >= MIN_N_FLAG and r.recency_ratio <= COLD_T:
            fl = add_flag(fl, "RECENCY-COLD")
        elif r.mix_bbe_21 < MIN_N_THIN:
            fl = add_flag(fl, "RECENCY-THIN")
        out_flags.append(fl)
    df["flags"] = out_flags
    n_hot = sum("RECENCY-HOT" in f for f in out_flags)
    n_cold = sum("RECENCY-COLD" in f for f in out_flags)
    print(f"recency filter: {n_hot} HOT, {n_cold} COLD, "
          f"median ratio {df.recency_ratio.median():.3f}")
    return df
