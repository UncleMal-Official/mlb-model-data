"""MLB DFS Model v1 — barrel/pitch-mix layer live (repo data). 2026-08-07."""
import json, math
import pandas as pd
import numpy as np

D = "/home/claude/mlb_model"
R = "/home/claude/mlb-model-data/data"
park = json.load(open(f"{D}/config/park_factors_2026-08-07.json"))["games"]
scoring = json.load(open(f"{D}/config/scoring.json"))["prizepicks_batter_fantasy_score"]
bpp = json.load(open(f"{D}/config/bpp_features_2026-08-07.json"))
sides_csv = pd.read_csv(f"{D}/data/bat_sides.csv", dtype={"player_id": str}).set_index("player_id")["bat_side"].to_dict()
b21_csv = pd.read_csv(f"{D}/data/batters_21d.csv", dtype={"player_id": str}).drop_duplicates("player_id").set_index("player_id")

ba = pd.read_parquet(f"{R}/batter_agg.parquet")
pa_ = pd.read_parquet(f"{R}/pitcher_agg.parquet")
hb = pd.read_parquet(f"{R}/hitting_basic.parquet")
sch = json.load(open(f"{R}/schedule.json"))

TEAM_ABBR = {"New York Mets":"NYM","Pittsburgh Pirates":"PIT","Toronto Blue Jays":"TOR","Philadelphia Phillies":"PHI",
"Cincinnati Reds":"CIN","Washington Nationals":"WSH","Atlanta Braves":"ATL","New York Yankees":"NYY",
"Athletics":"ATH","Boston Red Sox":"BOS","Los Angeles Angels":"LAA","Miami Marlins":"MIA",
"Cleveland Guardians":"CLE","Chicago White Sox":"CHW","Minnesota Twins":"MIN","Milwaukee Brewers":"MIL",
"Chicago Cubs":"CHC","Kansas City Royals":"KC","Colorado Rockies":"COL","St. Louis Cardinals":"STL",
"Baltimore Orioles":"BAL","Texas Rangers":"TEX","Los Angeles Dodgers":"LAD","Arizona Diamondbacks":"ARI",
"Houston Astros":"HOU","San Diego Padres":"SD","Tampa Bay Rays":"TB","Seattle Mariners":"SEA",
"Detroit Tigers":"DET","San Francisco Giants":"SF"}
GKEY = {("NYM","PIT"):"NYM@PIT",("TOR","PHI"):"TOR@PHI",("CIN","WSH"):"CIN@WAS",("ATL","NYY"):"ATL@NYY",
("ATH","BOS"):"ATH@BOS",("LAA","MIA"):"LAA@MIA",("CLE","CHW"):"CLE@CHW",("MIN","MIL"):"MIN@MIL",
("CHC","KC"):"CHC@KC",("COL","STL"):"COL@STL",("BAL","TEX"):"BAL@TEX",("LAD","ARI"):"LAD@ARI",
("HOU","SD"):"HOU@SD",("TB","SEA"):"TB@SEA",("DET","SF"):"DET@SF"}

# ---------------- league baselines from statcast aggregates (season, ALL sides/groups)
lg = ba[(ba.window=="season") & (ba.pitch_grp=="ALL")]
LG_PA = lg.pa.sum()
L = dict(b1=lg.b1.sum()/LG_PA, b2=lg.b2.sum()/LG_PA, b3=lg.b3.sum()/LG_PA, hr=lg.hr.sum()/LG_PA,
         bb=(lg.bb.sum()+lg.hbp.sum())/LG_PA, k=lg.k.sum()/LG_PA)
LG_XW = lg.xwobacon_sum.sum()/lg.bbe.sum()
LG_BRL = lg.barrels.sum()/lg.bbe.sum()
LG_WHIFF = lg.whiffs.sum()/lg.swings.sum()
LG_EV = lg.ev_sum.sum()/lg.bbe.sum()
LG_LA = lg.la_sum.sum()/lg.bbe.sum()
HR_LA_CENTER, HR_LA_WIDTH = 24.0, 15.0
def la_fit(la): return math.exp(-(((la - HR_LA_CENTER)/HR_LA_WIDTH)**2))
LG_LAFIT = la_fit(LG_LA)
HBP0 = lg.hbp.sum()/LG_PA
sb_league = 0.0187
print(f"league: {({k: round(v,4) for k,v in L.items()})} xwCON {LG_XW:.3f} brl {LG_BRL:.3f} whiff {LG_WHIFF:.3f}")

def rates_from(g, kcap):
    n = g.pa.sum()
    if n == 0: return None
    w = n/(n+kcap)
    def sh(x, lgv): return w*(x/n) + (1-w)*lgv
    return dict(b1=sh(g.b1.sum(),L["b1"]), b2=sh(g.b2.sum(),L["b2"]), b3=sh(g.b3.sum(),L["b3"]),
                hr=sh(g.hr.sum(),L["hr"]), bb=sh(g.bb.sum()+g.hbp.sum(),L["bb"]), k=sh(g.k.sum(),L["k"]), n=int(n))

def batter_side_rates(bid, hand):
    """blend w21/w45/season vs pitcher hand."""
    out, tot_w = None, 0
    for win, wgt, kcap in [("w21",0.30,30),("w45",0.40,45),("season",0.30,90)]:
        g = ba[(ba.batter==bid)&(ba.p_throws==hand)&(ba.pitch_grp=="ALL")&(ba.window==win)]
        r = rates_from(g, kcap)
        if r is None: continue
        if out is None: out = {k:0.0 for k in ["b1","b2","b3","hr","bb","k"]}; out["n"]=0
        for k in ["b1","b2","b3","hr","bb","k"]: out[k] += wgt*r[k]
        out["n"] = max(out["n"], r["n"]); tot_w += wgt
    if out is None or tot_w == 0: return None
    for k in ["b1","b2","b3","hr","bb","k"]: out[k] /= tot_w
    return out

def batter_overall_rates(bid):
    g = ba[(ba.batter==bid)&(ba.pitch_grp=="ALL")&(ba.window=="w45")]
    r = rates_from(g, 60)
    if r is None:
        g = ba[(ba.batter==bid)&(ba.pitch_grp=="ALL")&(ba.window=="season")]
        r = rates_from(g, 60)
    return r

def pitcher_side_rates(pid, side):
    out, tot_w = None, 0
    for win, wgt, kcap in [("w45",0.35,50),("season",0.65,90)]:
        g = pa_[(pa_.pitcher==pid)&(pa_.stand==side)&(pa_.pitch_type=="ALL")&(pa_.window==win)]
        r = rates_from(g, kcap)
        if r is None: continue
        if out is None: out = {k:0.0 for k in ["b1","b2","b3","hr","bb","k"]}; out["n"]=0
        for k in ["b1","b2","b3","hr","bb","k"]: out[k] += wgt*r[k]
        out["n"] = max(out["n"], r["n"]); tot_w += wgt
    if out is None or tot_w == 0: return None
    for k in ["b1","b2","b3","hr","bb","k"]: out[k] /= tot_w
    g = pa_[(pa_.pitcher==pid)&(pa_.stand==side)&(pa_.pitch_type=="ALL")&(pa_.window=="season")]
    out["bb_raw"] = float(g.bb.sum()/max(g.pa.sum(),1))
    return out

def pitcher_mix(pid, side):
    g = pa_[(pa_.pitcher==pid)&(pa_.stand==side)&(pa_.window=="season")&(pa_.pitch_type.str.startswith("GRP_"))]
    tot = g.pitches.sum()
    if tot < 30: return None
    return {r.pitch_type[4:]: r.pitches/tot for _, r in g.iterrows() if r.pitch_type[4:] != "OTH"}

def batter_group_dmg(bid, hand, grp):
    """xwobacon + barrel% + EV/LA vs pitch group from that hand; w45 primary w/ season backstop."""
    for win, kcap in [("w45",12),("season",25)]:
        g = ba[(ba.batter==bid)&(ba.p_throws==hand)&(ba.pitch_grp==grp)&(ba.window==win)]
        bbe = g.bbe.sum(); sw = g.swings.sum()
        if bbe >= 4:
            w = bbe/(bbe+kcap)
            xw = w*(g.xwobacon_sum.sum()/bbe) + (1-w)*LG_XW
            brl = w*(g.barrels.sum()/bbe) + (1-w)*LG_BRL
            ev = w*(g.ev_sum.sum()/bbe) + (1-w)*LG_EV
            la = w*(g.la_sum.sum()/bbe) + (1-w)*LG_LA
            whiff = (g.whiffs.sum()/sw) if sw >= 10 else LG_WHIFF
            return xw, brl, ev, la, whiff, int(bbe)
    return LG_XW, LG_BRL, LG_EV, LG_LA, LG_WHIFF, 0

# ---------------- build lineups (posted from repo schedule; else projected)
games = []
for g in sch["games"]:
    a, h = TEAM_ABBR[g["away"]["team"]], TEAM_ABBR[g["home"]["team"]]
    gk = GKEY[(a,h)]
    games.append(dict(gk=gk, away=a, home=h,
        away_sp=g["away"]["probable_id"], away_sp_name=g["away"]["probable_name"], away_throws=g["away"]["probable_throws"],
        home_sp=g["home"]["probable_id"], home_sp_name=g["home"]["probable_name"], home_throws=g["home"]["probable_throws"],
        away_lu=g.get("away_lineup") or [], home_lu=g.get("home_lineup") or []))

def projected_lineup(team):
    sub = b21_csv[b21_csv.team==team].sort_values("pa", ascending=False).head(9)
    return [dict(id=int(i), name=r["name"], bat_side=sides_csv.get(str(i),"R"), position="", order=j+1)
            for j,(i,r) in enumerate(sub.iterrows())]

PA_BY_SLOT = [4.70,4.55,4.42,4.30,4.18,4.05,3.92,3.80,3.68]
R_SHARE  = [0.135,0.130,0.125,0.118,0.111,0.104,0.096,0.092,0.089]
RBI_SHARE= [0.096,0.116,0.128,0.133,0.120,0.109,0.102,0.100,0.096]
STARTER_SHARE = 0.62

bpp_hr_odds = {x["batter"]: x["odds"] for x in bpp["most_likely_hr_bp_odds"]}
bpp_match = {x["batter"]: x for x in bpp["notable_matchups"]}
bpp_hrsolid = {x["batter"]: x for x in bpp["hr_solid_matchups"]}
bvp = {x["batter"]: x for x in bpp["bvp_history"]}

hb21 = hb[hb.window=="last21"].set_index("player_id")
rows = []
for g in games:
    pk = park[g["gk"]]
    for side in ["away","home"]:
        team = g[side]
        sp_id = g["home_sp" if side=="away" else "away_sp"]
        sp_name = g["home_sp_name" if side=="away" else "away_sp_name"]
        p_throws = g["home_throws" if side=="away" else "away_throws"] or "R"
        lu = g[f"{side}_lu"]
        posted = len(lu) == 9
        if not posted: lu = projected_lineup(team)
        runs_mult = pk["runs"]
        for p in lu:
            slot = p["order"] - 1
            bid = int(p["id"]); pname = p["name"]
            bs = p.get("bat_side") or sides_csv.get(str(bid), "R")
            eff = ("L" if p_throws=="R" else "R") if bs=="S" else bs
            br = batter_side_rates(bid, p_throws)
            bo = batter_overall_rates(bid)
            pr = pitcher_side_rates(sp_id, eff)
            if bo is None: bo = dict(**{k: L.get(k, 0) for k in ["b1","b2","b3","hr","bb","k"]}, n=0)
            if br is None: br = dict(bo); br["n"] = 0
            if pr is None: pr = dict(**{k: L.get(k,0) for k in ["b1","b2","b3","hr","bb","k"]}, n=0, bb_raw=L["bb"])
            mix = pitcher_mix(sp_id, eff)
            mix_str, mix_xw, mix_brl, mix_ev, mix_la, mix_whiff, mix_bbe = "", LG_XW, LG_BRL, LG_EV, LG_LA, LG_WHIFF, 0
            if mix:
                mix_str = "/".join(f"{k} {round(100*v)}" for k,v in sorted(mix.items(), key=lambda x:-x[1]))
                mix_xw = mix_brl = mix_ev = mix_la = mix_whiff = 0.0
                for grp, u in mix.items():
                    xw, brl, ev, la, wh, nb = batter_group_dmg(bid, p_throws, grp)
                    mix_xw += u*xw; mix_brl += u*brl; mix_ev += u*ev; mix_la += u*la; mix_whiff += u*wh; mix_bbe += nb
                s = sum(mix.values())
                mix_xw /= s; mix_brl /= s; mix_ev /= s; mix_la /= s; mix_whiff /= s
            # XBH damage from overall contact quality (xwOBAcon vs mix)
            dmg_mult = float(np.clip((mix_xw/LG_XW)**0.6, 0.65, 1.55))
            # HR-specific quality: barrels + exit velo + launch-angle fit vs this mix/platoon (Mike spec)
            ev_norm = float(np.clip((mix_ev - (LG_EV - 4)) / 4.0, 0.6, 1.4))   # ~1.0 at league EV
            la_norm = float(np.clip(la_fit(mix_la)/LG_LAFIT, 0.6, 1.35))
            hr_mult = float(np.clip(((mix_brl/LG_BRL)**0.50) * (ev_norm**0.15) * (la_norm**0.15), 0.60, 1.45))
            k_mult = float(np.clip((mix_whiff/LG_WHIFF)**0.35, 0.8, 1.25))
            def cmb(k):
                stat = L[k] * ((max(br[k],1e-4)/L[k])**0.55) * ((max(pr[k],1e-4)/L[k])**0.45)
                return STARTER_SHARE*stat + (1-STARTER_SHARE)*bo[k]
            r1, r2, r3, rhr, rbb = cmb("b1"), cmb("b2"), cmb("b3"), cmb("hr"), cmb("bb")
            rhr *= hr_mult; r2 *= dmg_mult**0.5; r3 *= dmg_mult**0.5
            r1 *= (2.0 - k_mult) ** 0.3
            r1 *= pk["single"]; r2 *= pk["xbh"]; r3 *= pk["xbh"]; rhr *= pk["hr"]
            rhr = min(rhr, 2.6*L["hr"])
            exp_pa = PA_BY_SLOT[slot] * (runs_mult**0.35)
            team_runs = 4.55 * runs_mult
            e1,e2,e3,ehr = r1*exp_pa, r2*exp_pa, r3*exp_pa, rhr*exp_pa
            ebb = (rbb + HBP0) * exp_pa
            onbase_q = (br["bb"]+br["b1"]+br["b2"]+br["hr"]) / (L["bb"]+L["b1"]+L["b2"]+L["hr"])
            er = team_runs * R_SHARE[slot] * (0.85 + 0.15*min(onbase_q,1.6))
            erbi = team_runs * RBI_SHARE[slot] * (0.8 + 6.0*rhr)
            sb_rate = sb_league
            if bid in hb21.index:
                h21 = hb21.loc[bid]
                if float(h21.pa or 0) > 10: sb_rate = min(float(h21.sb)/float(h21.pa), 0.055)
            esb = sb_rate * exp_pa * 0.9
            fp = (scoring["single"]*e1 + scoring["double"]*e2 + scoring["triple"]*e3 + scoring["home_run"]*ehr
                  + scoring["walk"]*ebb + scoring["run"]*er + scoring["rbi"]*erbi + scoring["stolen_base"]*esb)
            etb = e1+2*e2+3*e3+4*ehr
            hr_prob = 1-math.exp(-ehr); hit_prob = 1-math.exp(-(e1+e2+e3+ehr))
            # naive/public projection: overall 21d surface + generic platoon, no pitcher/mix/park
            same_hand = (eff == p_throws)
            nb = bo; plat = 0.92 if same_hand else 1.06
            npa = PA_BY_SLOT[slot]
            nfp = (3*nb["b1"]*plat**0.5 + 5*nb["b2"]*plat + 8*nb["b3"]*plat + 10*nb["hr"]*plat + 2*(nb["bb"]+HBP0))*npa \
                  + 2*4.55*R_SHARE[slot] + 2*4.55*RBI_SHARE[slot] + 5*sb_rate*npa*0.9
            pgs = fp - nfp
            free_swing = (pr.get("bb_raw", L["bb"]) < 0.065) and (bo["bb"] < 0.075)
            tags = []
            if pname in bpp_match:
                m = bpp_match[pname]; tags.append(f"BPP: RC+{m['rc']}% HR+{m['hr']}%")
            if pname in bpp_hrsolid: tags.append(f"BPP HR-solid +{bpp_hrsolid[pname]['starter']}% SP factor")
            if pname in bpp_hr_odds: tags.append(f"BPP HR +{bpp_hr_odds[pname]}")
            if pname in bvp and sp_name and bvp[pname]["pitcher"] in sp_name:
                v = bvp[pname]; tags.append(f"BvP {v['hits']}/{v['ab']} {v['hr']}HR {v['ops']}OPS")
            flags = []
            if not posted: flags.append("LINEUP-PROJ")
            if pk.get("rain_flag"): flags.append("RAIN")
            if pr["n"] < 90: flags.append("SP-SMALL-SAMPLE")
            if br["n"] < 40: flags.append("THIN-SIDE-SAMPLE")
            conf = "HIGH" if (br["n"]>=60 and pr["n"]>=120 and mix_bbe>=12) else ("MED" if (br["n"]>=30 and pr["n"]>=50) else "LOW")
            rows.append(dict(
                game=g["gk"], team=team, slot=slot+1, pos=p.get("position",""), player=pname, player_id=bid, bats=bs,
                opp_sp=sp_name, sp_throws=p_throws, matchup=("SAME" if same_hand else "OPP"),
                side_form=f"{br['n']}PA", side_pa=br["n"], sp_pa=pr["n"],
                mix=mix_str, mix_xwobacon=round(mix_xw,3), mix_barrel=round(100*mix_brl,1),
                mix_ev=round(mix_ev,1), mix_la=round(mix_la,1), mix_bbe=mix_bbe,
                dmg_mult=round(dmg_mult,2), hr_mult=round(hr_mult,2), k_mult=round(k_mult,2),
                exp_pa=round(exp_pa,2), proj_fp=round(fp,2), naive_fp=round(nfp,2), pgs=round(pgs,2),
                proj_tb=round(etb,2), proj_hits=round(e1+e2+e3+ehr,2), proj_bb=round(ebb,2),
                proj_r=round(er,2), proj_rbi=round(erbi,2), proj_sb=round(esb,2),
                hr_prob=round(hr_prob*100,1), hit_prob=round(hit_prob*100,1),
                park_hr=pk["hr"], park_xbh=pk["xbh"], park_1b=pk["single"], park_runs=pk["runs"],
                free_swinger="YES" if free_swing else "", conf=conf,
                bpp_tags="; ".join(tags), flags=",".join(flags)))

df = pd.DataFrame(rows)
df["pgs"] = (df["pgs"] - df["pgs"].mean()).round(2)
# Recency-vs-mix filter (permanent, 8/8): audits every row's last-21 damage against
# tonight's specific arsenal, adjusts projections + PGS, flags RECENCY-HOT/COLD/THIN.
from recency import apply_recency
_spmap = {v["name"]: int(k) for k, v in json.load(open(f"{D}/data/pitcher_splits_2026.json"))["pitchers"].items()}
df = apply_recency(df, ba, pa_, _spmap, scoring)
df["rev_platoon"] = np.where((df.matchup=="SAME") & (df.pgs > 0.5), "YES", "")
df = df.sort_values("proj_fp", ascending=False).reset_index(drop=True)
df.to_csv(f"{D}/output/full_board_v1_2026-08-07.csv", index=False)
print("rows:", len(df))
print("\nTOP 20:")
print(df.head(20)[["player","team","slot","bats","opp_sp","matchup","mix_barrel","dmg_mult","proj_fp","pgs","hr_prob","conf","flags"]].to_string())
print("\nTOP PGS:")
print(df.nlargest(12,"pgs")[["player","team","opp_sp","matchup","mix_barrel","dmg_mult","proj_fp","pgs","conf"]].to_string())
print("\nTOP HR:")
print(df.nlargest(12,"hr_prob")[["player","team","opp_sp","mix_barrel","mix_ev","mix_la","hr_mult","park_hr","hr_prob","bpp_tags"]].to_string())
print("\nRADAR CHECK:")
for n in ["Alvarez","Adell","Jordan Walker","Bleday","Witt","Rafaela","Abrams","Perez","Springer","Schwarber","Ohtani"]:
    m = df[df.player.str.contains(n)]
    if len(m):
        x = m.iloc[0]
        print(f"  {x.player:22s} {x.team:4s} vs {x.opp_sp:18s} {x.matchup:4s} mixBrl={x.mix_barrel:4.1f}% dmg={x.dmg_mult:4.2f} fp={x.proj_fp:5.2f} pgs={x.pgs:5.2f} hr={x.hr_prob:4.1f}% {x.conf}")
