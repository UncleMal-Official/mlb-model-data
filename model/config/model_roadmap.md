# Model Roadmap — signals Mike has specified

## Live tonight (v0)
- Pitcher platoon splits (BAA/OBP/SLG, BB%, K%, HR by side) + reverse-split detection
- 21-day batter form, batting sides, lineup slots (posted or projected), exp PA by slot
- Ballpark Pal park/environment multipliers + rain flags + BPP grades/sims/BvP/outlier corroboration
- PrizePicks fantasy scoring; PGS (projection gap vs public read); free-swinger × low-BB-pitcher flag
- Confidence tiers; overs-first orientation

## Arrives with GitHub fetcher (v1)
- Barrel% / EV / xwOBAcon vs pitch group, by platoon, by window (the Gameday Insights layer)
- Batter vs LHP/RHP actual splits (fixes LHB-vs-LHP calls like Alvarez–Ray)
- Pitch mix by handedness for every starter; batter damage vs that specific mix
- Season baselines for regression/perception context; day/night splits
- PrizePicks lines + names: Line/Edge columns, demon-goblin ladders, calibration of FP scale to PP lines

## Specified by Mike — build into v1.x (from Bleday walkthrough 8/7)
1. **Home/away splits with aggression read**: BB% home vs road (Bleday walks far more at home; "more aggressive on the road"). Flag when tonight's venue context contradicts the headline walk/contact profile.
2. **Road-trip day 1 effect**: performance in first away game after a homestand (and general homestand/road-trip transition trends).
3. **Batter park history ratings**: how each hitter has performed at tonight's specific park (career + recent).
4. **Data-skew / integrity flags**: when a hot surface trend is built on a context that doesn't travel (e.g., heating up during a long homestand; platoon sample mostly one venue), mark the input as SKEW-RISK and discount, showing why.
5. **Market arbitration**: for each player, compute EV across ALL their PrizePicks markets (Fantasy Score vs H+R+RBI vs TB vs hits...) from the same component projections (hits, XBH, BB, R, RBI) and recommend WHICH market is the value expression, not just over/under.
6. **Team context around the player**: on-base quality of the 2-3 slots ahead (RBI flow) and behind (R flow), so FS-vs-HRR decisions account for runners environment.
7. **Lineup slot & position splits** (already specced): production by batting-order slot, position that day (C/DH rest-day dynamics), slot-movement flag.
8. **K%/BB% edge logic** (already specced): BB points for FS value; low-BB%-pitcher × free-swinger for TB/hits.
9. **League-relative context on EVERY quoted split (specced 8/8, live in card composer)**: never call a number a
   leak/strength without the same-platoon league baseline (e.g., .386 SLG vs LHB looks bad but the RHP-league
   norm is .396 - the REAL Burns leaks are HR/PA 3.6% vs 3.1% lg and BB 11.3% vs 9.1% lg). Card composer now
   auto-picks the receipt that is a true league-relative leak (SLG > lg+15pts, else HR/PA > lg+0.4%, else
   BB/PA > lg+1%) and prints the lg anchor in the support line. Extend to board tabs + Top-6 bullets next.

## Output targets
- Tabs: CORE (FS-first + traditional props −125→EVEN) / Demons +105-up / HRs (model prob + outlier odds value) / Stacks / Full Board / Tracking
- Top-6 plays with data-backed bullets each day
- App-style daily view (persisted artifact) + sharable spreadsheet
- Daily results tracking: log every board, grade vs box scores, learn which signals earn

## Day After tool (specced 8/7, live with fetch.py v4)
- Yesterday's unlucky hard contact: barrels + near-barrels (98+ EV in 18-38 deg), deep outs (370+ ft), max EV/distance, xwOBAcon vs actual — with 0 HR = "unlucky" flag.
- Today similarity: platoon alignment + overlap between the pitch groups the player was barreling yesterday and today's SP mix (>=35% overlap threshold).
- Output: Day After tab with yesterday's contact line, today's matchup, DA score; DA tag surfaces on CORE/HR tabs. Counter-programming public recency bias.
- Tracking will grade DA candidates separately to measure the real effect size.

## Spreadsheet UX (specced 8/7, live in v1 board)
- AutoFilter on every column of every table tab.
- Frozen header row + frozen identity columns (Rank/Player) so who/what stays visible while scrolling.

## What Changed tab (specced 8/7, live v1.3.1)
- Every versioned board diff-tracks vs the prior snapshot: scratches (UGLY), projection/slot downgrades and projection-misses (BAD), upgrades and new entrants (GOOD), with was/now values and a why (lineup confirmed, slot move, SP change, confidence change, data refresh).
- build pipeline snapshots each board run so diffs are always available.

## Auto-POTD card (specced 8/7 night, live)
- On the FIRST board build of each day, make_potd_card.py picks the single best CORE value
  (z(PGS)*0.45 + z(projFP)*0.35 + HIGH-conf bonus + BPP/Day-After corroboration - rain/proj-lineup/small-sample penalties),
  runs a 20k-sim on its components, composes the receipts (BvP, barrels-vs-mix, SP platoon leak, park, L21),
  and renders the branded 1080x1350 card automatically.
- Mike reviews the PNG before posting; overrides: --player, --line, --market, --photo.
- Line on the card is MODEL-ESTIMATED (sim median - 1.0, PP-style 0.5 rounding) until the PrizePicks
  feed lands; run summary flags it; --line sets the real number. Real lines auto-fill once PP joins.
- Photo waterfall (fetch.py v6 data/headshots/): manual upload > {id}_hero.jpg (MLB action shot,
  auto-fetched for schedule players when it exists) > {id}_silo.png (transparent cutout, guaranteed
  for every batter) > branded initials medallion. Zero-upload daily default; manual photo = premium look.

## Savant deep-capture (fetch.py v5, specced 8/7)
Verified against Savant csv-docs. New per-pitch fields captured: bat_speed, swing_length, squared_up, attack_angle, hc_x/hc_y (spray), estimated_ba, release_speed/spin/extension, arm_angle, api break fields.
New aggregates and what each sharpens (by platoon x pitch group x window where applicable):
- sweet-spot% (8-32 deg LA share) -> XBH/TB model beyond barrels
- pull-air% (pulled + LA>10) -> HR model x park dimensions (short-porch fit)
- squared-up rate + bat speed -> contact quality & fatigue/decline detection ahead of results
- xBA sums -> hit-probability calibration for hits/TB markets
- pitcher velo/spin/extension per pitch type + NEW pitcher_starts.parquet (per-game velo, whiffs, barrels-against) -> velo-trend fatigue flags, "escaped" starts, Day After pitcher targeting
- zone% -> free-swinger x zone-filler logic upgrade
- sprint_speed leaderboard -> SB model (5-pt PrizePicks steals) + infield hits
Still OUR system: every new stat feeds an existing pillar (barrels-by-mix, HR model, FS floor, PGS, Day After) - no black box.

## PrizePicks lines pipeline (live 8/8)
- NO fully-automated path exists: PP runs DataDome + CF JS challenge (403s container, GH runners, Zapier);
  browser extension declines gambling sites by policy. Accepted.
- Daily 20-second flow: Mike saves api.prizepicks.com/projections?league_id=2&per_page=500 from his browser
  (returned ALL 9,039 projections in one page incl. demons/goblins), drops JSON in chat ->
  join_pp_lines.py parses, name-normalizes, joins FS lines, MARKET-CALIBRATES (global k so mean P(over)~50%;
  8/8 k=0.842), writes board_pp_<date>.csv with win_pct (cal), win_pct_raw, edge_vs_flex.
- Workbook CORE gets PP Line / Win% / Edge columns (green = edge >= +3). Card takes --line and --cal-k.
- Low-line (3.0-4.5) role-player "edges" are PAPER-TRACKED as a cohort before real money.
- All 810 FS + TB/H+R+RBI/HR/BB/SB lines are in pp_lines_<date>.csv -> full market arbitration next.
