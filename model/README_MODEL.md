# CashCord MLB Model — Autopilot

## What runs automatically
After every "Daily MLB Data Fetch" completes (8:10 AM / 3:00 PM / 5:40 PM CT), the
**Model Autopilot** workflow:
1. refreshes platoon splits for today's probables
2. builds the full board — barrel-by-mix, platoon, park, PGS, **recency-vs-mix filter**
   (RECENCY-HOT / COLD / THIN flags on every player)
3. grades YESTERDAY's board vs real box scores → results CSV + tracking.csv
4. runs Day After + Yesterday's Lessons
5. joins PrizePicks lines **if** you've uploaded today's JSON (see below)
6. writes the workbook + auto-POTD card (auto player photo from data/headshots/)
7. commits everything to `outputs/<date>/` — view/download from any device

## Install (one time, ~2 minutes)
1. GitHub → your repo → "Add file → Upload files" → drag the **model** folder in → commit.
2. Same again for `.github/workflows/model.yml` (create path `.github/workflows/` if asked).
3. Actions tab → "Model Autopilot" → "Run workflow" once to verify.

## Your two daily touches (both optional)
- **PrizePicks lines**: save `https://api.prizepicks.com/projections?league_id=2&per_page=500`
  from your browser as `pp_<date>.json` and upload to `data/pp/` (works from phone via
  github.com). Next model run joins Line / Win% / Edge.
- **Ballpark Pal sheets**: post them in chat with Claude — park + BPP configs get written and
  committed; without them the board runs park-neutral with a PENDING flag.

## Notes
- Card line is model-estimated until PP lines are joined; review the PNG before posting.
- tracking.csv is the permanent grading log — every cohort, every day.
