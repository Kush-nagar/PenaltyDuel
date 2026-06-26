# PenaltyDuel — Next Steps

All planned steps are complete. System is end-to-end functional.
Run `uvicorn src.api.main:app --reload --port 8000` + `cd frontend && npm run dev` to start.

---

## Priority 1: Fix Enrichment Coverage (Biggest Impact)

**Problem:** 50% of penalties have no `preferred_foot` for shooter.
Root cause: StatsBomb stores full legal names ("Lionel Andrés Messi Cuccittini") but TM/FBref
use short names ("Lionel Messi"). `token_sort_ratio` scores them ~57%, below threshold 90.
Result: 408 players in `outputs/enrichment/entity_resolution/unresolved.csv` including
Messi (82 penalties!), Ronaldinho (16), Neymar (15), Kim Little (10), Dybala (7).

**Fix:** In `src/enrichment/entity_resolution.py`:
- Replace `fuzz.token_sort_ratio` with `fuzz.token_set_ratio` for the fuzzy match step
  (token_set_ratio handles subset names — "lionel messi" tokens all appear in full legal name)
- Add short-name variant to TM search: try searching first + last token of normalized name
  as a fallback when full-name search returns no candidates
- Keep corroboration requirement (nationality or team overlap) before auto-accepting

Expected outcome: coverage from 50% → ~70%+ foot data. Retrain model after.

Key files:
- `src/enrichment/entity_resolution.py` — fuzzy matching logic, FUZZY_THRESHOLD = 90
- `src/ingestion/transfermarkt.py` — `_search_candidates()` method, builds TM search URL
- `pipelines/enrich_players.py` — orchestrator, run after fixing resolution

---

## Priority 2: Fix Frontend Bugs (Quick Wins)

**Bug 1:** Matchup builder cards show "? PENALTIES RECORDED" after player selected.
File: `frontend/src/pages/Home.jsx` — shooter/keeper profile not propagating to card display.

**Bug 2:** Clicking PREDICT nav link with no `?shooter=X&keeper=Y` params crashes Prediction page.
File: `frontend/src/pages/Prediction.jsx` — needs empty/prompt state when params missing.

**Bug 3:** 3-panel grid (zone heatmap / dive bars / SHAP terminal) breaks on mobile.
File: `frontend/src/pages/Prediction.jsx` — grid needs responsive breakpoint (stack vertically on small screens).

---

## Priority 3: Expand Dive Data

**Problem:** Only 14 Serie A keepers have real dive labels from Kaggle dataset.
All 379 keepers not in that set fall back to league/global Dirichlet prior.

**Options:**
- Source more Kaggle PK datasets with dive direction labels
- Manually label 50-100 top keepers from YouTube penalty compilations
- Kaggle dataset used: `data/kaggle/real_data.xlsx` (474 kicks) + `data/kaggle/WorldCupShootouts.csv` (279 kicks)

Key files:
- `src/ingestion/kaggle_dive.py` — parses dive datasets
- `src/models/dive_prior.py` — Dirichlet shrinkage for dive distributions
- `pipelines/enrich_dive.py` — orchestrator, ~5s runtime

---

## Priority 4: Deploy (Make It Publicly Shareable)

No code changes needed — just config.

**Backend (FastAPI):** Deploy to Render or Railway (both have free tiers)
- Build command: `pip install -r requirements.txt && python pipelines/train.py`
- Start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- Need to commit model artifacts or run train.py at deploy time

**Frontend (React/Vite):** Deploy to Vercel
- Build command: `npm run build`
- Output dir: `dist`
- Set `VITE_API_BASE` env var to point at deployed backend URL
- Update `frontend/vite.config.js` proxy to use env var

---

## Priority 5: Model Feature Improvements

Low-medium impact, add after enrichment coverage is fixed (more foot data = these features matter more).

- **Height differential:** `shooter_height_cm - keeper_height_cm` — taller keepers cover more goal area
- **Foot × zone interaction:** Left-footed shooters bias toward low-right; add as a feature or improve zone prior
- **League-level priors:** Currently priors are `foot × position × league`. Could add era/year as dimension.

Key files:
- `src/features/build.py` — feature assembly
- `src/models/combo.py` — `COMBO_FEATURE_NAMES` list, `build_combo_features()`
- Retrain: `python pipelines/train.py`

---

## Priority 6: More StatsBomb Data

StatsBomb open data has more competitions than currently loaded.
Check `data/competitions.json` for full list — add unlisted comp IDs to `conf/settings.py`.
Re-run `python pipelines/build_dataset.py` then full enrichment + train cycle.

Current: 1,477 penalties · 21 competitions
Target: 2,000+ penalties with more competitions

---

## Current Architecture (for context)

```
StatsBomb data (read-only)
    ↓ pipelines/build_dataset.py
outputs/statsbomb/penalties_statsbomb_clean.parquet
    ↓ pipelines/enrich_players.py (FBref + TM scrape, ~60min first run)
    ↓ pipelines/enrich_dive.py   (Kaggle dive data, ~5s)
outputs/enrichment/penalties_enriched.parquet   ← USE THIS for modeling
outputs/enrichment/keeper_dive_distributions.json
outputs/enrichment/dive_resolution_table.json
    ↓ pipelines/train.py
outputs/model/pipeline.pkl   ← loaded by FastAPI at startup
    ↓ uvicorn src.api.main:app --port 8000
API endpoints: /health /players/shooters /players/keepers /predict
    ↓ Vite proxy (/api/* → localhost:8000)
frontend/   ← npm run dev (localhost:5173)
```

## Key Stats

- 1,477 penalties · 772 shooters · 379 keepers · 21 competitions
- 73.9% global conversion rate
- 50% `preferred_foot` coverage (408 players unresolved — see Priority 1)
- 14 keepers with real dive labels (rest use Dirichlet prior)
- Model: LogisticRegression on 6 combo features (shrunk rates + zone entropy + shootout flag + log sample sizes)
- SHAP: LinearExplainer, probability space, 6 factors per prediction
