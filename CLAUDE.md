# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
For more information and context look inside the `context_files/` folder — multiple documents covering architecture, implementation plan, and StatsBomb workflow.

## Project

**PenaltyDuel** — a hierarchical, explainable matchup prediction system for football penalty outcomes. Given a shooter and a goalkeeper, it predicts: goal probability (with calibrated uncertainty), shot placement (6-zone heatmap), keeper dive tendency, and a plain-language explanation card.

Three reference documents in `context_files/` — read these before making architectural decisions:
- `penalty_shootout_predictor_blueprint.md` — full architecture, schema definitions, feature catalog, modeling strategy
- `penalty_predictor_implementation_plan.md` — step-by-step build order, definition-of-done per step, common mistakes
- `penalty_statsbomb_colab_workflow.md` — StatsBomb data schema details and the 9-step ingestion workflow

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the StatsBomb data pipeline (ingestion → cleaning → outputs)
# Runtime: ~5 min. Writes to outputs/statsbomb/
PYTHONIOENCODING=utf-8 python pipelines/build_dataset.py

# Run enrichment pipeline (FBref + Transfermarkt player attributes)
# First full run: ~60 min (TM rate-limited). Resumable — re-run safely.
python pipelines/enrich_players.py

# Enrichment options:
python pipelines/enrich_players.py --offline      # cache-only, instant (~30s)
python pipelines/enrich_players.py --skip-fbref   # TM live + FBref from cache
python pipelines/enrich_players.py --skip-tm      # FBref fetch only

# Run dive enrichment pipeline (Kaggle dive datasets -> keeper dive distributions)
# Requires: data/kaggle/real_data.xlsx + data/kaggle/WorldCupShootouts.csv
# Runtime: ~5s. Outputs to outputs/enrichment/.
PYTHONIOENCODING=utf-8 python pipelines/enrich_dive.py

# Train model artifact (required before starting API)
PYTHONIOENCODING=utf-8 python pipelines/train.py

# Start API server (http://localhost:8000)
uvicorn src.api.main:app --reload --port 8000

# Start React frontend (http://localhost:5173) — run from frontend/
cd frontend && npm run dev
```

All commands run from the project root (`D:\Coding\Soccer`). No test runner or linter is configured yet.

## Architecture

### Current state

**All planned steps complete. System is end-to-end functional.**

- **Steps 1–3 (StatsBomb pipeline):** Complete. 1,477 penalties · 772 shooters · 379 keepers · 21 competitions at `outputs/statsbomb/`.
- **Steps 4–7 (EDA, features, baselines, LightGBM model):** Complete. Temporal split, shrinkage, leakage test, model beats keeper-shrunk floor.
- **Phase 2 (enrichment pipeline):** Complete. FBref + Transfermarkt data at `outputs/enrichment/`. 734 high/medium entity matches. 408 unresolved.
- **Steps 8, 8.5 (enrichment + dive):** Complete. `preferred_foot` + `height_cm` features (50% coverage) + Dirichlet-shrunk keeper dive distributions (379 keepers) + 18-cell zone×dive resolution table from Kaggle WC data.
- **Step 11 (serving layer):** Complete. FastAPI at `src/api/main.py`, model artifact at `outputs/model/pipeline.pkl`. Endpoints: `/health`, `/players/shooters`, `/players/keepers`, `/predict`.
- **Step 9 (SHAP cards):** Complete. `src/explain/shap_wrap.py` + `src/explain/cards.py` + 28 honesty tests in `tests/test_explanation_honesty.py`. `/predict` response includes full `shap_card` (headline, confidence, low-data warning, factors, shooter/keeper summaries).
- **Step 10 (React dashboard):** Complete. Vite + React at `frontend/`. 3 pages: Home/Matchup Builder, Prediction Results, Player Explorer. Retro 16-bit pixel design. Dev server at `http://localhost:5173`.

### Known issues / gaps

- **Enrichment coverage 50%:** 408 players unresolved — entity resolution fails because StatsBomb stores full legal names (e.g. "Lionel Andrés Messi Cuccittini") while TM/FBref use shortened names. `token_sort_ratio` at threshold 90 can't bridge this. Fix: use `token_set_ratio` (subset matching) + short-name fallback variant.
- **Dive data thin:** Only 14 Serie A keepers in Kaggle dive set. All other keepers fall back to league/global prior. WC shootout data adds 279 kicks but limited keeper diversity.
- **Frontend minor bugs:** (1) Matchup builder cards show "? PENALTIES RECORDED" on player selection — shooter profile not propagating to card display. (2) `/predict` nav link with no params crashes Prediction page. (3) 3-panel grid breaks on mobile screens.
- **No deployment:** App only runs locally. No cloud hosting, no public URL.

### Dependency direction

```
conf/settings.py
    ↑
src/ingestion/statsbomb.py   ←─── src/cleaning/validate.py
    ↑
pipelines/build_dataset.py
    ↑
pipelines/enrich_players.py  ←─── src/ingestion/fbref.py
                             ←─── src/ingestion/transfermarkt.py
                             ←─── src/enrichment/entity_resolution.py
```

`conf/` and `src/` must be importable from the project root. `build_dataset.py` inserts `ROOT` into `sys.path` at startup.

### Key design rules (from the blueprint)

**A penalty decomposes into three models, not one:**
1. `P(zone | shooter)` — placement distribution (Dirichlet-shrunk, 6 zones)
2. `P(dive | keeper)` — dive distribution (Dirichlet-shrunk, L/C/R)
3. `P(goal | zone, dive, segment)` — resolution table (empirical, smoothed)

Composed headline: `P(goal) = Σ_z Σ_d P(zone=z|shooter) · P(dive=d|keeper) · P(goal|z,d,segment)`

**Sparsity is the central problem.** Most players have <10 career penalties. Every player-level rate uses Beta-Binomial (for scalars) or Dirichlet-multinomial (for zone/dive vectors) shrinkage toward a `foot × position × league` prior. Raw rates must never be used as model features. The fallback chain is: player → (foot × position × league) → (foot × position) → (foot) → global.

**Temporal leakage is the cardinal sin.** All history features must be computed as-of the kick date (expanding window only). A `tests/test_no_leakage.py` must pass before any modeling proceeds.

**Never drop rows globally** for a missing field that only one sub-model uses (e.g., missing `keeper_dive` still trains the outcome model).

### Data layer

```
data/                        # StatsBomb Open Data (read-only)
  competitions.json
  matches/{comp_id}/{season_id}.json
  events/{match_id}.json     # ~3,961 files, scanned for Shot+Penalty events
  lineups/{match_id}.json    # ~3,820 files, used to resolve keeper identity

outputs/statsbomb/           # StatsBomb pipeline outputs (generated, not committed)
  penalties_statsbomb_clean.parquet
  penalties_statsbomb_with_freeze_frame.parquet
  matches_statsbomb.parquet
  lineups_statsbomb_penalty_matches.parquet

outputs/enrichment/          # enrichment pipeline outputs (generated, not committed)
  penalties_enriched.parquet          ← USE THIS for all modeling
  player_attributes.parquet           # preferred_foot, dob, height per player
  player_career_penalty_stats.parquet # FBref career PK counts per player-season
  keeper_dive_distributions.json      # P(dive|keeper) for 379 keepers (Dirichlet-shrunk)
  dive_resolution_table.json          # 18-cell zone×dive resolution table (WC Kaggle data)
  entity_resolution/
    name_map.csv                      # StatsBomb → FBref/TM identity links
    unresolved.csv                    # players needing human review

outputs/model/               # trained model artifact (generated, not committed)
  pipeline.pkl               # serialized OutcomePipeline (joblib)
  metadata.json              # training stats: n_penalties, global_rate, dive_table_cells
```

**Current dataset stats:** 1,477 penalties · 772 shooters · 379 keepers · 21 competitions · 73.9% conversion rate.

**Enrichment coverage:** 618/1,146 unique players have `preferred_foot` (53% shooters, 55% keepers) · 54,556 career PK stat rows · 734 high/medium entity matches · 402 unresolved.

### Critical schema facts

- `keeper_id` is **not** on the shot event — it is derived from the opposing team's lineup by matching the goalkeeper position window to the kick's period. See `src/ingestion/statsbomb.py:_resolve_keeper()`.
- `keeper_dive_direction` is **always NaN** in this StatsBomb-only dataset. Dive labels require a future enrichment source (Kaggle hand-coded sets or the video pipeline).
- `gk_freeze_location_x/y` is the keeper's position **at the instant of the kick** (pre-dive stance), not the completed dive direction. Do not label or use it as dive direction.
- `is_shootout = (period == 5)` — StatsBomb convention, validated at pipeline runtime.
- `shot_zone` is a 6-bin classification: `{low,high} × {left,center,right}` from the keeper's POV. Boundary constants live in `conf/settings.py`. Raw `(x,y)` coordinates are preserved for re-binning.

### Completed build steps

| Step | Status | What was built | Key files |
|------|--------|---------------|-----------|
| 1–3 | ✅ Done | StatsBomb ingestion pipeline | `pipelines/build_dataset.py`, `src/ingestion/statsbomb.py` |
| 4 | ✅ Done | EDA notebook | `notebooks/01_eda.ipynb` |
| 5 | ✅ Done | Shrinkage utils + as-of feature builder + leakage test | `src/features/` |
| 6 | ✅ Done | Baselines + evaluation framework | `src/models/baselines.py`, `src/evaluation/` |
| 7 | ✅ Done | LightGBM outcome model + calibration | `src/models/outcome_lgbm.py`, `src/models/calibration.py` |
| Phase 2 | ✅ Done | FBref + Transfermarkt enrichment pipeline | `pipelines/enrich_players.py`, `src/ingestion/fbref.py`, `src/ingestion/transfermarkt.py` |
| 8 | ✅ Done | Foot/height features + retrain | `src/features/build.py` |
| 8.5 | ✅ Done | Kaggle dive datasets → keeper dive distributions + resolution table | `src/ingestion/kaggle_dive.py`, `src/models/dive_prior.py`, `pipelines/enrich_dive.py` |
| 11 | ✅ Done | FastAPI serving layer | `src/serving/predict.py`, `src/api/main.py`, `pipelines/train.py` |
| 9 | ✅ Done | SHAP explanation cards | `src/explain/shap_wrap.py`, `src/explain/cards.py` |
| 10 | ✅ Done | React dashboard (Retro 16-bit pixel design) | `frontend/` |

### Improvement opportunities (priority order)

| Priority | Area | Impact | What to do |
|----------|------|--------|-----------|
| 1 | Enrichment coverage | High | Fix entity resolution: `token_set_ratio` + short-name variant for legal-name players. Unlocks foot/height for Messi, Ronaldinho, Neymar, ~350 more players (50% → ~70%+ coverage) |
| 2 | Frontend bug fixes | Medium | Fix matchup card display bug, `/predict` crash with no params, mobile grid layout |
| 3 | Dive data expansion | Medium | Source more keeper dive labels (Kaggle PK datasets, manually labeled video). Current: 14 keepers. Target: 100+ |
| 4 | Deployment | Medium | Deploy API to Render/Railway, frontend to Vercel. Make app publicly shareable |
| 5 | Model features | Low-Medium | Add height differential (shooter_height - keeper_height), foot × zone interaction, league-level priors |
| 6 | More StatsBomb data | Low | StatsBomb has more open competitions. Add them to expand 1,477 → 2,000+ penalties |

### Evaluation rules

- Primary metric: **log loss + Brier score** (never accuracy — 73.9% base rate makes accuracy useless).
- Primary split: **temporal** (train before cutoff, test after). Never random split.
- The bar to beat: **shrunk-marginal baselines** (shooter-only shrunk, keeper-only shrunk) — not the global rate.
- Always report **calibration** (reliability diagram + ECE). This is a probability product.

### Video upgrade seam (install now, use later)

When adding video features: implement a `FeatureProvider` interface (`src/features/providers.py`) so tabular and video features are separate providers concatenated at assembly time. An empty `video_features` table (keyed by `penalty_id`) should be added to the DB schema. Models consume a feature dict and treat missing video features as `null + video_is_missing=True`.
