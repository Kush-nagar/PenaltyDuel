# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
For more information adn context towards the project look inside the project_context_files folder (D:\Coding\Soccer\project_context_files) where there are multiple files which will help in understanding the project a bit more in depth and will answer most of the questions.

## Project

**PenaltyDuel** — a hierarchical, explainable matchup prediction system for football penalty outcomes. Given a shooter and a goalkeeper, it predicts: goal probability (with calibrated uncertainty), shot placement (6-zone heatmap), keeper dive tendency, and a plain-language explanation card.

Three reference documents live at the project root — read these before making architectural decisions:
- `penalty_shootout_predictor_blueprint.md` — full architecture, schema definitions, feature catalog, modeling strategy
- `penalty_predictor_implementation_plan.md` — step-by-step build order, definition-of-done per step, common mistakes
- `penalty_statsbomb_colab_workflow.md` — StatsBomb data schema details and the 9-step ingestion workflow

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the data pipeline (Steps 1–3 of the plan: ingestion → cleaning → outputs)
# Runtime: ~5 min. Writes to outputs/statsbomb/
python pipelines/build_dataset.py

# Run the pipeline with correct encoding on Windows
PYTHONIOENCODING=utf-8 python pipelines/build_dataset.py
```

All commands run from the project root (`D:\Coding\Soccer`). No test runner or linter is configured yet.

## Architecture

### Current state (Phase A complete)

The data pipeline is built and the clean dataset exists. Modeling, serving, and frontend are not yet implemented.

### Dependency direction

```
conf/settings.py
    ↑
src/ingestion/statsbomb.py   ←─── src/cleaning/validate.py
    ↑
pipelines/build_dataset.py
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

outputs/statsbomb/           # pipeline outputs (generated, not committed)
  penalties_statsbomb_clean.parquet     ← USE THIS for all modeling
  penalties_statsbomb_with_freeze_frame.parquet
  matches_statsbomb.parquet
  lineups_statsbomb_penalty_matches.parquet
```

**Current dataset stats:** 1,477 penalties · 772 shooters · 379 keepers · 21 competitions · 73.9% conversion rate.

### Critical schema facts

- `keeper_id` is **not** on the shot event — it is derived from the opposing team's lineup by matching the goalkeeper position window to the kick's period. See `src/ingestion/statsbomb.py:_resolve_keeper()`.
- `keeper_dive_direction` is **always NaN** in this StatsBomb-only dataset. Dive labels require a future enrichment source (Kaggle hand-coded sets or the video pipeline).
- `gk_freeze_location_x/y` is the keeper's position **at the instant of the kick** (pre-dive stance), not the completed dive direction. Do not label or use it as dive direction.
- `is_shootout = (period == 5)` — StatsBomb convention, validated at pipeline runtime.
- `shot_zone` is a 6-bin classification: `{low,high} × {left,center,right}` from the keeper's POV. Boundary constants live in `conf/settings.py`. Raw `(x,y)` coordinates are preserved for re-binning.

### Next build steps (from the implementation plan)

| Step | What to build | Key files |
|------|--------------|-----------|
| 4 | EDA notebook | `notebooks/01_eda.ipynb`, `src/viz/goalmouth.py` |
| 5 | Shrinkage utils + as-of feature builder + leakage test | `src/features/shrinkage.py`, `src/features/as_of.py`, `src/features/build.py`, `tests/test_no_leakage.py` |
| 6 | Baselines + evaluation framework | `src/models/baselines.py`, `src/evaluation/splitters.py`, `src/evaluation/metrics.py` |
| 7 | LightGBM outcome model + calibration | `src/models/outcome_lgbm.py`, `src/models/calibration.py` |
| 11 | FastAPI serving layer | `src/serving/predict.py`, `src/api/main.py` |
| 10 | React dashboard | `frontend/` |

### Evaluation rules

- Primary metric: **log loss + Brier score** (never accuracy — 73.9% base rate makes accuracy useless).
- Primary split: **temporal** (train before cutoff, test after). Never random split.
- The bar to beat: **shrunk-marginal baselines** (shooter-only shrunk, keeper-only shrunk) — not the global rate.
- Always report **calibration** (reliability diagram + ECE). This is a probability product.

### Video upgrade seam (install now, use later)

When adding video features: implement a `FeatureProvider` interface (`src/features/providers.py`) so tabular and video features are separate providers concatenated at assembly time. An empty `video_features` table (keyed by `penalty_id`) should be added to the DB schema. Models consume a feature dict and treat missing video features as `null + video_is_missing=True`.
