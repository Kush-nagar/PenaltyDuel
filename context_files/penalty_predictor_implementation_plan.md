# Penalty Shootout Predictor — Step-by-Step Implementation Plan

*Zero → final product, in execution order. This is the "what to do, in what order, and how to know each piece is done" companion to the architecture blueprint. Where this plan says "see Blueprint §X," that refers to the full reference document (schemas, feature catalog, architecture) — this plan stays focused on **sequence, definition-of-done, and mistakes to avoid** so you can actually build.*

---

## How to use this plan

Each step uses the same template so you can treat it like a ticket:

- **Do** — the concrete actions.
- **Why it matters** — so you don't cut the wrong corner.
- **Data / tools** — what's needed.
- **Files / modules to create** — exact paths.
- **Common mistakes** — the traps that quietly sink student projects.
- **Done looks like** — the gate you must pass before moving on (commit it to git).

Two rules that override everything: **(1) ship a working thin slice before adding depth** — a calibrated headline probability in a basic UI is a real project; polish comes after. **(2) Guard against data leakage from day one** — it is the single failure that makes impressive-looking results worthless.

### Step map at a glance

| Step | Output | Effort | On the critical path to a demo? |
|---|---|---|---|
| 1. Problem & success criteria | One-page spec + numeric success bar | ◐ 0.5 day | Yes |
| 2. Gather data | Raw snapshots + coverage report | ◑ 2–4 days | Yes |
| 3. Clean & organize | `penalties.parquet` + player map | ◑ 2–4 days | Yes |
| 4. Explore (EDA) | Notebook + "data realities" memo | ◐ 1–2 days | Partly |
| 5. Engineer features | Leak-safe modeling table | ● 3–5 days | Yes |
| 6. Baseline model | Baseline metrics table (the floor) | ◐ 1 day | Yes |
| 7. Stronger model | Calibrated LightGBM + profiles | ● 4–7 days | Yes |
| 8. Evaluate properly | Results + ablation + generalization | ◑ 2–4 days | Partly |
| 9. Explainability | SHAP-grounded explanation cards | ◑ 2–3 days | Yes (for the "why") |
| 10. Dashboard | Searchable UI w/ heatmaps + uncertainty | ● 5–8 days | Yes |
| 11. Backend | FastAPI service + DB | ◑ 3–5 days | Yes |
| 12. Train/version/retrain | Reproducible pipeline + tracking | ◑ 2–3 days | Partly |
| 13. Video upgrade seam | Interfaces in place now | ◐ 0.5 day | Seam only |
| 14. Exact order | The master sequence | — | — |

*Effort: ◐ low · ◑ medium · ● high, for a motivated student.*

---

## Step 1 — Define the problem and success criteria

**Do.** Write a single-page spec and a quantitative success bar, then commit them. Decide scope explicitly:
- **What the model predicts:** (1) probability of scoring; (2) shot placement (6-zone distribution); (3) keeper dive direction (L/C/R); plus (4) a plain-language explanation and (5) a searchable dashboard.
- **Version 1 (non-video) includes:** calibrated **scoring probability** + **placement heatmap** + **explanation card** + a working **two-search dashboard**. Treat **dive direction as a stretch goal within v1** (data is scarce — see Step 2).
- **Final version includes:** the full composed matchup model (placement ⊗ dive ⊗ resolution), hierarchical-Bayesian uncertainty surfaced in the UI, dive panel where data allows, leave-players-out generalization study, deployment, and a written report/model card.
- **Success bar (numeric, not vibes):** beat the **shrunk shooter/keeper marginal baselines** on **log loss and Brier** across folds (positive Brier Skill Score); **calibration** ECE low with a near-diagonal reliability diagram; **placement top-2 accuracy** clearly above the ~33% chance line; dashboard returns a matchup in <1s with visible uncertainty.

**Why it matters.** Scope creep and "chase a higher accuracy number" are what kill these projects. A written success bar gives your eventual write-up its thesis ("a modest, well-calibrated, honest improvement over strong baselines") and tells you when to *stop* tuning.

**Data / tools.** None — just a markdown editor.

**Files / modules.**
```
docs/PROJECT_SPEC.md      # the one-pager: outputs, v1 vs final scope
docs/SUCCESS_CRITERIA.md  # the numeric bar + chosen metrics
MODEL_CARD.md             # stub now; fill as you go (data, assumptions, limits, biases)
README.md                 # project intro + how to run (stub)
```

**Common mistakes.**
- Defining success as **accuracy** — at a ~78% base rate, "always predict scored" already scores ~78%. Useless. Use log loss / Brier / calibration.
- Promising **dive direction in v1** as a core feature when free dive-label data is thin — set it as a stretch goal so a data gap can't derail the whole project.
- Not writing it down, then tinkering forever with no finish line.

**Done looks like.** A committed one-page spec with explicit v1/final scope and concrete numeric success thresholds. You can answer "what does winning look like?" in two sentences.

---

## Step 2 — Gather the data

**Do.** Start with **one source (StatsBomb open data)** end-to-end before adding others. Then layer in player histories and, optionally, location and dive-labeled sets. Snapshot every raw pull (date-stamped) and produce a **field-coverage report** before you model anything.

**Sources, fields to pull, and status:**

| Source | Pull these fields | Free? | Use it for |
|---|---|---|---|
| **StatsBomb Open Data** (`statsbombpy`) | event type=Shot where `shot.type=Penalty`; `location`/end-location, `shot.outcome`, freeze-frame (keeper position), player/team IDs, competition, match date, period | **Free** (non-commercial) | The core: outcomes **+ (x,y) locations** + partial keeper positioning. Start here. |
| **FBref** (`worldfootballR` or careful scraping) | per-player `PKatt`, `PK` (made), minutes, position, foot, DOB, club-season | **Free** (respect ToS, rate-limit) | Shooter/keeper **penalty histories + attributes** for priors. |
| **Understat** (community wrapper) | shot-level `X`,`Y`,`result`,`xG`, `situation=Penalty`, player, season | **Free** (scrape politely) | Extra **(x,y) placement** at scale for big-5 leagues. |
| **Kaggle penalty/shootout datasets** | placement label, **keeper dive direction**, outcome, kick order | **Free** (quality varies) | Often the **only free dive-direction labels** — for the dive sub-model. |
| **Transfermarkt** | foot, height, position, DOB | **Free** (scrape politely) | Filling player attributes / disambiguation. |
| **Opta / StatsPerform, Wyscout, StatsBomb full** | dense keeper actions, placement qualifiers | **Paid — optional** | Only if you get access; not required. |

**Handling inconsistent names & missing values (do this as you ingest, not later):**
- Normalize names (lowercase, strip diacritics, unify "Last, First"), keep an `aliases` list, and **never auto-merge on name alone** — require a second signal (birth year, nationality, club-season overlap). Use RapidFuzz to *propose*, not to *decide*.
- For every field that can be absent, emit the value **plus a companion `*_is_missing` flag**. Don't silently fill.

**Merging multiple sources:** write a thin **adapter per source** that maps its raw fields into one canonical penalty schema (Blueprint §3.3 has the full column list). All downstream code sees only the canonical schema. Dedup overlapping penalties on `(date, shooter_id, keeper_id, competition, minute±tol)`, resolving conflicts by a **source-trust ranking** (StatsBomb > Understat > FBref-derived > Kaggle-community) and logging disagreements.

**Files / modules.**
```
src/ingestion/statsbomb.py      # FIRST: raw StatsBomb -> canonical rows
src/ingestion/fbref.py          # player histories + attributes
src/ingestion/understat.py      # optional: (x,y) at scale
src/ingestion/kaggle_dive.py    # optional: dive-direction labels
src/ingestion/base.py           # shared adapter interface + canonical schema
pipelines/build_dataset.py      # orchestrates pulls -> data/raw/ snapshots
reports/coverage_report.py      # % non-null per field, per source
```

**Common mistakes.**
- **Scraping FBref/Transfermarkt too fast** → IP ban. Rate-limit hard, cache responses, prefer `worldfootballR`.
- **Trusting names for joins** → two different "Danny Williams" merged into one phantom player. Require a second signal.
- **Not snapshotting raw** → results become irreproducible when a site changes. Save date-stamped raw, track with DVC.
- **Starting with five sources at once** → drowning in reconciliation. Get StatsBomb working solo first.

**Done looks like.** Date-stamped raw snapshots in `data/raw/`, an adapter that turns StatsBomb into canonical rows, and a **coverage report** that tells you honestly what fraction of penalties have `shot_zone` and `keeper_dive` (this number shapes the rest of the project).

---

## Step 3 — Clean and organize the dataset

**Do.** Turn the messy multi-source pile into **one validated, one-row-per-penalty dataset** plus a **canonical player/keeper mapping table**.

- **Remove/repair bad rows:** drop impossible records (outcome not in vocabulary, kick_index < 1, nonsensical dates); **exclude retakes by default** (or keep only the decisive attempt) and log the choice; quarantine cross-source conflicts to a review queue rather than guessing a label.
- **Standardize columns:** enforce the canonical schema and types with a validation library (`pandera` or Great Expectations). Discretize end-locations into the **6-zone** scheme (keeper's POV) while **retaining raw (x,y)** for later re-binning and the future regression target.
- **Canonical player/keeper map:** build `name_map` (`player_id ↔ display_name + aliases + foot + position + DOB + nationality`). Keep an `unresolved.csv` for the long tail and review it.
- **Final dataset:** write `data/processed/penalties.parquet` — one row per penalty, fully typed, with provenance (`source`, `source_event_id`) and all `*_is_missing` flags.

**Folder structure (create it now, in full):**
```
penalty-predictor/
├── data/            raw/  interim/  processed/      # DVC-tracked; contents gitignored
├── src/
│   ├── ingestion/  cleaning/  entity_resolution/
│   ├── features/   models/    evaluation/
│   ├── explain/    serving/   api/
├── pipelines/       build_dataset.py  train.py  export_profiles.py
├── frontend/        # React/Vite app (later)
├── models/          # exported artifacts (DVC-tracked)
├── notebooks/       # EDA + figures (not on the import path)
├── tests/           # pytest incl. the leakage test
├── conf/            params.yaml      # paths, hyperparams, model_version
├── docs/            MODEL_CARD.md  PROJECT_SPEC.md
├── dvc.yaml  params.yaml  pyproject.toml  README.md
```

**Files / modules.**
```
src/cleaning/validate.py          # schema/range checks, dedup, retake handling
src/cleaning/zoning.py            # (x,y) -> 6-zone; keep raw coords
src/entity_resolution/name_map.py # normalization + fuzzy proposals + review queue
pipelines/build_dataset.py        # extend to emit processed/penalties.parquet
```

**Common mistakes.**
- **Global row-dropping** for a field only one sub-model needs (e.g., dropping a penalty for missing `keeper_dive` even though it still trains the outcome model). Drop per-sub-model, never globally.
- **Editing raw files in place** — raw is read-only; all cleaning writes to `interim/`/`processed/`.
- **Auto-accepting fuzzy name matches** without review → silent label corruption.
- **Throwing away raw (x,y)** after binning — you'll want it for the video phase and for re-binning experiments.

**Done looks like.** `penalties.parquet` passes schema validation, the player map is reviewed (no name-only auto-merges), the full folder structure exists, and the dataset is DVC-tracked and tagged (e.g., `data-v2024-07`).

---

## Step 4 — Explore the data

**Do.** Run a focused EDA whose purpose is to (a) surface data problems before they corrupt the model and (b) produce a "data realities" memo that becomes part of your model card and write-up.

**Patterns to quantify:**
- Overall conversion rate (sanity-check it lands in the well-known high-70s%); conversion **by 6-zone**, **by shooter foot**, **by stage/competition**.
- Keeper **save rate** distribution and **dive-direction** distribution (and how many keepers have enough samples to say anything).
- **Pressure structure:** conversion by `stakes` (routine vs. must-score-to-survive vs. can-win-it), by `shootout_kick_index`, and the **first-kicker** effect.
- **Sparsity reality:** histogram of penalties-per-player and penalties-faced-per-keeper. This single chart justifies the entire shrinkage strategy — feature it.
- **Coverage:** % of rows with `shot_zone`, with `keeper_dive`, by source and era.

**Charts to create:**
- Penalties-per-player histogram (the sparsity story).
- League-wide **goal-mouth heatmap** of placement (and conversion by zone) — also a UI-component dry run.
- Conversion-by-stakes bar chart with confidence intervals.
- Coverage bars per field/source.
- Reliability of the **naive** shooter rate (does a raw 5-pen rate predict the next pen? — it won't, motivating shrinkage).

**Weird cases / problems to check:** 100%-on-2-penalties players (sparsity trap), missing/unknown keepers, retakes still present, obvious era drift in conversion, duplicated penalties across sources, mislabeled outcomes.

**Files / modules.**
```
notebooks/01_eda.ipynb            # the figures
notebooks/02_coverage_sparsity.ipynb
docs/DATA_REALITIES.md            # written summary feeding MODEL_CARD.md
src/viz/goalmouth.py              # reusable 6-zone heatmap (used again in the UI)
```

**Common mistakes.**
- Drawing player-level conclusions from **tiny samples** during EDA (the same trap the model must avoid).
- Computing EDA aggregates over the **whole** dataset and mentally "learning" them, then leaking that intuition into feature design — keep EDA descriptive, not predictive.
- **Skipping the coverage check** and discovering mid-modeling that 80% of rows lack dive labels.

**Done looks like.** An EDA notebook with the key figures, a `DATA_REALITIES.md` memo (sparsity levels, coverage, pressure effects, known label issues), and a reusable goal-mouth plotting function you'll reuse in the dashboard.

---

## Step 5 — Engineer features

**Do.** Build features in dependency order: **shrinkage utilities → as-of feature builder → the leakage test**, then the feature groups. Every history feature must be computed **as-of the kick date** (only data strictly before the row), and must **fall back to a prior** when the sample is thin. (Full feature catalog in Blueprint §5; below is the build-ordered essentials.)

**Feature groups (build the v1 core first):**

| Group | Key features | v1 priority |
|---|---|---|
| **Shooter** | `conv_rate_shrunk`, `n_pens`, `zone_probs` (Dirichlet-shrunk), `pref_side_natural`, `zone_entropy` | **Core** |
| **Keeper** | `save_rate_shrunk`, `n_faced`, `dive_probs` (shrunk), `guess_correct_rate`, `stay_center_rate` | **Core** (dive parts where data allows) |
| **Matchup (interaction)** | `expected_guess_correct`, `side_conflict_score` (derived from the *shrunk* per-player distributions, **not** raw head-to-head — most pairs have 0 meetings) | **Core — the differentiator** |
| **Context** | `is_shootout`, `stakes` categorical, `stage`, `home_away`, era bucket, `competition_importance` | **Core** (stakes + is_shootout) |
| **Pressure** | `shootout_kick_index`, `must_score`/`must_not_concede`, `shooter_under_pressure_delta` (shrink hard) | Strong version |
| **Form / streaks** | `shooter_recent_conv`, `missed_last_pen`, `minutes_played`, `days_rest` | Strong version (prove via ablation) |

**The v1 "core 7" to ship first:** `shooter_conv_rate_shrunk`, `shooter_n_pens`, `shooter_zone_probs`, `keeper_save_rate_shrunk`, `keeper_n_faced`, `expected_guess_correct`, and `stakes`+`is_shootout`. Add `shooter_foot`/`position` as the prior backbone.

**Players with few penalties (the core problem):**
- **Beta-Binomial** shrinkage for rates: posterior mean = (α + scored)/(α + β + n), with (α, β) from the population or the `foot × position` subgroup. A 2-for-2 shooter is **not** 100%.
- **Dirichlet-multinomial** shrinkage for `zone_probs`/`dive_probs`.
- **Fallback chain:** player → (foot × position × league) → (foot × position) → (foot) → global.
- **Cold start (0 penalties):** predict entirely from the subgroup prior and flag it. The system never refuses.

**Encoding:** one-hot low-cardinality categoricals (`stakes`, `stage`, `home_away`); **represent players through their shrunk numeric profiles, not one-hot/embeddings** (sparsity would destroy per-player parameters in v1); pair every imputed value with its `*_is_imputed` flag; bucket `season` into eras; use expanding windows with optional exponential recency-weighting.

**Files / modules.**
```
src/features/shrinkage.py     # FIRST: Beta-Binomial + Dirichlet utils + fallback chain
src/features/as_of.py         # expanding-window, leak-safe history aggregation
src/features/build.py         # assembles the modeling table from feature groups
src/features/providers.py     # FeatureProvider interface (tabular now, video later) — see Step 13
tests/test_no_leakage.py      # WRITE THIS NOW: assert no feature uses data dated >= the kick
```

**Common mistakes.**
- **The cardinal sin:** computing `groupby('shooter').conv_rate.mean()` over the **entire** dataset — it leaks the test penalty's own outcome into its features. Use as-of expanding windows and enforce with the leakage test.
- **One-hot or embedding player IDs** in v1 → severe overfit on tiny data.
- **Forgetting `*_is_imputed` flags** → the model and the dashboard can't tell real from prior-based.
- **Deriving interaction features from raw head-to-head counts** → mostly zeros and noise; derive from shrunk distributions instead.

**Done looks like.** A leak-safe modeling table built from the core feature groups, shrinkage utilities tested on toy inputs, and **`tests/test_no_leakage.py` passing**. This test passing is the gate for all modeling.

---

## Step 6 — Build the baseline model

**Do.** Build the baselines that every later model must beat, and set up **safe splits** once so all models reuse them.

- **Baselines (in order of strength):** (1) global conversion rate; (2) shooter-only rate (raw, then **shrunk**); (3) keeper-only rate (raw, then **shrunk**). The **shrunk single-factor models are the real bar** — they're deceptively strong.
- **Splits (set up once, reuse everywhere):**
  - **Temporal (primary):** train before a cutoff date, test after — with all features computed as-of.
  - **Grouped-by-player** (leave-players-out) and **grouped-by-competition** for generalization (used in Step 8).
- **Metrics:** **log loss**, **Brier score**, **Brier Skill Score vs. the shrunk-marginal baseline**, **ROC-AUC / PR-AUC**, and **calibration (reliability diagram + ECE)**. Accuracy is reported only as a footnote.

**Files / modules.**
```
src/evaluation/splitters.py   # temporal + grouped (player/competition) splitters
src/evaluation/metrics.py     # log loss, Brier, BSS, ECE, reliability, AUC
src/models/baselines.py       # global / shooter-shrunk / keeper-shrunk
reports/baseline_table.md     # the comparison floor (committed)
```

**Common mistakes.**
- **Random train/test split** instead of temporal → leakage and an inflated, dishonest score.
- **Optimizing accuracy** or using **SMOTE/oversampling** → wrecks the calibration this product depends on.
- Comparing only against the **global** rate (easy to beat) and skipping the **shrunk-marginal** baseline (the honest, hard bar).

**Done looks like.** A committed baseline table (all three baselines × all metrics, with CIs from repeated/grouped CV) and reusable splitter + metric modules. You now have the floor every advanced model is measured against.

---

## Step 7 — Build the stronger model

**Do.** Train the main tabular outcome model, the hierarchical profiles that supply shrinkage + uncertainty, and the composed matchup prediction; then calibrate.

### Which algorithm? (direct answer)

| Option | Verdict |
|---|---|
| **LightGBM** | **Primary.** Best-in-class on small/medium tabular data, native missing-value handling, **monotonic constraints** (force "higher shooter conversion ⇒ higher predicted goal prob," which prevents nonsense fits on small data), fast, SHAP-friendly. |
| **CatBoost** | **Strong alternative / cross-check.** Its **ordered boosting** resists overfitting on small data, and it handles raw categoricals with little tuning. In *this* design players are represented via shrunk numeric profiles (so the categorical advantage matters less), but CatBoost's overfit-resistance makes it a legitimate second model — and a good pick if you ever keep raw categorical player/team features. |
| **XGBoost** | **Third cross-check.** Solid and well-documented; use it to confirm LightGBM isn't overfitting (if all three agree within CI, you can trust the result). |
| **Regularized logistic regression** | **Keep as an interpretable sanity check.** If logistic ≈ trees on log loss, there's little nonlinearity to exploit on data this small — an honest, common finding. |
| **Bayesian / hierarchical** | **Use as a layer, not a competitor** (below) — it supplies the shrinkage and the uncertainty bands. |
| **Hybrid (recommended overall)** | **LightGBM for the live headline number + hierarchical Bayesian (offline) for profiles/priors/CIs + an analytic composition for placement⊗dive⊗resolution + a tiny meta-blender.** This is the system. |

**Don't** ensemble all three boosters for v1 — diminishing returns and added complexity. Pick LightGBM, keep CatBoost/XGBoost as cross-checks for your evaluation table.

### Combining shooter + keeper + context
1. **Hierarchical Bayesian (PyMC/NumPyro, offline):** multilevel logistic for conversion + Dirichlet-multinomial for zones/dives, partially pooled across `foot × position × league`. Outputs the **shrunk profiles + credible intervals** stored for serving.
2. **Resolution table:** empirical, smoothed `P(goal | zone, dive, segment)` (optionally split by foot).
3. **Composition (live):** `P(goal) = Σ_z Σ_d P(zone|shooter)·P(dive|keeper)·P(goal|z,d,segment)` → also yields the heatmap + "most likely zone vs. dive."
4. **Meta-blender:** a small logistic over {composed prob, direct LightGBM prob} → the single calibrated headline number. Reconcile and report both; large disagreement = an uncertainty signal.

### Anti-overfitting (settings, not vibes)
Shallow trees (`max_depth` 3–4), large `min_data_in_leaf`/`min_child_samples`, strong `lambda_l1/l2`, low learning rate + **early stopping on a temporal fold**, **monotonic constraints** on monotone features, **grouped-by-player CV**, and **few well-engineered (already-shrunk) features** over many raw ones.

### Calibration (non-negotiable for a probability product)
Fit on a **held-out** fold: **isotonic** if data allows, **Platt/sigmoid** if thin (isotonic overfits small sets). Validate with reliability diagram + ECE (before/after). **Calibrate the composed probability too** — multiplying sub-probabilities can drift.

**Files / modules.**
```
src/models/outcome_lgbm.py    # LightGBM + monotonic constraints + early stopping
src/models/bayesian.py        # PyMC hierarchy -> shooter/keeper profiles + CIs
src/models/placement.py       # multiclass zone model (or Dirichlet posterior)
src/models/dive.py            # multiclass dive model (data-permitting)
src/models/resolution.py      # P(goal | zone, dive, segment) table
src/models/blender.py         # meta-blend of composed + direct
src/models/calibration.py     # isotonic/Platt + reliability/ECE
conf/params.yaml              # hyperparameters + model_version
```

**Common mistakes.**
- **Over-tuning** a giant hyperparameter grid on tiny data — that's overfitting the validation set. Keep the search small and sensible (Optuna, few params).
- **Deep trees / no regularization / no monotonic constraints** → memorization.
- **Calibrating on the training data** → meaningless calibration. Use a held-out fold.
- **Skipping calibration of the composed probability** → the headline and the breakdown disagree.

**Done looks like.** A LightGBM model that **beats the shrunk-marginal baseline** on temporal CV, hierarchical profiles + CIs exported, a working composition that produces headline + heatmap + dive, calibrated probabilities (reliability near-diagonal), and all artifacts saved with a `model_version`.

---

## Step 8 — Evaluate the model properly

**Do.** Run the full evaluation that turns "a model" into "a defensible result." This is where the research credibility lives.

- **Metrics:** the Step-6 set (log loss, Brier, BSS, AUC, ECE) for outcome; **multiclass log loss + top-1/top-2 + macro-F1** for placement/dive; **Spearman/Kendall** of predicted vs. realized conversion for ranking sanity.
- **Generalization to unseen players/keepers:** run **leave-players-out** and compare the full model vs. a **priors-only** model on those unseen players. Report metric **bucketed by `n_pens`** (0, 1–3, 4–10, 11+) to show *where* personal signal actually helps. Run **leave-competition-out** for cross-league transfer. **Don't average the split types into one number** — temporal is the headline; grouped splits back the generalization claims.
- **Leakage check:** the automated leakage test must pass; treat a suspiciously high AUC as a red flag to investigate, not celebrate.
- **Ablations (toggle one group, report ΔBrier/Δlog-loss/ΔECE):** − matchup-interaction (does the "matchup" claim hold?), − pressure, − keeper (shooter-only), − shooter-tendency (rate-only), − form/streaks (likely small — report honestly), shrinkage on vs. off (off should be worse on sparse players), zoning 3 vs. 6 vs. 9.
- **What "good" looks like:** a **modest but consistent** Brier/log-loss improvement over the shrunk-marginal baseline with **good calibration**, plus a clean monotone ablation story. On data this noisy that is a *strong* result — frame it that way rather than chasing an implausible AUC.

**Files / modules.**
```
src/evaluation/ablation.py     # toggles feature groups, runs the suite
src/evaluation/generalization.py  # leave-players-out / leave-competition-out
reports/results_table.md       # baselines vs models × all metrics (w/ CIs)
reports/ablation_table.md
reports/figures/               # reliability diagram, calibration, metric-vs-n_pens
```

**Common mistakes.**
- **One mushy averaged number** across split types — keep them separate and labeled.
- **Reporting only AUC** (ignores calibration, which is the product's whole value).
- **No ablation** → you can't substantiate the "matchup model, not xG" claim.
- **Evaluating placement with accuracy only** — use top-2 and macro-F1 (zones are genuinely close).

**Done looks like.** A complete results table (baselines vs. {logistic, LightGBM, CatBoost/XGBoost cross-checks, composed, blended} × all metrics with CIs), an ablation table, a generalization analysis with metric-vs-`n_pens`, and the calibration/reliability figures — all committed. This is the spine of your write-up.

---

## Step 9 — Add explainability

**Do.** Make every prediction justify itself in plain language, grounded in real model internals.

- **Methods:** **SHAP (TreeExplainer on LightGBM)** for per-prediction attributions; **global SHAP summary** for the "About the model" page; **Bayesian posteriors** for interpretable effect sizes with uncertainty; **rule-based explanation cards** that *translate* SHAP/posterior values into sentences (the cards are grounded in stored numbers, never free-generated).
- **What the explanation panel shows:** headline probability + range; a shooter line (conversion, sample size, favored zone); a keeper line (save rate, sample size, dive tendency); a matchup line ("+X% vs. an average keeper" because the keeper's favored side is opposite the shooter's favored corner); a context line (stakes); and a **confidence** statement tied to sample size.
- **Honesty rules (enforce in code):** never present an **imputed/prior** value as personal ("typical for players like X," not "X does Y"); always show uncertainty and **widen it visibly when support is low**; **no decimals** in the headline (false precision misleads); attribute effects, never predict destiny.

**Example card (well-sampled):**
> **Predicted 81% goal** (74–86%). Shooter converts **84%** over **22** pens, favors the **bottom-left** (41%). Keeper saves **19%** over **31** faced, tends to **dive right early** (54%). The keeper's favored side is **opposite** the shooter's favored corner → **+4% vs. an average keeper**. Shootout, kick 4, level (routine). **Confidence: high.**

**Example card (low-data):**
> **Predicted 76% goal** (61–88%). ⚠️ **Low data:** only **3** recorded penalties, so this leans on typical right-footed forwards. Treat the placement map as indicative, not personal.

**Files / modules.**
```
src/explain/shap_wrap.py    # TreeExplainer wrapper -> per-feature contributions
src/explain/cards.py        # SHAP/posteriors -> templated, honesty-aware sentences
tests/test_explanation_honesty.py  # asserts imputed values are never phrased as personal
```

**Common mistakes.**
- **Free-text/LLM explanations not grounded in the model** → confident-sounding fiction. Template from stored SHAP/posterior values.
- **Presenting prior-based numbers as personal facts** → misleading. Gate on `*_is_imputed`.
- **Hiding uncertainty** to look more impressive → the opposite of what reviewers reward.

**Done looks like.** Explanation cards generated for a set of sample matchups (well-sampled and low-data), a **faithfulness spot-check** (SHAP attributions track leave-one-feature-out deltas), and an honesty test that fails if an imputed value is ever phrased as personal.

---

## Step 10 — Design the dashboard

**Do.** Build a searchable "shooter vs. keeper" interface that returns a result instantly and shows uncertainty everywhere. (Full UX in Blueprint §9.)

- **The interaction:** two prominent autocomplete boxes (**Shooter**, **Keeper**) + a collapsible **Context** strip (competition, home/away, shootout toggle → kick index & running score, pressure auto-derived). Auto-predict when both are selected.
- **Above the fold on a matchup:** the **headline probability** (large, with a **visible range bar** and a **confidence chip**); the **shooter goal-mouth heatmap** (6-zone, most-likely zone highlighted); the **keeper dive panel** (same goal-mouth, arrows/shading, most-likely dive highlighted); a **one-line verdict**; and the **expandable explanation card**. Plus two comparison cards (shooter/keeper recent form).
- **Pages:** Matchup (home) · Player profile · Keeper profile · Leaderboards (optional: clutch / unpredictable / hardest-to-beat, with CIs + min-sample filters) · About/Methodology (model card + global SHAP + reliability diagram).
- **Showing uncertainty (the differentiator):** range bars on every probability; confidence chips driven by `n_pens`/`n_faced`; **greyed/hatched** styling for prior-based values; heatmap **opacity scaled by support**; explicit ⚠️ low-data banners.
- **Visualizations:** the **goal-mouth grid heatmap** (signature, reused everywhere); dive direction as a mirrored heatmap or fan/arrow diagram; probability as a **range bar** (point + whiskers); history as sparklines / outcome-colored strip plots; an overlaid "where he aims vs. where keeper goes" mini-graphic to make the edge intuitive. Avoid 3D, pie charts for zones, and dense tables where a heatmap is faster.

**Files / modules.**
```
frontend/  (React + Vite + TypeScript + Tailwind)
  src/components/SearchBox.tsx       # autocomplete (shooter/keeper)
  src/components/GoalMouthHeatmap.tsx# the signature 6-zone SVG (port src/viz/goalmouth.py)
  src/components/DivePanel.tsx
  src/components/ProbabilityBar.tsx  # point + CI whiskers + confidence chip
  src/components/ExplanationCard.tsx
  src/pages/Matchup.tsx  PlayerProfile.tsx  KeeperProfile.tsx  About.tsx
  src/lib/api.ts                     # typed client for the backend
```

**Common mistakes.**
- **Querying raw data / aggregating at request time** → slow UI. Read precomputed profiles only.
- **Decimals in the headline** and **no uncertainty** → false precision; the opposite of credible.
- **Pie charts for zones / 3D** → harder to read than the heatmap.
- Building polish before the data/model is right → polish a wrong number and you've wasted the most time-consuming step.

**Done looks like.** A working UI where selecting both players renders headline + heatmap + dive panel + explanation in <1s, with uncertainty visible throughout and low-data states clearly flagged. Screenshot-ready.

---

## Step 11 — Build the backend

**Do.** Stand up a thin FastAPI service that reads precomputed profiles and composes predictions live. (Full architecture + endpoint specs in Blueprint §10.)

- **Endpoints:**
  ```
  GET  /api/players?search=&role=shooter|keeper   -> autocomplete
  GET  /api/shooter/{id}  ·  GET /api/keeper/{id}  -> profiles (+ CIs)
  POST /api/predict  {shooter_id, keeper_id, context} ->
        {p_goal, ci_low, ci_high, confidence, zone_probs, most_likely_zone,
         dive_probs, most_likely_dive, interaction_edge_pct, explanation, model_version}
  GET  /api/matchup/{sid}/{kid}  ·  GET /api/leaderboard  ·  GET /api/health, /api/model-card
  ```
- **Stored in the DB vs. computed live:**
  - **Stored (precomputed offline):** `dim_players`, `shooter_profiles`, `keeper_profiles` (+ CIs, vectors), `resolution_table`, autocomplete index, optional `matchup_cache`. These change only on retrain.
  - **Computed live (cheap):** the composition (zone⊗dive⊗resolution), meta-blend, calibration mapping, and the SHAP explanation for that feature vector. Sub-second, no GPU.
- **Framework & layering:** FastAPI + Pydantic + Uvicorn; SQLAlchemy (+ Alembic) over **SQLite/DuckDB (dev) → PostgreSQL (prod)**. Keep `api/` thin — it calls a **pure, importable `serving/predict.py`** so the prediction logic is unit-testable and reusable in batch. Dependency direction: `api/ → serving/ → models/ + features/`, never upward.

**Files / modules.**
```
src/serving/predict.py    # pure composition + blend + explanation assembly
src/api/main.py           # FastAPI app + routers
src/api/schemas.py        # Pydantic request/response models
src/api/db.py             # SQLAlchemy models + session
pipelines/export_profiles.py  # loads profiles + resolution into the DB; bumps model_version
tests/test_api.py  tests/test_serving.py
```

**Common mistakes.**
- **A fat API that aggregates raw data per request** → slow and fragile. Precompute offline; serve reads.
- **No input validation** (e.g., kick_index without is_shootout) → garbage predictions. Validate with Pydantic.
- **500-ing on unknown players** → return a structured fallback (predict from prior, `fallback_used: true`).
- **`model_version` not embedded** → UI/cache drift after retrain.

**Done looks like.** A running API that serves `/players` and `/predict` in <1s, validates input, returns structured fallbacks for unknown players, embeds `model_version`, and is covered by serving + API tests. The frontend talks to it end-to-end.

---

## Step 12 — Train, version, and retrain the model

**Do.** Make the whole pipeline reproducible and tracked so results are defensible and updates are painless.

- **Train the first version:** orchestrate the stages as a single pipeline (`dvc repro`): build dataset → clean/resolve → **leak-safe** features (+ leakage test) → fit Bayesian profiles → train LightGBM (+ placement/dive) → calibrate → evaluate → export artifacts → publish profiles to DB.
- **Save models & datasets:** DVC-track raw snapshots + processed table (tag `data-vYYYY-MM`); DVC/MLflow-track model artifacts (LightGBM, calibrator, profiles, resolution table) with a `model_version` embedded in every API response and in `matchup_cache`.
- **Version experiments:** log every run to **MLflow** (or W&B free tier) — params, all Step-8 metrics, calibration plots, ablation deltas, the artifact — tagged by hypothesis ("added entropy features") so your results table is a query over runs.
- **Retraining cadence:** **quarterly**, or on-demand after a tournament adds a batch of shootouts. Nightly retraining adds nothing — the data barely moves. Automate with a scheduled `dvc repro` if deployed.
- **Updating over time:** on retrain, bump `model_version`, re-export profiles, and **invalidate `matchup_cache`**; keep the environment locked (`uv`/lockfile) so any run reproduces.

**Files / modules.**
```
dvc.yaml                  # the pipeline DAG
pipelines/train.py        # train + calibrate + evaluate + export
conf/params.yaml          # hyperparameters + model_version (single source of truth)
src/tracking/mlflow_utils.py
pyproject.toml / uv.lock  # locked environment
```

**Common mistakes.**
- **Untracked experiments** → "which settings produced that number?" is unanswerable. Track everything.
- **Cache not invalidated on retrain** → users see stale predictions from an old model.
- **Unlocked environment** → "works on my machine," not reproducible for a reviewer.
- **Retraining too often** → wasted effort and churn with no signal gain.

**Done looks like.** `dvc repro` rebuilds the project end-to-end; every experiment is logged in MLflow; artifacts and datasets are versioned and tagged; the API reports `model_version`; and a retrain cleanly bumps the version and refreshes the cache.

---

## Step 13 — Plan the future video upgrade

**Do.** You won't build video now — but you **install the seams now** so adding it later is a join, not a rewrite. (Full roadmap in Blueprint §13.)

- **What video adds that tabular can't:** pre-contact "tells" — run-up speed/angle, **hip/shoulder orientation**, **plant-foot angle**, gaze, last-stride deceleration; **keeper pre-movement / weight-shift / commit-time**; and **true continuous (x,y) ball end-coordinates** + speed/spin. It also lets you de-confound "did the keeper guess vs. react."
- **CV features to extract later:** pose keypoints (shooter & keeper) → joint/hip/shoulder angles at contact; ball tracking → trajectory, end-coordinate, speed; keeper tracking → dive vector + onset frame + reaction latency; scene homography → pixel→metric mapping.
- **Tooling (later):** MediaPipe / MMPose / ViTPose (pose), YOLO + ByteTrack/DeepSORT (detection/tracking), TrackNet-style for the ball, OpenCV homography; one modern GPU (Colab/Kaggle free tier) suffices for a prototype on a few hundred clips.
- **Seams to put in the non-video code today (cheap, high-credit):**
  1. A **`FeatureProvider` interface** (`build(penalty_ids) -> DataFrame`) — tabular is one provider; video becomes another. The feature assembler concatenates all available providers. *(You created this stub in Step 5.)*
  2. An **empty `video_features` table** keyed by `penalty_id` (pose/ball/keeper-timing columns) — so enriching a kick later is a join.
  3. **Keep raw `(x,y)`** from day one so the placement model can switch from 6-zone classification to (x,y) **regression** without interface changes.
  4. **Modality-agnostic models** — they consume a feature dict; missing video features are just `null` + `video_is_missing=true` (same missingness machinery as everywhere else).
  5. The **API and dashboard don't change** — richer inputs just tighten the numbers and may add explanation lines ("body shape suggests a left-side kick").

**Files / modules (stubs now).**
```
src/features/providers.py     # FeatureProvider base + TabularProvider (+ VideoProvider stub)
src/api/db.py                 # add empty video_features table to the schema
docs/VIDEO_ROADMAP.md         # what the CV phase will add + the seams it relies on
```

**Common mistakes.**
- **Not leaving the seam now** → a painful rewrite later. The interface + empty table cost almost nothing today.
- **Assuming clips are easy to align** to specific penalties — they're not (per-stadium camera calibration, clip-to-event matching). Plan a small curated proof-of-concept first.

**Done looks like.** The `FeatureProvider` interface, the empty `video_features` table, retained raw `(x,y)`, and a short `VIDEO_ROADMAP.md` — proof to any reviewer that the system was designed to grow into video without re-architecture.

---

## Step 14 — The exact implementation order

This is the **critical path**. Do it in this order; each chunk ends in something you can show. Earlier chunks unblock later ones, so resist jumping ahead.

### Phase A — Foundations (Steps 1–3) · ~1 week
1. **Repo + environment + spec.** Create the folder structure (Step 3), `uv`/lockfile, `docs/PROJECT_SPEC.md` + `SUCCESS_CRITERIA.md` (Step 1). Init Git + DVC.
2. **StatsBomb ingestion only.** `src/ingestion/statsbomb.py` → canonical rows; snapshot raw (Step 2). One source, working end-to-end.
3. **Clean + canonical map + processed dataset.** `validate.py`, `zoning.py`, `name_map.py` → `penalties.parquet`; run the **coverage report** (Steps 2–3).
   - *Demo checkpoint:* a clean, validated `penalties.parquet` + a coverage report you can describe.

### Phase B — The honest core model (Steps 4–7) · ~1.5–2 weeks
4. **EDA + sparsity/coverage memo** (Step 4) — and reuse the goal-mouth plot later.
5. **Shrinkage utils → as-of features → leakage test.** `shrinkage.py`, `as_of.py`, `build.py`, and **`tests/test_no_leakage.py` passing** (Step 5). *Do not proceed until this test passes.*
6. **Splitters + metrics + baselines.** `splitters.py`, `metrics.py`, `baselines.py` → committed baseline table (Step 6). This is your floor.
7. **LightGBM outcome model + calibration**, beating the shrunk-marginal baseline (Step 7, the headline first). Add CatBoost/XGBoost only as cross-checks.
   - *Demo checkpoint:* a calibrated headline goal probability that beats baselines on a temporal split — **this is already a real project.**

### Phase C — Thin end-to-end product (Steps 11 → 10, minimal) · ~1.5 weeks
8. **Profiles → DB + serving function + minimal API.** Fit Bayesian profiles (or empirical-Bayes for speed), `export_profiles.py`, `serving/predict.py`, FastAPI `/players` + `/predict` (Step 11).
9. **Minimal dashboard.** Two search boxes + headline probability + range bar (Step 10, stripped down). Wire it to the API.
   - *Demo checkpoint:* **search any shooter vs. keeper → calibrated probability with uncertainty, in the browser.** The fastest path to an impressive demo ends here.

### Phase D — The signature features (Steps 7→3, 9, 10) · ~1.5–2 weeks
10. **Placement + dive sub-models + resolution table + composition** (Step 7) → real `zone_probs`/`dive_probs`.
11. **Goal-mouth heatmap + dive panel** in the UI (Step 10) — the highest wow-per-hour visuals. Build the heatmap first.
12. **SHAP explanation cards** with low-data honesty (Step 9), shown in the matchup panel.
    - *Demo checkpoint:* full matchup view — probability + heatmap + dive panel + honest explanation + visible uncertainty.

### Phase E — Research credibility + polish (Steps 8, 12, 13) · ~1.5 weeks
13. **Full evaluation:** generalization (leave-players-out / leave-competition-out), ablations, metric-vs-`n_pens`, calibration figures (Step 8). Produce the results + ablation tables.
14. **Reproducibility + tracking:** `dvc repro` pipeline, MLflow logging, versioned artifacts, `MODEL_CARD.md` (Step 12).
15. **Video seams + polish:** `FeatureProvider` + empty `video_features` table + `VIDEO_ROADMAP.md` (Step 13); profile/leaderboard pages, About/methodology page, deployment if desired.
    - *Demo checkpoint:* deployable product + a write-up/poster with results, ablations, and a reliability diagram.

### Fastest path to a strong demo (if you must cut)
**Chunks 1 → 9** (Phases A, B, and the minimal Phase C) get you to *"search any shooter vs. keeper → calibrated probability with uncertainty"* — a credible, defensible demo. Then add the **goal-mouth heatmap (chunk 11) first**, because it's the single most impressive visual per hour of work and what makes screenshots compelling.

---

## Pre-launch checklist

Everything that must be true before you call it done and show it.

**Data & cleaning**
- [ ] Canonical `penalties.parquet` validates against schema; DVC-tracked and tagged.
- [ ] Coverage report exists; `shot_zone` and `keeper_dive` coverage acknowledged in the model card.
- [ ] Player/keeper map reviewed; `unresolved.csv` triaged; **no name-only auto-merges**.
- [ ] Retakes handled; cross-source conflicts reviewed, not guessed.

**Modeling & evaluation**
- [ ] **`tests/test_no_leakage.py` passing**; features confirmed as-of-time.
- [ ] Baselines committed; the advanced model **beats shrunk-marginal baselines** on log loss/Brier across folds (with CIs).
- [ ] Probabilities **calibrated**; reliability diagram near-diagonal; ECE recorded before/after; composed probability calibrated too.
- [ ] Placement/dive evaluated (top-2, macro-F1); low-support keepers fall back to prior + flagged.
- [ ] Ablation table complete; generalization (leave-players-out, leave-competition-out) reported with metric-vs-`n_pens`.

**Explainability**
- [ ] Explanation cards grounded in SHAP/posteriors; **faithfulness spot-check** passes.
- [ ] Honesty test passes: **imputed values are never phrased as personal**; headline shows no false-precision decimals.

**Product & backend**
- [ ] Dashboard returns a matchup in <1s; both searches work; heatmap + dive panel + explanation render.
- [ ] Uncertainty visible everywhere; ⚠️ low-data banners working; opacity scaled by support.
- [ ] API validates input and returns structured fallbacks for unknown players (`fallback_used: true`).
- [ ] `model_version` embedded in responses and cache; cache invalidation on retrain verified.

**Reproducibility & docs**
- [ ] `dvc repro` rebuilds end-to-end; experiments logged in MLflow; environment locked.
- [ ] `README.md` (how to run) and `MODEL_CARD.md` (data, assumptions, limitations, biases) written.

**Future-proofing**
- [ ] **Video seam present:** `FeatureProvider` interface + empty `video_features` table + retained raw `(x,y)` + `VIDEO_ROADMAP.md`.

---

### One-paragraph framing for your application/write-up
Lead with the three honest strengths, because they're exactly what serious reviewers reward: it's a **matchup model, not an xG model** (the interaction is first-class); it's **statistically disciplined** (shrinkage + calibration + leakage-controlled, grouped evaluation, with uncertainty shown to users); and it's **explainable and product-complete**. A *modest, well-calibrated* improvement over strong baselines — presented with clear ablations and visible uncertainty — reads as more credible and more mature than an implausibly high accuracy number, and it's the version of this project that survives expert questions.
