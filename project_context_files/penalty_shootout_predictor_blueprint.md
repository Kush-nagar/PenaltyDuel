# Penalty Shootout Predictor — Full Project Blueprint

*A research-grade, explainable matchup prediction system for football/soccer penalties. Non-video version is the first deliverable; a computer-vision upgrade path is designed in from day one.*

---

## How to read this document

This is a build blueprint, not an overview. Every section is written so you can open VS Code and start implementing. Where a common approach is weak for *this specific problem*, I say so and give the better alternative. Assumptions are marked **[Assumption]**. Things that are genuinely hard or under-supported by free data are marked **[Reality check]** so you don't discover them three weeks in.

The single most important idea in the whole project, stated up front so everything else makes sense:

> **A penalty is not one prediction. It is two simultaneous decisions (where the shooter aims, where the keeper goes) plus a physics-of-the-duel resolution (does it go in given those two choices).** Model those three things separately and combine them. This is what makes the project more than an xG model, and it is what produces the placement heatmap and keeper-dive panel your dashboard needs.

---

## 1) Executive summary

**The project in one paragraph.** This system predicts the outcome of a one-on-one penalty kick for any specified shooter-versus-goalkeeper matchup. A user searches a shooter and a keeper (optionally adding context such as competition, score state, and kick number in a shootout), and the system returns: the probability the penalty is scored, a calibrated uncertainty band on that probability, a heatmap of where the shooter is likely to aim, a panel of where the keeper is likely to dive, the single most-likely zone-versus-dive interaction, and a plain-language explanation of *why* the model predicted what it did. Under the hood it is a hierarchical, probabilistic matchup model: it decomposes the kick into shooter placement tendencies, keeper movement tendencies, and a learned resolution of goal probability conditional on both, with shrinkage toward sensible priors so that players with only a handful of career penalties still get reasonable predictions.

**Why this is strong and unique.** Almost every student "penalty" or "xG" project predicts a single scalar (goal probability) from a flat feature table and stops. This project is differentiated on four axes that admissions readers, research-fair judges, and portfolio reviewers actually reward: (1) it is a *matchup* model — it explicitly represents the interaction between a specific shooter and a specific keeper, not just two independent skill ratings; (2) it is *multi-output* — it predicts placement and dive direction, which is both harder and far more visually compelling; (3) it is *statistically honest about sparsity* — penalties are rare per player, so it uses hierarchical Bayesian shrinkage and reports uncertainty rather than pretending a 4-penalty sample is a stable rate; and (4) it is *explainable by construction*, so the dashboard can justify every number in language a non-expert understands. That combination — interaction modeling + multi-output + proper uncertainty + explainability + a polished product — is publication-adjacent and is rarely executed well at the student level.

**Non-video vs. future video version.** The **non-video version** (the first deliverable) learns from *tabular event data*: historical penalty records, shot end-locations where available, keeper save records, and match/competition context. It can model *tendencies and outcomes* extremely well ("this shooter scores 84% and favors the bottom-left; this keeper saves 21% and commits early to his right"), but it cannot see the kick itself, so it cannot use the body-language and run-up cues that elite analysts use. The **future video version** adds a computer-vision pipeline that extracts run-up speed, hip and shoulder orientation, plant-foot angle, gaze, keeper weight-shift and pre-movement, and true ball trajectory from broadcast or training footage. Video adds *signal the non-video version structurally cannot access* — the moment-to-moment "tells" that precede contact — and it converts coarse zone labels into continuous (x, y) end-coordinates. Crucially (see §13), the non-video architecture is designed so video features are added as a new feature group behind the same interface, without rewriting the models, API, or dashboard.

---

## 2) Problem formulation

**What the system predicts.** Given a shooter *s*, a keeper *k*, and a context vector *c* (competition, home/away, score differential, shootout kick index, fatigue proxies, etc.), the system predicts a joint description of the kick:

- **Outcome** `Y_goal ∈ {scored, not_scored}` — the headline number. (A finer-grained version is `{scored, saved, missed/post}` because "not scored" conflates a keeper save with a shooter skying it, and those have different causes and different explanations.)
- **Placement** `Y_zone` — a categorical over goal-mouth zones (where the *shot ends up*).
- **Keeper action** `Y_dive` — a categorical over keeper movement (direction, and ideally height).

**Subproblems.** Decompose into three estimators that are trained and reasoned about separately, then composed:

1. **Placement model** — `P(Y_zone | s, c)`: where this shooter tends to put penalties, shrunk toward foot/position/league priors.
2. **Dive model** — `P(Y_dive | k, c)`: where this keeper tends to go, shrunk toward keeper-population priors. Optionally conditioned on a *read* of the shooter (some keepers ball-watch and react; in the non-video version this is a weak, aggregate effect).
3. **Resolution model** — `P(Y_goal | Y_zone, Y_dive, s, k, c)`: the physics-and-skill of the duel. Bottom-corner shots with the keeper going the wrong way are ~near-certain goals; the same shot with the keeper going the right way is a coin-flip; central high shots beat a diving keeper but risk the bar. This term carries the *interaction*.

The headline goal probability is then the marginalization

```
P(goal | s, k, c) = Σ_z Σ_d  P(zone=z | s, c) · P(dive=d | k, c) · P(goal | z, d, s, k, c)
```

**[Assumption]** Shot and dive are treated as *conditionally independent given the players and context* — the keeper commits essentially as the ball is struck and cannot fully react to its flight, so the dominant dependence is "does the keeper's chosen side match the shot's side," which lives in the resolution term, not in correlated choices. This is a standard and defensible simplification for the non-video version; the video version relaxes it by modeling keeper reaction latency directly.

**Which ML framing is correct.** Reviewers will ask you to justify this, so be precise:

| Candidate framing | Verdict for this project |
|---|---|
| Pure **binary classification** (goal/no-goal) | Necessary but insufficient. It gives the headline number but no placement, no dive, no interaction — i.e., an xG model. Use it as a *baseline* and as a *sanity check* on the composed probability. |
| **Multiclass classification** | The right tool for the placement and dive sub-models (zones, dive directions). |
| **Regression** | Wrong for outcome (binary), but the natural framing for the *future* continuous (x, y) end-location once video exists. |
| **Bayesian / hierarchical** | The right tool for the *sparsity* problem (few penalties per player) and for producing the uncertainty bands the product promises. Applied as a layer over the above, not as a replacement. |
| **Ranking / learning-to-rank** | Not the primary task, but a useful *evaluation* lens ("does the model rank shooters by true conversion?") and a nice secondary product feature ("most clutch penalty takers"). |

**Recommended formulation — a hybrid, hierarchical, multi-output system.** Use **multiclass models for placement and dive**, a **classification model for the conditional resolution**, **compose them analytically** for the headline probability, and wrap the player-level parameters in a **hierarchical Bayesian shrinkage layer** so sparse players borrow strength from their position/foot/league. This is the formulation that simultaneously (a) produces every output the dashboard needs, (b) is honest about uncertainty, and (c) makes the matchup interaction a first-class modeled quantity rather than an afterthought.

**Alternative considered:** a single end-to-end multi-task neural network with shared embeddings and three heads (zone, dive, outcome). Briefly attractive, but rejected as the *primary* approach for the non-video version because (1) penalty datasets are far too small to fit player embeddings without severe overfitting, and (2) it is much harder to make honestly explainable and well-calibrated. Keep it on the roadmap as an "elite/experimental" comparison once the decomposed system works.

---

## 3) Data requirements

### 3.1 Categories of data, what they represent, why they matter

| Category | What it is | Why it matters | Required? |
|---|---|---|---|
| **Penalty event records** | One row per penalty taken (shootout *and* in-game), with outcome | The core label source. Without this there is no project. | **Required** |
| **Shot end-location / zone** | Where the ball crossed the goal line (zone or (x,y)) | Trains the placement model and the resolution model; powers the heatmap. | **Required for full system; degrade gracefully if absent** |
| **Keeper action** | Dive direction (L/C/R), ideally height; whether keeper guessed correctly | Trains the dive model; powers the keeper panel. | **Optional but high-value [Reality check: scarce in free data]** |
| **Shooter identity & attributes** | Player ID, name aliases, foot, position, age, height | Keys every shooter feature; foot/position drive priors for sparse players. | **Required** |
| **Keeper identity & attributes** | Player ID, name aliases, height, reach, age | Keys every keeper feature; height/reach are weak priors for save ability. | **Required** |
| **Shooter penalty history** | All prior penalties by this shooter (as-of the kick) | Conversion rate, zone tendencies, predictability — computed *as-of-time* to avoid leakage. | **Required** |
| **Keeper penalty history** | All prior penalties faced by this keeper | Save rate, dive tendencies, early/late commitment proxies. | **Required** |
| **Match context** | Competition, season, date, home/away, venue, attendance, stage | Knockout/final pressure, crowd effects, era effects. | **Required (most fields)** |
| **Competition metadata** | League/cup tier, importance weight, country | Lets the model weight a Champions League final differently from a friendly. | **Optional (nice prior)** |
| **Shootout sequence / order** | Kick index (1–5+), running shootout score, sudden death flag, must-score/must-save flag | Pressure is highly structured in shootouts; "must score or lose" kicks convert measurably worse. | **Required for shootout mode** |
| **Score & game state (in-game pens)** | Minute, score differential before the kick, importance | Late equalizers and winners carry different pressure. | **Optional** |
| **Recent form / fatigue proxies** | Minutes played in match, recent penalty results, team form | Streak and fatigue effects. | **Optional** |
| **Referee / retake flags** | Retakes, encroachment, VAR | Data-quality hygiene; ambiguous outcomes. | **Optional (for cleaning)** |

### 3.2 What to do when features are missing

Missingness is the normal case here, not the exception. Adopt a **three-tier policy** for every feature:

1. **Impute with an informative prior, and flag it.** Add a companion boolean `*_is_imputed` for every feature that can be missing. Never silently fill — the model can learn from the missingness flag, and the dashboard can warn the user.
2. **Fall back up the hierarchy.** If a shooter's personal zone distribution is unavailable (too few/zero penalties), fall back to `foot × position` prior → `foot` prior → global prior. Same idea for keepers.
3. **Degrade the output, not the system.** If shot-location data is entirely absent for a region of the data, the system still returns a calibrated headline goal probability; it just labels the placement heatmap as "low-confidence / prior-based" rather than fabricating precision. Build the API so any sub-model can return "insufficient data" without breaking the response.

### 3.3 Ideal training dataset schema (`penalties` fact table)

One row per penalty. Aliases/IDs resolved during ingestion (see §4.3).

| Column | Type | Description / notes |
|---|---|---|
| `penalty_id` | str (PK) | Stable surrogate key (hash of source + source_event_id). |
| `source` | enum | `statsbomb`, `understat`, `fbref`, `kaggle_xxx`, `manual`. Track provenance always. |
| `date` | date | Kick date. Drives temporal splits and as-of feature windows. |
| `season` | str | e.g., `2023/2024`. |
| `competition_id` | str (FK) | Joins to `competitions`. |
| `stage` | enum | `group`, `r16`, `qf`, `sf`, `final`, `league`, `friendly`. |
| `is_shootout` | bool | Shootout vs. in-run-of-play penalty. |
| `shootout_kick_index` | int (nullable) | 1..N within the shootout. |
| `shootout_score_for` / `_against` | int (nullable) | Running shootout tally *before* this kick. |
| `is_sudden_death` | bool (nullable) | Past the first five. |
| `must_score` / `must_not_concede` | bool (nullable) | Kick decides the tie if missed/scored. High-pressure flag. |
| `match_minute` | int (nullable) | For in-game penalties. |
| `score_diff_before` | int (nullable) | Taker's team score minus opponent, before the kick. |
| `home_away` | enum | `home`, `away`, `neutral`. |
| `shooter_id` | str (FK) | Resolved player ID. |
| `shooter_foot` | enum | `left`, `right`, `unknown`. |
| `shooter_position` | enum | `FW`, `MF`, `DF`, `GK`. |
| `keeper_id` | str (FK) | Resolved keeper ID. |
| `keeper_height_cm` | int (nullable) | Weak prior on reach. |
| `outcome_3` | enum | `scored`, `saved`, `missed` (off-target/woodwork). **Primary fine label.** |
| `outcome_bin` | bool | Derived: `scored` vs. not. **Headline label.** |
| `shot_zone` | enum (nullable) | Discretized end-location (see zoning below). |
| `shot_x` / `shot_y` | float (nullable) | Continuous end-location if source provides it (Understat/StatsBomb). |
| `keeper_dive` | enum (nullable) | `left`, `center`, `right` (× `low`/`high` if available). |
| `keeper_guessed_side` | bool (nullable) | Derived: dive side == shot side. |
| `foot_relative_zone` | enum (nullable) | Zone re-expressed as "natural side / open side" relative to shooter's foot — see §5. |
| `retake_flag` | bool | Exclude or special-case in cleaning. |
| `ingested_at` | timestamp | For dataset versioning. |

**Goal-mouth zoning [Assumption].** Use a **6-zone grid** for v1: columns {left, center, right} × rows {low, high}, from the *keeper's* perspective, with "left/right" later re-expressed relative to the shooter's strong foot. Six zones balance signal against sparsity (9 zones fragment tiny samples; 3 zones throw away the high/low distinction that drives whether a correctly-guessing keeper actually saves it). Keep the raw (x, y) when available so you can re-bin later without re-collecting.

### 3.4 Live lookup database schema (serves the dashboard)

The dashboard does **not** query the raw fact table at request time. It queries precomputed, denormalized *profile* tables plus small lookup tables. Predictions are composed on the fly from these (cheap matrix math); the heavy aggregation is done offline in the training/feature pipeline.

**`dim_players`** (autocomplete + profile)

| Column | Type | Notes |
|---|---|---|
| `player_id` | str (PK) | |
| `display_name` | str | Canonical name. |
| `aliases` | str[] | For fuzzy search. |
| `foot`, `position`, `height_cm`, `team`, `nationality`, `photo_url` | — | Profile chrome. |
| `is_keeper` | bool | Filters the two search boxes. |

**`shooter_profiles`** (one row per shooter, precomputed)

| Column | Type | Notes |
|---|---|---|
| `player_id` | str (PK/FK) | |
| `n_pens` | int | Drives the "low-data" warning. |
| `conv_rate_raw` | float | Empirical conversion. |
| `conv_rate_shrunk` | float | Posterior mean after shrinkage. |
| `conv_rate_ci_low/high` | float | 90% credible interval. |
| `zone_probs` | json | 6-vector, shrunk. |
| `zone_probs_ci` | json | Per-zone interval. |
| `predictability_entropy` | float | Shannon entropy of zone dist (low = predictable). |
| `under_pressure_delta` | float (nullable) | Conversion in high-pressure vs. baseline. |
| `last_updated` | timestamp | |

**`keeper_profiles`** (one row per keeper, precomputed)

| Column | Type | Notes |
|---|---|---|
| `player_id` | str (PK/FK) | |
| `n_pens_faced` | int | |
| `save_rate_raw` / `save_rate_shrunk` / `_ci_low/high` | float | |
| `dive_probs` | json | 3- or 6-vector, shrunk. |
| `dive_probs_ci` | json | |
| `guess_correct_rate` | float | How often the keeper picks the right side. |
| `early_commit_proxy` | float (nullable) | If derivable. |
| `last_updated` | timestamp | |

**`resolution_table`** (the duel physics — small, global, optionally segmented by foot/era)

| Column | Type | Notes |
|---|---|---|
| `shot_zone` | enum | |
| `keeper_dive` | enum | |
| `segment` | enum | e.g., `all`, `left_foot`, `right_foot`. |
| `p_goal` | float | Smoothed P(goal | zone, dive, segment). |
| `n` | int | Support, for trust. |

**`matchup_cache`** (optional, for popular pairs)

| Column | Type | Notes |
|---|---|---|
| `shooter_id`, `keeper_id`, `context_hash` | composite PK | |
| `p_goal`, `ci_low`, `ci_high`, `zone_probs`, `dive_probs`, `explanation_json` | — | Cached prediction. |
| `model_version`, `computed_at` | — | Invalidate on retrain. |

---

## 4) Data sources

### 4.1 The realistic free core

| Source | What you get | Penalty-relevant? | Cost | How to access |
|---|---|---|---|---|
| **StatsBomb Open Data** (GitHub) | Event data incl. shots with **(x, y) locations** and **freeze-frames** (positions of players incl. keeper at the shot) for a curated set of competitions (World Cups men's/women's, some leagues, historical comps). Includes penalties and some shootouts. | **Yes — the crown jewel.** Shot location is present; keeper *position at shot* is partially inferable from freeze-frames. | **Free** (non-commercial license) | Clone the repo; parse JSON, or use the `statsbombpy` Python package. |
| **Understat** | Shot-level data with **(x, y)**, xG, result, for the big-5 European leagues + a few others, multiple seasons. Penalties are flagged. | **Yes** — great for shot locations at scale, but **no keeper dive direction**. | **Free** (scraping; be polite) | `understatr`-style scraping or community Python wrappers. |
| **FBref (StatsPerform data)** | Per-player **penalty attempts and conversions**, season aggregates, plus rich player metadata (foot, position, minutes). No shot locations, no dive direction. | **Yes — for histories & priors.** | **Free** (respect ToS; rate-limit) | `worldfootballR` (R) or careful Python scraping. |
| **Transfermarkt** | Player bios (foot, height, position, DOB), some penalty records, transfer/market context. | **Partial** — identities and attributes; thin on event detail. | **Free** (scraping) | Community scrapers; rate-limit hard. |
| **Kaggle penalty datasets** | Several community datasets exist with **placement and keeper-direction labels** (e.g., World Cup penalty-shootout compilations, hand-coded sets). Quality varies. | **Yes — often the only free source of dive direction.** | **Free** | Kaggle download. |
| **Wikipedia / RSSSF / national archives** | Historical shootout results and orderings. | **Yes — for historical outcome + order**, no locations. | **Free** | Scrape/parse; cite. |

### 4.2 Optional / paid (mark clearly as not required)

| Source | Why you'd want it | Status |
|---|---|---|
| **StatsBomb full / IQ** | Dense event + 360 freeze-frame data, more competitions, richer keeper detail. | **Paid / academic license — optional.** |
| **Opta / StatsPerform**, **Wyscout** | Industry-grade event data incl. detailed keeper actions, shot placement, qualifiers. | **Paid — optional.** |
| **Sky / broadcaster tracking, Second Spectrum** | Tracking data (positions over time). | **Paid — optional; mostly relevant to the video phase.** |

**[Reality check] Keeper dive direction is the data bottleneck, not outcome or placement.** Outcome is everywhere; shot location is available at scale (Understat, StatsBomb); **dive direction is rare in free data.** Plan around this explicitly: (a) source dive labels from StatsBomb freeze-frames (infer the side the keeper has moved to at the shot frame) and from the best Kaggle hand-coded sets; (b) accept that your *dive model* will train on a smaller, noisier corpus than your *outcome model*; (c) make the dashboard's keeper panel honestly label low-support keepers as "tendency uncertain"; (d) treat *precise* dive modeling as the headline win of the future video phase, where you can extract it directly.

### 4.3 Combining sources with inconsistent naming / missing fields

This is real engineering work; budget for it.

- **Entity resolution (player & team names).** Build a `name_map` keyed by `(normalized_name, birth_year?, nationality?, position?)`. Normalize with: lowercase, strip diacritics (`José`→`jose`), unify separators, handle "Last, First" vs "First Last", and known nickname tables (`Cristiano Ronaldo` vs `Ronaldo` vs `CR7`). Use fuzzy matching (RapidFuzz token-set ratio) to *propose* matches, but require a second signal (birth year, club-season overlap, nationality) before auto-accepting. Keep a human-reviewable `unresolved.csv` for the long tail. **Do not trust names alone — there are many "Danny Williams."**
- **Schema harmonization.** Write a thin *adapter* per source that maps its raw fields into the canonical `penalties` schema, emitting `null` + an `*_is_imputed`/`*_is_missing` flag for anything it can't supply. All downstream code sees only the canonical schema.
- **Deduplication.** The same penalty may appear in multiple sources. Dedup on `(date, shooter_id, keeper_id, competition, minute±tolerance)`. When sources disagree on a field, prefer by a **source-trust ranking** (e.g., StatsBomb > Understat > FBref-derived > Kaggle-community), and log conflicts.
- **Provenance & reconciliation.** Keep `source` and `source_event_id` on every row. When two sources give different outcomes, surface it to a review queue rather than silently picking one — outcome label noise directly corrupts training.

### 4.4 If a source lacks shot location or dive direction

- **No shot location:** the penalty still contributes to the **outcome model** and to **conversion-rate priors**. It simply doesn't contribute to the placement model. Tag it `shot_zone = null, shot_zone_is_missing = true`. The shooter's `zone_probs` are then estimated from whatever located penalties exist, shrunk hard toward foot/position priors when located samples are tiny.
- **No dive direction:** the penalty contributes to outcome and placement but **not** to the dive model. The keeper still gets a `save_rate` (outcome-based) even with zero usable dive labels; the dive panel falls back to the keeper-population prior and is labeled low-confidence.
- **General rule:** *every* sub-model is trained on the subset of rows that have the columns it needs. Never drop a row globally for missing a field that only one sub-model uses.

---

## 5) Feature engineering

Design principle: features should be **as-of-the-kick** (computed only from data strictly *before* the kick's date) to prevent temporal leakage (see §7), and should **degrade to a prior** when the underlying sample is thin.

### (a) Shooter skill & tendencies — *"how good and how predictable is this taker?"*
- `shooter_conv_rate_shrunk` (Beta-Binomial posterior mean), `shooter_n_pens`, `shooter_conv_ci_width`.
- `shooter_zone_probs` (6-vector, Dirichlet-shrunk) — the heart of placement.
- `shooter_pref_side_natural` — share to the "natural side" (a right-footer's natural-power side is the keeper's right, low). Re-expressing zones relative to foot makes left- and right-footers comparable and densifies the data.
- `shooter_high_share` — propensity to go high (riskier, but beats diving keepers).
- **Intuition:** a taker's *placement distribution and its concentration* matter as much as raw conversion. Two 80%-takers differ if one is unpredictable and one always hits the same corner.

### (b) Goalkeeper skill & tendencies — *"how good, and where does he go?"*
- `keeper_save_rate_shrunk`, `keeper_n_faced`, `keeper_save_ci_width`.
- `keeper_dive_probs` (3- or 6-vector, Dirichlet-shrunk).
- `keeper_guess_correct_rate` — how often the chosen side matches the shot side (skill at reading).
- `keeper_stay_center_rate` — relevant because central shots punish early divers.
- `keeper_height_cm`, `keeper_reach_proxy` — weak priors when history is thin.
- **Intuition:** save rate alone hides *how* a keeper saves. A keeper who guesses correctly often is dangerous in a way a tall-but-static keeper isn't.

### (c) Matchup interaction features — *the differentiator*
- `side_conflict_score` = alignment between the shooter's preferred side distribution and the keeper's dive distribution (high when the shooter likes the side the keeper tends to cover).
- `entropy_advantage` = keeper's read skill × shooter's predictability (a predictable shooter facing a good guesser is the worst case).
- `expected_guess_correct` = Σ_zones P(shot to zone | shooter) · P(keeper covers that side | keeper) — an *a priori* "will the keeper be there" estimate, fed into both the resolution term and the explanation.
- **Intuition:** this is the feature group that makes it a matchup model. It is also the most leakage-prone and sparsity-prone, so it is *derived from the already-shrunk per-player distributions*, not from raw head-to-head counts (most shooter-keeper pairs have **zero** prior meetings).

### (d) Match context — *situational baseline shifts*
- `competition_importance_weight`, `stage` (one-hot), `is_neutral_venue`, `home_away`, `season`/era bucket, `attendance_bucket` (nullable).
- **Intuition:** conversion drifts with era and stage; encode it so the model doesn't attribute a stage effect to a player.

### (e) Psychological / pressure features — *structured, not hand-wavy*
- **Shootout pressure index:** engineered from `must_score`, `must_not_concede`, `shootout_kick_index`, and `shootout_score_diff`. The cleanest single feature is a **"stakes" categorical**: `{routine, advantage, level, must_score_to_survive, can_win_with_score}`. Empirically, "must-score-or-lose" kicks convert worse and "can-win-it" kicks are mixed — let the model learn it rather than asserting a sign.
- **In-game pressure:** `abs(score_diff_before)`, `late_game_flag (minute≥80)`.
- `shooter_under_pressure_delta` (player-specific high-pressure conversion vs. their baseline, shrunk — sparse, so shrink hard).
- **Intuition:** pressure in penalties is unusually *measurable* because the shootout state is fully observed. This is a genuine, defensible feature group, unlike vague "morale."

### (f) Recent form & streaks
- `shooter_last5_pen_results` → `shooter_recent_conv`, `shooter_missed_last_pen` (does a recent miss depress the next?).
- `team_recent_form`, `minutes_played_in_match` (fatigue), `days_rest`.
- **Intuition:** weak signal individually and easy to overfit; include but regularize, and test in an ablation whether it actually helps (§7). Be ready to report "form features did not improve calibration" as an honest finding.

### (g) Randomness / entropy / predictability
- `shooter_zone_entropy` (Shannon entropy of the shrunk zone distribution).
- `keeper_dive_entropy`.
- `mixed_strategy_gap` — distance of the shooter's distribution from the game-theoretic optimal mix given the keeper (a nod to the literature treating penalties as a mixed-strategy equilibrium).
- **Intuition:** an unpredictable shooter is harder to save against *independent of raw skill*. This group is genuinely novel-feeling for a portfolio and ties to real economics/game-theory research on penalties.

### (h) Penalty order & shootout state
- `shootout_kick_index`, `is_sudden_death`, `running_score_state`, `kicks_remaining`, `is_first_kick` (taking first correlates with winning shootouts — a documented effect).
- **Intuition:** sequence position carries real predictive weight; encode it explicitly for the shootout mode of the product.

### Which features matter most for v1 (ship these first)
1. `shooter_conv_rate_shrunk`, `shooter_n_pens`
2. `shooter_zone_probs` (shrunk)
3. `keeper_save_rate_shrunk`, `keeper_n_faced`
4. `keeper_dive_probs` (shrunk)
5. `expected_guess_correct` / `side_conflict_score` (the interaction)
6. `stakes` categorical + `is_shootout`
7. `shooter_foot`, `shooter_position` (prior backbone)

Everything in (f)/(g) beyond entropy is "strong/elite version" — add once the core is calibrated, and *prove* each adds value via ablation rather than assuming.

### Encoding categorical / numerical / time features
- **Categorical:** one-hot for low-cardinality (`stage`, `home_away`, `stakes`). For player identity, do **not** one-hot or embed in v1 — represent players *through their shrunk profile features*, which is the whole point of the hierarchical design and avoids the sparsity catastrophe of per-player parameters.
- **Numerical:** keep rates as probabilities in [0,1]; standardize only for the linear/Bayesian models, not for tree models (trees are scale-invariant). Always pair an imputed value with its `*_is_imputed` flag.
- **Time-based:** bucket `season` into eras to absorb rule/trend drift; compute all history features over an **expanding window up to the kick date**; add `days_since_last_pen` and recency-weighted variants (exponential decay) so a 10-year-old penalty counts less than a recent one.

### Representing players with sparse histories (the core problem)
- **Empirical-Bayes / Beta-Binomial shrinkage** for rates: posterior mean = (α + scored)/(α + β + n), with (α, β) fit from the population (or from the `foot × position` subgroup). A 2-for-2 shooter is *not* a 100% shooter; shrinkage pulls them to ~the prior with wide CIs.
- **Dirichlet-multinomial shrinkage** for zone/dive distributions, with the prior set by the relevant subgroup. A shooter with 3 located penalties gets a distribution that is mostly the foot/position prior with a small personal tilt.
- **Hierarchical fallback chain:** player → (foot × position × league) → (foot × position) → (foot) → global. Use the most specific level with adequate support.
- **Cold start (zero penalties):** predict entirely from the subgroup prior, and have the dashboard say so. This is a *feature* (the system never refuses), not a bug.

---

## 6) Modeling strategy

The pipeline climbs a ladder: each rung is a working system you can demo, and each rung is the *baseline* that the next rung must beat. Never skip to the top — half the research value is the comparison.

### Rung 0 — Naive baselines (a day of work, but you must report them)
- **Global rate:** predict the overall conversion (~the long-run penalty conversion baseline, well known to sit in the high-70s%). This is the "dumb" floor every model must clear.
- **Shooter-only rate** and **keeper-only rate** (raw, then shrunk). The shrunk single-factor models are surprisingly strong and are the *real* bar to beat.
- **Why:** if your fancy model can't beat shrunk marginal rates on log loss and calibration, it isn't earning its complexity. Reviewers love seeing this honesty.

### Rung 1 — Strong tabular model for the headline outcome
- **Algorithm: gradient-boosted trees — LightGBM (primary), XGBoost (cross-check).** Reasons: best-in-class on small/medium heterogeneous tabular data, native handling of missing values, monotonic-constraint support (you can force "higher shooter conversion ⇒ higher predicted goal prob," which prevents nonsensical fits on small data), fast, and SHAP-friendly.
- **Target:** `outcome_bin` (and a 3-class variant for `outcome_3`).
- **Regularization for sparsity:** shallow trees (`max_depth` 3–4), strong `min_child_samples`/`min_data_in_leaf`, heavy `lambda_l1/l2`, low learning rate with early stopping on a temporal validation fold, and **monotonic constraints** on the obviously-monotone features. Prefer fewer, well-engineered (already-shrunk) features over many raw ones.
- **Alternative:** regularized logistic regression as an interpretable secondary — and as a reality check that the trees aren't overfitting (if logistic ≈ trees in log loss, the trees aren't finding much nonlinearity, which on data this small is often the truth).

### Rung 2 — Probabilistic / hierarchical Bayesian layer (the statistical backbone)
- **Tool: PyMC or NumPyro.** Fit a **hierarchical (multilevel) logistic regression** for conversion with **partial pooling** across players: each shooter's log-odds offset is drawn from a foot×position×league hyper-distribution; same for keepers. This *is* the shrinkage from §5, done properly with full posteriors.
- **Placement & dive:** **Dirichlet-multinomial** hierarchical models for `zone_probs` and `dive_probs`, partially pooled the same way.
- **Why Bayesian here specifically:** (1) it gives you *uncertainty for free* — the credible intervals the product promises come straight out of the posterior; (2) partial pooling is the principled answer to "4-penalty player"; (3) it's interpretable (coefficients have meaning), which feeds explainability. This is also the part that elevates the project from "ML app" to "statistically sound research."
- **Pragmatic note:** if full MCMC is too slow/finicky for the live product, fit the hierarchy offline to produce the `*_shrunk` profile features and CIs, store them, and let the fast LightGBM/logistic model consume them at request time. **The Bayesian model produces the priors; the tabular model (or the analytic composition) produces the live answer.** Best of both.

### Rung 3 — The composed matchup model (the product's actual prediction)
Combine the three sub-models analytically (the formula in §2):
```
P(goal) = Σ_z Σ_d  P(zone=z | shooter)·P(dive=d | keeper)·P(goal | z, d, segment)
```
- `P(zone | shooter)` from the shooter placement model; `P(dive | keeper)` from the keeper dive model; `P(goal | z, d, segment)` from the **`resolution_table`** (empirical, smoothed, optionally split by foot — e.g., low-corner + keeper-wrong-way ≈ near-1; low-corner + keeper-right-way ≈ coin-flip; high-center + keeper-dives ≈ high; high-anywhere risks the miss).
- **Reconcile two estimates of the headline number:** the *composed* probability (interpretable, drives the panels) and the *direct* LightGBM probability (often slightly better calibrated). Strategy: use a **stacked/blended** final probability (a tiny logistic meta-model over the two), and show the user the composed one for the visual breakdown. Report both in evaluation; if they disagree a lot for a matchup, that itself is a useful uncertainty signal.

### How the three predictions combine — summary
- **Shooter model →** placement heatmap + zone vector.
- **Keeper model →** dive panel + dive vector.
- **Resolution table →** turns (zone vector ⊗ dive vector) into the headline goal probability and the "most likely zone vs. most likely dive" call-out.
- **Meta-blender →** the single calibrated number shown big at the top.

### Avoiding overfitting on sparse penalty data (consolidated checklist)
- Represent players via **shrunk profile features**, never per-player one-hots/embeddings (v1).
- **Partial pooling** for all player/keeper effects.
- **Monotonic constraints** on obviously-monotone features.
- Shallow trees, strong leaf minimums, L1/L2, early stopping on a **temporal** fold.
- **Nested / grouped cross-validation** (group = player) so a player can't appear in train and validation simultaneously.
- Keep the live feature set small and meaningful; prove additions help via ablation.
- Recency-weight history; bucket eras to avoid spurious trend-fitting.

### Class imbalance & small samples
- Penalties are ~78% scored — imbalanced but not extreme. **Do not SMOTE / oversample;** it corrupts probability calibration, which is the whole point of this product. Instead: optimize **log loss / Brier** directly (proper scoring rules), use `scale_pos_weight`/`is_unbalance` *only if* it improves validation log loss (usually it doesn't here), and rely on calibration (below) rather than resampling.

### Probability calibration (non-negotiable for a probability product)
- After training, fit a calibrator on a held-out fold: **isotonic regression** if you have enough data, **Platt/sigmoid** if data is thin (isotonic overfits on small sets). 
- Validate with a **reliability diagram** and **Expected Calibration Error (ECE)**; report both before/after.
- The Bayesian models are typically better-calibrated out of the box — use that as a calibration reference.
- **Calibrate the composed probability too,** not just the direct model, since multiplying sub-probabilities can drift.

---

## 7) Evaluation plan

Penalties are sparse and leakage is sneaky, so the *evaluation design* is itself a headline contribution. Make the splitting strategy a section of your write-up.

### Metrics

| Output | Primary metrics | Why |
|---|---|---|
| **Outcome (binary)** | **Log loss** and **Brier score** (proper scoring rules), **ROC-AUC** and **PR-AUC** (discrimination), **calibration: reliability diagram + ECE**, plus a **skill score vs. the shrunk-marginal baseline** (Brier Skill Score). | Probability quality > accuracy. Accuracy is nearly useless here (always-predict-"scored" gets ~78%). |
| **Placement (multiclass)** | **Multiclass log loss**, **top-1 / top-2 accuracy**, **macro-F1**, per-zone reliability. | Top-2 matters because "bottom corners" are genuinely close. |
| **Dive (multiclass)** | Same as placement; also **balanced accuracy** (dives are imbalanced toward L/R). | Center dives are rare; don't let them vanish. |
| **Ranking** | **Spearman / Kendall** between predicted shooter conversion and realized conversion on held-out kicks; **NDCG** for "most clutch" leaderboards. | Validates the *ordering* the product implies, not just pointwise probs. |
| **Explanation quality** | **Faithfulness** (do SHAP attributions match leave-one-feature-out deltas?), **stability** (similar matchups → similar explanations), and a small **human-rater rubric** ("is the card accurate and non-misleading?"). | Explanations are a product surface; treat them as evaluable, not decorative. |

### What "good" looks like
- **Beat the shrunk-marginal baseline** on log loss/Brier by a clear, repeated margin across folds (positive Brier Skill Score). On data this noisy, a *modest but consistent* improvement with good calibration is a strong, honest result — and you should frame it that way rather than chasing an implausibly high AUC.
- **ECE low and reliability diagram near the diagonal** post-calibration.
- **Placement top-2 meaningfully above chance** (chance for 6 zones ≈ 33% for top-2); dive direction clearly above the majority-class rate.
- A clean, monotone **ablation story** (below).

### Splitting carefully (avoid leakage — this is where most projects quietly cheat)
- **Temporal split (primary):** train on kicks before a cutoff date, test after. *All* history features must be computed **as-of the kick date** — a shooter's conversion rate in a test row must use only penalties before that row. Build the feature pipeline to enforce this (expanding windows keyed on date); a single global `groupby('shooter').mean()` over the whole dataset is the classic leak.
- **Grouped-by-player split:** hold out *entire players* (leave-players-out) to measure **generalization to unseen shooters/keepers** — i.e., does the system still work via priors when a player has no rows in training? This directly tests the cold-start machinery.
- **Grouped-by-competition split:** train on some competitions, test on others, to measure transfer across leagues/eras.
- **Combine them:** the headline evaluation is **temporal**; the generalization claims come from **grouped** splits. Report all three; don't average them into one mushy number.

### Generalization to unseen players/keepers
- Run the leave-players-out split and compare: (a) full model vs. (b) priors-only model on unseen players. If the full model barely beats priors-only on unseen players, say so — it means the personal signal is real only with enough history, which is true and worth stating.
- Track metric vs. `n_pens` buckets (0, 1–3, 4–10, 11+) to show *where* the model earns its keep.

### Ablation studies (plan these as experiments, not afterthoughts)
Toggle one group at a time and report ΔBrier / Δlog-loss / ΔECE:
1. − matchup-interaction features (does the "matchup" claim hold up?).
2. − pressure/shootout-state features.
3. − keeper features (shooter-only).
4. − shooter tendency/zone features (rate-only).
5. − recent-form/streak features (likely small — report honestly).
6. shrinkage on vs. off (expect off = worse calibration on sparse players — a clean result).
7. zoning granularity 3 vs. 6 vs. 9 zones.

### Baseline-vs-advanced comparison
Single results table, same folds, same metrics: **{global rate, shooter-only-shrunk, keeper-only-shrunk, logistic, LightGBM, composed matchup, blended}** × **{log loss, Brier, BSS, ECE, AUC}**, with confidence intervals from repeated/grouped CV. This table is the spine of your results section.

---

## 8) Explainability and interpretability

The product promises *why*, so explainability is a feature, not a compliance checkbox.

### Methods (and what each is for)
- **SHAP (TreeExplainer on LightGBM):** per-prediction feature attributions — the engine behind "this matchup is +X% because…". Fast and exact for trees.
- **Global feature importance + SHAP summary plot:** for an "About the model" page and your write-up.
- **Partial dependence / ICE:** to show how predicted goal prob moves with, say, keeper save rate or pressure — good for a methodology appendix.
- **The Bayesian coefficients & posteriors:** inherently interpretable effect sizes with uncertainty (e.g., "must-score-to-survive kicks: −X log-odds, 90% CI […]").
- **Rule-based explanation cards (the user-facing layer):** translate the above into templated natural language. The cards are *grounded in* SHAP/posterior values, not generated freely — this keeps them faithful.

### Presenting it in plain language
The dashboard explanation card should read like a knowledgeable friend, with every claim backed by a stored number:

> **Predicted: 81% goal** (range 74–86%). 
> **Shooter** — Converts **84%** over **22** career penalties; strongly favors the **bottom-left** corner (41% of shots there) and rarely goes high. 
> **Keeper** — Saves **19%** over **31** penalties faced; tends to **dive early to his right** (54%) and rarely stays central. 
> **Matchup** — The keeper's favorite side is the **opposite** of the shooter's favorite corner, so the keeper is unlikely to be there: this **adds ~+4%** versus an average keeper. 
> **Context** — Shootout, kick 4, level scores (routine pressure): negligible effect. 
> **Confidence** — High (both players well-sampled). The wide-ish range reflects normal penalty randomness, not missing data.

For a **low-data** matchup the same card must change tone:

> **Predicted: 76% goal** (range 61–88%). ⚠️ **Low data:** this shooter has only **3** recorded penalties, so the estimate leans on typical right-footed forwards. Treat the placement map as indicative, not personal.

### Keeping explanations honest (anti-misleading rules)
- **Never present an imputed/prior-based number as personal.** If `*_is_imputed`, the card says "typical for players like X," not "X does Y."
- **Always show uncertainty width**, and widen it visibly when support is low.
- **Attribute, don't editorialize.** Cards report effect sizes ("+4% vs. average keeper"), not destiny ("he'll score").
- **Reconcile the two probability estimates:** if composed and direct disagree materially, the card flags higher uncertainty rather than hiding it.
- **No spurious precision:** round to whole percent in the headline; the false-precision of "81.37%" misleads non-experts.
- **Faithfulness check in CI:** periodically verify SHAP attributions track leave-one-feature-out deltas, so the explanation reflects the model, not a comforting story.

---

## 9) Dashboard and product design

### The core interaction
Two prominent autocomplete search boxes — **Shooter** and **Goalkeeper** — plus an optional, collapsible **Context** strip (competition, home/away, shootout toggle → kick index & running score, pressure auto-derived). Hitting "Predict" (or auto-predict on both-selected) renders the result instantly (sub-second; it's matrix math over cached profiles).

### Pages / sections
1. **Matchup (home).** Search + context + the result view (below). This is 90% of usage.
2. **Player profile.** Deep-dive for one shooter: zone heatmap, conversion over time, pressure splits, predictability/entropy, recent penalties — with "low data" honesty throughout.
3. **Keeper profile.** Save rate, dive distribution, guess-correct rate, notable saves, support warnings.
4. **Leaderboards (optional, delightful).** "Most clutch takers," "hardest keepers to beat," "most unpredictable," each with CIs and min-sample filters.
5. **About / methodology.** Model card, data sources, limitations, SHAP global importance — the part research judges read.

### What's visible immediately on a matchup (above the fold)
- **Headline goal probability**, large, with a **visible uncertainty band** (e.g., 81% with a 74–86% range bar) and a **confidence chip** (High/Medium/Low).
- **Shooter goal-mouth heatmap** (6-zone, from the keeper's POV) with the **most-likely zone** highlighted.
- **Keeper dive panel** (same goal-mouth, arrows/shading for dive tendencies) with the **most-likely dive** highlighted, and a subtle overlay showing where shot-likelihood and dive-likelihood collide.
- **One-line verdict** ("Slight edge to the shooter; keeper unlikely to cover the favored corner.").
- **Explanation card** (expandable) as in §8.
- **Two comparison cards:** shooter recent form / keeper recent form.

### Showing uncertainty (do this everywhere, it's a differentiator)
- Range bars on every probability; **confidence chips** driven by `n_pens`/`n_faced`.
- **Greyed, hatched** styling for prior-based (imputed) values; tooltip explains.
- Heatmap **opacity scaled by support** — a thinly-sampled shooter's map looks appropriately faint.
- Explicit ⚠️ **low-data banners** rather than silent guessing.

### Displaying shot-zone & keeper-dive probabilities (visualizations)
- **Goal-mouth grid heatmap** (custom SVG of a goal with a 3×2 or 3×3 grid; color = probability; label = %). This is the signature visual — build it as a reusable component used in matchup, shooter profile, and keeper panel.
- **Dive direction** as either a mirrored goal-mouth heatmap or a small **fan/arrow diagram** (left/center/right with thickness = probability).
- **Probability + uncertainty:** a horizontal **range bar** (point estimate + CI whiskers), not a bare number.
- **History:** small-multiples / sparkline of conversion or save rate over time; a **strip plot** of past penalties colored by outcome.
- **Matchup interaction:** a tiny **side-by-side or overlaid** goal-mouth showing "where he aims" vs. "where the keeper goes," which makes the +/− edge intuitive at a glance.
- **Calibration (methodology page):** the reliability diagram, so technical viewers can trust the numbers.
- **Avoid:** 3D charts, pie charts for the zone grid, and dense tables where a heatmap communicates faster.

### Microcopy & honesty in the UI
Every probability gets a plain-language qualifier; every low-support element is visibly flagged; the headline never shows decimals; the "verdict" sentence never overclaims certainty. The product's credibility *is* its honesty about uncertainty.

---

## 10) Backend and system architecture

### Production-style architecture (but right-sized for a student project)
```
                 ┌─────────────────────────────────────────────────────┐
                 │  OFFLINE (batch, runs on your machine / a cron job)  │
                 │                                                     │
 raw sources ──► │  ingestion/  ──►  cleaning/  ──►  entity_resolution/ │
 (StatsBomb,     │     (adapters)     (validate)      (name_map)        │
  Understat,     │                                                     │
  FBref,         │            ▼                                        │
  Kaggle…)       │       canonical penalties.parquet  ──►  features/   │
                 │                                          (as-of)    │
                 │            ▼                                ▼        │
                 │   hierarchical Bayesian fit         LightGBM + cal.  │
                 │   (shooter/keeper profiles, CIs)    (outcome model)  │
                 │            └──────────────┬───────────────┘          │
                 │                           ▼                          │
                 │      writes:  shooter_profiles, keeper_profiles,     │
                 │      resolution_table, dim_players  +  model/ artifacts│
                 └───────────────────────────┬─────────────────────────┘
                                             ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  ONLINE (serves requests in <1s)                              │
        │                                                              │
        │   PostgreSQL  ◄──reads──  FastAPI service  ──►  React/Vite UI │
        │  (profiles,               (loads model/                       │
        │   resolution,              composes prediction,               │
        │   dim_players)             SHAP explain, cache)               │
        └──────────────────────────────────────────────────────────────┘
```
**Key architectural decision — what's precomputed vs. on-the-fly:**
- **Precomputed offline & stored in DB:** all heavy aggregation — shrunk player/keeper profiles, their CIs, zone/dive vectors, the resolution table, autocomplete index. These change only on retrain.
- **Computed live per request (cheap):** the *composition* (zone⊗dive⊗resolution), the meta-blend, calibration mapping, and the SHAP explanation for that specific feature vector. All of this is small matrix math + one model call; sub-second without GPUs.
- **Cached:** popular matchup results in `matchup_cache`, invalidated on `model_version` bump.

This split is what keeps it fast *and* keeps the live service simple (it never touches raw data or fits anything).

### Practical stack (VS Code–friendly)

| Layer | Choice | Why |
|---|---|---|
| Language (core) | **Python 3.11+** | One language across data, models, API. |
| Data wrangling | **pandas** (+ **polars** if data grows), **pyarrow/parquet** | Standard; parquet for fast, typed intermediate storage. |
| Local analytical store | **DuckDB** (dev) | Query parquet with SQL, zero setup — ideal in VS Code. |
| Modeling | **scikit-learn**, **LightGBM**, **XGBoost**, **PyMC**/**NumPyro** | Covers baseline→trees→Bayesian. |
| Explainability | **SHAP** | Tree attributions for cards. |
| Experiment tracking | **MLflow** (local) or **Weights & Biases** (free tier) | Runs, params, metrics, artifacts. |
| Data/model versioning | **DVC** (+ Git) | Reproducible datasets and model artifacts. |
| API | **FastAPI** + **Pydantic** + **Uvicorn** | Async, typed, auto-generated OpenAPI docs, perfect for a model service. |
| Frontend | **React + Vite + TypeScript**, **Tailwind** | Fast dev, typed, the SVG goal-mouth component is easy here. |
| Charts | **D3** (custom goal-mouth/heatmap) + **Recharts** (standard charts) | D3 for the signature visuals; Recharts for quick line/bar. |
| Production DB | **PostgreSQL** | Robust, free, great with FastAPI/SQLAlchemy. |
| ORM / access | **SQLAlchemy** (+ **Alembic** migrations) | Clean DB layer. |
| Deployment | **Docker**; backend on **Render/Railway/Fly.io** (free-ish), frontend on **Vercel/Netlify** | One-command demos; or just run locally for a fair. |
| Version control | **Git + GitHub** | Plus a clean README and the model card. |

### API design (example endpoints)
```
GET  /api/players?search=mes&role=shooter        → [{player_id, display_name, team, n_pens}]
GET  /api/players?search=neu&role=keeper         → [{player_id, display_name, team, n_faced}]
GET  /api/shooter/{player_id}                     → full shooter_profile (+ CIs, zone vector)
GET  /api/keeper/{player_id}                      → full keeper_profile (+ CIs, dive vector)

POST /api/predict
  body: { shooter_id, keeper_id,
          context: { competition_id?, home_away?, is_shootout?,
                     shootout_kick_index?, shootout_score_for?, shootout_score_against? } }
  resp: { p_goal, ci_low, ci_high, confidence: "high|med|low",
          zone_probs: {z1..z6}, most_likely_zone,
          dive_probs: {left, center, right}, most_likely_dive,
          interaction_edge_pct, explanation: { headline, shooter, keeper, matchup, context, warnings[] },
          model_version }

GET  /api/matchup/{shooter_id}/{keeper_id}        → cached prediction + any real prior meetings
GET  /api/leaderboard?metric=clutch&min_pens=10   → ranked list w/ CIs
GET  /api/health, GET /api/model-card             → ops + transparency
```
- **Validation:** Pydantic models reject bad context (e.g., kick_index without is_shootout).
- **Errors:** return structured `{error, fallback_used}` rather than 500s when a player is unknown — predict from priors and say so.

### Codebase structure (clean, layered)
```
penalty-predictor/
├── data/                  # DVC-tracked; raw/, interim/, processed/ (gitignored contents)
├── src/
│   ├── ingestion/         # one adapter module per source -> canonical rows
│   ├── cleaning/          # validation, dedup, retake handling
│   ├── entity_resolution/ # name_map, fuzzy matching, review queue
│   ├── features/          # as-of feature builders (leak-safe), shrinkage utils
│   ├── models/            # baselines, lgbm, bayesian (pymc), resolution, blender, calibration
│   ├── evaluation/        # splitters (temporal/grouped), metrics, ablation runner
│   ├── explain/           # shap wrappers, card templating
│   ├── serving/           # composition logic, prediction service (pure, importable)
│   └── api/               # FastAPI app, routers, schemas (thin; calls serving/)
├── pipelines/             # scripts: build_dataset.py, train.py, export_profiles.py
├── frontend/              # React/Vite app
├── models/                # exported artifacts (DVC-tracked): lgbm.txt, calibrator.pkl, etc.
├── notebooks/             # EDA + figure generation (not in the import path)
├── tests/                 # pytest: leakage tests, schema tests, API tests
├── conf/                  # config (paths, hyperparams, model_version) via Hydra/pydantic-settings
├── docker/                # Dockerfiles, compose
├── dvc.yaml, params.yaml  # pipeline + experiment params
└── README.md, MODEL_CARD.md
```
- **Layering rule:** `api/` depends on `serving/`, which depends on `models/` + `features/`; nothing depends "upward." The prediction logic in `serving/` is a pure, importable function so you can unit-test it and reuse it in batch.
- **Leakage as a test:** add a pytest that asserts no feature for a kick uses any row dated ≥ that kick. This single test prevents the most common silent failure.

---

## 11) Training workflow

### Step-by-step (one command per stage; orchestrate with `dvc repro`)
1. **Collect** (`pipelines/build_dataset.py`): pull each source via its adapter → raw/ (DVC-tracked snapshots with date stamps).
2. **Clean & resolve:** validate schemas, run entity resolution, dedup across sources, handle retakes/ambiguous outcomes, emit canonical `penalties.parquet` + an `unresolved.csv` review queue.
3. **Feature engineering (leak-safe):** compute as-of expanding-window features and the shrinkage inputs; emit a modeling table. Run the **leakage test** here.
4. **Fit hierarchy (Bayesian):** estimate shooter/keeper posteriors → write `shooter_profiles`, `keeper_profiles`, CIs, and the `resolution_table`.
5. **Train tabular model:** LightGBM on the outcome with temporal early-stopping; train placement & dive multiclass models.
6. **Tune:** Optuna over a *small, sensible* search space (depth, leaves, min_data, L1/L2, learning rate), scored by **temporal-CV log loss**, with grouped-by-player CV as a guard. Keep the search modest — over-tuning on tiny data is itself overfitting.
7. **Calibrate:** fit isotonic/Platt on a held-out fold; record reliability diagram + ECE.
8. **Evaluate:** run the full metrics + ablation suite; generate the results table and figures.
9. **Export:** save model artifacts, calibrator, profiles, `resolution_table`, and a `model_version` (e.g., `v0.3.1+2024data`). Write/refresh `MODEL_CARD.md`.
10. **Publish to DB:** `pipelines/export_profiles.py` loads profiles + resolution into Postgres; bump `model_version`; invalidate `matchup_cache`.

### Retraining cadence
- **[Assumption]** Data refreshes are infrequent (penalties accrue slowly). **Retrain quarterly**, or on-demand after a major tournament adds a batch of shootouts. There's no value in nightly retraining — the data barely moves week to week. Automate with a single `dvc repro` + a scheduled job if deployed.

### Versioning
- **Datasets:** DVC tracks raw snapshots and the processed table; tag each dataset (`data-v2024-07`). 
- **Models:** DVC/MLflow track artifacts; embed `model_version` in every API response and in `matchup_cache` so the UI and cache are always consistent with the deployed model.
- **Reproducibility:** `params.yaml` + pinned environment (`uv`/`pip-tools` lockfile) so any run reproduces.

### Experiment tracking
- Log every run to **MLflow/W&B**: params, all §7 metrics, calibration plots, ablation deltas, and the trained artifact. Tag experiments by hypothesis ("add entropy features"), so your write-up's results table is just a query over tracked runs.

---

## 12) Data quality and risk management

| Risk | Why it bites here | Mitigation | How to detect |
|---|---|---|---|
| **Tiny per-player samples** | Most players have <10 penalties; raw rates are noise. | Hierarchical shrinkage; report CIs; cold-start priors; metric-vs-`n_pens` reporting. | Histogram of `n_pens`; compare raw vs. shrunk rate variance. |
| **Missing keeper dive data** | Free dive labels are scarce — the dive model trains on little. | Train dive model only on labeled rows; fall back to keeper-population prior; flag low-confidence; earmark for video phase. | Coverage report: % rows with `keeper_dive`. |
| **Missing shot location** | Many sources lack (x,y)/zone. | Outcome model still trained; placement falls back to prior; opacity-scaled heatmaps. | Coverage report: % rows with `shot_zone`. |
| **Inconsistent names / wrong joins** | Merging sources silently mismaps players (two "Danny Williams"). | Multi-signal entity resolution (name + birth year + nationality + club-season); human review queue; never auto-accept name-only. | Spot-check merged profiles; flag players with improbable penalty counts. |
| **Label noise (wrong outcomes/zones)** | Source disagreements corrupt training labels. | Source-trust ranking; conflict review queue; exclude retakes by default. | Cross-source agreement rate on overlapping penalties. |
| **Temporal leakage** | Career-aggregate features computed over all time leak the future. | As-of expanding windows; **automated leakage unit test**; temporal split as primary eval. | The leakage test; suspiciously high AUC is a red flag. |
| **Overfitting on sparse data** | Flexible models memorize. | Monotonic constraints, shallow trees, regularization, grouped CV, ablation discipline, prefer shrunk features over raw. | Train-vs-val gap; grouped-CV degradation; logistic≈trees check. |
| **Miscalibration** | A probability product that's overconfident is worse than useless. | Isotonic/Platt + reliability diagram + ECE; calibrate composed prob too. | ECE; reliability diagram off-diagonal. |
| **Survivorship / selection bias** | Coaches pick confident takers; data over-represents specialists. | Acknowledge in the model card; condition on position/role; don't extrapolate to non-takers without flagging. | Compare taker population vs. general squad attributes. |
| **Era/rule drift** | Penalty norms and conversion shift over decades. | Era buckets; recency-weighting; report era as a covariate. | Conversion-by-season trend plot. |
| **Ambiguous/incomplete penalties** | Retakes, encroachment, abandoned shootouts, unknown keeper. | Explicit handling rules (below); exclude or special-case; never guess outcomes. | Validation rules flagging impossible/instate rows. |

### Detecting bad data (build these as automated checks)
- **Schema & range validators** (e.g., `pandera`/`great_expectations`): outcomes in the allowed set, kick_index ≥ 1, dates plausible, zone in vocabulary.
- **Coverage dashboards:** per-field non-null %, per-source.
- **Cross-source agreement** on overlapping penalties.
- **Outlier profiles:** players with implausible counts/rates surfaced for review.

### Handling ambiguous outcomes / incomplete metadata
- **Retakes:** default-exclude (or keep only the decisive attempt) and record the choice; never count both.
- **Off-target vs. saved:** keep them distinct in `outcome_3`; if a source only says "missed," set `outcome_bin` confidently but `outcome_3 = unknown_nonscore` and exclude from the *resolution* table (which needs the cause), not from the outcome model.
- **Unknown keeper / shooter:** still usable for the *other* side's history; predict the matchup via the known side + prior for the unknown side, flagged low-confidence.
- **Abandoned/atypical shootouts:** drop with a logged reason.
- **Golden rule:** when in doubt about a *label*, exclude rather than guess — label noise is the most corrosive error for this product.

---

## 13) Future video / computer-vision upgrade

The non-video version models *tendencies and outcomes*. The video version adds *the kick itself* — the pre-contact "tells" and the true continuous geometry — which the tabular system structurally cannot see.

### What video adds that tabular cannot
- **Pre-contact intent cues:** run-up angle and speed, **hip and shoulder orientation**, **plant-foot position/angle**, **gaze/head direction**, last-stride deceleration — the signals real analysts and keepers read to anticipate side.
- **Keeper behavior in time:** **early movement / weight-shift / pre-dive**, reaction latency, whether the keeper committed before contact — turning the coarse "dive direction" label into a timed, observed action.
- **True ball geometry:** continuous **(x, y) end-coordinates**, ball **speed and spin**, trajectory and time-to-contact — replacing 6-zone bins with real placement and enabling a regression target.
- **De-confounding the duel:** with timing, you can finally model "did the keeper genuinely react vs. guess," relaxing the conditional-independence assumption from §2.

### Video features to extract later
- **Pose (shooter & keeper):** joint angles, hip/shoulder yaw, plant-foot vector, approach kinematics → per-frame and at-contact summaries.
- **Ball tracking:** release point, trajectory, speed, end-coordinate, spin proxy.
- **Keeper tracking:** position over time, commit-time, dive vector, reach.
- **Event timing:** contact frame, keeper-move-onset frame, latency between them.
- **Scene geometry:** homography mapping pixels → real goal/pitch coordinates so positions are metric, not pixel.

### Tooling (concrete)
- **Pose estimation:** MediaPipe (fast, easy) or MMPose/ViTPose (accurate, research-grade).
- **Detection & tracking:** YOLO (v8/v11) for players/ball + **ByteTrack/DeepSORT** for identity persistence; **TrackNet**-style models for small-fast-ball tracking.
- **Homography/calibration:** detect goal posts / pitch lines → compute the pixel↔world transform (OpenCV).
- **Action/timing:** lightweight temporal CNN or rules over keypoints to find contact and keeper-onset frames.
- **Compute:** a single modern GPU (Colab/Kaggle free tiers are enough for a prototype on a few hundred clips).

### Designing the non-video architecture so video slots in cleanly (do this now)
This is the part that earns "thought about the future" credit — and it costs almost nothing today:
1. **A `feature_modality` dimension everywhere.** Every feature has a provider tag (`tabular` | `video`). The modeling table is built by *joining feature groups*, so adding a `video_features` table keyed by `penalty_id` is a join, not a rewrite.
2. **A `FeatureProvider` interface.** Each provider (tabular, later video) implements `build(penalty_ids) -> DataFrame[penalty_id, features...]`. The feature assembler calls all available providers and concatenates. Video becomes a new provider class.
3. **Modality-agnostic model layer.** Models consume a feature dict/vector and never assume which providers filled it; missing video features are just `null` + `video_is_missing=true` (and pre-video rows simply lack them — the same missingness machinery from §5 applies).
4. **Storage seam:** add an empty `video_features` table now (schema: `penalty_id`, `pose_*`, `ball_*`, `keeper_timing_*`, `homography_ok`). The serving layer reads "best available features for this kick," so a video-enriched kick automatically gets richer predictions while legacy kicks fall back gracefully.
5. **Target seam:** keep raw `(x, y)` from day one so that, when video supplies continuous coordinates, you can switch the placement model from 6-zone classification to (x, y) **regression** without changing the interface — the dashboard heatmap just renders a continuous density instead of a grid.
6. **The API and dashboard don't change.** `/api/predict` returns the same shape; richer inputs simply tighten the numbers and may add new explanation lines ("body shape suggests a left-side kick"). No frontend rewrite.

**[Reality check]** The video phase's hard part is **data**: getting clips aligned to specific penalties, plus calibration per stadium/camera. Start with a small, hand-curated clip set (e.g., a single tournament's shootouts from public footage you're permitted to use) to prove the pose→feature→prediction loop end-to-end before scaling. The architecture above means that proof-of-concept plugs straight into the existing system.

---

## 14) Recommended tech stack (consolidated, with must-have vs. optional)

| Need | Must-have | Optional / upgrade | Why this fit |
|---|---|---|---|
| **Data handling** | Python, **pandas**, **pyarrow/parquet**, **DuckDB** | **polars** (scale), **pandera/great_expectations** (validation) | Parquet+DuckDB gives SQL-on-files with zero infra in VS Code; validation libs catch bad data automatically. |
| **Modeling** | **scikit-learn**, **LightGBM** | **XGBoost** (cross-check), **PyMC**/**NumPyro** (hierarchy), **Optuna** (tuning) | Trees dominate small tabular; Bayesian layer supplies shrinkage + uncertainty that define the project. |
| **Explainability** | **SHAP** | partial-dependence utilities | Exact, fast tree attributions power the explanation cards. |
| **Experiment tracking** | **MLflow** (local) | **Weights & Biases** (free tier) | Turns your results section into a query over logged runs. |
| **Data/model versioning** | **Git + GitHub**, **DVC** | MLflow Model Registry | Reproducibility is a grading criterion; DVC versions data+models cleanly. |
| **Backend API** | **FastAPI** + **Pydantic** + **Uvicorn** | **SQLAlchemy** + **Alembic** | Typed, async, auto-docs; perfect for a thin model-serving service. |
| **Frontend dashboard** | **React + Vite + TypeScript**, **Tailwind** | component lib (shadcn/ui) | Fast to build; the custom SVG goal-mouth is straightforward here. |
| **Database** | **SQLite/DuckDB** (dev) → **PostgreSQL** (prod) | Redis (cache) | Start file-based; Postgres when you deploy; profiles are tiny so this is light. |
| **Visualization** | **D3** (signature goal-mouth/heatmaps) + **Recharts** | Plotly (quick interactive) | D3 for the bespoke visuals that make the demo memorable. |
| **Deployment** | **Docker**; **Render/Railway/Fly.io** (API) + **Vercel/Netlify** (UI) | GitHub Actions CI | One-command live demo; or run locally for a fair with no hosting cost. |
| **Dev environment** | **VS Code** + Python/Pylance, Ruff (lint+format), **uv** (env/deps), **pytest** | Dev Containers, Jupyter ext | Cohesive, fast, reproducible local workflow. |

**Minimum viable stack to a demo:** Python + pandas + DuckDB + LightGBM + scikit-learn (calibration) + FastAPI + a small React/Vite app + SQLite. Everything else (PyMC, DVC, MLflow, Postgres, deployment) is an upgrade that makes it *elite*, not a prerequisite to *working*.

---

## 15) Project roadmap

Complexity is rated **◐ low / ◑ medium / ● high** in effort for a motivated student.

### Phase 1 — MVP (fastest path to a strong demo) ◐–◑
**Goal:** a working "search shooter vs. keeper → calibrated goal probability + basic explanation" web app.
- StatsBomb open data + FBref-derived histories; canonical schema + entity resolution for the covered players. ◑
- Baselines (global, shooter-only-shrunk, keeper-only-shrunk) + **Beta-Binomial shrinkage** for rates. ◐
- LightGBM outcome model on the core 7 features; **calibration** + reliability diagram. ◑
- FastAPI `/players`, `/predict`; minimal React UI with two search boxes, headline probability + uncertainty bar, and a templated text explanation. ◑
- **Deliverables:** working app, calibrated headline number, baseline-vs-model table, README. *This alone is already a solid project.*

### Phase 2 — Strong version ◑–●
**Goal:** the full multi-output matchup experience.
- **Placement model** (6-zone) + **goal-mouth heatmap** component; **dive model** (where data allows) + keeper panel. ●
- **Composed matchup prediction** + **resolution table**; interaction features and the "+X% vs. average keeper" edge. ●
- **Hierarchical Bayesian** profiles (PyMC) producing shrunk vectors + **credible intervals**; uncertainty surfaced everywhere in the UI. ●
- **SHAP** explanation cards with low-data honesty; player & keeper profile pages. ◑
- Temporal **and** grouped-by-player evaluation; **ablation studies**; MLflow tracking; DVC versioning. ◑
- **Deliverables:** heatmaps, dive panels, uncertainty, explanation cards, full results+ablation tables.

### Phase 3 — Elite version ●
**Goal:** publication-style polish.
- Meta-blended final probability; calibration of the composed prob; leave-players-out and by-competition generalization studies with metric-vs-`n_pens` analysis. ●
- Leaderboards (clutch/unpredictable/hardest-to-beat) with CIs and min-sample filters. ◑
- Deployment (Docker + hosted API + UI), CI, model card, and a written report/poster with the reliability diagram and ablation story. ◑
- Optional experimental comparison: multi-task NN with embeddings, reported *honestly* against the decomposed system. ●
- **Deliverables:** deployed product, research write-up/poster, model card, reproducible pipeline.

### Phase 4 — Optional video upgrade ● (research extension)
- Curate a small permitted clip set aligned to specific penalties; build pose + ball + keeper-timing extraction; populate `video_features` via a new `FeatureProvider`; retrain with the video group; add body-shape explanation lines; switch placement toward (x, y) regression. ●
- **Deliverables:** end-to-end CV proof-of-concept on a focused dataset, plus an ablation showing what video adds over tabular.

### What to build first
Do Phase 1 in this order for the quickest credible demo: **(1)** canonical dataset + entity resolution on StatsBomb, **(2)** shrunk rate baselines, **(3)** LightGBM + calibration, **(4)** FastAPI `/predict`, **(5)** minimal React search + headline + uncertainty. Then add the **goal-mouth heatmap** first in Phase 2 — it's the single highest-wow-per-hour feature and it's what makes screenshots compelling.

---

## 16) Final deliverables

**a. Project title** 
**PenaltyDuel — A Hierarchical, Explainable Matchup Model for Football Penalty Outcomes**

**b. One-sentence research question** 
*Can a hierarchical, multi-output model that represents shooter placement tendencies, goalkeeper movement tendencies, and their interaction predict penalty outcomes and placement more accurately — and more honestly about uncertainty — than shooter- or keeper-only baselines, while remaining fully explainable?*

**c. One-paragraph abstract** 
Penalty kicks are decided by a simultaneous duel between a shooter's placement choice and a goalkeeper's movement choice, yet most predictive models reduce them to a single conversion probability. We present PenaltyDuel, a matchup prediction system that decomposes a penalty into three estimable parts — a shooter placement distribution, a goalkeeper dive distribution, and a conditional resolution of goal probability given both — and composes them into a calibrated outcome prediction along with a shot-placement heatmap and a keeper-dive panel. Because penalties are sparse per player, all player-level effects are estimated with hierarchical Bayesian partial pooling, yielding shrunken estimates and credible intervals that the system surfaces directly to users; players with little or no history are handled via foot/position/league priors. Predictions are explained per-matchup with SHAP-grounded, plain-language cards that explicitly flag low-data cases. Evaluated with proper scoring rules and calibration metrics under strict leakage-controlled temporal and grouped (leave-players-out, leave-competition-out) splits, and stress-tested with ablations, the system is delivered as a searchable dashboard. The architecture isolates feature provision behind a modality-agnostic interface so that a future computer-vision pipeline — extracting run-up kinematics, body orientation, keeper pre-movement, and true ball trajectory — can be added as a new feature group without reworking the models, API, or product.

**d. Suggested folder structure** — see §10 (`penalty-predictor/` tree). Create it verbatim; it already separates ingestion, features, models, serving, API, and frontend with the correct dependency direction.

**e. Exact files/modules to create first (in order)**
1. `conf/params.yaml` + `src/conf.py` — paths, `model_version`, hyperparameters.
2. `src/ingestion/statsbomb.py` — adapter: raw StatsBomb → canonical penalty rows.
3. `src/ingestion/fbref.py` — adapter for player penalty histories + attributes.
4. `src/entity_resolution/name_map.py` — normalization + fuzzy matching + review queue.
5. `src/cleaning/validate.py` — schema/range checks, dedup, retake handling.
6. `src/features/shrinkage.py` — Beta-Binomial + Dirichlet shrinkage utilities.
7. `src/features/build.py` — **leak-safe, as-of** feature assembly (+ the `FeatureProvider` seam).
8. `tests/test_no_leakage.py` — assert no feature uses data dated ≥ the kick. *(Write this early.)*
9. `src/models/baselines.py` — global / shooter-only / keeper-only (shrunk).
10. `src/models/outcome_lgbm.py` — LightGBM + monotonic constraints.
11. `src/models/calibration.py` — isotonic/Platt + reliability/ECE.
12. `src/models/resolution.py` — build the `P(goal | zone, dive, segment)` table.
13. `src/serving/predict.py` — pure composition + blend function (importable, testable).
14. `src/explain/cards.py` — SHAP → templated, honesty-aware explanation cards.
15. `src/api/main.py` + `src/api/schemas.py` — FastAPI `/players`, `/predict`.
16. `pipelines/build_dataset.py`, `pipelines/train.py`, `pipelines/export_profiles.py`.
17. `frontend/` — Vite app: search boxes, headline+uncertainty, then the SVG goal-mouth heatmap.

**f. Pre-launch checklist**
- [ ] Canonical `penalties` dataset built; coverage report for `shot_zone` and `keeper_dive` produced and acknowledged in the model card.
- [ ] Entity resolution reviewed; `unresolved.csv` triaged; no name-only auto-merges.
- [ ] **Leakage test passing**; features confirmed as-of-time.
- [ ] Baselines reported; advanced models **beat shrunk-marginal baselines** on log loss/Brier across folds (with CIs).
- [ ] Probabilities **calibrated**; reliability diagram near-diagonal; ECE recorded (before/after).
- [ ] Placement/dive models evaluated (top-2, macro-F1); low-support keepers fall back to prior + flagged.
- [ ] Composed + direct probabilities reconciled; blend in place; both reported.
- [ ] Ablation table complete; generalization (leave-players-out, by-competition) reported with metric-vs-`n_pens`.
- [ ] Explanation cards faithful (SHAP vs. leave-one-feature-out spot-check) and **never present imputed values as personal**.
- [ ] UI surfaces uncertainty everywhere; ⚠️ low-data banners working; headline shows no false-precision decimals.
- [ ] API validates input, returns structured fallbacks for unknown players (predict-from-prior, flagged).
- [ ] `model_version` embedded in responses + cache; cache invalidation on retrain verified.
- [ ] DVC/MLflow runs reproducible; environment locked; `README.md` + `MODEL_CARD.md` written (data sources, assumptions, limitations, biases).
- [ ] **Video seam present** (`video_features` table + `FeatureProvider` interface + raw `(x,y)` retained) so the CV phase needs no rewrite.

---

### Closing note on framing for an application/portfolio
Lead your write-up with the three honest strengths, because they are exactly what distinguishes serious work: **it's a matchup model, not an xG model** (interaction is first-class); **it's statistically disciplined** (shrinkage + calibration + leakage-controlled, grouped evaluation, with uncertainty shown to users); and **it's explainable and product-complete**. A *modest but well-calibrated* improvement over strong baselines, presented with clear ablations and visible uncertainty, reads as more credible and more mature than an implausibly high accuracy number — and it's the version of this project that holds up to expert questions.
