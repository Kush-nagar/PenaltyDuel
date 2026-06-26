"""
Ingestion adapters for Kaggle dive-direction datasets.

Dataset 1 (rodrigoarede2003): real_data.xlsx
  - EPL / Primeira Liga / Serie A, 2020–2025, ~482 kicks
  - Has keeper/shooter names + keeper dive direction (L/R/C)
  - No 2D shot zone — only kick direction (L/R/C)
  - Used for: per-keeper dive frequency tables

Dataset 2 (pablollanderos33): WorldCupShootouts.csv
  - World Cup shootouts 1982–2022, 304 kicks
  - Has 9-zone shot placement + keeper dive (L/R/C)
  - No keeper/shooter names (team-level only)
  - Used for: population prior + zone×dive resolution table
"""

from __future__ import annotations

import pandas as pd

from src.enrichment.entity_resolution import normalize_name

DIVE_DIRECTIONS = ["left", "center", "right"]

# DS1 Country → canonical competition name (matches StatsBomb competition_name where possible)
_COUNTRY_MAP: dict[str, str] = {
    "italy": "Serie A",
    "england": "Premier League",
    "portugal": "Primeira Liga",
}

# DS2 9-zone grid → 6-zone schema
# Layout: rows = high(7-9) / mid(4-6) / low(1-3), cols = left/center/right
_ZONE9_MAP: dict[int, str] = {
    1: "low_left",
    2: "low_center",
    3: "low_right",
    4: "low_left",   # mid collapses into low
    5: "low_center",
    6: "low_right",
    7: "high_left",
    8: "high_center",
    9: "high_right",
}


def normalize_dive(raw: object) -> str | None:
    """Map raw dive label to canonical left/center/right."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().upper()
    if s in ("L", "I"):   # I = Izquierda (Spanish for left)
        return "left"
    if s == "C":
        return "center"
    if s == "R":
        return "right"
    return None


def _parse_season_year(label: object) -> int | None:
    """'24/25' → 2024, '20/21' → 2020."""
    if pd.isna(label):
        return None
    parts = str(label).strip().split("/")
    try:
        prefix = int(parts[0])
        return 2000 + prefix if prefix < 100 else prefix
    except (ValueError, IndexError):
        return None


def _normalize_foot(raw: object) -> str | None:
    if pd.isna(raw):
        return None
    s = str(raw).strip().upper()
    if s == "R":
        return "right"
    if s == "L":
        return "left"
    return None


def load_rodrigo(path: str) -> pd.DataFrame:
    """
    Load and normalize the rodrigoarede2003 Excel dataset.

    Returns canonical DataFrame with columns:
      source, keeper_name_normalized, shooter_name_normalized,
      keeper_dive_direction, foot, outcome_bin, competition, season_approx,
      kick_direction
    """
    raw = pd.read_excel(path)

    # Rows with fractional Outcome values are data errors (Excel date artifacts)
    outcome_clean = pd.to_numeric(raw["Outcome"], errors="coerce")
    keep = outcome_clean.isin([0.0, 1.0])
    df = raw.loc[keep].copy()

    df["source"] = "kaggle_rodrigo"
    df["keeper_name_normalized"] = df["goalkeer_name"].map(normalize_name)
    df["shooter_name_normalized"] = df["player_name"].map(normalize_name)
    df["keeper_dive_direction"] = df["Goalie_Side"].map(normalize_dive)
    df["foot"] = df["Kicker_Foot"].map(_normalize_foot)
    df["kick_direction"] = df["Kicker_Side"].map(normalize_dive)  # L/C/R for kick side
    df["outcome_bin"] = outcome_clean.loc[keep].astype(int)
    df["competition"] = (
        df["Country"].astype(str).str.strip().str.lower().map(_COUNTRY_MAP)
    )
    df["season_approx"] = df["Unnamed: 0"].map(_parse_season_year)
    df["is_shootout"] = False
    df["shot_zone"] = None   # not available — only kick direction

    keep_cols = [
        "source",
        "keeper_name_normalized",
        "shooter_name_normalized",
        "keeper_dive_direction",
        "foot",
        "kick_direction",
        "outcome_bin",
        "competition",
        "season_approx",
        "is_shootout",
        "shot_zone",
    ]
    return df[keep_cols].reset_index(drop=True)


def load_pablo(path: str) -> pd.DataFrame:
    """
    Load and normalize the pablollanderos33 World Cup Shootouts CSV.

    Returns canonical DataFrame with columns:
      source, keeper_dive_direction, shot_zone, foot, outcome_bin,
      is_shootout, kick_index_in_series, is_elimination_kick
    No keeper/shooter names available.
    """
    raw = pd.read_csv(path)

    # Drop rows where core fields are missing (25 NaN rows in original)
    df = raw.dropna(subset=["Goal", "Keeper", "Zone"]).copy()

    df["source"] = "kaggle_pablo"
    df["keeper_dive_direction"] = df["Keeper"].map(normalize_dive)
    df["foot"] = df["Foot"].map(_normalize_foot)
    df["outcome_bin"] = pd.to_numeric(df["Goal"], errors="coerce").astype("Int64")
    df["is_shootout"] = True
    df["kick_index_in_series"] = pd.to_numeric(
        df["Penalty_Number"], errors="coerce"
    ).astype("Int64")
    df["is_elimination_kick"] = pd.to_numeric(
        df.get("Elimination", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0).astype(int).astype(bool)

    zone_raw = pd.to_numeric(df["Zone"], errors="coerce").astype("Int64")
    df["shot_zone"] = zone_raw.map(lambda z: _ZONE9_MAP.get(int(z)) if pd.notna(z) else None)

    # keeper/shooter names not available
    df["keeper_name_normalized"] = None
    df["shooter_name_normalized"] = None
    df["competition"] = "FIFA World Cup"
    df["season_approx"] = None

    keep_cols = [
        "source",
        "keeper_name_normalized",
        "shooter_name_normalized",
        "keeper_dive_direction",
        "shot_zone",
        "foot",
        "outcome_bin",
        "competition",
        "season_approx",
        "is_shootout",
        "kick_index_in_series",
        "is_elimination_kick",
    ]
    return df[keep_cols].reset_index(drop=True)


def build_keeper_dive_counts(rodrigo_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Aggregate DS1 kicks into per-keeper dive counts.

    Returns {keeper_name_normalized: {"left": n, "center": n, "right": n}}.
    Only keepers with at least one valid dive label are included.
    """
    counts: dict[str, dict[str, float]] = {}
    labeled = rodrigo_df[rodrigo_df["keeper_dive_direction"].isin(DIVE_DIRECTIONS)]
    for keeper, grp in labeled.groupby("keeper_name_normalized"):
        if not keeper:
            continue
        vc = grp["keeper_dive_direction"].value_counts()
        counts[keeper] = {d: float(vc.get(d, 0)) for d in DIVE_DIRECTIONS}
    return counts


def build_population_prior_counts(pablo_df: pd.DataFrame) -> dict[str, float]:
    """
    Aggregate DS2 WC kicks into population-level dive counts.

    These become the Dirichlet prior for per-keeper estimation.
    """
    labeled = pablo_df[pablo_df["keeper_dive_direction"].isin(DIVE_DIRECTIONS)]
    vc = labeled["keeper_dive_direction"].value_counts()
    total = len(labeled)
    if total == 0:
        return {d: 1.0 for d in DIVE_DIRECTIONS}
    # Normalize to a prior strength of ~10 pseudo-counts total
    prior_strength = 10.0
    return {d: float(vc.get(d, 0)) / total * prior_strength for d in DIVE_DIRECTIONS}
