"""
Step 4 EDA for the StatsBomb penalty dataset.

Run from the project root:
    python reports/step4_eda.py

Outputs:
    reports/figures/*.png
    docs/DATA_REALITIES.md
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.viz.goalmouth import ZONE_ORDER, plot_goalmouth_heatmap


DATA_PATH = ROOT / "outputs" / "statsbomb" / "penalties_statsbomb_clean.parquet"
FIGURES_DIR = ROOT / "reports" / "figures"
MEMO_PATH = ROOT / "docs" / "DATA_REALITIES.md"


def pct(value: float | int | np.floating | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def count_pct(count: int, total: int) -> str:
    share = count / total if total else 0.0
    return f"{count:,} ({pct(share)})"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in df[columns].iterrows()
    ]
    return "\n".join([header, divider, *rows])


def summarize_binary_rate(
    df: pd.DataFrame,
    group_col: str,
    label_col: str | None = None,
    min_count: int = 1,
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)["outcome_bin"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={group_col: "group", "count": "n", "sum": "goals", "mean": "conversion"})
    )
    grouped = grouped[grouped["n"] >= min_count].copy()
    grouped["conversion"] = grouped["conversion"].map(lambda x: pct(x))
    grouped["goals"] = grouped["goals"].astype(int)
    grouped["group"] = grouped["group"].fillna("missing").astype(str)
    if label_col:
        grouped = grouped.rename(columns={"group": label_col})
    return grouped.sort_values("n", ascending=False)


def add_shootout_first_kicker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["shootout_first_kicker"] = pd.NA
    shootouts = df[df["is_shootout"]].sort_values(["match_id", "shootout_kick_index"])
    first_team_by_match = shootouts.groupby("match_id")["shooter_team_id"].first()
    first_team = df["match_id"].map(first_team_by_match)
    mask = df["is_shootout"] & first_team.notna()
    df.loc[mask, "shootout_first_kicker"] = (
        df.loc[mask, "shooter_team_id"].astype("Int64") == first_team.loc[mask].astype("Int64")
    )
    return df


def add_pressure_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pressure_proxy"] = "in_game"
    shootout = df["is_shootout"]
    early = shootout & df["shootout_kick_index"].between(1, 5, inclusive="both")
    late = shootout & (df["shootout_kick_index"] > 5)
    df.loc[early, "pressure_proxy"] = "shootout_kicks_1_5"
    df.loc[late, "pressure_proxy"] = "shootout_kicks_6_plus"
    return df


def save_sparsity_histograms(df: pd.DataFrame) -> None:
    shooter_counts = df.groupby("shooter_id").size()
    keeper_counts = df.dropna(subset=["keeper_id"]).groupby("keeper_id").size()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(1, max(shooter_counts.max(), 2) + 2) - 0.5
    ax.hist(shooter_counts, bins=bins, color="#4C78A8", edgecolor="white")
    ax.set_title("Penalties Per Shooter")
    ax.set_xlabel("Recorded penalties")
    ax.set_ylabel("Shooters")
    ax.set_xlim(left=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "penalties_per_shooter_hist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if keeper_counts.empty:
        ax.text(0.5, 0.5, "No resolved keepers", ha="center", va="center")
        ax.set_axis_off()
    else:
        bins = np.arange(1, max(keeper_counts.max(), 2) + 2) - 0.5
        ax.hist(keeper_counts, bins=bins, color="#F58518", edgecolor="white")
        ax.set_title("Penalties Faced Per Resolved Keeper")
        ax.set_xlabel("Recorded penalties faced")
        ax.set_ylabel("Keepers")
        ax.set_xlim(left=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "penalties_faced_per_keeper_hist.png", dpi=160)
    plt.close(fig)


def save_goalmouth_figures(df: pd.DataFrame) -> None:
    zone_counts = df["shot_zone"].value_counts().reindex(ZONE_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_goalmouth_heatmap(
        zone_counts.to_dict(),
        ax=ax,
        title="League-wide penalty placement counts",
        cmap="Blues",
        value_format="{:.0f}",
        cbar_label="Kicks",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "goalmouth_placement_counts.png", dpi=160)
    plt.close(fig)

    zone_conversion = df.groupby("shot_zone")["outcome_bin"].mean().reindex(ZONE_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_goalmouth_heatmap(
        zone_conversion.fillna(0).to_dict(),
        ax=ax,
        title="Conversion rate by shot zone",
        cmap="Greens",
        value_format="{:.1%}",
        cbar_label="Conversion",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "goalmouth_conversion_by_zone.png", dpi=160)
    plt.close(fig)


def save_coverage_bars(df: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "shot_zone",
        "keeper_id",
        "keeper_dive_direction",
        "gk_freeze_location_x",
        "gk_freeze_location_y",
        "shootout_kick_index",
    ]
    coverage = pd.DataFrame(
        {
            "field": fields,
            "non_null": [int(df[field].notna().sum()) for field in fields],
            "coverage": [float(df[field].notna().mean()) for field in fields],
        }
    )
    coverage["missing"] = len(df) - coverage["non_null"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(coverage["field"], coverage["coverage"], color="#54A24B")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rows non-null")
    ax.set_title("Key field coverage")
    ax.xaxis.set_major_formatter(lambda value, _: pct(value, digits=0))
    for idx, row in coverage.iterrows():
        ax.text(
            row["coverage"] + 0.01,
            idx,
            f"{row['non_null']:,}/{len(df):,}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "coverage_bars.png", dpi=160)
    plt.close(fig)
    return coverage


def save_pressure_chart(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["in_game", "shootout_kicks_1_5", "shootout_kicks_6_plus"]
    for label in order:
        subset = df[df["pressure_proxy"] == label]
        n = len(subset)
        goals = int(subset["outcome_bin"].sum())
        low, high = wilson_interval(goals, n)
        rows.append(
            {
                "pressure_proxy": label,
                "n": n,
                "goals": goals,
                "conversion_value": goals / n if n else np.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    pressure = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = np.vstack(
        [
            pressure["conversion_value"] - pressure["ci_low"],
            pressure["ci_high"] - pressure["conversion_value"],
        ]
    )
    ax.bar(pressure["pressure_proxy"], pressure["conversion_value"], color="#B279A2")
    ax.errorbar(
        pressure["pressure_proxy"],
        pressure["conversion_value"],
        yerr=yerr,
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Conversion rate")
    ax.set_title("Conversion by pressure proxy")
    ax.yaxis.set_major_formatter(lambda value, _: pct(value, digits=0))
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "conversion_by_pressure_proxy.png", dpi=160)
    plt.close(fig)

    pressure["conversion"] = pressure["conversion_value"].map(pct)
    pressure["95pct_ci"] = pressure.apply(
        lambda row: f"{pct(row['ci_low'])} - {pct(row['ci_high'])}",
        axis=1,
    )
    return pressure


def save_naive_shooter_reliability(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["match_date", "match_id", "index_in_match"]).copy()
    ordered["shooter_prev_n"] = ordered.groupby("shooter_id").cumcount()
    ordered["shooter_prev_goals"] = (
        ordered.groupby("shooter_id")["outcome_bin"].cumsum() - ordered["outcome_bin"]
    )
    ordered["shooter_prev_rate_raw"] = ordered["shooter_prev_goals"] / ordered["shooter_prev_n"]
    eligible = ordered[ordered["shooter_prev_n"] >= 2].copy()

    if eligible.empty:
        reliability = pd.DataFrame(
            columns=["raw_rate_bin", "n", "mean_raw_rate", "next_conversion"]
        )
    else:
        eligible["raw_rate_bin"] = pd.cut(
            eligible["shooter_prev_rate_raw"],
            bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
            labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        )
        reliability = (
            eligible.groupby("raw_rate_bin", observed=False)
            .agg(
                n=("outcome_bin", "size"),
                mean_raw_rate=("shooter_prev_rate_raw", "mean"),
                next_conversion=("outcome_bin", "mean"),
            )
            .reset_index()
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    if reliability.empty:
        ax.text(0.5, 0.5, "Not enough repeat shooters", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        sizes = reliability["n"].clip(lower=1) * 8
        ax.scatter(
            reliability["mean_raw_rate"],
            reliability["next_conversion"],
            s=sizes,
            color="#E45756",
            alpha=0.8,
        )
        for _, row in reliability.iterrows():
            ax.text(
                row["mean_raw_rate"],
                row["next_conversion"],
                f"n={int(row['n'])}",
                fontsize=8,
                ha="left",
                va="bottom",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Raw prior shooter conversion rate")
        ax.set_ylabel("Next-penalty conversion rate")
        ax.set_title("Naive raw shooter rate reliability")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "naive_shooter_rate_reliability.png", dpi=160)
    plt.close(fig)

    if not reliability.empty:
        reliability["mean_raw_rate"] = reliability["mean_raw_rate"].map(pct)
        reliability["next_conversion"] = reliability["next_conversion"].map(pct)
    return reliability


def build_memo(
    df: pd.DataFrame,
    coverage: pd.DataFrame,
    pressure: pd.DataFrame,
    reliability: pd.DataFrame,
) -> str:
    total = len(df)
    date_min = df["match_date"].min()
    date_max = df["match_date"].max()
    shooter_counts = df.groupby("shooter_id").size()
    keeper_counts = df.dropna(subset=["keeper_id"]).groupby("keeper_id").size()
    low_sample_shooters = int((shooter_counts <= 2).sum())
    low_sample_keepers = int((keeper_counts <= 2).sum())

    duplicate_penalty_ids = int(df["penalty_id"].duplicated().sum())
    duplicate_events = int(df["source_event_id"].duplicated().sum())
    retake_candidates = int(
        df.duplicated(
            subset=["match_id", "period", "minute", "shooter_id"],
            keep=False,
        ).sum()
    )
    unknown_outcomes = int((df["outcome_3"] == "unknown").sum())
    perfect_tiny = (
        df.groupby(["shooter_id", "shooter_name"], dropna=False)["outcome_bin"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    perfect_tiny = perfect_tiny[
        (perfect_tiny["count"] <= 2) & (perfect_tiny["mean"] == 1.0)
    ].sort_values(["count", "shooter_name"], ascending=[False, True])

    zone_summary = summarize_binary_rate(df, "shot_zone", label_col="shot_zone")
    foot_summary = summarize_binary_rate(df, "shot_body_part_name", label_col="body_part")
    stage_summary = summarize_binary_rate(
        df,
        "competition_stage_name",
        label_col="stage",
        min_count=10,
    ).head(12)
    first_kicker = summarize_binary_rate(
        df[df["is_shootout"]].dropna(subset=["shootout_first_kicker"]),
        "shootout_first_kicker",
        label_col="first_kicking_team",
    )

    coverage_display = coverage.copy()
    coverage_display["coverage"] = coverage_display["coverage"].map(pct)

    pressure_display = pressure[
        ["pressure_proxy", "n", "goals", "conversion", "95pct_ci"]
    ].copy()

    reliability_display = reliability.copy()
    if not reliability_display.empty:
        reliability_display["raw_rate_bin"] = reliability_display["raw_rate_bin"].astype(str)

    perfect_tiny_display = perfect_tiny.head(12).copy()
    perfect_tiny_display["record"] = perfect_tiny_display.apply(
        lambda row: f"{int(row['sum'])}-for-{int(row['count'])}",
        axis=1,
    )

    lines = [
        "# Data Realities - Step 4 EDA",
        "",
        "Generated by `python reports/step4_eda.py` from "
        "`outputs/statsbomb/penalties_statsbomb_clean.parquet`.",
        "",
        "## Dataset Snapshot",
        "",
        f"- Rows: {total:,} penalties from {date_min} to {date_max}.",
        f"- Overall conversion rate: {pct(df['outcome_bin'].mean())}.",
        f"- Unique shooters: {df['shooter_id'].nunique():,}.",
        f"- Resolved unique keepers: {df['keeper_id'].nunique():,}.",
        f"- Competitions covered: {df['competition_name'].nunique():,}.",
        f"- Shootout penalties: {count_pct(int(df['is_shootout'].sum()), total)}.",
        "",
        "## Coverage",
        "",
        markdown_table(coverage_display, ["field", "non_null", "missing", "coverage"]),
        "",
        "The StatsBomb-only dataset has no keeper dive labels: "
        f"`keeper_dive_direction` coverage is "
        f"{coverage_display.loc[coverage_display['field'] == 'keeper_dive_direction', 'coverage'].iloc[0]}. "
        "Do not train a dive-direction model from this table until another source enriches it.",
        "",
        "## Sparsity",
        "",
        f"- Median penalties per shooter: {shooter_counts.median():.0f}; "
        f"75th percentile: {shooter_counts.quantile(0.75):.0f}; "
        f"max: {shooter_counts.max():.0f}.",
        f"- Shooters with 1-2 penalties: {count_pct(low_sample_shooters, len(shooter_counts))}.",
        f"- Median penalties faced per resolved keeper: "
        f"{keeper_counts.median() if not keeper_counts.empty else 0:.0f}; "
        f"max: {keeper_counts.max() if not keeper_counts.empty else 0:.0f}.",
        f"- Resolved keepers with 1-2 faced penalties: "
        f"{count_pct(low_sample_keepers, len(keeper_counts)) if len(keeper_counts) else '0 (0.0%)'}.",
        "",
        "This is the central modeling constraint: raw player rates are mostly tiny-sample "
        "summaries, not stable player skill estimates. Step 5 should use shrinkage and "
        "as-of expanding windows before any player-level rates become features.",
        "",
        "## Conversion Patterns",
        "",
        "### By Shot Zone",
        "",
        markdown_table(zone_summary, ["shot_zone", "n", "goals", "conversion"]),
        "",
        "### By Body Part",
        "",
        markdown_table(foot_summary, ["body_part", "n", "goals", "conversion"]),
        "",
        "### By Stage",
        "",
        markdown_table(stage_summary, ["stage", "n", "goals", "conversion"]),
        "",
        "## Pressure Proxies",
        "",
        "A true `stakes` label is not present yet, so this EDA uses a conservative proxy: "
        "in-game penalties, shootout kicks 1-5, and shootout kicks 6+.",
        "",
        markdown_table(pressure_display, ["pressure_proxy", "n", "goals", "conversion", "95pct_ci"]),
        "",
        "### First-Kicker Effect In Shootouts",
        "",
        markdown_table(first_kicker, ["first_kicking_team", "n", "goals", "conversion"]),
        "",
        "## Naive Shooter Rate Reliability",
        "",
        "For each repeat shooter, the chart bins their raw prior conversion rate before a kick "
        "and compares it with the next-kick outcome. Sparse bins and visible deviations from "
        "the diagonal are the EDA-level warning that raw rates need shrinkage.",
        "",
        markdown_table(
            reliability_display,
            ["raw_rate_bin", "n", "mean_raw_rate", "next_conversion"],
        ),
        "",
        "## Weird Cases / Quality Checks",
        "",
        f"- Duplicate `penalty_id` rows: {duplicate_penalty_ids}.",
        f"- Duplicate `source_event_id` rows: {duplicate_events}.",
        f"- Same-match/minute/shooter retake candidates: {retake_candidates}.",
        f"- Unknown `outcome_3` rows: {unknown_outcomes}.",
        f"- 100% conversion on 1-2 recorded penalties: {len(perfect_tiny):,} shooters.",
        "",
        "Top tiny-sample perfect shooters:",
        "",
        markdown_table(perfect_tiny_display, ["shooter_name", "record"]),
        "",
        "## Figures",
        "",
        "- `reports/figures/penalties_per_shooter_hist.png`",
        "- `reports/figures/penalties_faced_per_keeper_hist.png`",
        "- `reports/figures/goalmouth_placement_counts.png`",
        "- `reports/figures/goalmouth_conversion_by_zone.png`",
        "- `reports/figures/conversion_by_pressure_proxy.png`",
        "- `reports/figures/coverage_bars.png`",
        "- `reports/figures/naive_shooter_rate_reliability.png`",
        "",
        "## Step 5 Implications",
        "",
        "- Use `penalties_statsbomb_clean.parquet` for modeling, but compute every history "
        "feature as-of the kick date.",
        "- Use Beta-Binomial or Dirichlet shrinkage for shooter/keeper rates and zone vectors.",
        "- Treat missing keeper dive labels as a data-source limitation, not a missing-value "
        "imputation problem.",
        "- Keep raw shot end coordinates because the six-zone bins may be revisited later.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date.astype(str)
    df = add_shootout_first_kicker(df)
    df = add_pressure_proxy(df)

    save_sparsity_histograms(df)
    save_goalmouth_figures(df)
    coverage = save_coverage_bars(df)
    pressure = save_pressure_chart(df)
    reliability = save_naive_shooter_reliability(df)

    MEMO_PATH.write_text(
        build_memo(df, coverage, pressure, reliability),
        encoding="utf-8",
    )
    print(f"Wrote {MEMO_PATH.relative_to(ROOT)}")
    print(f"Wrote figures to {FIGURES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
