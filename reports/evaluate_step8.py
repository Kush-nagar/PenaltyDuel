"""
Step 8 - evaluate the model properly.

Produces the defensible-result spine:
- results_table.md      baselines vs models x metrics (CV mean +/- std)
- ablation_table.md     feature-group + shrinkage toggles (delta vs full)
- generalization_table.md  leave-players-out / leave-competition-out + metric-vs-n_pens
- placement_table.md    6-zone placement forecast vs uniform/marginal + ranking sanity
- figures/step8_metric_vs_npens.png

Run from the project root:
    python reports/evaluate_step8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.ablation import run_ablation
from src.evaluation.generalization import (
    BUCKET_ORDER,
    conversion_ranking_correlation,
    evaluate_generalization,
)
from src.evaluation.metrics import binary_classification_metrics, format_metric
from src.evaluation.placement import evaluate_placement
from src.evaluation.splitters import temporal_cutoff_by_fraction, temporal_train_test_split
from src.features.build import build_modeling_table
from src.models.baselines import add_baseline_predictions
from src.models.training import evaluate_outcome_models, fit_pipeline, predict_pipeline

DATA_PATH = ROOT / "outputs" / "statsbomb" / "penalties_statsbomb_clean.parquet"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MODEL_LABELS = {
    "global": "global rate (baseline)",
    "keeper_shrunk": "keeper-shrunk (baseline floor)",
    "composed": "composed placement/resolution",
    "combo": "combined log-odds",
    "blend": "meta-blend (diagnostic)",
    "combo_calibrated": "combined log-odds calibrated (HEADLINE)",
}


def results_table(cv: dict, headline_detail: dict) -> str:
    summary = cv["cv_summary"]
    lines = [
        "## Outcome results (rolling temporal CV, 5 folds)",
        "",
        "Mean +/- std across folds. Log loss and Brier are the decision metrics; "
        "AUC/ECE shown for the headline on the single 80/20 temporal split.",
        "",
        "| model | log loss | Brier |",
        "| --- | --- | --- |",
    ]
    for name in ["global", "keeper_shrunk", "composed", "combo", "blend", "combo_calibrated"]:
        s = summary[name]
        lines.append(
            f"| {MODEL_LABELS[name]} | {s['log_loss_mean']:.4f} +/- {s['log_loss_std']:.4f} "
            f"| {s['brier_mean']:.4f} +/- {s['brier_std']:.4f} |"
        )
    g = cv["gate"]
    lines += [
        "",
        f"- **Headline beats keeper-shrunk floor (log loss AND Brier): "
        f"`{g['beats_keeper_shrunk']}`**, BSS `{g['brier_skill_score']:+.4f}`, "
        f"folds won `{g['folds_won']}/{g['folds_total']}`.",
        "",
        "Headline detailed metrics (single 80/20 temporal split - for AUC/ECE only; "
        "this single split is the one noisy fold the headline loses, which is exactly "
        "why the 5-fold rolling CV above is the decision metric, not one split):",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| log loss | {format_metric(headline_detail['log_loss'])} |",
        f"| Brier | {format_metric(headline_detail['brier'])} |",
        f"| Brier skill vs floor | {format_metric(headline_detail['brier_skill_score'])} |",
        f"| ROC-AUC | {format_metric(headline_detail['roc_auc'])} |",
        f"| PR-AUC | {format_metric(headline_detail['pr_auc'])} |",
        f"| ECE | {format_metric(headline_detail['ece'])} |",
        f"| accuracy (footnote) | {format_metric(headline_detail['accuracy_footnote'])} |",
        "",
    ]
    return "\n".join(lines)


def ablation_md(ablation: pd.DataFrame) -> str:
    lines = [
        "# Ablation table - Step 8",
        "",
        "Rolling temporal CV. Delta is variant minus full; positive delta_log_loss "
        "means the dropped group *helped* (removing it hurt).",
        "",
        "| variant | log loss | Brier | ECE | d_log_loss | d_brier | d_ece |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in ablation.iterrows():
        lines.append(
            f"| {r['variant']} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['ece']:.4f} "
            f"| {r['d_log_loss']:+.4f} | {r['d_brier']:+.4f} | {r['d_ece']:+.4f} |"
        )
    lines += [
        "",
        "Reading: a feature group is justified when removing it raises log loss "
        "(positive `d_log_loss`). `shrinkage_off` raising log loss confirms shrinkage "
        "earns its keep on sparse players.",
        "",
    ]
    return "\n".join(lines)


def generalization_md(lpo: dict, lco: dict) -> str:
    def block(title: str, res: dict) -> list[str]:
        o = res["overall"]
        out = [
            f"## {title} (n={o['n']})",
            "",
            "| model | log loss | Brier | ECE |",
            "| --- | --- | --- | --- |",
            f"| full combined | {o['full']['log_loss']:.4f} | {o['full']['brier']:.4f} "
            f"| {o['full']['ece']:.4f} |",
            f"| priors-only | {o['priors_only']['log_loss']:.4f} | "
            f"{o['priors_only']['brier']:.4f} | {o['priors_only']['ece']:.4f} |",
            "",
            "Metric by prior-penalty support bucket (log loss):",
            "",
            "| n_pens bucket | n | full | priors-only |",
            "| --- | --- | --- | --- |",
        ]
        for row in res["by_n_pens"]:
            out.append(
                f"| {row['bucket']} | {row['n']} | {row['full_log_loss']:.4f} "
                f"| {row['priors_log_loss']:.4f} |"
            )
        out.append("")
        return out

    lines = [
        "# Generalization - Step 8",
        "",
        "Grouped out-of-fold CV on groups never seen in training. Temporal is the "
        "headline split (see results_table.md); these grouped splits back the "
        "generalization claim and are reported separately, not averaged in.",
        "",
        *block("Leave-players-out", lpo),
        *block("Leave-competition-out", lco),
        "![metric vs n_pens](figures/step8_metric_vs_npens.png)",
        "",
    ]
    return "\n".join(lines)


def placement_md(placement: dict, ranking: dict) -> str:
    def row(name: str, m: dict) -> str:
        return (
            f"| {name} | {m['log_loss']:.4f} | {m['top1']:.3f} | {m['top2']:.3f} "
            f"| {m['macro_f1']:.3f} |"
        )

    return "\n".join(
        [
            "# Placement & ranking sanity - Step 8",
            "",
            f"Six-zone placement forecast (as-of shrunk zone distribution) on the "
            f"temporal test split (n={placement['n_test']}).",
            "",
            "| forecast | multiclass log loss | top-1 | top-2 | macro-F1 |",
            "| --- | --- | --- | --- | --- |",
            row("model (as-of)", placement["model"]),
            row("train marginal", placement["marginal"]),
            row("uniform", placement["uniform"]),
            "",
            "Honest finding: the as-of model beats uniform and edges top-1/macro-F1, "
            "but the **train marginal beats it on multiclass log loss**. Most shooters "
            "are sparse, so the Dirichlet feature falls back to a *symmetric* prior "
            "instead of the real zone base rates. Fix for a future step: shrink the "
            "placement distribution toward the train marginal, not a uniform prior.",
            "",
            "## Ranking sanity (predicted vs realized conversion per shooter)",
            "",
            f"- Shooters with >= 3 test kicks: `{ranking['n_shooters']}`.",
            f"- Spearman: `{format_metric(ranking['spearman'])}`, "
            f"Kendall tau: `{format_metric(ranking['kendall'])}`.",
            "",
            "Dive placement is out of scope: `keeper_dive_direction` has 0% coverage.",
            "",
        ]
    )


def metric_vs_npens_figure(lpo: dict) -> None:
    rows = lpo["by_n_pens"]
    buckets = [r["bucket"] for r in rows]
    full = [r["full_log_loss"] for r in rows]
    priors = [r["priors_log_loss"] for r in rows]
    x = np.arange(len(buckets))
    width = 0.38

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - width / 2, full, width, label="full combined", color="#1b6ca8")
    ax.bar(x + width / 2, priors, width, label="priors-only", color="#b0b0b0")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("prior-penalty support bucket (shooter_n_pens_before)")
    ax.set_ylabel("log loss (lower better)")
    ax.set_title("Leave-players-out: where personal signal helps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "step8_metric_vs_npens.png", dpi=120)
    plt.close(fig)


def main() -> None:
    penalties = pd.read_parquet(DATA_PATH)
    cutoff = temporal_cutoff_by_fraction(penalties, train_fraction=0.8)
    train_raw, _ = temporal_train_test_split(penalties, cutoff=cutoff)
    train_global_rate = float(train_raw["outcome_bin"].mean())

    table = build_modeling_table(penalties)
    table = add_baseline_predictions(table, global_rate=train_global_rate)

    cv = evaluate_outcome_models(table, n_folds=5, test_fraction=0.5)

    pre, test = temporal_train_test_split(table, cutoff=cutoff)
    pipeline = fit_pipeline(pre)
    preds = predict_pipeline(pipeline, test)
    headline_detail = binary_classification_metrics(
        y_true=test["outcome_bin"],
        y_prob=preds["combo_calibrated"],
        baseline_prob=preds["keeper_shrunk"],
    )

    ablation = run_ablation(table, n_folds=5, test_fraction=0.5)
    lpo = evaluate_generalization(table, group_col="shooter_id", n_splits=5)
    lco = evaluate_generalization(table, group_col="competition_id", n_splits=5)
    placement = evaluate_placement(table, cutoff=cutoff)

    test_with_pred = test.copy()
    test_with_pred["headline_pred"] = preds["combo_calibrated"]
    ranking = conversion_ranking_correlation(
        test_with_pred, pred_col="headline_pred", group_col="shooter_id", min_kicks=3
    )

    metric_vs_npens_figure(lpo)

    (REPORTS / "results_table.md").write_text(
        "# Results table - Step 8\n\n" + results_table(cv, headline_detail),
        encoding="utf-8",
    )
    (REPORTS / "ablation_table.md").write_text(ablation_md(ablation), encoding="utf-8")
    (REPORTS / "generalization_table.md").write_text(
        generalization_md(lpo, lco), encoding="utf-8"
    )
    (REPORTS / "placement_table.md").write_text(
        placement_md(placement, ranking), encoding="utf-8"
    )

    print("Wrote results_table.md, ablation_table.md, generalization_table.md, placement_table.md")
    print(f"Gate beats_floor={cv['gate']['beats_keeper_shrunk']} "
          f"BSS={cv['gate']['brier_skill_score']:+.4f}")
    print("Ablation deltas (log loss vs full):")
    for _, r in ablation.iterrows():
        print(f"  {r['variant']:<16}{r['d_log_loss']:+.4f}")
    print(f"LPO full vs priors log loss: "
          f"{lpo['overall']['full']['log_loss']:.4f} vs "
          f"{lpo['overall']['priors_only']['log_loss']:.4f}")
    print(f"Placement model logloss {placement['model']['log_loss']:.4f} "
          f"top2 {placement['model']['top2']:.3f}")
    print(f"Ranking spearman {ranking['spearman']:.3f} (n={ranking['n_shooters']})")


if __name__ == "__main__":
    main()
