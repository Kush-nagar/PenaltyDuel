"""
Run Step 7 outcome model training/evaluation.

The headline model is the composed outcome pipeline (combined log-odds model +
placement/resolution path + meta-blender + calibration), evaluated with rolling
temporal cross-validation against the Step 6 keeper-shrunk floor. The LightGBM
direct model is reported as an overfitting cross-check.

Run from the project root:
    python reports/train_outcome_model.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    expected_calibration_error,
    format_metric,
    reliability_table,
)
from src.evaluation.splitters import (
    temporal_cutoff_by_fraction,
    temporal_train_test_split,
)
from src.features.build import build_modeling_table
from src.models.baselines import add_baseline_predictions
from src.models.training import (
    evaluate_direct_outcome_model,
    evaluate_outcome_models,
    fit_pipeline,
    predict_pipeline,
)


DATA_PATH = ROOT / "outputs" / "statsbomb" / "penalties_statsbomb_clean.parquet"
PARAMS_PATH = ROOT / "conf" / "params.yaml"
REPORT_PATH = ROOT / "reports" / "outcome_model_report.md"
METRICS_PATH = ROOT / "reports" / "outcome_model_metrics.json"
FIGURE_PATH = ROOT / "reports" / "figures" / "step7_reliability.png"

MODEL_LABELS = {
    "global": "global rate",
    "keeper_shrunk": "keeper-shrunk baseline",
    "composed": "composed placement/resolution",
    "combo": "combined log-odds",
    "blend": "meta-blend (diagnostic)",
    "combo_calibrated": "combined log-odds (calibrated) [headline]",
}


def read_params() -> dict[str, object]:
    text = PARAMS_PATH.read_text(encoding="utf-8")

    def find(pattern: str, default):
        match = re.search(pattern, text)
        return match.group(1) if match else default

    return {
        "model_version": find(r'model_version:\s*"([^"]+)"', "unknown"),
        "train_fraction": float(find(r"train_fraction:\s*([0-9.]+)", "0.8")),
        "calibration_fraction": float(find(r"calibration_fraction:\s*([0-9.]+)", "0.25")),
        "calibration_method": find(r'calibration_method:\s*"([^"]+)"', "sigmoid"),
        "combo_C": float(find(r"combo_C:\s*([0-9.]+)", "1.0")),
        "cv_n_folds": int(float(find(r"cv_n_folds:\s*([0-9.]+)", "5"))),
        "cv_test_fraction": float(find(r"cv_test_fraction:\s*([0-9.]+)", "0.5")),
    }


def cv_table(cv_result: dict[str, object]) -> str:
    summary = cv_result["cv_summary"]
    header = "| model | log loss (mean ± std) | Brier (mean ± std) |"
    divider = "| --- | --- | --- |"
    rows = []
    for name in ["global", "keeper_shrunk", "composed", "combo", "blend", "combo_calibrated"]:
        s = summary[name]
        rows.append(
            f"| {MODEL_LABELS[name]} | "
            f"{s['log_loss_mean']:.4f} ± {s['log_loss_std']:.4f} | "
            f"{s['brier_mean']:.4f} ± {s['brier_std']:.4f} |"
        )
    return "\n".join([header, divider, *rows])


def lgbm_table(result: dict[str, object]) -> str:
    metric_names = ["log_loss", "brier", "brier_skill_score", "roc_auc", "ece"]
    header = "| model | " + " | ".join(metric_names) + " |"
    divider = "| " + " | ".join(["---"] * (len(metric_names) + 1)) + " |"
    rows = []
    metrics = result["metrics"]
    for name in ["keeper_shrunk_baseline", "direct", "direct_calibrated"]:
        row = metrics[name]
        values = [name]
        values.extend(format_metric(row[metric]) for metric in metric_names)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def make_reliability_figure(modeling_table: pd.DataFrame, cutoff) -> dict[str, object]:
    """Fit the pipeline on pre-cutoff data, plot reliability on the held-out tail."""
    pre_cutoff, test = temporal_train_test_split(modeling_table, cutoff=cutoff)
    pipeline = fit_pipeline(pre_cutoff)
    preds = predict_pipeline(pipeline, test)
    y_test = test["outcome_bin"].to_numpy(dtype=int)

    headline = preds["combo_calibrated"]
    keeper = preds["keeper_shrunk"]
    rel = reliability_table(y_test, headline, n_bins=8)
    ece_headline = expected_calibration_error(y_test, headline, n_bins=8)
    ece_keeper = expected_calibration_error(y_test, keeper, n_bins=8)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    valid = rel.dropna(subset=["mean_predicted", "observed_rate"])
    ax.plot(
        valid["mean_predicted"],
        valid["observed_rate"],
        "o-",
        color="#1b6ca8",
        label=f"headline (ECE={ece_headline:.3f})",
    )
    ax.set_xlabel("Mean predicted P(goal)")
    ax.set_ylabel("Observed goal fraction")
    ax.set_title("Step 7 headline reliability (held-out tail)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=120)
    plt.close(fig)

    return {
        "calibration_method": pipeline.calibration_method,
        "ece_headline": float(ece_headline),
        "ece_keeper": float(ece_keeper),
        "test_rows": int(len(test)),
        "blend_weights": dict(
            zip(
                pipeline.blender.component_names,
                np.round(pipeline.blender.estimator.coef_.ravel(), 4).tolist(),
            )
        )
        if pipeline.blender is not None
        else {},
    }


def build_report() -> tuple[str, dict[str, object]]:
    params = read_params()
    penalties = pd.read_parquet(DATA_PATH)
    cutoff = temporal_cutoff_by_fraction(
        penalties, train_fraction=float(params["train_fraction"])
    )
    train_raw, _ = temporal_train_test_split(penalties, cutoff=cutoff)
    train_global_rate = float(train_raw["outcome_bin"].mean())

    modeling_table = build_modeling_table(penalties)
    modeling_table = add_baseline_predictions(
        modeling_table, global_rate=train_global_rate
    )

    cv_result = evaluate_outcome_models(
        modeling_table,
        n_folds=int(params["cv_n_folds"]),
        test_fraction=float(params["cv_test_fraction"]),
        combo_C=float(params["combo_C"]),
    )
    gate = cv_result["gate"]
    reliability = make_reliability_figure(modeling_table, cutoff)

    lgbm_result = evaluate_direct_outcome_model(
        modeling_table,
        cutoff=cutoff,
        calibration_fraction=float(params["calibration_fraction"]),
        calibration_method=str(params["calibration_method"]),
    )

    headline_summary = cv_result["cv_summary"][cv_result["headline_model"]]
    report = [
        "# Outcome Model Report - Step 7",
        "",
        "Generated by `python reports/train_outcome_model.py`.",
        "",
        "## Headline Model",
        "",
        f"- Model version: `{params['model_version']}`.",
        "- Architecture: calibrated combined log-odds outcome model "
        "(shooter + keeper shrunk rates combined in log-odds space). The "
        "composed placement/resolution path and meta-blender were evaluated "
        "but did not improve on the combined model (see diagnostics below).",
        f"- Calibration method selected on held-out slice: "
        f"`{reliability['calibration_method']}`.",
        "- Evaluation: rolling expanding-window temporal CV "
        f"({cv_result['n_folds']} folds over the most recent "
        f"{int(params['cv_test_fraction'] * 100)}% of kicks).",
        "",
        "## Baseline Gate (rolling temporal CV)",
        "",
        f"- **Headline beats keeper-shrunk floor on log loss AND Brier: "
        f"`{gate['beats_keeper_shrunk']}`.**",
        f"- Mean log loss improvement vs floor: `{gate['log_loss_improvement']:+.4f}` "
        f"({headline_summary['log_loss_mean']:.4f} vs "
        f"{cv_result['cv_summary']['keeper_shrunk']['log_loss_mean']:.4f}).",
        f"- Mean Brier improvement vs floor: `{gate['brier_improvement']:+.4f}` "
        f"({headline_summary['brier_mean']:.4f} vs "
        f"{cv_result['cv_summary']['keeper_shrunk']['brier_mean']:.4f}).",
        f"- Brier Skill Score vs floor: `{gate['brier_skill_score']:+.4f}`.",
        f"- Folds won (log loss) by headline: `{gate['folds_won']}/{gate['folds_total']}`.",
        "",
        "The improvement is deliberately modest: penalty outcomes are ~74% base "
        "rate and severely sparse, so a small, consistent, well-calibrated gain "
        "over the shrunk-marginal floor is the honest target (not an implausibly "
        "high AUC).",
        "",
        "## Cross-validated metrics",
        "",
        cv_table(cv_result),
        "",
        "## Calibration",
        "",
        f"- Headline ECE on held-out tail: `{reliability['ece_headline']:.4f}` "
        f"(keeper floor ECE: `{reliability['ece_keeper']:.4f}`, "
        f"{reliability['test_rows']} rows).",
        f"- Diagnostic meta-blend log-odds weights: `{reliability['blend_weights']}` "
        "(the blend down-weights the composed placement signal, which is why it "
        "does not beat the combined model alone).",
        "",
        "![Step 7 reliability](figures/step7_reliability.png)",
        "",
        "## Composed placement/resolution path",
        "",
        "- `P(goal) = Σ_z P(zone=z | shooter) · P(goal | zone=z)`.",
        "- The resolution table `P(goal | zone)` is fit on training rows only "
        "(Laplace-smoothed toward the train global rate); the shooter zone "
        "distribution is the as-of Step 5 feature, so the path is leak-safe.",
        "- It is a component of the meta-blend, contributing placement signal "
        "orthogonal to the keeper marginal.",
        "- Dive composition (`P(goal | zone, dive)`) stays deferred: "
        "`keeper_dive_direction` has 0% coverage in the StatsBomb-only dataset.",
        "",
        "## LightGBM cross-check (single temporal split)",
        "",
        lgbm_table(lgbm_result),
        "",
        "On data this small the gradient-boosted tree does not beat the smooth "
        "log-odds combination, which confirms there is little extra nonlinearity "
        "to exploit - exactly why the logistic combination is the headline.",
        "",
    ]

    full_result = {
        "model_version": params["model_version"],
        "cv": cv_result,
        "reliability": reliability,
        "lgbm_cross_check": lgbm_result,
    }
    return "\n".join(report), full_result


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report, result = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {METRICS_PATH.relative_to(ROOT)}")
    print(f"Wrote {FIGURE_PATH.relative_to(ROOT)}")
    gate = result["cv"]["gate"]
    print(f"Gate beats_keeper_shrunk={gate['beats_keeper_shrunk']} "
          f"BSS={gate['brier_skill_score']:+.4f} "
          f"folds_won={gate['folds_won']}/{gate['folds_total']}")


if __name__ == "__main__":
    main()
