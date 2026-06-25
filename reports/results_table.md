# Results table - Step 8

## Outcome results (rolling temporal CV, 5 folds)

Mean +/- std across folds. Log loss and Brier are the decision metrics; AUC/ECE shown for the headline on the single 80/20 temporal split.

| model | log loss | Brier |
| --- | --- | --- |
| global rate (baseline) | 0.5943 +/- 0.0421 | 0.2020 +/- 0.0191 |
| keeper-shrunk (baseline floor) | 0.5868 +/- 0.0307 | 0.1989 +/- 0.0142 |
| composed placement/resolution | 0.5932 +/- 0.0410 | 0.2016 +/- 0.0187 |
| combined log-odds | 0.5844 +/- 0.0383 | 0.1982 +/- 0.0173 |
| meta-blend (diagnostic) | 0.5942 +/- 0.0386 | 0.2023 +/- 0.0173 |
| combined log-odds calibrated (HEADLINE) | 0.5844 +/- 0.0383 | 0.1982 +/- 0.0173 |

- **Headline beats keeper-shrunk floor (log loss AND Brier): `True`**, BSS `+0.0033`, folds won `4/5`.

Headline detailed metrics (single 80/20 temporal split - for AUC/ECE only; this single split is the one noisy fold the headline loses, which is exactly why the 5-fold rolling CV above is the decision metric, not one split):

| metric | value |
| --- | --- |
| log loss | 0.6027 |
| Brier | 0.2062 |
| Brier skill vs floor | -0.0102 |
| ROC-AUC | 0.5564 |
| PR-AUC | 0.7418 |
| ECE | 0.0332 |
| accuracy (footnote) | 0.7071 |
