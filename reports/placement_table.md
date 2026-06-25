# Placement & ranking sanity - Step 8

Six-zone placement forecast (as-of shrunk zone distribution) on the temporal test split (n=297).

| forecast | multiclass log loss | top-1 | top-2 | macro-F1 |
| --- | --- | --- | --- | --- |
| model (as-of) | 1.7291 | 0.438 | 0.593 | 0.164 |
| train marginal | 1.3419 | 0.424 | 0.788 | 0.099 |
| uniform | 1.7918 | 0.424 | 0.502 | 0.099 |

Honest finding: the as-of model beats uniform and edges top-1/macro-F1, but the **train marginal beats it on multiclass log loss**. Most shooters are sparse, so the Dirichlet feature falls back to a *symmetric* prior instead of the real zone base rates. Fix for a future step: shrink the placement distribution toward the train marginal, not a uniform prior.

## Ranking sanity (predicted vs realized conversion per shooter)

- Shooters with >= 3 test kicks: `10`.
- Spearman: `0.6877`, Kendall tau: `0.6875`.

Dive placement is out of scope: `keeper_dive_direction` has 0% coverage.
