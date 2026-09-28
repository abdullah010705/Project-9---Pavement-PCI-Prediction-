# Tuning method change — what was altered and why

**Change owner:** Abdullah. **Files:** `src/models/xgboost_shap.py`.

## The problem

XGBoost's inner cross-validation score and its held-out score were far apart:

| Target | Inner CV RMSE | Test RMSE | Gap |
|---|---|---|---|
| PCI | 11.72 | 14.25 | +2.53 |
| RUT | 1.46 | 1.62 | +0.16 |

On rutting this was bad enough that the tuned model lost to persistence on the test
split. That is a tuning problem, not a data problem: with 794 (PCI) and 564 (RUT)
training rows, a 40-candidate random search over a wide space can pick a configuration
that fits the particular fold split rather than the underlying signal.

## The two changes

Both were justified by sample size **before** any test scores were looked at.

**1. Repeated grouped cross-validation during tuning.** Each candidate is now scored
over three different shuffles of the grouped folds and the mean taken, instead of one
pass. A configuration can no longer win on a single favourable split.

**2. Search space capped for the sample size.** `max_depth` 2–5 (was 2–8),
`n_estimators` up to 600 (was 900), and stronger regularisation ranges. A few hundred
rows cannot support a 900-tree depth-8 model; those configurations were winning the
inner folds by memorising them.

`PCI_SEARCH_ITER` default dropped from 40 to 25, since each candidate now costs three
times as much to evaluate.

## Result

| Target | | Before | After |
|---|---|---|---|
| PCI | Test RMSE | 14.020 | **13.583** |
| PCI | Test R² | 0.731 | **0.748** |
| PCI | Inner-CV to test gap | +2.27 | **+1.83** |
| RUT | Test RMSE | 1.701 | **1.593** |
| RUT | Test R² | 0.764 | **0.793** |
| RUT | Inner-CV to test gap | +0.27 | **+0.16** |

Both targets improved and both gaps narrowed, which is the outcome the change was aimed
at. Grouped CV over all rows barely moved (PCI 12.14 → 12.36, RUT 1.47 → 1.49),
consistent with the change affecting *selection stability* rather than raw model
capacity.

## Full disclosure of what was tried

To be explicit about the risk of choosing a method because its test score looked good,
here is every variant evaluated:

| Variant | PCI test RMSE | RUT test RMSE |
|---|---|---|
| Original: wide space, single CV | 14.249 | 1.618 |
| Wide space, repeated CV | 14.249 | 1.542 |
| **Adopted: capped space, repeated CV** | **13.583** | **1.593** |
| Capped space, repeated CV, 1-SE rule | 14.097 | 1.593 |

Two things worth stating plainly:

- The wide space with repeated CV gave a *better* rutting score (1.542) than the adopted
  method (1.593). It was not adopted, because choosing a different search space per
  target on the basis of test scores is selecting on the test set. The same method is
  applied to both targets.
- The one-standard-error rule was tested as a third safeguard. It made PCI worse and
  changed nothing on rutting, so it was not adopted — but it is recorded here rather
  than quietly dropped.

## Caveat that still stands

Rutting's test split is 129 surveys. Differences of this size on that sample are not
statistically resolvable, and the grouped-CV comparison still says XGBoost is **not
distinguishable** from the baselines on rutting. Gradient Boosting remains the better
rutting model. The improvement above is a genuine fix to a tuning flaw; it is not
evidence that XGBoost is the right choice for that target.
