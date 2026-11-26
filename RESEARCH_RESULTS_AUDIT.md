# SENTINEL — Research Results Audit

## Scope

This audit records the saved NSL-KDD native five-family benchmark bundled with this release. It is a single saved experiment using the Decision Tree configuration and the official train/test files. The three approaches were evaluated on the same held-out test records.

## Measured saved results

| Approach | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Average features processed |
|---|---:|---:|---:|---:|---:|---:|
| All features | 72.88% | 62.98% | 51.48% | 51.01% | 69.38% | 122 |
| Fixed feature selection | 72.48% | 66.57% | 51.65% | 51.69% | 68.96% | 42 |
| Dynamic feature selection | **76.98%** | **68.50%** | **55.15%** | **52.98%** | **72.87%** | 54 |

Dynamic feature selection is the measured leader in accuracy, macro precision, macro recall, macro F1 and weighted F1 in this saved NSL-KDD run. It processes approximately 55.7% fewer features than the all-feature approach (54 versus 122).

It does **not** use fewer features than fixed selection (54 versus 42), and its recorded model-only inference time (0.01955 s for the test set) is slower than fixed selection (0.00848 s) and slightly slower than all features (0.01907 s). Therefore this saved experiment does not establish that Dynamic wins every resource metric. The UI now displays feature counts and model-only inference per record alongside the accuracy/F1 metrics so this trade-off is visible.

## Important distinction: poster figures

The earlier project poster labels its results table **Illustrative** and lists 96.10% / 97.35% / 98.72% accuracy for All / Fixed / Dynamic. Those figures are illustrative targets in the poster, not measured outputs from the saved NSL-KDD experiment above. They must not be presented as experimental results unless a reproducible run using the stated protocol produces them.

## UNSW-NB15 binary result

The separate 90.83% UNSW-NB15 binary LightGBM validation is a different task from the native 10-category UNSW-NB15 classifier and from the NSL-KDD five-family task. It must remain explicitly labelled binary and must not be mixed into the native multiclass benchmark.

## Inference correctness fix

Dynamic manual prediction now passes expert inputs through the same model-specific feature conversion used during training and batch evaluation. This is important for CatBoost's categorical feature representation and prevents a train/inference representation mismatch.

## Next legitimate optimization work

- Add an internal validation split from the training file for router/expert feature-budget and hyperparameter selection.
- Compare feature budgets and router/expert classifier choices using validation macro F1 with an explicit feature/latency cost constraint.
- Retrain the final selected pipeline on the full official training split.
- Evaluate once on the untouched official test split.
- Keep per-class recall and resource metrics visible; do not select a model by test-set performance.
